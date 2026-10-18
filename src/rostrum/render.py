"""Render: frame-exact trim + concat of the kept segments with ffmpeg, plus cutmap.json and render.json.

Keep points are snapped to the source's frame grid (an exact Fraction, so 29.97 stays 30000/1001). Video and
audio are cut at the same instant, audio gets short fades at every join, and the result is re-encoded once.
"""
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from fractions import Fraction
from pathlib import Path

from rostrum import __version__
from rostrum.cutmap import FILE_NAME as CUTMAP_FILE
from rostrum.cutmap import CutMapBuilder, CutMapFile, TimeText
from rostrum.edl import FrameGrid
from rostrum.media import MEDIA_DEFAULTS, Prober, Rational, Runner

RENDER_JSON = "render.json"
EDL_JSON = "edl.json"
RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{n}" for n in range(1, 10)), *(f"LPT{n}" for n in range(1, 10))}
MUXERS = {"mp4": "mp4", "m4v": "mp4", "mov": "mov", "mkv": "matroska"}
FASTSTART_CONTAINERS = {"mp4", "m4v", "mov"}
CHANNEL_LAYOUTS = {1: "mono", 2: "stereo"}
COPYABLE_AUDIO = {"aac", "mp3", "opus", "alac", "ac3", "eac3"}
# ffprobe prints times to the microsecond, so a stream that is exactly N frames long can read a hair short.
FRAME_EPSILON = Fraction(1, 1000)

RENDER_DEFAULTS = {
    "videoCodec": "libx264",
    "crf": 18,
    "preset": "veryfast",
    "profile": "high",
    "pixelFormat": "yuv420p",
    "keyframeSec": 2.0,
    "scThreshold": 0,
    # null copies the first source's color tags; "none" leaves the tag out.
    "colorPrimaries": None,
    "colorTrc": None,
    "colorspace": None,
    "colorRange": None,
    "audioCodec": "aac",
    "audioBitrate": "192k",
    "audioRate": 48000,
    "audioChannels": 2,
    "audioFadeMs": 15,
    "faststart": True,
    "container": "mp4",
    "stripMetadata": True,
    # EBU R128 two-pass loudnorm. null leaves the loudness alone.
    "loudnessLufs": None,
    "loudnessTruePeak": -1.5,
    "loudnessRange": 11.0,
    "loudnessLinear": True,
    "name": "{speaker}-{title}",
    "nameFallback": "{talk}",
    "nameSpace": "_",
    "runDir": "render-{n}",
    "workDir": None,
    "keepWork": False,
    "normalizeVfr": True,
    "normalizeCodec": "libx264",
    "normalizeCrf": 12,
    "normalizePreset": "veryfast",
    "normalizeAudioBitrate": "320k",
    "standardRates": ["24000/1001", "24", "25", "30000/1001", "30", "50", "60000/1001", "60"],
    "maxEndOverrunSec": 2.0,
    "progressPeriodSec": 5,
    "logLevel": "warning",
    "extraOutputArgs": [],
    **MEDIA_DEFAULTS,
}


class RenderError(RuntimeError):
    pass


class StderrReport:
    def __call__(self, message):
        sys.stderr.write(message + "\n")
        sys.stderr.flush()


class Clock:
    def __init__(self):
        self.started = time.monotonic()
        self.last = self.started

    def lap(self):
        now = time.monotonic()
        elapsed = now - self.last
        self.last = now
        return round(elapsed, 3)

    def total(self):
        return round(time.monotonic() - self.started, 3)


class Seconds:
    def arg(self, value):
        """Seconds for ffmpeg options. ffmpeg reads them as whole microseconds and rounds them to the stream's
        time base, so 6 decimals land on the exact frame or sample."""
        return f"{float(value):.6f}"


@dataclass
class PlannedSegment:
    source_index: int
    fps: Fraction
    anchor: Fraction
    start_frame: int
    end_frame: int
    timeline_offset: float
    first_usable: int
    last_usable: int
    index: int = 0
    output_frame: int = 0

    @property
    def frames(self):
        return self.end_frame - self.start_frame

    @property
    def length(self):
        return float(Fraction(self.frames) / self.fps)

    def local_time(self, frame):
        return self.anchor + Fraction(frame) / self.fps

    @property
    def local_start(self):
        return float(self.local_time(self.start_frame))

    @property
    def local_end(self):
        return float(self.local_time(self.end_frame))

    @property
    def timeline_start(self):
        return self.timeline_offset + self.local_start

    @property
    def timeline_end(self):
        return self.timeline_offset + self.local_end

    @property
    def output_start(self):
        return float(Fraction(self.output_frame) / self.fps)

    @property
    def output_end(self):
        return float(Fraction(self.output_frame + self.frames) / self.fps)

    @property
    def starts_at_source_start(self):
        return self.start_frame <= self.first_usable

    @property
    def ends_at_source_end(self):
        return self.end_frame >= self.last_usable


@dataclass
class Join:
    index: int
    before: PlannedSegment
    after: PlannedSegment

    @property
    def output_frame(self):
        return self.after.output_frame

    @property
    def output_time(self):
        return self.after.output_start

    @property
    def kind(self):
        """'cut' when time was removed here; 'file' when the talk just continues in the next source file."""
        half_frame = 0.5 / float(self.before.fps)
        return "cut" if self.after.timeline_start - self.before.timeline_end > half_frame else "file"


@dataclass
class RenderPlan:
    fps: Fraction
    width: int
    height: int
    segments: list
    sources: list
    media: list
    warnings: list = field(default_factory=list)

    @property
    def frames(self):
        return sum(segment.frames for segment in self.segments)

    @property
    def duration(self):
        return float(Fraction(self.frames) / self.fps)

    def joins(self):
        pairs = zip(self.segments, self.segments[1:])
        return [Join(number, before, after) for number, (before, after) in enumerate(pairs, start=1)]


class SourcePolicy:
    """Variable frame rate sources are planned and rendered as constant-rate copies at the nearest standard rate."""

    def __init__(self, settings):
        self.settings = settings
        self.rational = Rational()

    def planning_media(self, media_list):
        planned = []
        for media in media_list:
            self.check_streams(media)
            if not media.vfr:
                planned.append(media)
                continue
            if not self.settings["normalizeVfr"]:
                raise RenderError(
                    f"{media.name} has a variable frame rate ({media.vfr_reason}). Set render.normalizeVfr "
                    "to true to convert it to a constant rate first, or convert it yourself."
                )
            planned.append(self.normalized(media))
        return planned

    def check_streams(self, media):
        if media.video is None:
            raise RenderError(f"{media.name} has no video stream.")
        if media.audio is None:
            raise RenderError(f"{media.name} has no audio stream.")

    def target_rate(self, media):
        mean = media.video.avg_frame_rate or media.video.fps or media.video.r_frame_rate
        rates = [rate for rate in (self.rational.parse(text) for text in self.settings["standardRates"]) if rate]
        if not rates:
            raise RenderError("render.standardRates has no valid rate.")
        if mean is None:
            return rates[0]
        return min(rates, key=lambda rate: abs(rate - mean))

    def normalized(self, media):
        """The media as it will look after normalizing: constant target rate, video starting at the file start.

        The copy holds only the mapped streams, so its video is stream 0 and its audio stream 1.
        """
        target = self.target_rate(media)
        video = replace(media.video, index=0, fps=target, r_frame_rate=target, avg_frame_rate=target, start=media.start)
        audio = replace(media.audio, index=1)
        return replace(media, video=video, audio=audio, vfr=False,
                       vfr_reason=f"normalized to {self.rational.text(target)} fps")


class Planner:
    def __init__(self, settings):
        self.settings = settings
        self.rational = Rational()
        self.text = TimeText()

    def plan(self, edl, media):
        if len(media) != len(edl.sources):
            raise RenderError(f"expected {len(edl.sources)} probed sources, got {len(media)}")
        fps = self.common_rate(media)
        width, height = self.common_size(media)
        warnings = []
        segments = []
        for keep in edl.segments():
            segment = self.snap(keep, edl.sources[keep.source_index], media[keep.source_index], fps, warnings)
            if segment is None:
                continue
            if self.touches_previous(segments, segment):
                segments[-1].end_frame = max(segments[-1].end_frame, segment.end_frame)
                continue
            segments.append(segment)
        if not segments:
            raise RenderError("Nothing to render: no keep range is left after snapping to frames.")
        output_frame = 0
        for index, segment in enumerate(segments):
            segment.index = index
            segment.output_frame = output_frame
            output_frame += segment.frames
        return RenderPlan(fps, width, height, segments, list(edl.sources), list(media), warnings)

    def common_rate(self, media):
        missing = [item.name for item in media if item.fps is None]
        if missing:
            raise RenderError(f"No frame rate found for {', '.join(missing)}.")
        rates = {item.fps for item in media}
        if len(rates) > 1:
            listing = ", ".join(f"{item.name} {self.rational.text(item.fps)} fps" for item in media)
            raise RenderError(
                f"Sources have different frame rates ({listing}). One output needs one constant rate; "
                "convert the sources to the same rate first."
            )
        return rates.pop()

    def common_size(self, media):
        sizes = {(item.video.width, item.video.height) for item in media}
        if len(sizes) > 1:
            listing = ", ".join(f"{item.name} {item.video.width}x{item.video.height}" for item in media)
            raise RenderError(f"Sources have different video sizes ({listing}). Scale them to one size first.")
        return sizes.pop()

    def snap(self, keep, source, media, fps, warnings):
        usable_end = media.usable_end
        if keep.end - usable_end > float(self.settings["maxEndOverrunSec"]):
            raise RenderError(
                f"A keep range ends at {self.text.hms(keep.end)} in {media.name}, but the file only has "
                f"video and audio up to {self.text.hms(usable_end)}."
            )
        grid = FrameGrid(fps)
        anchor = Fraction(media.video_offset).limit_denominator(1_000_000)
        both_streams_start = Fraction(max(media.video_offset, media.audio_offset))
        first_usable = math.ceil((both_streams_start - anchor) * fps - FRAME_EPSILON)
        last_usable = math.floor((Fraction(usable_end) - anchor) * fps + FRAME_EPSILON)
        start_frame = max(self.frame(grid, keep.start - float(anchor), fps), first_usable)
        end_frame = min(self.frame(grid, keep.end - float(anchor), fps), last_usable)
        if end_frame - start_frame < 1:
            warnings.append(
                f"Keep range {self.text.hms(source.offset + keep.start)}-{self.text.hms(source.offset + keep.end)} "
                "is shorter than one frame after snapping; skipped."
            )
            return None
        return PlannedSegment(
            source_index=keep.source_index,
            fps=fps,
            anchor=anchor,
            start_frame=start_frame,
            end_frame=end_frame,
            timeline_offset=source.offset,
            first_usable=first_usable,
            last_usable=last_usable,
        )

    def frame(self, grid, seconds, fps):
        return round(Fraction(grid.snap(seconds)) * fps)

    def touches_previous(self, segments, segment):
        if not segments:
            return False
        previous = segments[-1]
        return (previous.source_index == segment.source_index and
                segment.start_frame <= previous.end_frame)


class FilterGraph:
    """trim/atrim per segment from one decode of each input (split/asplit), audio fades, concat."""

    def __init__(self, settings):
        self.settings = settings
        self.seconds = Seconds()

    def build(self, plan, media, audio_tail, with_video=True):
        lines = []
        video_inputs = self.split(plan, media, "v", lines) if with_video else {}
        audio_inputs = self.split(plan, media, "a", lines)
        last = len(plan.segments) - 1
        pads = []
        for segment in plan.segments:
            number = segment.index
            start = self.seconds.arg(segment.local_time(segment.start_frame))
            end = self.seconds.arg(segment.local_time(segment.end_frame))
            if with_video:
                lines.append(f"{video_inputs[number]}trim=start={start}:end={end},setpts=PTS-STARTPTS[v{number}];")
                pads.append(f"[v{number}]")
            lines.append(f"{audio_inputs[number]}{self.audio_chain(segment, start, end, last)}[a{number}];")
            pads.append(f"[a{number}]")
        count = len(plan.segments)
        if with_video:
            lines.append(f"{''.join(pads)}concat=n={count}:v=1:a=1[vc][ac];")
            fps = plan.fps
            lines.append(f"[vc]fps={fps.numerator}/{fps.denominator},format={self.settings['pixelFormat']}[vout];")
        else:
            lines.append(f"{''.join(pads)}concat=n={count}:v=0:a=1[ac];")
        lines.append(f"[ac]{audio_tail}[aout]")
        return "\n".join(lines)

    def split(self, plan, media, kind, lines):
        by_source = {}
        for segment in plan.segments:
            by_source.setdefault(segment.source_index, []).append(segment.index)
        labels = {}
        for source_index, numbers in by_source.items():
            stream = media[source_index].video if kind == "v" else media[source_index].audio
            pad = f"[{source_index}:{stream.index}]"
            if len(numbers) == 1:
                labels[numbers[0]] = pad
                continue
            outputs = [f"[{kind}{source_index}_{position}]" for position in range(len(numbers))]
            splitter = "split" if kind == "v" else "asplit"
            lines.append(f"{pad}{splitter}={len(numbers)}{''.join(outputs)};")
            labels.update(zip(numbers, outputs))
        return labels

    def audio_chain(self, segment, start, end, last):
        parts = [f"atrim=start={start}:end={end}", "asetpts=PTS-STARTPTS", self.audio_format()]
        length = Fraction(segment.frames) / segment.fps
        fade = min(Fraction(self.settings["audioFadeMs"]) / 1000, length / 2)
        if fade <= 0:
            return ",".join(parts)
        fade_in = (segment.index > 0 or
                   not segment.starts_at_source_start)
        fade_out = (segment.index < last or
                    not segment.ends_at_source_end)
        if fade_in:
            parts.append(f"afade=t=in:st=0:d={self.seconds.arg(fade)}")
        if fade_out:
            parts.append(f"afade=t=out:st={self.seconds.arg(length - fade)}:d={self.seconds.arg(fade)}")
        return ",".join(parts)

    def audio_format(self):
        layout = CHANNEL_LAYOUTS.get(int(self.settings["audioChannels"]))
        text = f"aformat=sample_rates={int(self.settings['audioRate'])}"
        return f"{text}:channel_layouts={layout}" if layout else text


class LoudnessMeter:
    """EBU R128 two-pass loudnorm: pass 1 measures the edited audio, pass 2 applies it during the render."""

    def __init__(self, settings, runner, report):
        self.settings = settings
        self.runner = runner
        self.report = report

    def target(self):
        return (f"I={float(self.settings['loudnessLufs'])}:TP={float(self.settings['loudnessTruePeak'])}"
                f":LRA={float(self.settings['loudnessRange'])}")

    def measure(self, plan, media, inputs, script):
        graph = FilterGraph(self.settings).build(plan, media, f"loudnorm={self.target()}:print_format=json", with_video=False)
        script.write_text(graph, encoding="utf-8")
        args = ["-hide_banner", "-nostdin", "-v", "info", "-nostats"]
        for path in inputs:
            args += ["-i", str(path)]
        args += [self.runner.filter_script_option(), str(script), "-map", "[aout]", "-f", "null", "-"]
        self.report("loudness: measuring the edited audio (pass 1 of 2)...")
        stderr = self.runner.ffmpeg(args).stderr
        return self.parse(stderr)

    def parse(self, stderr):
        key = stderr.rfind('"input_i"')
        start = stderr.rfind("{", 0, key)
        end = stderr.find("}", key)
        if min(key, start, end) < 0:
            raise RenderError("loudnorm printed no measurement.")
        return json.loads(stderr[start:end + 1])

    def usable(self, measured):
        return all(self.finite(measured.get(key)) for key in ("input_i", "input_tp", "input_lra", "input_thresh"))

    def finite(self, value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def apply_filter(self, measured, audio_format):
        linear = "true" if self.settings["loudnessLinear"] else "false"
        return (
            f"loudnorm={self.target()}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}:linear={linear}:print_format=summary,"
            f"aresample={int(self.settings['audioRate'])},{audio_format}"
        )


class VfrNormalizer:
    """Transcodes a variable frame rate source to a constant rate once, into the work dir, and reuses it."""

    def __init__(self, settings, runner, report):
        self.settings = settings
        self.runner = runner
        self.report = report

    def normalize(self, media, target, work_dir):
        source = Path(media.path).resolve()
        stat = source.stat()
        key = hashlib.sha1(f"{source}|{stat.st_size}|{stat.st_mtime_ns}|{target}".encode("utf-8")).hexdigest()[:10]
        output = work_dir / f"{source.stem}.cfr-{target.numerator}-{target.denominator}.{key}.mp4"
        if output.exists():
            self.report(f"normalize: reusing the constant-rate copy {output.name}")
            return output
        work_dir.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(f"{output.stem}.partial.mp4")
        if media.audio.codec in COPYABLE_AUDIO:
            audio = ["-c:a", "copy"]
        else:
            audio = ["-c:a", "aac", "-b:a", str(self.settings["normalizeAudioBitrate"])]
        args = [
            "-hide_banner", "-nostdin", "-y", "-v", str(self.settings["logLevel"]), "-nostats",
            "-progress", "pipe:1", "-stats_period", str(self.settings["progressPeriodSec"]),
            "-i", str(source), "-map", f"0:{media.video.index}", "-map", f"0:{media.audio.index}",
            "-vf", f"fps=fps={target.numerator}/{target.denominator}:start_time=0",
            "-c:v", str(self.settings["normalizeCodec"]), "-preset", str(self.settings["normalizePreset"]),
            "-crf", str(self.settings["normalizeCrf"]), "-pix_fmt", str(self.settings["pixelFormat"]),
            *audio, "-f", "mp4", str(partial),
        ]
        progress = EncodeProgress(self.report, media.duration, f"normalize {source.name}")
        try:
            self.runner.run_with_progress(args, progress)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        partial.rename(output)
        return output


class EncodeProgress:
    def __init__(self, report, total, label):
        self.report = report
        self.total = max(float(total), 1e-6)
        self.label = label
        self.text = TimeText()

    def __call__(self, block):
        micros = block.get("out_time_us") or block.get("out_time_ms") or ""
        if not micros.strip().isdigit():
            return
        done = int(micros) / 1_000_000
        percent = min(100.0, 100.0 * done / self.total)
        speed = block.get("speed", "").strip()
        self.report(f"{self.label}: {percent:5.1f}%  {self.text.hms(done)} of {self.text.hms(self.total)}  speed {speed}")


class EncodeCommand:
    def __init__(self, settings, runner):
        self.settings = settings
        self.runner = runner

    def args(self, plan, inputs, script, partial):
        settings = self.settings
        args = [
            "-hide_banner", "-nostdin", "-y", "-v", str(settings["logLevel"]), "-nostats",
            "-progress", "pipe:1", "-stats_period", str(settings["progressPeriodSec"]),
        ]
        for path in inputs:
            args += ["-i", str(path)]
        args += [self.runner.filter_script_option(), str(script), "-map", "[vout]", "-map", "[aout]"]
        args += ["-c:v", str(settings["videoCodec"])]
        args += self.optional("-preset", settings["preset"])
        args += self.optional("-crf", settings["crf"])
        args += self.optional("-profile:v", settings["profile"])
        args += ["-pix_fmt", str(settings["pixelFormat"])]
        args += ["-g", str(max(1, round(Fraction(str(settings["keyframeSec"])) * plan.fps)))]
        args += self.optional("-sc_threshold", settings["scThreshold"])
        args += self.color_args(plan.media[0].video.color)
        args += ["-c:a", str(settings["audioCodec"])]
        args += self.optional("-b:a", settings["audioBitrate"])
        args += ["-ar", str(int(settings["audioRate"])), "-ac", str(int(settings["audioChannels"]))]
        if settings["stripMetadata"]:
            args += ["-map_metadata", "-1", "-map_chapters", "-1"]
        container = self.container()
        if (settings["faststart"] and
                container in FASTSTART_CONTAINERS):
            args += ["-movflags", "+faststart"]
        args += [str(value) for value in settings["extraOutputArgs"]]
        args += ["-f", MUXERS.get(container, container), str(partial)]
        return args

    def container(self):
        return str(self.settings["container"]).lower().lstrip(".")

    def optional(self, option, value):
        if value in (None, ""):
            return []
        return [option, str(value)]

    def color_args(self, probed):
        pairs = [
            ("-color_primaries", "colorPrimaries", "primaries"),
            ("-color_trc", "colorTrc", "transfer"),
            ("-colorspace", "colorspace", "space"),
            ("-color_range", "colorRange", "range"),
        ]
        args = []
        for option, key, probed_key in pairs:
            value = self.settings.get(key) or probed.get(probed_key)
            if value in (None, "", "none", "unknown"):
                continue
            args += [option, str(value)]
        return args


class OutputName:
    FIELDS = ("talk", "speaker", "title", "event", "date")

    def __init__(self, settings):
        self.settings = settings

    def make(self, fields, talk):
        values = {key: "" for key in self.FIELDS}
        values.update({key: str(value).strip() for key, value in (fields or {}).items() if value is not None})
        values["talk"] = values["talk"] or str(talk or "")
        values["date"] = values["date"] or date.today().isoformat()
        name = self.fill(self.settings["name"], values) or self.fill(self.settings["nameFallback"], values)
        return self.clean(name or "output")

    def fill(self, template, values):
        try:
            text = str(template or "").format_map(values)
        except KeyError as error:
            raise RenderError(
                f"render.name uses an unknown field {{{error.args[0]}}}; use {{talk}}, {{speaker}}, {{title}}, "
                "{event} or {date}."
            ) from error
        except (ValueError, IndexError) as error:
            raise RenderError(f"render.name is not a valid template: {template!r} ({error})") from error
        text = re.sub(r"([-_.])(\s*[-_.])+", r"\1", text)
        return text.strip(" \t-_.")

    def clean(self, name):
        text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", name)
        text = re.sub(r"\s+", str(self.settings["nameSpace"]), text.strip())
        text = text.strip(" .")[:150] or "output"
        if text.split(".")[0].upper() in RESERVED_NAMES:
            text = f"{text}_"
        return text


class UniquePath:
    def make(self, path):
        path = Path(path)
        candidate = path
        number = 2
        while candidate.exists():
            candidate = path.with_name(f"{path.stem}-{number}{path.suffix}")
            number += 1
        return candidate


class RunDirs:
    """Each render gets a new numbered folder (render-1, render-2, ...); nothing is ever overwritten."""

    def __init__(self, settings):
        template = str(settings["runDir"] or "render-{n}")
        self.template = template if "{n}" in template else f"{template}-{{n}}"

    def create(self, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        pattern = re.compile("^" + re.escape(self.template).replace(re.escape("{n}"), r"(\d+)") + "$")
        numbers = [int(match.group(1)) for child in out_dir.iterdir() if (match := pattern.match(child.name))]
        number = max(numbers, default=0) + 1
        while True:
            candidate = out_dir / self.template.format(n=number)
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                number += 1


class RenderFolders:
    """The numbered render folders in a folder (render.runDir, default render-{n}). A failed render keeps a
    render.json with "ok": false."""

    def __init__(self, render_settings):
        template = str(render_settings.get("runDir") or "render-{n}")
        self.template = template if "{n}" in template else f"{template}-{{n}}"

    def all(self, folder):
        """[(number, render folder, ok)], newest first."""
        folder = Path(folder)
        if not folder.is_dir():
            return []
        pattern = re.compile("^" + re.escape(self.template).replace(re.escape("{n}"), r"(\d+)") + "$")
        found = []
        for child in folder.iterdir():
            match = pattern.match(child.name)
            render_json = child / RENDER_JSON
            if (match and
                    render_json.is_file()):
                found.append((int(match.group(1)), child, self.worked(render_json)))
        return sorted(found, key=lambda item: item[0], reverse=True)

    def worked(self, render_json):
        try:
            return bool(json.loads(render_json.read_text(encoding="utf-8")).get("ok", True))
        except (OSError, ValueError):
            return False

    def newest(self, folder):
        """(the newest render that worked or None, the newer renders that failed)."""
        failed = []
        for _, child, ok in self.all(folder):
            if ok:
                return child, failed
            failed.append(child)
        return None, failed


@dataclass
class RenderJob:
    edl: object
    media: list
    planning: list
    plan: RenderPlan
    work_dir: Path
    clock: Clock
    warnings: list
    timings: dict = field(default_factory=dict)
    inputs: list = field(default_factory=list)
    run_dir: Path | None = None
    output: Path | None = None
    scratch: list = field(default_factory=list)
    loudness: dict | None = None


class Renderer:
    def __init__(self, settings, tools, report=None):
        self.settings = {**RENDER_DEFAULTS, **(settings or {})}
        self.tools = tools
        self.runner = Runner(tools)
        self.report = report or StderrReport()
        self.prober = Prober(tools, self.settings, self.runner)
        self.text = TimeText()
        self.rational = Rational()

    def render(self, edl, out_dir, name_fields=None):
        """Renders the EDL into a new numbered folder inside out_dir. Returns paths, counts, timings and warnings."""
        out_dir = Path(out_dir)
        job = self.prepare(edl, out_dir)
        job.run_dir = RunDirs(self.settings).create(out_dir)
        try:
            self.encode(job, name_fields)
            return self.finish(job)
        except BaseException as error:
            self.write_failure(job, error)
            raise

    def prepare(self, edl, out_dir):
        clock = Clock()
        media = [self.prober.probe(source.path) for source in edl.sources]
        policy = SourcePolicy(self.settings)
        planning = policy.planning_media(media)
        plan = Planner(self.settings).plan(edl, planning)
        work_dir = Path(self.settings["workDir"]) if self.settings["workDir"] else out_dir / ".work"
        job = RenderJob(edl, media, planning, plan, work_dir, clock, list(plan.warnings))
        job.timings["plan"] = clock.lap()
        self.report(
            f"plan: {len(plan.segments)} segments, {len(plan.joins())} joins, {self.text.hms(plan.duration)} kept of "
            f"{self.text.hms(edl.duration)}, {self.rational.text(plan.fps)} fps"
        )
        job.inputs = self.inputs(media, policy, work_dir, job.warnings)
        job.timings["normalize"] = clock.lap()
        return job

    def encode(self, job, name_fields):
        name = OutputName(self.settings).make(name_fields, job.edl.talk)
        command = EncodeCommand(self.settings, self.runner)
        extension = command.container()
        job.output = UniquePath().make(job.run_dir / f"{name}.{extension}")
        self.check_not_source(job.output, job.edl)
        job.work_dir.mkdir(parents=True, exist_ok=True)
        audio_tail = self.loudness_tail(job)
        job.timings["loudness"] = job.clock.lap()

        script = UniquePath().make(job.work_dir / f"{job.run_dir.name}-filter.txt")
        script.write_text(FilterGraph(self.settings).build(job.plan, job.planning, audio_tail), encoding="utf-8")
        job.scratch.append(script)
        partial = UniquePath().make(job.run_dir / f"{name}.partial.{extension}")
        args = command.args(job.plan, job.inputs, script, partial)
        self.report(f"render: {job.output.name}, {self.text.hms(job.plan.duration)} of video, preset {self.settings['preset']}")
        try:
            self.runner.run_with_progress(args, EncodeProgress(self.report, job.plan.duration, "render"))
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        partial.rename(job.output)
        job.timings["encode"] = job.clock.lap()

    def loudness_tail(self, job):
        audio_tail = FilterGraph(self.settings).audio_format()
        if self.settings["loudnessLufs"] is None:
            return audio_tail
        meter = LoudnessMeter(self.settings, self.runner, self.report)
        script = UniquePath().make(job.work_dir / f"{job.run_dir.name}-loudness.txt")
        job.scratch.append(script)
        measured = meter.measure(job.plan, job.planning, job.inputs, script)
        if not meter.usable(measured):
            job.warnings.append(f"loudness: the audio is silent (integrated {measured.get('input_i')} LUFS); left unchanged.")
            return audio_tail
        job.loudness = measured
        return meter.apply_filter(measured, audio_tail)

    def finish(self, job):
        run_dir = job.run_dir
        cutmap_path = CutMapFile().save(CutMapBuilder().build(job.plan, job.edl, job.output.name), run_dir / CUTMAP_FILE)
        edl_path = run_dir / EDL_JSON
        self.resolved(job.edl).save(edl_path)
        if not self.settings["keepWork"]:
            self.clean(job.scratch, job.work_dir)
        job.timings["total"] = job.clock.total()
        render_json = run_dir / RENDER_JSON
        render_json.write_text(json.dumps(self.summary(job), indent=2, default=str), encoding="utf-8")
        encode_seconds = job.timings["encode"]
        speed = job.plan.duration / encode_seconds if encode_seconds else None
        speed_text = f" ({speed:.1f}x real time)" if speed else ""
        self.report(f"render: done in {self.text.hms(encode_seconds)}{speed_text}")
        return {
            "ok": True,
            "dir": str(run_dir),
            "output": str(job.output),
            "cutmap": str(cutmap_path),
            "render_json": str(render_json),
            "edl": str(edl_path),
            "fps": self.rational.text(job.plan.fps),
            "frames": job.plan.frames,
            "duration": round(job.plan.duration, 6),
            "segments": len(job.plan.segments),
            "joins": len(job.plan.joins()),
            "size": job.output.stat().st_size,
            "speed": round(speed, 2) if speed else None,
            "timings": job.timings,
            "warnings": job.warnings,
            "loudness": job.loudness,
        }

    def inputs(self, media, policy, work_dir, warnings):
        paths = []
        for item in media:
            if not item.vfr:
                paths.append(Path(item.path))
                continue
            target = policy.target_rate(item)
            message = (f"{item.name} has a variable frame rate ({item.vfr_reason}); converting it to "
                       f"{self.rational.text(target)} fps first.")
            warnings.append(message)
            self.report(f"warning: {message}")
            normalized = VfrNormalizer(self.settings, self.runner, self.report).normalize(item, target, work_dir)
            self.check_normalized(normalized, target, warnings)
            paths.append(normalized)
        return paths

    def check_normalized(self, path, target, warnings):
        result = self.prober.probe(path)
        drifted = (result.vfr or
                   result.fps != target or
                   result.video_offset > 0)
        if drifted:
            warnings.append(f"The constant-rate copy {path.name} is not exactly {self.rational.text(target)} fps from 0 s.")

    def check_not_source(self, output, edl):
        sources = {Path(source.path).resolve() for source in edl.sources}
        if output.resolve() in sources:
            raise RenderError(f"Refusing to write over a source file: {output}")

    def resolved(self, edl):
        sources = [replace(source, path=str(Path(source.path).resolve())) for source in edl.sources]
        return replace(edl, sources=sources)

    def clean(self, scratch, work_dir):
        for path in scratch:
            path.unlink(missing_ok=True)
        try:
            work_dir.rmdir()
        except OSError:
            pass

    def summary(self, job):
        sources = []
        for item, path in zip(job.media, job.inputs):
            data = item.summary()
            if Path(path) != Path(item.path):
                data["rendered_from"] = str(path)
            sources.append(data)
        plan = job.plan
        return {
            "schema": "rostrum.render/1",
            "ok": True,
            "talk": job.edl.talk,
            "output": job.output.name,
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "rostrum": __version__,
            "ffmpeg": self.runner.version(),
            "fps": self.rational.text(plan.fps),
            "width": plan.width,
            "height": plan.height,
            "frames": plan.frames,
            "duration": round(plan.duration, 6),
            "kept_before_snapping": round(job.edl.kept, 6),
            "segments": len(plan.segments),
            "joins": len(plan.joins()),
            "sources": sources,
            "loudness": job.loudness,
            "settings": self.settings,
            "timings": job.timings,
            "warnings": job.warnings,
            "edl": EDL_JSON,
            "cutmap": CUTMAP_FILE,
        }

    def write_failure(self, job, error):
        data = {
            "schema": "rostrum.render/1",
            "ok": False,
            "talk": job.edl.talk,
            "error": str(error) or type(error).__name__,
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "settings": self.settings,
            "warnings": job.warnings,
            "timings": job.timings,
        }
        (job.run_dir / RENDER_JSON).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
