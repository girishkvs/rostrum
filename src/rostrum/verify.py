"""Verify a render against its EDL: duration, frame and packet continuity, full decode, every join, A/V offset.

`fast` checks timestamps at every join and measures the audio offset at a sample of joins; `full` measures every
join and writes before/after PNG pairs.
"""
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from rostrum.cutmap import CutMapBuilder, TimeText
from rostrum.media import Prober, Rational, Runner
from rostrum.render import RENDER_DEFAULTS, RENDER_JSON, Clock, Planner, SourcePolicy, StderrReport

VERIFY_JSON = "verify-{mode}.json"
MODES = ("fast", "full")
SIDES = ("before", "after")

VERIFY_DEFAULTS = {
    "mode": "fast",
    "durationToleranceFrames": 1,
    "avLengthToleranceMs": 50,
    "frameStepToleranceTicks": 1,
    "frameCountTolerance": 0,
    "audioStepToleranceTicks": 1,
    # fast mode also runs the full decode check unless this is false; full mode always runs it.
    "fastDecode": True,
    "decodeFrameTolerance": 0,
    "playEndToleranceSec": 0.2,
    "joinVideoStepToleranceMs": 1.0,
    "joinFrameTimeToleranceMs": 2.0,
    "joinAudioStepToleranceMs": 1.0,
    "fastSampleJoins": 12,
    "avOffsetLimitMs": 5.0,
    "avWindowSec": 3.0,
    "avMinWindowSec": 0.5,
    "avEdgeMarginSec": 0.2,
    "avGuardSec": 0.05,
    "avMaxLagSec": 0.1,
    "avSampleRate": 8000,
    # Audio windows starting closer than this to the file start are decoded from 0 instead of seeking.
    "avNoSeekBelowSec": 1.0,
    "avQuietRms": 0.002,
    "avMinCorrelation": 0.5,
    "pngs": True,
    "pngDir": "verify-png",
    "pngWidth": None,
    "workers": 4,
}


class VerifyError(RuntimeError):
    pass


class PacketStream:
    """Packet timestamps of one stream, sorted, without packets flagged discard (such as AAC priming)."""

    def __init__(self, pts, durations, time_base):
        order = np.argsort(np.asarray(pts, dtype=np.int64), kind="stable")
        self.pts = np.asarray(pts, dtype=np.int64)[order]
        self.durations = np.asarray(durations, dtype=np.int64)[order]
        self.time_base = time_base
        self.seconds_per_tick = float(time_base)

    @property
    def count(self):
        return int(self.pts.size)

    @property
    def start(self):
        return float(self.pts[0]) * self.seconds_per_tick if self.count else 0.0

    @property
    def end(self):
        if not self.count:
            return 0.0
        return float(self.pts[-1] + self.durations[-1]) * self.seconds_per_tick

    def time(self, index):
        return float(self.pts[index]) * self.seconds_per_tick

    def step_errors(self, step):
        """Distance of every step between consecutive timestamps from the expected constant step, in ticks."""
        return np.diff(self.pts) - step

    def gap_errors(self):
        """Where each packet starts compared with where the previous one ended, in ticks."""
        return self.pts[1:] - (self.pts[:-1] + self.durations[:-1])


class PacketReader:
    def __init__(self, runner):
        self.runner = runner

    def read(self, path, media):
        result = self.runner.ffprobe([
            "-v", "error", "-show_entries", "packet=stream_index,pts,duration,flags", "-of", "compact=p=0", str(path),
        ])
        wanted = {media.video.index: ([], []), media.audio.index: ([], [])}
        for line in result.stdout.splitlines():
            fields = dict(part.split("=", 1) for part in line.split("|") if "=" in part)
            stream = int(fields.get("stream_index", -1))
            pts = fields.get("pts", "N/A")
            skip = (stream not in wanted or
                    "D" in fields.get("flags", "") or
                    not pts.lstrip("-").isdigit())
            if skip:
                continue
            duration = fields.get("duration", "0")
            wanted[stream][0].append(int(pts))
            wanted[stream][1].append(int(duration) if duration.isdigit() else 0)
        video = PacketStream(*wanted[media.video.index], media.video.time_base)
        audio = PacketStream(*wanted[media.audio.index], media.audio.time_base)
        return video, audio


class Correlator:
    def best_lag(self, window, reference, max_lag):
        """Normalized cross-correlation of window against reference (window + 2 * max_lag long), via FFT.

        Returns (lag index, correlation, RMS of window); lag index max_lag means no offset.
        """
        size = len(window)
        lags = min(2 * max_lag, len(reference) - size)
        if (size == 0 or
                lags < 0):
            return None
        energy = float(np.dot(window, window))
        length = 1 << int(math.ceil(math.log2(len(reference) + size)))
        spectrum = np.fft.rfft(reference, length) * np.conj(np.fft.rfft(window, length))
        products = np.fft.irfft(spectrum, length)[: lags + 1]
        squares = np.concatenate(([0.0], np.cumsum(reference * reference)))
        energies = squares[size: size + lags + 1] - squares[: lags + 1]
        denominator = np.sqrt(np.maximum(energy * energies, 0.0))
        scores = np.divide(products, denominator, out=np.zeros_like(products), where=denominator > 0)
        best = int(np.argmax(scores))
        return best, float(scores[best]), math.sqrt(energy / size)


class AudioWindows:
    """Decodes short mono PCM windows through ffmpeg pipes and measures where output audio sits in the source."""

    def __init__(self, runner, settings):
        self.runner = runner
        self.settings = settings
        self.rate = int(settings["avSampleRate"])

    def pcm(self, path, stream_index, start, duration):
        wanted = int(round(duration * self.rate))
        lead = 0
        if start < 0:
            lead = min(wanted, int(round(-start * self.rate)))
            duration += start
            start = 0.0
        samples = self.decode(path, stream_index, start, duration) if duration > 0 else np.zeros(0)
        data = np.concatenate((np.zeros(lead), samples))[:wanted]
        if len(data) < wanted:
            data = np.concatenate((data, np.zeros(wanted - len(data))))
        return data

    def decode(self, path, stream_index, start, duration):
        """Seeking into the first audio frame of a file (AAC priming) shifts the decode by up to a frame,
        so windows that start near the file start decode from 0 and drop the leading samples instead."""
        near_start = start < float(self.settings["avNoSeekBelowSec"])
        seek = [] if near_start else ["-ss", f"{start:.6f}"]
        length = start + duration if near_start else duration
        result = self.runner.ffmpeg([
            "-v", "error", "-nostdin", *seek, "-i", str(path), "-t", f"{length:.6f}",
            "-map", f"0:{stream_index}", "-ac", "1", "-ar", str(self.rate), "-f", "s16le", "-",
        ], binary_output=True)
        samples = np.frombuffer(result.stdout, dtype="<i2").astype(np.float64) / 32768.0
        if near_start:
            samples = samples[int(round(start * self.rate)):]
        return samples

    def offset(self, output, output_stream, source, source_stream, output_start, source_start, window):
        """offset_ms 0: the output audio at output_start is exactly the source audio at source_start.
        Positive: the output audio comes from later in the source than the EDL says."""
        lag = float(self.settings["avMaxLagSec"])
        lag_samples = int(round(lag * self.rate))
        heard = self.pcm(output, output_stream, output_start, window)
        reference = self.pcm(source, source_stream, source_start - lag, window + 2 * lag)
        best = Correlator().best_lag(heard, reference, lag_samples)
        if best is None:
            return {"status": "short", "offset_ms": None, "correlation": None, "rms": None}
        index, correlation, rms = best
        quiet = (rms < float(self.settings["avQuietRms"]) or
                 correlation < float(self.settings["avMinCorrelation"]))
        return {
            "status": "quiet" if quiet else "measured",
            "offset_ms": None if quiet else round((index - lag_samples) * 1000.0 / self.rate, 3),
            "correlation": round(correlation, 4),
            "rms": round(rms, 5),
        }


class JoinSample:
    def pick(self, count, wanted):
        """About `wanted` joins spread evenly from the first to the last (0-based indexes)."""
        wanted = min(int(wanted), count)
        if wanted <= 0:
            return set()
        if wanted == 1:
            return {0}
        return {int(round(position * (count - 1) / (wanted - 1))) for position in range(wanted)}


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, ok, detail):
        self.items.append({"name": name, "ok": bool(ok), "detail": detail})

    @property
    def ok(self):
        return all(item["ok"] for item in self.items)


@dataclass
class VerifyJob:
    mode: str
    edl: object
    output: Path
    plan: object
    sources: list
    media: object
    clock: Clock
    checks: Checks = field(default_factory=Checks)
    rows: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    video: PacketStream | None = None
    audio: PacketStream | None = None


class Verifier:
    def __init__(self, settings, tools, render_settings=None, report=None):
        """render_settings: used to re-plan the render when the output has no render.json next to it."""
        self.settings = {**VERIFY_DEFAULTS, **(settings or {})}
        self.tools = tools
        self.runner = Runner(tools)
        self.render_settings = render_settings
        self.report = report or StderrReport()
        self.text = TimeText()
        self.rational = Rational()

    def verify(self, edl, output_path, mode):
        job = self.prepare(edl, Path(output_path), mode or self.settings["mode"])
        if self.has_streams(job):
            self.report(f"verify {job.mode}: reading packets of {job.output.name}...")
            job.video, job.audio = PacketReader(self.runner).read(job.output, job.media)
            job.timings["packets"] = job.clock.lap()
            self.stream_checks(job)
            if (job.mode == "full" or
                    self.settings["fastDecode"]):
                self.decode_check(job)
                job.timings["decode"] = job.clock.lap()
            job.rows = self.join_rows(job)
            self.join_timestamp_check(job)
            self.offset_check(job)
            job.timings["offsets"] = job.clock.lap()
            if (job.mode == "full" and
                    self.settings["pngs"]):
                self.png_check(job)
                job.timings["pngs"] = job.clock.lap()
        return self.write(job)

    def prepare(self, edl, output, mode):
        clock = Clock()
        if mode not in MODES:
            raise VerifyError(f"unknown verify mode {mode!r}; use fast or full")
        if not output.is_file():
            raise VerifyError(f"output not found: {output}")
        render_settings = {**RENDER_DEFAULTS, **self.planning_settings(output)}
        prober = Prober(self.tools, render_settings, self.runner)
        sources = [prober.probe(source.path) for source in edl.sources]
        plan = Planner(render_settings).plan(edl, SourcePolicy(render_settings).planning_media(sources))
        job = VerifyJob(mode, edl, output, plan, sources, prober.probe(output, check_vfr=False), clock)
        job.summary.update({
            "fps": self.rational.text(plan.fps),
            "expected_frames": plan.frames,
            "expected_duration": round(plan.duration, 6),
            "segments": len(plan.segments),
            "joins": len(plan.joins()),
        })
        job.timings["plan"] = clock.lap()
        return job

    def planning_settings(self, output):
        recorded = output.parent / RENDER_JSON
        if recorded.is_file():
            data = json.loads(recorded.read_text(encoding="utf-8"))
            if (data.get("output") == output.name and
                    isinstance(data.get("settings"), dict)):
                return data["settings"]
        return dict(self.render_settings or {})

    def write(self, job):
        job.timings["total"] = job.clock.total()
        result = {
            "schema": "rostrum.verify/1",
            "ok": job.checks.ok,
            "mode": job.mode,
            "output": str(job.output),
            "seconds": job.timings["total"],
            "checks": job.checks.items,
            "summary": job.summary,
            "timings": job.timings,
            "joins": job.rows,
        }
        verify_json = job.output.parent / VERIFY_JSON.format(mode=job.mode)
        verify_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["verify_json"] = str(verify_json)
        return result

    def has_streams(self, job):
        present = (job.media.video is not None and
                   job.media.audio is not None)
        if not present:
            detail = f"video: {job.media.video is not None}, audio: {job.media.audio is not None}"
            job.checks.add("Output has video and audio", False, detail)
        return present

    def stream_checks(self, job):
        plan, video, audio = job.plan, job.video, job.audio
        start = min(video.start, audio.start)
        duration = max(video.end, audio.end) - start
        tolerance = float(self.settings["durationToleranceFrames"]) / float(plan.fps)
        difference = duration - plan.duration
        job.checks.add(
            "Duration = kept time",
            abs(difference) <= tolerance + 1e-9,
            f"{duration:.3f} s vs kept {plan.duration:.3f} s (diff {difference:+.3f} s, limit {tolerance:.3f} s)",
        )
        job.checks.add(
            "Audio and video same length",
            abs(video.end - audio.end) <= float(self.settings["avLengthToleranceMs"]) / 1000.0,
            f"video {video.start:.3f}-{video.end:.3f} s, audio {audio.start:.3f}-{audio.end:.3f} s",
        )
        step = self.frame_ticks(plan.fps, video.time_base)
        uneven = int(np.sum(np.abs(video.step_errors(step)) > int(self.settings["frameStepToleranceTicks"])))
        frames_ok = (uneven == 0 and
                     abs(video.count - plan.frames) <= int(self.settings["frameCountTolerance"]))
        job.checks.add(
            f"Video frames continuous, {self.rational.text(plan.fps)} fps",
            frames_ok,
            f"{video.count} frames (expected {plan.frames}), {uneven} uneven steps, "
            f"output rate {self.rational.text(job.media.video.fps)}",
        )
        gaps = audio.gap_errors()
        bad_gaps = int(np.sum(np.abs(gaps) > int(self.settings["audioStepToleranceTicks"])))
        worst_ms = float(np.max(np.abs(gaps))) * audio.seconds_per_tick * 1000.0 if gaps.size else 0.0
        job.checks.add(
            "Audio continuous, no gaps or overlaps",
            bad_gaps == 0,
            f"{audio.count} packets, {bad_gaps} uneven steps, max step error {worst_ms:.2f} ms",
        )
        job.summary.update({
            "duration": round(duration, 6),
            "frames": video.count,
            "video": [round(video.start, 6), round(video.end, 6)],
            "audio": [round(audio.start, 6), round(audio.end, 6)],
            "audio_packets": audio.count,
        })

    def frame_ticks(self, fps, time_base):
        ticks = 1 / (fps * time_base)
        return ticks.numerator if ticks.denominator == 1 else round(ticks)

    def decode_check(self, job):
        self.report(f"verify: full decode of {job.output.name}...")
        clock = Clock()
        result = self.runner.ffmpeg([
            "-v", "error", "-nostdin", "-nostats", "-progress", "pipe:1", "-i", str(job.output),
            "-map", f"0:{job.media.video.index}", "-map", f"0:{job.media.audio.index}", "-f", "null", "-",
        ], check=False)
        seconds = clock.total()
        errors = [line.strip() for line in result.stderr.splitlines() if line.strip()]
        frames = [int(value) for value in re.findall(r"^frame=(\d+)", result.stdout, re.M)]
        times = [int(value) for value in re.findall(r"^out_time_us=(\d+)", result.stdout, re.M)]
        decoded_frames = frames[-1] if frames else -1
        decoded_time = times[-1] / 1_000_000 if times else -1.0
        decoded_cleanly = (result.returncode == 0 and
                           not errors)
        detail = f"{len(errors)} error lines, exit {result.returncode}, {seconds:.1f} s"
        if errors:
            detail += ": " + "; ".join(errors[:3])
        job.checks.add("Full decode, no errors", decoded_cleanly, detail)
        plays_through = ("progress=end" in result.stdout and
                         abs(decoded_frames - job.video.count) <= int(self.settings["decodeFrameTolerance"]) and
                         abs(decoded_time - job.summary["duration"]) <= float(self.settings["playEndToleranceSec"]))
        job.checks.add("Plays start to end", plays_through, f"decoded {decoded_frames} frames to {self.text.hms(decoded_time)}")
        job.summary["decode_seconds"] = seconds

    def join_rows(self, job):
        packet_seconds = self.typical_packet_seconds(job.audio)
        return [self.join_row(job, join, packet_seconds) for join in job.plan.joins()]

    def join_row(self, job, join, packet_seconds):
        video = job.video
        frame = join.output_frame
        row = {
            "join": join.index,
            "kind": join.kind,
            "output_time": round(join.output_time, 6),
            "output_frame": frame,
            "cut_start": round(join.before.timeline_end, 6),
            "cut_end": round(join.after.timeline_start, 6),
            "reasons": self.reasons(job.edl, join),
            "source_before": join.before.source_index,
            "source_after": join.after.source_index,
            "video_step_ms": None,
            "frame_time_error_ms": None,
            "audio_step_error_ms": None,
            "offset_before": "not sampled",
            "offset_after": "not sampled",
            "offset_before_ms": None,
            "offset_after_ms": None,
            "png_before": None,
            "png_after": None,
            "notes": [],
        }
        if 1 <= frame < video.count:
            step_ms = (video.time(frame) - video.time(frame - 1)) * 1000.0
            frame_time_error_ms = (video.time(frame) - video.start - join.output_time) * 1000.0
            row["video_step_ms"] = round(step_ms, 3)
            row["frame_time_error_ms"] = round(frame_time_error_ms, 3)
            video_off = (abs(step_ms - 1000.0 / float(job.plan.fps)) > float(self.settings["joinVideoStepToleranceMs"]) or
                         abs(frame_time_error_ms) > float(self.settings["joinFrameTimeToleranceMs"]))
            if video_off:
                row["notes"].append("video timestamps off at the join")
        else:
            row["notes"].append("join frame not found in the output")
        row["audio_step_error_ms"] = round(self.audio_step_error(job.audio, join.output_time, packet_seconds), 3)
        if row["audio_step_error_ms"] > float(self.settings["joinAudioStepToleranceMs"]):
            row["notes"].append("audio gap or overlap at the join")
        return row

    def typical_packet_seconds(self, audio):
        if not audio.count:
            return 0.0
        values, counts = np.unique(audio.durations, return_counts=True)
        return float(values[int(np.argmax(counts))]) * audio.seconds_per_tick

    def audio_step_error(self, audio, join_time, packet_seconds):
        if (audio.count < 2 or
                packet_seconds <= 0):
            return 0.0
        center = int(math.floor((join_time - audio.start) / packet_seconds))
        first = max(1, center - 1)
        last = min(audio.count - 1, center + 2)
        gaps = audio.gap_errors()[first - 1: last]
        if not gaps.size:
            return 0.0
        return float(np.max(np.abs(gaps))) * audio.seconds_per_tick * 1000.0

    def reasons(self, edl, join):
        if join.kind == "file":
            return ["file boundary"]
        return CutMapBuilder().reasons(edl, join.before.timeline_end, join.after.timeline_start)

    def join_timestamp_check(self, job):
        failed = [row["join"] for row in job.rows if row["notes"]]
        detail = f"{len(job.rows)} joins checked (video frame step, frame time and audio packet step at each join)"
        if failed:
            detail += f"; failed at joins {', '.join(str(number) for number in failed[:10])}"
        job.checks.add("Timestamps continuous at every join", not failed, detail)

    def offset_check(self, job):
        joins = job.plan.joins()
        if job.mode == "full":
            selected = sorted(range(len(joins)))
        else:
            selected = sorted(JoinSample().pick(len(joins), self.settings["fastSampleJoins"]))
        tasks = []
        for position in selected:
            for side in SIDES:
                window = self.window(joins[position], side)
                if window is None:
                    job.rows[position][f"offset_{side}"] = "short"
                    continue
                tasks.append((position, side, window))
        self.report(f"verify: A/V offset at {len(selected)} joins ({len(tasks)} windows)...")
        self.measure(job, tasks)
        self.offset_result(job, selected)

    def window(self, join, side):
        """(output start, source index, source local start, length) of the audio window on one side of a join."""
        guard = float(self.settings["avGuardSec"])
        segment = join.before if side == "before" else join.after
        length = min(float(self.settings["avWindowSec"]), segment.length - float(self.settings["avEdgeMarginSec"]))
        if length < float(self.settings["avMinWindowSec"]):
            return None
        if side == "before":
            return join.output_time - guard - length, segment.source_index, segment.local_end - guard - length, length
        return join.output_time + guard, segment.source_index, segment.local_start + guard, length

    def measure(self, job, tasks):
        windows = AudioWindows(self.runner, self.settings)
        limit = float(self.settings["avOffsetLimitMs"])
        with ThreadPoolExecutor(max_workers=max(1, int(self.settings["workers"]))) as pool:
            futures = []
            for position, side, (output_start, source_index, source_start, length) in tasks:
                source = job.sources[source_index]
                future = pool.submit(windows.offset, job.output, job.media.audio.index, source.path,
                                     source.audio.index, output_start, source_start, length)
                futures.append((position, side, future))
            for position, side, future in futures:
                measured = future.result()
                row = job.rows[position]
                row[f"offset_{side}"] = measured["status"]
                row[f"offset_{side}_ms"] = measured["offset_ms"]
                row[f"correlation_{side}"] = measured["correlation"]
                off_limit = (measured["offset_ms"] is not None and
                             abs(measured["offset_ms"]) > limit)
                if off_limit:
                    row["notes"].append(f"audio {side} the join is off by {measured['offset_ms']:.1f} ms")
                    row["offset_off"] = True

    def offset_result(self, job, selected):
        rows = [job.rows[position] for position in selected]
        statuses = [row[f"offset_{side}"] for row in rows for side in SIDES]
        offsets = [abs(row[f"offset_{side}_ms"]) for row in rows for side in SIDES if row[f"offset_{side}_ms"] is not None]
        bad = [row["join"] for row in rows if row.get("offset_off")]
        limit = float(self.settings["avOffsetLimitMs"])
        worst = max(offsets, default=0.0)
        name = "A/V in sync at every join" if job.mode == "full" else f"A/V in sync ({len(selected)} sampled joins)"
        detail = (f"audio offset vs source measured on {statuses.count('measured')} windows "
                  f"(limit {limit:g} ms, max {worst:.2f} ms), {statuses.count('quiet')} quiet and "
                  f"{statuses.count('short')} short windows skipped")
        if bad:
            detail += f"; off at joins {', '.join(str(number) for number in bad[:10])}"
        job.checks.add(name, not bad, detail)
        job.summary.update({
            "offset_joins": len(selected),
            "offset_windows_measured": statuses.count("measured"),
            "offset_windows_quiet": statuses.count("quiet"),
            "offset_windows_short": statuses.count("short"),
            "max_abs_offset_ms": round(worst, 3),
        })

    def png_check(self, job):
        folder = job.output.parent / str(self.settings["pngDir"])
        folder.mkdir(parents=True, exist_ok=True)
        self.report(f"verify: PNG pairs at {len(job.rows)} joins into {folder}...")
        with ThreadPoolExecutor(max_workers=max(1, int(self.settings["workers"]))) as pool:
            futures = []
            for row in job.rows:
                for side, frame in (("before", row["output_frame"] - 1), ("after", row["output_frame"])):
                    path = folder / f"join-{row['join']:03d}-{side}.png"
                    futures.append((row, side, path, pool.submit(self.png, job, frame, path)))
            for row, side, path, future in futures:
                if future.result():
                    row[f"png_{side}"] = str(path)
        expected = 2 * len(job.rows)
        written = sum(1 for row in job.rows for side in SIDES if row[f"png_{side}"])
        job.checks.add("PNG pair at every join", written == expected, f"{written} of {expected} PNGs in {folder}")
        job.summary["png_dir"] = str(folder)

    def png(self, job, frame, path):
        """Extracts output frame `frame`: seeks half a frame before it, so the first frame decoded is that one."""
        seek = job.media.video_offset + float((Fraction(frame) - Fraction(1, 2)) / job.plan.fps)
        args = ["-v", "error", "-nostdin", "-y", "-ss", f"{max(0.0, seek):.6f}", "-i", str(job.output),
                "-map", f"0:{job.media.video.index}", "-frames:v", "1", "-update", "1"]
        if self.settings["pngWidth"]:
            args += ["-vf", f"scale={int(self.settings['pngWidth'])}:-2"]
        args.append(str(path))
        result = self.runner.ffmpeg(args, check=False)
        return (result.returncode == 0 and
                path.is_file() and
                path.stat().st_size > 0)
