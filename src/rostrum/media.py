"""Media probing through ffprobe, and a small ffmpeg/ffprobe process runner.

Times are file-relative seconds: 0 is the start of the file, the way ffmpeg's -ss and filters see it.
"""
import json
import re
import statistics
import subprocess
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

MEDIA_DEFAULTS = {
    # A stream is variable frame rate when avg_frame_rate and r_frame_rate differ by more than this share...
    "vfrRateTolerance": 0.01,
    # ...or when more than vfrJitterRatio of the sampled frame steps are off the median step by more than
    # vfrJitterFrames of a frame.
    "vfrJitterFrames": 0.1,
    "vfrJitterRatio": 0.02,
    "vfrSampleFrames": 300,
}

ERROR_TAIL_LINES = 15


class MediaError(RuntimeError):
    pass


class FfmpegError(RuntimeError):
    pass


class Rational:
    def parse(self, text):
        """'30000/1001' -> Fraction(30000, 1001). Missing or zero rates ('0/0') -> None."""
        if text in (None, "", "N/A"):
            return None
        numerator, _, denominator = str(text).partition("/")
        try:
            value = Fraction(int(numerator), int(denominator)) if denominator else Fraction(numerator)
        except (ValueError, ZeroDivisionError):
            return None
        return value if value > 0 else None

    def text(self, value):
        if value is None:
            return ""
        return f"{value.numerator}/{value.denominator}"


class Runner:
    """Runs ffmpeg and ffprobe with argument lists (no shell), so it behaves the same on every OS."""

    def __init__(self, tools):
        self.tools = tools
        self.paths = {}
        self.version_line = None

    def binary(self, name):
        if name not in self.paths:
            self.paths[name] = self.tools.find(name)
        return self.paths[name]

    def ffmpeg(self, args, binary_output=False, check=True):
        return self.run([self.binary("ffmpeg"), *args], binary_output, check)

    def ffprobe(self, args, check=True):
        return self.run([self.binary("ffprobe"), *args], False, check)

    def run(self, command, binary_output=False, check=True):
        completed = subprocess.run(command, capture_output=True, stdin=subprocess.DEVNULL)
        stdout = completed.stdout if binary_output else completed.stdout.decode("utf-8", errors="replace")
        stderr = completed.stderr.decode("utf-8", errors="replace")
        if (check and
                completed.returncode != 0):
            raise FfmpegError(self.failure(command, completed.returncode, stderr))
        return subprocess.CompletedProcess(command, completed.returncode, stdout, stderr)

    def run_with_progress(self, args, on_progress):
        """Runs ffmpeg with `-progress pipe:1` in args and calls on_progress(block) for every progress block."""
        command = [self.binary("ffmpeg"), *args]
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        errors = []
        reader = threading.Thread(target=self.drain, args=(process.stderr, errors), daemon=True)
        reader.start()
        block = {}
        try:
            for raw in process.stdout:
                key, separator, value = raw.decode("utf-8", errors="replace").strip().partition("=")
                if not separator:
                    continue
                block[key] = value
                if key == "progress":
                    on_progress(block)
                    block = {}
            process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            reader.join(timeout=10)
        stderr = "".join(errors)
        if process.returncode != 0:
            raise FfmpegError(self.failure(command, process.returncode, stderr))
        return stderr

    def drain(self, stream, sink):
        for raw in stream:
            sink.append(raw.decode("utf-8", errors="replace"))

    def failure(self, command, code, stderr):
        tail = "\n".join(stderr.strip().splitlines()[-ERROR_TAIL_LINES:])
        arguments = " ".join(str(part) for part in command[1:])
        if len(arguments) > 600:
            arguments = arguments[:600] + " ..."
        return f"{Path(command[0]).stem} failed (exit {code}): {arguments}\n{tail}".rstrip()

    def version(self):
        if self.version_line is None:
            result = self.ffmpeg(["-hide_banner", "-version"], check=False)
            lines = result.stdout.strip().splitlines()
            self.version_line = lines[0] if lines else "unknown"
        return self.version_line

    def major_version(self):
        """9 for 'ffmpeg version 9.0.2-...'. None for git builds ('N-12345-g...'), which are recent."""
        match = re.search(r"version\s+n?(\d+)\.", self.version())
        return int(match.group(1)) if match else None

    def filter_script_option(self):
        """ffmpeg 7 replaced -filter_complex_script with the -/filter_complex <file> form; 9 removed the old one."""
        major = self.major_version()
        if (major is not None and
                major < 7):
            return "-filter_complex_script"
        return "-/filter_complex"


@dataclass
class VideoStream:
    index: int
    codec: str
    width: int
    height: int
    pix_fmt: str
    time_base: Fraction
    r_frame_rate: Fraction | None
    avg_frame_rate: Fraction | None
    fps: Fraction | None
    start: float
    duration: float | None
    frames: int | None
    color: dict = field(default_factory=dict)


@dataclass
class AudioStream:
    index: int
    codec: str
    sample_rate: int
    channels: int
    channel_layout: str
    time_base: Fraction
    start: float
    duration: float | None


@dataclass
class Media:
    path: str
    format: str
    start: float
    duration: float
    size: int
    video: VideoStream | None = None
    audio: AudioStream | None = None
    vfr: bool = False
    vfr_reason: str = ""
    timing: dict = field(default_factory=dict)

    @property
    def name(self):
        return Path(self.path).name

    @property
    def fps(self):
        return self.video.fps if self.video else None

    @property
    def video_offset(self):
        return self.offset_of(self.video)

    @property
    def audio_offset(self):
        return self.offset_of(self.audio)

    @property
    def usable_end(self):
        """Where both video and audio still have data."""
        ends = [self.end_of(stream) for stream in (self.video, self.audio) if stream is not None]
        return min(ends) if ends else self.duration

    def offset_of(self, stream):
        if stream is None:
            return 0.0
        return max(0.0, stream.start - self.start)

    def end_of(self, stream):
        if (stream is None or
                stream.duration is None):
            return self.duration
        return self.offset_of(stream) + stream.duration

    def stream_ends(self):
        streams = [stream for stream in (self.video, self.audio) if stream is not None]
        return [self.end_of(stream) for stream in streams if stream.duration is not None]

    def summary(self):
        rational = Rational()
        data = {
            "path": self.path,
            "format": self.format,
            "duration": round(self.duration, 6),
            "size": self.size,
            "vfr": self.vfr,
        }
        if self.vfr_reason:
            data["vfr_reason"] = self.vfr_reason
        if self.video:
            data["video"] = {
                "codec": self.video.codec,
                "width": self.video.width,
                "height": self.video.height,
                "pix_fmt": self.video.pix_fmt,
                "fps": rational.text(self.video.fps),
                "r_frame_rate": rational.text(self.video.r_frame_rate),
                "avg_frame_rate": rational.text(self.video.avg_frame_rate),
                "start": self.video.start,
                "duration": self.video.duration,
                "frames": self.video.frames,
            }
        if self.audio:
            data["audio"] = {
                "codec": self.audio.codec,
                "sample_rate": self.audio.sample_rate,
                "channels": self.audio.channels,
                "channel_layout": self.audio.channel_layout,
                "start": self.audio.start,
                "duration": self.audio.duration,
            }
        return data


class Prober:
    def __init__(self, tools, settings=None, runner=None):
        self.runner = runner or Runner(tools)
        self.settings = {**MEDIA_DEFAULTS, **(settings or {})}
        self.rational = Rational()

    def probe(self, path, check_vfr=True):
        file_path = Path(path)
        if not file_path.is_file():
            raise MediaError(f"media file not found: {file_path}")
        result = self.runner.ffprobe(
            ["-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(file_path)],
            check=False,
        )
        if result.returncode != 0:
            raise MediaError(f"ffprobe cannot read {file_path}: {result.stderr.strip()[-500:]}")
        data = json.loads(result.stdout or "{}")
        media = self.media(file_path, data)
        if (check_vfr and
                media.video is not None):
            VfrDetector(self.runner, self.settings).apply(media)
        return media

    def media(self, file_path, data):
        container = data.get("format", {})
        media = Media(
            path=str(file_path),
            format=container.get("format_name", ""),
            start=self.number(container.get("start_time"), 0.0),
            duration=self.number(container.get("duration"), 0.0),
            size=int(container.get("size") or file_path.stat().st_size),
        )
        for stream in data.get("streams", []):
            kind = stream.get("codec_type")
            is_first_video = (kind == "video" and
                              media.video is None and
                              not stream.get("disposition", {}).get("attached_pic"))
            is_first_audio = (kind == "audio" and
                              media.audio is None)
            if is_first_video:
                media.video = self.video_stream(stream)
            elif is_first_audio:
                media.audio = self.audio_stream(stream)
        if media.duration <= 0:
            media.duration = max(media.stream_ends(), default=0.0)
        return media

    def video_stream(self, stream):
        real_rate = self.rational.parse(stream.get("r_frame_rate"))
        average_rate = self.rational.parse(stream.get("avg_frame_rate"))
        frames = stream.get("nb_frames")
        return VideoStream(
            index=int(stream["index"]),
            codec=stream.get("codec_name", ""),
            width=int(stream.get("width") or 0),
            height=int(stream.get("height") or 0),
            pix_fmt=stream.get("pix_fmt", ""),
            time_base=self.rational.parse(stream.get("time_base")) or Fraction(1, 1000),
            r_frame_rate=real_rate,
            avg_frame_rate=average_rate,
            fps=real_rate or average_rate,
            start=self.number(stream.get("start_time"), 0.0),
            duration=self.number(stream.get("duration"), None),
            frames=int(frames) if str(frames or "").isdigit() else None,
            color={
                "primaries": stream.get("color_primaries"),
                "transfer": stream.get("color_transfer"),
                "space": stream.get("color_space"),
                "range": stream.get("color_range"),
            },
        )

    def audio_stream(self, stream):
        return AudioStream(
            index=int(stream["index"]),
            codec=stream.get("codec_name", ""),
            sample_rate=int(stream.get("sample_rate") or 0),
            channels=int(stream.get("channels") or 0),
            channel_layout=stream.get("channel_layout", ""),
            time_base=self.rational.parse(stream.get("time_base")) or Fraction(1, 1000),
            start=self.number(stream.get("start_time"), 0.0),
            duration=self.number(stream.get("duration"), None),
        )

    def number(self, value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


class VfrDetector:
    """Variable frame rate: avg_frame_rate vs r_frame_rate mismatch, or uneven timestamps on sampled frames."""

    def __init__(self, runner, settings):
        self.runner = runner
        self.settings = settings
        self.rational = Rational()

    def apply(self, media):
        video = media.video
        reasons = []
        mismatch = self.rate_mismatch(video)
        rates_disagree = (mismatch is not None and
                          mismatch > float(self.settings["vfrRateTolerance"]))
        if rates_disagree:
            reasons.append(
                f"avg_frame_rate {self.rational.text(video.avg_frame_rate)} differs from "
                f"r_frame_rate {self.rational.text(video.r_frame_rate)} by {mismatch:.1%}"
            )
        jitter = self.jitter(self.sample_steps(media))
        if jitter["ratio"] > float(self.settings["vfrJitterRatio"]):
            reasons.append(f"{jitter['uneven']} of {jitter['steps']} sampled frame steps are uneven")
        if (video.r_frame_rate is None and
                video.avg_frame_rate is None):
            reasons.append("the stream has no frame rate")
        media.vfr = bool(reasons)
        media.vfr_reason = "; ".join(reasons)
        media.timing = jitter
        if media.vfr:
            video.fps = video.avg_frame_rate or video.r_frame_rate
        else:
            video.fps = video.r_frame_rate or video.avg_frame_rate
        return media

    def rate_mismatch(self, video):
        if (video.r_frame_rate is None or
                video.avg_frame_rate is None):
            return None
        return float(abs(video.r_frame_rate - video.avg_frame_rate) / video.avg_frame_rate)

    def sample_steps(self, media):
        count = int(self.settings["vfrSampleFrames"])
        intervals = [f"%+#{count}"]
        sample_seconds = count / float(media.fps or 30)
        if media.duration > 4 * sample_seconds:
            intervals.append(f"{media.start + media.duration / 2:.3f}%+#{count}")
        steps = []
        for interval in intervals:
            result = self.runner.ffprobe([
                "-v", "error", "-select_streams", str(media.video.index), "-read_intervals", interval,
                "-show_entries", "packet=pts", "-of", "csv=p=0", media.path,
            ], check=False)
            stamps = sorted(int(value) for value in self.first_fields(result.stdout) if value.lstrip("-").isdigit())
            steps.extend(later - earlier for earlier, later in zip(stamps, stamps[1:]))
        return steps

    def first_fields(self, text):
        return [line.split(",")[0].strip() for line in text.splitlines() if line.strip()]

    def jitter(self, steps):
        if len(steps) < 3:
            return {"steps": len(steps), "uneven": 0, "ratio": 0.0, "median_step": None}
        median = statistics.median(steps)
        if median <= 0:
            return {"steps": len(steps), "uneven": len(steps), "ratio": 1.0, "median_step": median}
        limit = float(self.settings["vfrJitterFrames"]) * median
        uneven = sum(1 for step in steps if abs(step - median) > limit)
        return {"steps": len(steps), "uneven": uneven, "ratio": uneven / len(steps), "median_step": median}
