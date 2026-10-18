"""The speech-to-text contract: audio goes in, words with start, end and probability come out."""
import os
from dataclasses import dataclass, field
from pathlib import Path

TRANSCRIPT_SCHEMA = "rostrum.transcript/1"
AUTO_WORKERS_MAX = 6
CPUS_PER_WORKER = 4
GB_PER_WORKER = 2.0
GB = 1024 ** 3


class EngineError(RuntimeError):
    pass


class WorkerCount:
    """speech.workers: a number, or auto = min(max(1, CPUs // 4), available RAM GB // 2, 6), and at least 1.
    speech.threadsPerWorker: a number, or auto = CPUs // workers, and at least 1."""

    def resolve(self, value):
        if value != "auto":
            return max(1, int(value))
        cpus = os.cpu_count() or 1
        by_memory = int(self.available_gb() // GB_PER_WORKER)
        return max(1, min(max(1, cpus // CPUS_PER_WORKER), by_memory, AUTO_WORKERS_MAX))

    def threads(self, value, workers):
        if value != "auto":
            return max(1, int(value))
        return max(1, (os.cpu_count() or 1) // workers)

    def available_gb(self):
        try:
            import psutil
        except ImportError:
            return GB_PER_WORKER * AUTO_WORKERS_MAX
        return psutil.virtual_memory().available / GB

    def describe(self, value, threads_value="auto"):
        count = self.resolve(value)
        threads = self.threads(threads_value, count)
        chosen = f"{count} worker(s) x {threads} thread(s)"
        if value != "auto":
            return f"{chosen} (speech.workers {value})"
        return (f"{chosen} (speech.workers auto: {os.cpu_count() or 1} CPUs, "
                f"{self.available_gb():.1f} GB available)")


@dataclass
class Transcript:
    """Words on the source file's own clock: [{"start", "end", "word", "probability"}]."""
    words: list
    segments: list = field(default_factory=list)
    duration: float | None = None
    failed: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return {"schema": TRANSCRIPT_SCHEMA, "duration": self.duration, "meta": self.meta, "failed": self.failed,
                "segments": self.segments, "words": self.words}


class TranscriptData:
    def from_dict(self, data):
        return Transcript(words=data["words"], segments=data.get("segments", []), duration=data.get("duration"),
                          failed=[list(item) for item in data.get("failed", [])], meta=data.get("meta", {}))


@dataclass
class SourceInput:
    index: int
    path: Path
    duration: float
    scratch: Path
    transcript: Path | None = None
    # Seconds of silence before the audio stream starts, so word times are file times.
    audio_offset: float = 0.0


class SpeechEngine:
    """Base class. Engines get the full effective settings; most read only settings["speech"]."""

    name = ""
    supports_cue_pass = False
    # Part of the transcript cache key: bump it when the adapter's output changes for the same input.
    adapter_version = 1

    def __init__(self, settings, tools, report=None):
        self.settings = settings
        self.speech = settings["speech"]
        self.tools = tools
        self.report = report or (lambda message: None)

    def transcribe(self, source):
        raise NotImplementedError

    def check(self):
        """For `rostrum doctor`: {"ok": bool, "detail": str}."""
        return {"ok": True, "detail": "built in"}

    def cache_inputs(self, source):
        """Extra inputs that should invalidate a cached transcript when they change."""
        return {}
