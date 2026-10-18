"""import engine: use a transcript you already have (words JSON, SRT or VTT)."""
import hashlib
from pathlib import Path

from rostrum.engines.base import EngineError, SpeechEngine
from rostrum.engines.formats import TranscriptParser

DISCOVER_SUFFIXES = (".words.json", ".json", ".srt", ".vtt")


class ImportEngine(SpeechEngine):
    name = "import"

    def transcribe(self, source):
        path = self.transcript_path(source)
        transcript = TranscriptParser(self.speech).parse(path.read_text(encoding="utf-8-sig"), path.name)
        transcript.meta.update({"engine": self.name, "file": path.name})
        return transcript

    def transcript_path(self, source):
        if source.transcript is not None:
            if not Path(source.transcript).is_file():
                raise EngineError(f"transcript not found: {source.transcript}")
            return Path(source.transcript)
        media = Path(source.path)
        for suffix in DISCOVER_SUFFIXES:
            candidate = media.with_name(media.stem + suffix)
            if candidate.is_file():
                return candidate
        tried = ", ".join(media.stem + suffix for suffix in DISCOVER_SUFFIXES)
        raise EngineError(f"no transcript for {media.name}: set speech.transcripts, or put one of {tried} next to it")

    def cache_inputs(self, source):
        path = self.transcript_path(source)
        return {"transcript": hashlib.sha1(path.read_bytes()).hexdigest()}
