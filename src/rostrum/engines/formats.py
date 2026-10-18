"""Transcript files: words JSON (ours, an older words format, or any JSON through a field mapping), SRT and VTT."""
import json
import re

from rostrum.engines.base import EngineError, Transcript

ARROW = re.compile(r"(\S+)\s*-->\s*(\S+)")
TAG = re.compile(r"<[^>]+>")
FALLBACK_WORD_PATHS = ("words", "segments[].words[]", "")
DURATION_KEYS = ("duration_s",)
SEGMENT_SCORES = ("avg_logprob", "no_speech_prob", "compression_ratio")


class FieldPath:
    """Dotted paths into JSON. A part ending in [] flattens a list: segments[].words[]."""

    def values(self, data, path):
        if not path:
            return [data]
        nodes = [data]
        for part in path.split("."):
            flatten = part.endswith("[]")
            key = part[:-2] if flatten else part
            found = []
            for node in nodes:
                value = self.step(node, key)
                if value is None:
                    continue
                if flatten:
                    if isinstance(value, list):
                        found.extend(value)
                else:
                    found.append(value)
            nodes = found
        return nodes

    def step(self, node, key):
        if not key:
            return node
        if isinstance(node, dict):
            return node.get(key)
        return None

    def first(self, data, path, default=None):
        values = self.values(data, path)
        return values[0] if values else default


class WordText:
    def spaced(self, words):
        """Whisper-style words carry their leading space. Add it when an engine leaves it out."""
        if any(word["word"][:1].isspace() for word in words):
            return words
        return [dict(word, word=" " + word["word"]) for word in words]


class SubtitleCues:
    def parse(self, text):
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        cues = []
        index = 0
        while index < len(lines):
            match = ARROW.search(lines[index])
            if not match:
                index += 1
                continue
            start = self.seconds(match.group(1))
            end = self.seconds(match.group(2))
            index += 1
            body = []
            while (index < len(lines) and
                    lines[index].strip()):
                body.append(lines[index].strip())
                index += 1
            text_line = TAG.sub("", " ".join(body)).strip()
            if (text_line and
                    start is not None and
                    end is not None):
                cues.append((start, end, text_line))
        return cues

    def seconds(self, text):
        parts = text.replace(",", ".").split(":")
        try:
            numbers = [float(part) for part in parts]
        except ValueError:
            return None
        seconds = 0.0
        for number in numbers:
            seconds = seconds * 60 + number
        return seconds


class EstimatedWords:
    """Spreads a cue's time over its words by length. Used when a format has no word times."""

    def of(self, start, end, text):
        parts = text.split()
        if not parts:
            return []
        weights = [len(part) + 1 for part in parts]
        total = sum(weights)
        cursor = start
        words = []
        for part, weight in zip(parts, weights):
            length = (end - start) * weight / total
            words.append({"start": round(cursor, 3), "end": round(cursor + length, 3), "word": " " + part, "probability": 1.0})
            cursor += length
        return words


class TranscriptParser:
    def __init__(self, speech_settings):
        self.fields = speech_settings["fields"]
        self.format = speech_settings["format"]
        self.paths = FieldPath()

    def parse(self, text, name="transcript"):
        kind = self.detect(text) if self.format == "auto" else self.format
        if kind == "json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError as error:
                raise EngineError(f"{name}: not valid JSON (line {error.lineno}): {error.msg}") from error
            return self.from_json(data, name)
        return self.from_cues(SubtitleCues().parse(text), kind, name)

    def detect(self, text):
        stripped = text.lstrip("\ufeff \t\r\n")
        if stripped[:1] in ("{", "["):
            return "json"
        if stripped.startswith("WEBVTT"):
            return "vtt"
        return "srt"

    def from_json(self, data, name):
        fields = self.fields
        scale = float(fields["timeScale"])
        items = self.word_items(data)
        words = []
        for item in items:
            start = self.paths.first(item, fields["start"])
            end = self.paths.first(item, fields["end"])
            text = self.paths.first(item, fields["word"])
            if (start is None or
                    end is None or
                    text is None):
                continue
            probability = self.paths.first(item, fields["probability"])
            words.append({"start": float(start) * scale, "end": float(end) * scale, "word": str(text),
                          "probability": 1.0 if probability is None else float(probability)})
        segments = self.segments(data, scale)
        meta = {"format": "json"}
        if (not words and
                segments):
            for segment in segments:
                words.extend(EstimatedWords().of(segment["start"], segment["end"], segment["text"]))
            meta["wordTimes"] = "estimated"
        if not words:
            raise EngineError(f"{name}: no words found (fields.words is {fields['words']!r})")
        words.sort(key=lambda word: (word["start"], word["end"]))
        failed = data.get("failed", []) if isinstance(data, dict) else []
        return Transcript(words=WordText().spaced(words), segments=segments, duration=self.duration(data, scale),
                          failed=[list(item) for item in failed], meta=meta)

    def word_items(self, data):
        for path in (self.fields["words"],) + FALLBACK_WORD_PATHS:
            items = self.flat(self.paths.values(data, path))
            if items:
                return items
        return []

    def flat(self, nodes):
        """Word objects from the nodes at a path: a node may be one word or a list of words."""
        items = []
        for node in nodes:
            if isinstance(node, list):
                items.extend(item for item in node if isinstance(item, dict))
            elif isinstance(node, dict):
                items.append(node)
        return items

    def segments(self, data, scale):
        found = []
        for item in self.paths.values(data, self.fields["segments"]):
            if not isinstance(item, list):
                item = [item]
            for segment in item:
                has_times = (isinstance(segment, dict) and
                             "start" in segment and
                             "end" in segment)
                if has_times:
                    found.append(self.segment(segment, scale))
        return found

    def segment(self, segment, scale):
        """Times and text, plus the engine's own confidence figures when it gave them."""
        kept = {"start": float(segment["start"]) * scale, "end": float(segment["end"]) * scale,
                "text": str(segment.get("text", ""))}
        for key in SEGMENT_SCORES:
            if isinstance(segment.get(key), (int, float)):
                kept[key] = float(segment[key])
        return kept

    def duration(self, data, scale):
        if not isinstance(data, dict):
            return None
        value = self.paths.first(data, self.fields["duration"])
        if value is None:
            value = next((data[key] for key in DURATION_KEYS if key in data), None)
        try:
            return float(value) * scale if value is not None else None
        except (TypeError, ValueError):
            return None

    def from_cues(self, cues, kind, name):
        if not cues:
            raise EngineError(f"{name}: no {kind.upper()} cues found")
        words = []
        segments = []
        for start, end, text in cues:
            segments.append({"start": start, "end": end, "text": text})
            words.extend(EstimatedWords().of(start, end, text))
        return Transcript(words=words, segments=segments, duration=None, meta={"format": kind, "wordTimes": "estimated"})
