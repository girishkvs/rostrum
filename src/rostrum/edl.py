import hashlib
import json
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

SCHEMA = "rostrum.edl/1"


@dataclass
class Source:
    path: str
    duration: float
    offset: float = 0.0


@dataclass
class Cut:
    id: str
    start: float
    end: float
    type: str
    level: str = "auto"
    decision: str = "cut"
    reason: str = ""
    evidence: str = ""
    confidence: float = 1.0
    decided_by: str = "auto"

    @property
    def length(self):
        return self.end - self.start


@dataclass
class Segment:
    source_index: int
    start: float
    end: float
    timeline_start: float

    @property
    def length(self):
        return self.end - self.start


class Ranges:
    def merge(self, ranges):
        merged = []
        for start, end in sorted((float(a), float(b)) for a, b in ranges if b > a):
            if (merged and
                    start <= merged[-1][1]):
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return merged

    def total(self, ranges):
        return sum(end - start for start, end in self.merge(ranges))

    def complement(self, ranges, start, end):
        gaps = []
        cursor = start
        for range_start, range_end in self.merge(ranges):
            if range_start > cursor:
                gaps.append([cursor, min(range_start, end)])
            cursor = max(cursor, range_end)
        if cursor < end:
            gaps.append([cursor, end])
        return [gap for gap in gaps if gap[1] > gap[0]]

    def intersect_length(self, first, second):
        a = self.merge(first)
        b = self.merge(second)
        i = 0
        j = 0
        total = 0.0
        while (i < len(a) and
               j < len(b)):
            low = max(a[i][0], b[j][0])
            high = min(a[i][1], b[j][1])
            if high > low:
                total += high - low
            if a[i][1] < b[j][1]:
                i += 1
            else:
                j += 1
        return total


class FrameGrid:
    def __init__(self, fps):
        self.fps = fps if isinstance(fps, Fraction) else Fraction(str(fps))

    def snap(self, seconds):
        # Nearest frame in double precision. Edges written at 0.05 s steps land exactly on half
        # frames at 30 fps; this resolves them the same way as the original renderer, so golden
        # renders stay frame-identical.
        return float(round(float(seconds) * float(self.fps)) / self.fps)

    def frame_seconds(self):
        return float(1 / self.fps)


class CutIds:
    """Stable ids: same source, type, rounded range and evidence give the same id across re-runs."""

    def __init__(self, source_key, grid_seconds=0.5):
        self.source_key = source_key
        self.grid_seconds = grid_seconds

    def make(self, cut_type, start, end, evidence=""):
        rounded_start = round(start / self.grid_seconds) * self.grid_seconds
        rounded_end = round(end / self.grid_seconds) * self.grid_seconds
        normalized = " ".join(evidence.lower().split())
        key = f"{self.source_key}|{cut_type}|{rounded_start:.1f}|{rounded_end:.1f}|{normalized}"
        return f"{cut_type}-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}"


class SourceKey:
    """Cheap content key: size plus the first and last MiB. Full hashing a multi-GB take is too slow."""

    def __init__(self, chunk=1024 * 1024):
        self.chunk = chunk

    def of(self, paths):
        digest = hashlib.sha1()
        for path in paths:
            file_path = Path(path)
            size = file_path.stat().st_size
            digest.update(str(size).encode("ascii"))
            with file_path.open("rb") as handle:
                digest.update(handle.read(self.chunk))
                if size > self.chunk:
                    handle.seek(max(0, size - self.chunk))
                    digest.update(handle.read(self.chunk))
        return digest.hexdigest()[:12]


@dataclass
class Edl:
    talk: str
    sources: list
    keep: list
    cuts: list = field(default_factory=list)
    fps: str = ""
    schema: str = SCHEMA

    @property
    def duration(self):
        return sum(source.duration for source in self.sources)

    @property
    def kept(self):
        return Ranges().total(self.keep)

    def removed(self):
        return Ranges().complement(self.keep, 0.0, self.duration)

    def segments(self):
        """Keep ranges are on the talk timeline; split them at file boundaries into per-source segments."""
        segments = []
        timeline = 0.0
        for start, end in Ranges().merge(self.keep):
            for index, source in enumerate(self.sources):
                source_end = source.offset + source.duration
                overlap_start = max(start, source.offset)
                overlap_end = min(end, source_end)
                if overlap_end <= overlap_start:
                    continue
                segments.append(Segment(index, overlap_start - source.offset, overlap_end - source.offset, timeline))
                timeline += overlap_end - overlap_start
        return segments

    def to_dict(self):
        return {
            "schema": self.schema,
            "talk": self.talk,
            "fps": self.fps,
            "sources": [asdict(source) for source in self.sources],
            "keep": [[round(start, 6), round(end, 6)] for start, end in self.keep],
            "cuts": [asdict(cut) for cut in self.cuts],
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class EdlFile:
    def load(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise ValueError(f"{path}: expected schema {SCHEMA}, got {data.get('schema')!r}")
        return Edl(
            talk=data["talk"],
            sources=[Source(**source) for source in data["sources"]],
            keep=[list(pair) for pair in data["keep"]],
            cuts=[Cut(**cut) for cut in data.get("cuts", [])],
            fps=data.get("fps", ""),
        )

    def load_baseline(self, path, duration, source_path=None, talk="baseline"):
        """Reads the pre-rostrum format: {source, keep: [[s, e]], cuts: [{start, end, reason, before, after}]}."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        source = Source(path=str(source_path or data.get("source", "")), duration=float(duration))
        ids = CutIds(Path(path).stem)
        cuts = []
        for item in data.get("cuts", []):
            cut_type = str(item.get("reason", "cut")).split(":")[0].split()[0].lower() or "cut"
            evidence = f"{item.get('before', '')} | {item.get('after', '')}".strip(" |")
            cuts.append(Cut(
                id=ids.make(cut_type, item["start"], item["end"], evidence),
                start=float(item["start"]),
                end=float(item["end"]),
                type=cut_type,
                reason=str(item.get("reason", "")),
                evidence=evidence,
            ))
        return Edl(talk=talk, sources=[source], keep=[list(pair) for pair in data["keep"]], cuts=cuts)
