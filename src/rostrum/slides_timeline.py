"""Loads existing slide-timeline outputs (<label>-timeline.csv and friends). Detecting slides is a separate step."""
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from rostrum.config import ConfigError

TIMELINE_SUFFIX = "-timeline.csv"
EXTRA_TABLES = ("slides", "reveals", "rewinds", "backjumps")
TIME_COLUMNS = ("start", "end", "time", "final_start", "final_end")


@dataclass
class SlideVisit:
    start: float
    end: float
    slide: int | None
    entry: str = ""
    final: bool = False
    score: float | None = None

    @property
    def duration(self):
        return self.end - self.start


@dataclass
class SlideTimeline:
    """Slide visits on the talk timeline. `slide` is None where no slide matched (OFF)."""
    visits: list
    tables: dict = field(default_factory=dict)
    summaries: list = field(default_factory=list)
    samples: list = field(default_factory=list)

    def slides_present(self):
        return sorted({visit.slide for visit in self.visits if visit.slide})


class SlideName:
    def number(self, text):
        text = (text or "").strip()
        if (text[:1] in ("S", "s") and
                text[1:].isdigit()):
            return int(text[1:])
        return None

    def text(self, number):
        return f"S{number:02d}" if number else "OFF"


class SlideTimelineFiles:
    """`slides.timeline` is a path prefix (<folder>/<label>) or a -timeline.csv file; a list gives one per source."""

    def __init__(self, project_dir):
        self.project_dir = Path(project_dir)

    def load(self, value, sources):
        if not value:
            return None
        prefixes = [value] if isinstance(value, str) else list(value)
        if (len(prefixes) > 1 and
                len(prefixes) != len(sources)):
            raise ConfigError(f"slides.timeline has {len(prefixes)} entries for {len(sources)} sources")
        timeline = SlideTimeline(visits=[])
        for index, prefix in enumerate(prefixes):
            offset = sources[index].offset if index < len(sources) else 0.0
            self.add(timeline, self.prefix(prefix), offset)
        timeline.visits.sort(key=lambda visit: visit.start)
        return timeline

    def prefix(self, value):
        path = Path(value)
        if not path.is_absolute():
            path = self.project_dir / path
        text = str(path)
        if text.endswith(TIMELINE_SUFFIX):
            text = text[:-len(TIMELINE_SUFFIX)]
        return text

    def add(self, timeline, prefix, offset):
        csv_path = Path(prefix + TIMELINE_SUFFIX)
        if not csv_path.is_file():
            raise ConfigError(f"slide timeline not found: {csv_path}")
        names = SlideName()
        for row in self.rows(csv_path):
            timeline.visits.append(SlideVisit(
                start=float(row["start"]) + offset,
                end=float(row["end"]) + offset,
                slide=names.number(row.get("slide")),
                entry=row.get("entry", ""),
                final=row.get("final_pass", "") == "yes",
                score=float(row["score"]) if row.get("score") else None,
            ))
        for table in EXTRA_TABLES:
            path = Path(f"{prefix}-{table}.csv")
            if path.is_file():
                rows = [self.shift(row, offset) for row in self.rows(path)]
                timeline.tables.setdefault(table, []).extend(rows)
        summary = Path(prefix + "-summary.json")
        if summary.is_file():
            timeline.summaries.append(json.loads(summary.read_text(encoding="utf-8")))
        samples = Path(prefix + "-samples.npz")
        if samples.is_file():
            timeline.samples.append({"path": str(samples), "offset": offset})

    def rows(self, path):
        with open(path, newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    def shift(self, row, offset):
        shifted = dict(row)
        for column in TIME_COLUMNS:
            value = shifted.get(column)
            if value not in (None, ""):
                try:
                    shifted[column] = float(value) + offset
                except ValueError:
                    pass
        return shifted


class SlideSamples:
    """Per-frame match scores saved by the slide-timeline tool (t, scores, body, ink). Loaded only on request."""

    def load(self, entry):
        import numpy as np

        data = np.load(entry["path"])
        return {"t": data["t"] + entry["offset"], "scores": data["scores"], "body": data["body"], "ink": data["ink"]}
