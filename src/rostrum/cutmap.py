"""Cut map: where every kept segment sits in the output, on the talk timeline, and in its source file.

Times are seconds. "timeline" is the talk timeline (all source files back to back); "local" is the time
inside one source file; "output" is the time in the rendered video.
"""
import bisect
import json
import math
from pathlib import Path

SCHEMA = "rostrum.cutmap/1"
FILE_NAME = "cutmap.json"
# EDL cut notes within this distance of a removed range are listed as its reasons.
REASON_MATCH_SLACK_SEC = 0.1
EDGE_TOLERANCE_SEC = 1e-6


class MapError(ValueError):
    pass


class TimeText:
    def parse(self, text):
        """Accepts ss, mm:ss and hh:mm:ss, with decimals on the last part: '90', '23:30', '1:02:03.5'."""
        value = str(text).strip()
        parts = value.split(":")
        usage = f"not a time: {text!r}. Use ss, mm:ss or hh:mm:ss, for example 90.5, 23:30 or 1:02:03.5"
        if (not value or
                len(parts) > 3):
            raise MapError(usage)
        numbers = []
        for position, part in enumerate(parts):
            is_last = position == len(parts) - 1
            if not self.is_number(part, allow_decimals=is_last):
                raise MapError(usage)
            numbers.append(float(part))
        has_minutes_and_seconds = len(numbers) > 1
        if (has_minutes_and_seconds and
                numbers[-1] >= 60):
            raise MapError(f"{text!r}: seconds must be below 60")
        if (len(numbers) == 3 and
                numbers[1] >= 60):
            raise MapError(f"{text!r}: minutes must be below 60")
        total = 0.0
        for number in numbers:
            total = total * 60 + number
        return total

    def is_number(self, part, allow_decimals):
        whole, dot, fraction = part.partition(".")
        if not dot:
            return whole.isdigit()
        return (allow_decimals and
                (whole.isdigit() or whole == "") and
                fraction.isdigit())

    def hms(self, seconds):
        sign = "-" if seconds < 0 else ""
        total_ms = round(abs(seconds) * 1000)
        hours, rest = divmod(total_ms, 3_600_000)
        minutes, rest = divmod(rest, 60_000)
        whole_seconds, millis = divmod(rest, 1000)
        return f"{sign}{hours}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


class CutMapBuilder:
    """Builds cutmap.json data from a render plan (render.RenderPlan) and its EDL."""

    def build(self, plan, edl, output_name):
        return {
            "schema": SCHEMA,
            "talk": edl.talk,
            "output": output_name,
            "fps": f"{plan.fps.numerator}/{plan.fps.denominator}",
            "frames": plan.frames,
            "duration": self.round(plan.duration),
            "timeline_duration": self.round(edl.duration),
            "sources": [
                {"index": index, "path": source.path, "offset": source.offset, "duration": source.duration}
                for index, source in enumerate(edl.sources)
            ],
            "segments": [self.segment(segment) for segment in plan.segments],
            "cuts": self.cuts(plan, edl),
        }

    def segment(self, segment):
        return {
            "index": segment.index,
            "output_start": self.round(segment.output_start),
            "output_end": self.round(segment.output_end),
            "timeline_start": self.round(segment.timeline_start),
            "timeline_end": self.round(segment.timeline_end),
            "source": segment.source_index,
            "local_start": self.round(segment.local_start),
            "local_end": self.round(segment.local_end),
            "output_frame": segment.output_frame,
            "frames": segment.frames,
        }

    def cuts(self, plan, edl):
        half_frame = 0.5 / float(plan.fps)
        removed = []
        cursor = 0.0
        for segment in plan.segments:
            if segment.timeline_start - cursor > half_frame:
                removed.append((cursor, segment.timeline_start, segment.output_start))
            cursor = segment.timeline_end
        if edl.duration - cursor > half_frame:
            removed.append((cursor, edl.duration, plan.duration))
        return [
            {
                "timeline_start": self.round(start),
                "timeline_end": self.round(end),
                "output_time": self.round(output_time),
                "reasons": self.reasons(edl, start, end),
            }
            for start, end, output_time in removed
        ]

    def reasons(self, edl, start, end):
        found = []
        for cut in edl.cuts:
            overlaps = (cut.decision == "cut" and
                        cut.start < end + REASON_MATCH_SLACK_SEC and
                        cut.end > start - REASON_MATCH_SLACK_SEC)
            if (overlaps and
                    cut.reason not in found):
                found.append(cut.reason)
        return found

    def round(self, value):
        return round(float(value), 6)


class CutMapFile:
    def load(self, path):
        target = Path(path)
        if target.is_dir():
            target = target / FILE_NAME
        if not target.is_file():
            raise MapError(f"no cut map at {target}")
        data = json.loads(target.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise MapError(f"{target}: expected schema {SCHEMA}, got {data.get('schema')!r}")
        return CutMap(data, target)

    def save(self, data, path):
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        return Path(path)


class CutMap:
    def __init__(self, data, path=None):
        self.data = data
        self.path = path
        self.segments = data["segments"]
        self.sources = data["sources"]
        self.duration = float(data["duration"])
        self.timeline_duration = float(data["timeline_duration"])
        self.text = TimeText()

    def to_source(self, output_time):
        """Output time -> talk timeline time and (source file, local time)."""
        if not self.segments:
            raise MapError("the cut map has no segments")
        outside = (not math.isfinite(output_time) or
                   output_time < 0 or
                   output_time > self.duration + EDGE_TOLERANCE_SEC)
        if outside:
            raise MapError(f"output time {self.text.hms(output_time)} is outside the output (0 to {self.text.hms(self.duration)})")
        starts = [segment["output_start"] for segment in self.segments]
        segment = self.segments[max(0, bisect.bisect_right(starts, output_time) - 1)]
        delta = output_time - segment["output_start"]
        source = self.sources[segment["source"]]
        return {
            "direction": "output-to-source",
            "in_cut": False,
            "output": output_time,
            "timeline": segment["timeline_start"] + delta,
            "source": segment["source"],
            "source_path": source["path"],
            "local": segment["local_start"] + delta,
            "segment": segment["index"],
        }

    def to_output(self, time, source_index=None):
        """Talk timeline time (or local time in source_index) -> output time, or the cut that removed it."""
        timeline = time if source_index is None else self.timeline_of(source_index, time)
        outside = (not math.isfinite(timeline) or
                   timeline < 0 or
                   timeline > self.timeline_duration + EDGE_TOLERANCE_SEC)
        if outside:
            raise MapError(f"time {self.text.hms(timeline)} is outside the talk (0 to {self.text.hms(self.timeline_duration)})")
        starts = [segment["timeline_start"] for segment in self.segments]
        position = bisect.bisect_right(starts, timeline + EDGE_TOLERANCE_SEC) - 1
        if position >= 0:
            segment = self.segments[position]
            if timeline <= segment["timeline_end"] + EDGE_TOLERANCE_SEC:
                return self.kept(segment, timeline, time, source_index)
        previous = self.segments[position] if position >= 0 else None
        following = self.segments[position + 1] if position + 1 < len(self.segments) else None
        return self.removed(timeline, time, source_index, previous, following)

    def timeline_of(self, source_index, local):
        if not 0 <= source_index < len(self.sources):
            raise MapError(f"source {source_index} does not exist; this talk has {len(self.sources)} source file(s)")
        return float(self.sources[source_index]["offset"]) + local

    def kept(self, segment, timeline, time, source_index):
        delta = max(0.0, timeline - segment["timeline_start"])
        return {
            "direction": "source-to-output",
            "in_cut": False,
            "input": time,
            "input_source": source_index,
            "timeline": timeline,
            "output": min(segment["output_start"] + delta, segment["output_end"]),
            "source": segment["source"],
            "local": segment["local_start"] + delta,
            "segment": segment["index"],
        }

    def removed(self, timeline, time, source_index, previous, following):
        cut_start = previous["timeline_end"] if previous else 0.0
        cut_end = following["timeline_start"] if following else self.timeline_duration
        return {
            "direction": "source-to-output",
            "in_cut": True,
            "input": time,
            "input_source": source_index,
            "timeline": timeline,
            "cut": [cut_start, cut_end],
            "reasons": self.reasons_for(cut_start, cut_end),
            "previous_edge": self.edge(previous, "timeline_end", "output_end"),
            "next_edge": self.edge(following, "timeline_start", "output_start"),
        }

    def edge(self, segment, timeline_key, output_key):
        if segment is None:
            return None
        local_key = "local_end" if timeline_key == "timeline_end" else "local_start"
        return {
            "timeline": segment[timeline_key],
            "output": segment[output_key],
            "source": segment["source"],
            "local": segment[local_key],
            "segment": segment["index"],
        }

    def reasons_for(self, start, end):
        reasons = []
        for cut in self.data.get("cuts", []):
            overlaps = (cut["timeline_start"] < end + EDGE_TOLERANCE_SEC and
                        cut["timeline_end"] > start - EDGE_TOLERANCE_SEC)
            if overlaps:
                reasons.extend(reason for reason in cut.get("reasons", []) if reason not in reasons)
        return reasons
