"""A candidate cut found by a detector, before it gets a stable id and a decision."""
from dataclasses import dataclass, field

# Lower wins when cuts overlap (painting) and names a merged cut.
PRIORITY = {"lead-in": 0, "lead-out": 0, "retake": 1, "slide": 1, "repeat": 2, "cue": 3, "offtalk": 4, "filler": 4,
            "pause": 5, "gap": 6, "unclear": 7}
TRIM_TYPES = ("pause", "lead-in", "lead-out", "gap")


@dataclass
class Candidate:
    start: float
    end: float
    type: str
    group: str
    confidence: str
    note: str
    first_word: int | None = None
    end_word: int | None = None
    kind: str = ""
    fails: list = field(default_factory=list)
    merged: int = 1
    # A level for this candidate alone (a retake said again with different words), instead of its group's.
    level: str | None = None

    @property
    def length(self):
        return self.end - self.start

    @property
    def is_trim(self):
        return self.type in TRIM_TYPES

    def to_dict(self):
        return {"start": self.start, "end": self.end, "type": self.type, "group": self.group,
                "confidence": self.confidence, "note": self.note, "firstWord": self.first_word,
                "endWord": self.end_word, "kind": self.kind, "fails": list(self.fails), "merged": self.merged,
                "level": self.level}


class CandidateFile:
    def from_dict(self, data):
        return Candidate(start=data["start"], end=data["end"], type=data["type"], group=data["group"],
                         confidence=data["confidence"], note=data["note"], first_word=data.get("firstWord"),
                         end_word=data.get("endWord"), kind=data.get("kind", ""), fails=list(data.get("fails", [])),
                         merged=data.get("merged", 1), level=data.get("level"))


class Levels:
    """auto: strong candidates are cut. propose: everything is shown with keep selected. off: not run."""

    def __init__(self, settings):
        self.settings = settings

    def level(self, group):
        group_settings = self.settings.get(group)
        if not isinstance(group_settings, dict):
            return "auto"
        return group_settings.get("level", "auto")

    def of(self, candidate):
        return candidate.level or self.level(candidate.group)

    def enabled(self, group):
        return self.level(group) != "off"

    def applies(self, candidate):
        return (self.of(candidate) == "auto" and
                candidate.confidence == "high")
