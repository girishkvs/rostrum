"""First pass: run the detectors, merge and paint the applied cuts, add pause and lead trims, list proposals."""
import bisect
from dataclasses import dataclass, field, replace

from rostrum.detect.candidate import PRIORITY, Candidate, Levels
from rostrum.detect.cues import CueDetector
from rostrum.detect.fillers import FillerDetector
from rostrum.detect.offtalk import OffTalkDetector
from rostrum.detect.pauses import TrimDetector
from rostrum.detect.repeats import RepeatDetector
from rostrum.detect.retakes import RetakeDetector
from rostrum.detect.unclear import UnclearDetector


class Merger:
    """Overlapping applied cuts become one cut, named after the highest-priority type among them."""

    def merge(self, cuts):
        merged = []
        for cut in sorted(cuts, key=lambda item: item.start):
            if (merged and
                    cut.start < merged[-1].end):
                last = merged[-1]
                last.end = max(last.end, cut.end)
                if PRIORITY[cut.type] < PRIORITY[last.type]:
                    last.type = cut.type
                    last.group = cut.group
                last.merged += 1
            else:
                merged.append(replace(cut, merged=1))
        for cut in merged:
            if cut.merged > 1:
                cut.note += f" (+{cut.merged - 1} overlapping match)"
        return merged


class Painter:
    """Where cuts overlap, the higher-priority cut keeps the range and the other is split around it."""

    def __init__(self, min_piece):
        self.min_piece = min_piece

    def paint(self, cuts):
        covered = []
        painted = []
        for cut in sorted(cuts, key=lambda item: (PRIORITY[item.type], item.start)):
            pieces = [(cut.start, cut.end)]
            for low, high in covered:
                pieces = self.subtract(pieces, low, high)
            for start, end in pieces:
                if end - start >= self.min_piece:
                    painted.append(replace(cut, start=start, end=end))
                    covered.append((start, end))
        return sorted(painted, key=lambda item: item.start)

    def subtract(self, pieces, low, high):
        kept = []
        for start, end in pieces:
            if (high <= start or
                    low >= end):
                kept.append((start, end))
                continue
            if low > start:
                kept.append((start, low))
            if high < end:
                kept.append((high, end))
        return kept


class KeepRanges:
    """Keep is everything not cut. A kept sliver shorter than plan.minKeepSec with no word in it becomes a gap cut."""

    def __init__(self, total, word_middles, min_keep):
        self.total = total
        self.middles = word_middles
        self.min_keep = min_keep

    def compute(self, cuts):
        pieces = []
        cursor = 0.0
        for start, end in sorted((cut.start, cut.end) for cut in cuts):
            if start > cursor:
                pieces.append([cursor, start])
            cursor = max(cursor, end)
        if cursor < self.total:
            pieces.append([cursor, self.total])
        keep = []
        gaps = []
        for start, end in pieces:
            if (end - start < self.min_keep and
                    not self.has_words(start, end)):
                gaps.append(Candidate(start, end, "gap", "plan", "high", "tiny leftover between cuts"))
            else:
                keep.append([round(start, 3), round(end, 3)])
        return keep, gaps

    def has_words(self, start, end):
        index = bisect.bisect_left(self.middles, start)
        return (index < len(self.middles) and
                self.middles[index] < end)


class Coverage:
    def share(self, candidate, cuts, empty=1.0):
        length = candidate.end - candidate.start
        if length <= 0:
            return empty
        return self.covered(candidate, cuts) / length

    def ratio(self, candidate, cuts):
        """Like share, but very short candidates are measured against 10 ms."""
        return self.covered(candidate, cuts) / max(0.01, candidate.end - candidate.start)

    def covered(self, candidate, cuts):
        return sum(max(0.0, min(candidate.end, cut.end) - max(candidate.start, cut.start)) for cut in cuts)


class Proposals:
    """Candidates shown with keep selected: not already covered by cuts, one per rounded range."""

    def __init__(self, plan_settings):
        self.max_length = plan_settings["maxProposalSec"]
        self.covered_share = plan_settings["coveredShare"]

    def pick(self, candidates, painted):
        chosen = []
        seen = set()
        for candidate in sorted(candidates, key=lambda item: item.start):
            key = (round(candidate.start, 1), round(candidate.end, 1))
            if (key not in seen and
                    Coverage().share(candidate, painted) < self.covered_share):
                seen.add(key)
                chosen.append(candidate)
        return chosen


@dataclass
class PassResult:
    painted: list
    proposals: list
    keep: list
    content: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def applied(self):
        return [cut for cut in self.painted if cut.type != "gap"]


class Planner:
    def __init__(self, analysis):
        self.analysis = analysis
        self.settings = analysis.settings
        self.levels = Levels(analysis.settings)

    def content_candidates(self):
        """In this order: retakes (with prefix false starts), repeats, cues, off-talk; then fillers."""
        levels = self.levels
        found = []
        if (levels.enabled("retakes") or
                levels.enabled("repeats")):
            found.extend(RetakeDetector(self.analysis).find())
        if levels.enabled("repeats"):
            found.extend(RepeatDetector(self.analysis).find())
        if levels.enabled("cues"):
            found.extend(CueDetector(self.analysis).find())
        if levels.enabled("offtalk"):
            found.extend(OffTalkDetector(self.analysis).find())
        if levels.enabled("fillers"):
            found.extend(FillerDetector(self.analysis).find())
        if levels.enabled("unclear"):
            found.extend(UnclearDetector(self.analysis).find())
        return [candidate for candidate in found if levels.enabled(candidate.group)]

    def word_middles(self):
        return [word.middle for word in self.analysis.words]

    def run(self):
        plan = self.settings["plan"]
        content = self.content_candidates()
        applied = Merger().merge([candidate for candidate in content if self.levels.applies(candidate)])
        trims = TrimDetector(self.analysis, self.levels).find(applied)
        painted = Painter(plan["minPieceSec"]).paint(applied + [trim for trim in trims if self.levels.applies(trim)])
        keep, gaps = KeepRanges(self.analysis.total, self.word_middles(), plan["minKeepSec"]).compute(painted)
        painted = sorted(painted + gaps, key=lambda item: item.start)
        pending = [candidate for candidate in content + trims if self.pending(candidate, plan["maxProposalSec"])]
        proposals = Proposals(plan).pick(pending, painted)
        return PassResult(painted=painted, proposals=proposals, keep=keep, content=content)

    def pending(self, candidate, longest):
        """Not applied, and short enough to propose."""
        return (not self.levels.applies(candidate) and
                candidate.length <= longest)
