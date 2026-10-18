"""Second pass (reconcile):

1. a loose retake is cut when everything it drops is said again soon after (not one said again with swapped words,
   which keeps retakes.substitutionLevel);
2. a stand-alone cue ("Stop.") cuts back over the sentences that are said again after it;
3. with a slide timeline, slide visits that break the slide order or were abandoned are cut, except their unique words.

Pause and lead trims are then recomputed around the final content cuts.
"""
import bisect
from dataclasses import dataclass, replace

from rostrum.detect.candidate import Candidate, Levels
from rostrum.detect.cues import StopCueDetector
from rostrum.detect.pauses import TrimDetector
from rostrum.detect.planner import Coverage, KeepRanges, Merger, Painter, PassResult
from rostrum.detect.retakes import RetakeNotes
from rostrum.slides_timeline import SlideName

RECONSIDERED = ("retake", "repeat", "offtalk")
NO_FINAL_PASS = float("inf")
PIECE_MIN = 0.001
NEIGHBOUR_REACH = 1.0
NOTE_TOLERANCE = 0.2


class RetakePromoter:
    def __init__(self, analysis):
        self.analysis = analysis
        self.settings = analysis.settings["reconcile"]["retakes"]
        self.notes = RetakeNotes()

    def run(self, content):
        owner = self.analysis.unit_of_words()
        promoted = []
        unsure = []
        for candidate in content:
            if (candidate.type not in RECONSIDERED or
                    candidate.confidence == "high"):
                continue
            attempt = owner.get(candidate.first_word)
            retake = owner.get(candidate.end_word)
            stays_proposed = candidate.level == "propose"
            if (candidate.type != "retake" or
                    stays_proposed or
                    attempt is None or
                    retake is None or
                    retake <= attempt):
                unsure.append(candidate)
                continue
            problem = self.problem(candidate, attempt, retake)
            base_note = self.notes.base(candidate)
            if problem is None:
                promoted.append(replace(candidate, confidence="high", note="retake said again later: " + base_note))
            else:
                unsure.append(replace(candidate, note=base_note + f" [{problem}]"))
        return promoted, unsure

    def problem(self, candidate, attempt_index, retake_index):
        analysis = self.analysis
        settings = self.settings
        units = analysis.units
        attempt = units[attempt_index]
        retake = units[retake_index]
        span = retake.start - attempt.start
        opening = settings["sameOpeningWords"]
        same_opening = (len(attempt.tokens) >= opening and
                        attempt.tokens[:opening] == retake.tokens[:opening])
        if span > settings["maxSpanSec"]:
            return f"span > {settings['maxSpanSec']:g} s"
        before = units[attempt_index - 1] if attempt_index > 0 else None
        if (before is not None and
                before.content and
                not analysis.ends_sentence(before.text)):
            return "attempt starts mid-sentence"
        before_retake = units[retake_index - 1]
        if (before_retake.content and
                not analysis.ends_sentence(before_retake.text) and
                not same_opening):
            return "retake starts mid-sentence"
        loose_only = (candidate.kind == "fuzzy" and
                      not same_opening and
                      not analysis.same_script_line(analysis.script_map[attempt_index], analysis.script_map[retake_index]))
        if loose_only:
            return "loose match only"
        return self.unrepeated(candidate.kind, attempt_index, retake_index, span)

    def unrepeated(self, kind, attempt_index, retake_index, span):
        analysis = self.analysis
        settings = self.settings
        units = analysis.units
        region_end = units[retake_index].start + max(settings["regionMinSec"], settings["regionFactor"] * span)
        later = set()
        for unit in units[retake_index:]:
            if unit.start > region_end:
                break
            later |= unit.content
        for index in range(attempt_index, retake_index):
            unit = units[index]
            if (analysis.disposable(unit) or
                    not unit.content):
                continue
            first_and_firm = (index == attempt_index and
                              kind != "fuzzy")
            need = settings["firstUnitShare"] if first_and_firm else settings["unitShare"]
            if len(unit.content & later) / len(unit.content) < need:
                return f"not said again later: \u201c{unit.text[:60]}\u201d"
        return None


@dataclass
class SlidePiece:
    slide: int | None
    start: float
    end: float

    @property
    def length(self):
        return self.end - self.start


@dataclass
class Region:
    visit: object
    status: str
    cut_units: list
    keep_units: list


class SlideEvidence:
    def __init__(self, analysis, timeline, applied):
        self.analysis = analysis
        self.settings = analysis.settings["reconcile"]["slides"]
        self.visits = timeline.visits
        self.applied = applied
        self.names = SlideName()
        self.cut_units = {unit.index for unit in analysis.units if self.covered(unit)}

    def cover(self, start, end):
        return sum(max(0.0, min(end, cut.end) - max(start, cut.start)) for cut in self.applied)

    def covered(self, unit):
        length = max(0.01, unit.end - unit.start)
        return self.cover(unit.start, unit.end) / length >= self.settings["unitCutShare"]

    def is_cut(self, unit):
        return unit.index in self.cut_units

    def spoken_in(self, unit, visit):
        """A sentence said during the visit (by its middle) that no cut removes."""
        middle = (unit.start + unit.end) / 2
        return (visit.start <= middle < visit.end and
                not self.is_cut(unit))

    def later_content(self, start, end):
        content = set()
        for unit in self.analysis.units:
            if (start <= unit.start <= end and
                    not self.is_cut(unit)):
                content |= unit.content
        return content

    def resaid(self, unit, after, window):
        analysis = self.analysis
        settings = self.settings
        horizon = after + max(window, settings["repeatWindowSec"])
        for target, _ in analysis.matches[unit.index]:
            if after <= analysis.units[target].start <= horizon:
                return True, "repeat match"
        if analysis.disposable(unit):
            return True, "filler or aside"
        shared = len(unit.content & self.later_content(after, after + window))
        count = len(unit.content)
        enough = (count > 0 and
                  shared / count >= settings["resaidShare"] and
                  (shared >= settings["resaidMinWords"] or count == 1))
        if enough:
            return True, f"said again ({shared}/{count} key words)"
        return False, f"unique ({shared}/{count} key words said again)"

    def finals_after(self):
        result = []
        for index in range(len(self.visits)):
            following = next((visit for visit in self.visits[index:] if visit.final and visit.slide), None)
            result.append(following.slide if following else NO_FINAL_PASS)
        return result

    def classify(self):
        """Slide visits outside the final passes: keep them in order, or mark them for cutting."""
        settings = self.settings
        finals_after = self.finals_after()
        regions = []
        notes = []
        highest = 0
        for index, visit in enumerate(self.visits):
            slide = visit.slide
            if slide is None:
                continue
            units = [unit for unit in self.analysis.units if self.spoken_in(unit, visit)]
            if visit.final:
                highest = max(highest, slide)
                continue
            keepable = highest <= slide <= finals_after[index]
            window = max(settings["resaidMinWindowSec"], settings["resaidVisitFactor"] * (visit.end - visit.start))
            tests = [(unit, *self.resaid(unit, visit.end, window)) for unit in units]
            with_content = sum(1 for unit, said, _ in tests if unit.content)
            said_again = sum(1 for unit, said, _ in tests if unit.content and said)
            aborted = (with_content > 0 and
                       said_again / with_content >= settings["abortedShare"])
            if (keepable and
                    not aborted and
                    units):
                highest = max(highest, slide)
                continue
            if (keepable and
                    not units):
                status = "silent visit, cut"
            elif keepable:
                status = f"aborted start ({said_again}/{with_content} sentences said again later), cut"
            else:
                status = f"breaks slide order ({said_again}/{with_content} sentences said again later), cut"
            keep_units = [unit for unit, said, _ in tests if not said]
            cut_units = [unit for unit, said, _ in tests if said]
            name = self.names.text(slide)
            for unit, said, why in tests:
                if not said:
                    notes.append(Candidate(unit.start, unit.end, "slide", "slides", "low",
                                           f"{name} visit {visit.start:.1f}-{visit.end:.1f}s ({status}): words not said again, kept: "
                                           f"\u201c{unit.text[:120]}\u201d [{why}]", unit.first, unit.last + 1))
            if (keep_units and
                    not keepable):
                status += "; unique words kept, slide order will break here"
            elif (keepable and
                    keep_units):
                highest = max(highest, slide)
            regions.append(Region(visit, status, cut_units, keep_units))
        return regions, notes

    def snap(self, time, low, high):
        energy = self.analysis.energy
        radius = self.settings["snapRadiusSec"]
        first = int(max(low, time - radius) / energy.frame)
        last = int(min(high, time + radius) / energy.frame)
        if last <= first:
            return time
        frames = energy.db[first:last]
        return (first + int(frames.argmin()) + 0.5) * energy.frame

    def region_cuts(self, regions):
        settings = self.settings
        painter = Painter(0.0)
        radius = settings["snapRadiusSec"]
        found = []
        for region in regions:
            visit = region.visit
            low, high = visit.start, visit.end
            for unit in region.cut_units:
                low = min(low, unit.start - settings["unitPadSec"])
                high = max(high, unit.end + settings["unitPadSec"])
            pieces = [(low, high)]
            for protect_start, protect_end in self.protected(region, low, high):
                pieces = painter.subtract(pieces, protect_start, protect_end)
            for start, end in pieces:
                if end - start < settings["minCutSec"]:
                    continue
                snapped_start = self.snap(start, start - radius, start + radius)
                snapped_end = self.snap(end, end - radius, end + radius)
                if snapped_end - snapped_start < settings["minCutSec"]:
                    continue
                dropped = next((unit.text for unit in region.cut_units
                                if snapped_start - NOTE_TOLERANCE <= unit.start <= snapped_end), "")
                note = f"slide {self.names.text(visit.slide)} {visit.start:.1f}-{visit.end:.1f}s: {region.status}"
                if dropped:
                    note += f"; drops \u201c{dropped[:70]}\u201d"
                found.append(Candidate(snapped_start, snapped_end, "slide", "slides", "high", note))
        return found

    def protected(self, region, low, high):
        """Sentences near the region that stay: kept units and anything not already cut."""
        pad = self.settings["protectPadSec"]
        ranges = []
        for unit in self.analysis.units:
            if (unit.end < low - NEIGHBOUR_REACH or
                    unit.start > high + NEIGHBOUR_REACH):
                continue
            if (any(unit is cut for cut in region.cut_units) or
                    self.is_cut(unit)):
                continue
            ranges.append((unit.start - pad, unit.end + pad))
        return ranges

    def pieces(self, keep):
        found = []
        for start, end in keep:
            for visit in self.visits:
                low = max(start, visit.start)
                high = min(end, visit.end)
                if high - low > PIECE_MIN:
                    found.append(SlidePiece(visit.slide, low, high))
        return found

    def order(self, keep):
        sequence = []
        for piece in self.pieces(keep):
            if (sequence and
                    sequence[-1]["slide"] == piece.slide):
                sequence[-1]["seconds"] += piece.length
            else:
                sequence.append({"slide": piece.slide, "seconds": piece.length, "start": piece.start})
        return sequence

    def flash(self, pieces, index):
        piece = pieces[index]
        previous = pieces[index - 1].slide if index > 0 else None
        following = pieces[index + 1].slide if index + 1 < len(pieces) else None
        return (piece.length < self.settings["flashSec"] and
                piece.slide != previous and
                piece.slide != following)

    def fragments(self, keep):
        """Kept slivers that show the wrong slide and hold no speech."""
        middles = sorted(word.middle for word in self.analysis.words)
        pieces = self.pieces(keep)
        found = []
        for index, piece in enumerate(pieces):
            previous = pieces[index - 1].slide if index > 0 else None
            back = (previous is not None and
                    piece.slide is not None and
                    piece.slide < previous)
            wrong_slide = (back or
                           self.flash(pieces, index))
            if not wrong_slide:
                continue
            position = bisect.bisect_left(middles, piece.start)
            has_words = (position < len(middles) and
                         middles[position] < piece.end)
            if not has_words:
                found.append(Candidate(piece.start, piece.end, "slide", "slides", "high",
                                       f"wrong-slide fragment {self.names.text(piece.slide)} ({piece.length:.2f}s, no speech)"))
        return found

    def violations(self, keep):
        """Where the edit would still show a slide out of order, or flash one."""
        found = []
        highest = 0
        pieces = self.pieces(keep)
        for index, piece in enumerate(pieces):
            if piece.slide is None:
                continue
            back = piece.slide < highest
            if (back or
                    self.flash(pieces, index)):
                words = " ".join(word.text.strip() for word in self.analysis.words if piece.start <= word.middle < piece.end)
                kind = "back-jump" if back else "flash"
                found.append(Candidate(piece.start, piece.end, "slide", "slides", "low",
                                       f"slide {self.names.text(piece.slide)} shows for {piece.length:.1f}s ({kind}) because speech "
                                       f"runs across the slide change: \u201c{words or 'no words'}\u201d"))
            highest = max(highest, piece.slide)
        return found


class Reconciler:
    def __init__(self, analysis, timeline=None):
        self.analysis = analysis
        self.timeline = timeline
        self.settings = analysis.settings
        self.levels = Levels(analysis.settings)

    def base_cut(self, cut):
        """First-pass content cut, at the millisecond precision the first pass reports."""
        return Candidate(round(cut.start, 3), round(cut.end, 3), cut.type, cut.group, "high", cut.note)

    def run(self, first):
        analysis = self.analysis
        levels = self.levels
        plan = self.settings["plan"]
        base = [self.base_cut(cut) for cut in first.painted if not cut.is_trim]
        promoted, unsure = RetakePromoter(analysis).run(first.content) if levels.enabled("retakes") else ([], [])
        stops = StopCueDetector(analysis).find() if levels.enabled("cues") else []
        new_cuts = promoted + stops
        applied = base + [cut for cut in new_cuts if levels.applies(cut)]
        held = [cut for cut in new_cuts if not levels.applies(cut)]
        slides = None
        regions = []
        notes = []
        slide_cuts = []
        if (self.timeline is not None and
                levels.enabled("slides")):
            slides = SlideEvidence(analysis, self.timeline, applied)
            regions, notes = slides.classify()
            slide_cuts = slides.region_cuts(regions)
        content = Merger().merge(applied + [cut for cut in slide_cuts if levels.applies(cut)])
        held += [cut for cut in slide_cuts if not levels.applies(cut)]
        middles = [word.middle for word in analysis.words]
        for _ in range(self.settings["reconcile"]["slides"]["fragmentPasses"]):
            trims = TrimDetector(analysis, levels).find(content)
            painted = Painter(plan["minPieceSec"]).paint(content + [trim for trim in trims if levels.applies(trim)])
            keep, gaps = KeepRanges(analysis.total, middles, plan["minKeepSec"]).compute(painted)
            fragments = slides.fragments(keep) if slides else []
            if not fragments:
                break
            if not levels.applies(fragments[0]):
                held += fragments
                break
            content = Merger().merge(content + fragments)
        held += [trim for trim in trims if not levels.applies(trim)]
        painted = sorted(painted + gaps, key=lambda item: item.start)
        violations = slides.violations(keep) if slides else []
        proposals = self.proposals(notes, violations, unsure, held, first, painted, applied)
        info = {"promoted": len(promoted), "stopCues": len(stops), "slideCuts": len(slide_cuts)}
        if slides:
            sequence = slides.order(keep)
            shown = [item["slide"] for item in sequence if item["slide"] is not None]
            info["slideOrder"] = sequence
            info["monotonic"] = all(earlier < later for earlier, later in zip(shown, shown[1:]))
            info["regions"] = [{"slide": region.visit.slide, "start": region.visit.start, "end": region.visit.end,
                                "status": region.status} for region in regions]
        return PassResult(painted=painted, proposals=proposals, keep=keep, content=first.content, info=info)

    def proposals(self, notes, violations, unsure, held, first, painted, applied):
        share = self.settings["reconcile"]["proposalCoveredShare"]
        coverage = Coverage()
        groups = []
        for candidate in sorted(unsure, key=lambda item: item.start):
            if coverage.ratio(candidate, applied) >= share:
                continue
            if coverage.ratio(candidate, painted) >= share:
                continue
            if (groups and
                    candidate.start < groups[-1].end):
                groups[-1].end = max(groups[-1].end, candidate.end)
                groups[-1].merged += 1
            else:
                groups.append(replace(candidate, merged=1))
        for group in groups:
            if group.merged > 1:
                group.note += f" (+{group.merged - 1} overlapping)"
        needs = sorted(notes + violations + groups, key=lambda item: item.start)
        carried = [candidate for candidate in first.proposals if self.is_carried(candidate)]
        seen = {(round(item.start, 1), round(item.end, 1)) for item in needs}
        extra = []
        for candidate in sorted(held + carried, key=lambda item: item.start):
            key = (round(candidate.start, 1), round(candidate.end, 1))
            if (key in seen or
                    coverage.ratio(candidate, painted) >= share):
                continue
            seen.add(key)
            extra.append(candidate)
        return sorted(needs + extra, key=lambda item: item.start)

    def is_carried(self, candidate):
        """First-pass proposals that the reconcile pass does not reconsider."""
        return (candidate.type not in RECONSIDERED or
                candidate.confidence == "high")
