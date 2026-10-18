"""Off-talk: a short isolated aside ("Is this recording?", "One second, please") that is not part of the talk."""
from rostrum.detect.candidate import Candidate


class OffTalkDetector:
    def __init__(self, analysis):
        self.analysis = analysis
        self.settings = analysis.settings["offtalk"]
        self.clear_words = (analysis.offtalk_strong |
                            frozenset(self.settings["okWords"]) |
                            frozenset(analysis.settings["text"]["stopWords"]))

    def find(self):
        found = []
        for unit in self.analysis.units:
            candidate = self.candidate(unit)
            if candidate:
                found.append(candidate)
        return found

    def candidate(self, unit):
        analysis = self.analysis
        settings = self.settings
        tokens = unit.tokens
        count = len(tokens)
        if (count == 0 or
                count > settings["maxWords"] or
                unit.echo):
            return None
        if analysis.is_scripted(unit):
            return None
        if (count >= 2 and
                analysis.in_script(tokens)):
            return None
        previous_end = analysis.words[unit.first - 1].end if unit.first > 0 else 0.0
        next_start = analysis.words[unit.last + 1].start if unit.last + 1 < len(analysis.words) else analysis.total
        isolated = (unit.start - previous_end >= settings["isolationSec"] and
                    next_start - unit.end >= settings["isolationSec"])
        strong = any(token in analysis.offtalk_strong for token in tokens)
        if analysis.offtalk_greetings & set(tokens):
            return None
        suspicious = (strong or
                      unit.probability < settings["unclearBelow"])
        if (not suspicious or
                not isolated):
            return None
        clear = (strong and
                 settings["clearMinWords"] <= count <= settings["clearMaxWords"] and
                 all(token in self.clear_words for token in tokens))
        why = "off-talk words" if strong else "unclear speech, not in script"
        start, end = self.extend(unit)
        energy = analysis.energy
        search = settings["edgeSearchSec"]
        offset = settings["edgeOffsetSec"]
        radius = settings["edgeRadiusSec"]
        return Candidate(
            start=energy.quietest(start - search, start, start - offset, radius),
            end=energy.quietest(end, end + search, end + offset, radius),
            type="offtalk",
            group="offtalk",
            confidence="high" if clear else "low",
            note=f"offtalk ({why}): \u201c{unit.text[:60]}\u201d",
            first_word=unit.first,
            end_word=unit.last + 1,
        )

    def extend(self, unit):
        """Widen the aside over neighbouring frames that are still loud (breaths, trailing syllables)."""
        energy = self.analysis.energy
        frame = energy.frame
        loud = energy.speech - self.settings["extendDropDb"]
        reach = self.settings["extendSec"]
        start = unit.start
        end = unit.end
        while (start - frame >= unit.start - reach and
                energy.db[int((start - frame) / frame)] > loud):
            start -= frame
        while (end + frame <= unit.end + reach and
                int(end / frame) < len(energy.db) and
                energy.db[int(end / frame)] > loud):
            end += frame
        return start, end
