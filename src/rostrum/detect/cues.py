"""Spoken cues. Cue phrases ("let me redo that") in the first pass; stand-alone cues ("Stop.") in the reconcile pass.

Either way the cut goes back to where the abandoned attempt started, but only over sentences that are said again.
"""
import re

FAR_AWAY = 1e9


class CueDetector:
    def __init__(self, analysis):
        self.analysis = analysis
        self.settings = analysis.settings["cues"]
        patterns = [(re.compile(item["pattern"]), item.get("confidence", "high")) for item in self.settings["patterns"]]
        # A cue phrase that the script itself says is planned, never a cue.
        self.patterns = [(pattern, confidence) for pattern, confidence in patterns
                         if not pattern.search(analysis.script_padded)]

    def find(self):
        found = []
        for unit in self.analysis.units:
            text = " ".join(unit.tokens)
            for pattern, confidence in self.patterns:
                for match in pattern.finditer(text):
                    candidate = self.candidate(unit, text, match, confidence)
                    if candidate:
                        found.append(candidate)
        return found

    def candidate(self, unit, text, match, confidence):
        analysis = self.analysis
        units = analysis.units
        cue_first = len(text[:match.start()].split())
        cue_end = cue_first + len(match.group(0).split())
        lookback = self.settings["phraseLookbackSec"]
        first = unit.index
        while (first - 1 >= 0 and
                unit.start - units[first - 1].start <= lookback and
                analysis.survives(first - 1, unit.index + 1, unit.end + lookback)):
            first -= 1
        if first < unit.index:
            start_word = units[first].first
        else:
            start_word = unit.token_words[min(cue_first, len(unit.token_words) - 1)]
        rest = unit.tokens[cue_end:]
        if (not analysis.content_words.of(rest) and
                unit.index + 1 < len(units)):
            end_word = units[unit.index + 1].first
        elif cue_end < len(unit.token_words):
            end_word = unit.token_words[cue_end]
        else:
            end_word = unit.last + 1
        confident = (confidence == "high" or
                     first < unit.index)
        note = f"cue \u201c{match.group(0)}\u201d: \u201c{analysis.text_span(start_word, end_word)}\u201d"
        return analysis.content_cut(start_word, end_word, "cue", "cues", "high" if confident else "low", note)


class StopCueDetector:
    """A stand-alone cue ("Stop.", "Redo, okay.") or a sentence ending in a trailing cue word, then a retake."""

    def __init__(self, analysis):
        self.analysis = analysis
        self.settings = analysis.settings["cues"]
        self.leaders = frozenset(self.settings["words"])
        self.enders = frozenset(self.settings["endWords"])
        self.allowed = self.leaders | self.enders | frozenset(self.settings["companionWords"])
        self.trailing = frozenset(self.settings["trailingWords"])
        self.max_words = self.settings["maxCueWords"]

    def find(self):
        units = self.analysis.units
        found = []
        index = 0
        while index < len(units):
            cue = units[index]
            trailing = self.trailing_cue(index)
            is_cue = (self.standalone(cue) or
                      trailing)
            if not is_cue:
                index += 1
                continue
            group_end = index
            while (group_end + 1 < len(units) and
                    self.cue_words_only(units[group_end + 1])):
                group_end += 1
            if group_end + 1 >= len(units):
                break
            retake = units[group_end + 1]
            if (trailing and
                    not self.resaid(cue, retake.start)):
                index = group_end + 1
                continue
            first = self.attempt_start(index, retake)
            label = " ".join(units[position].text for position in range(index, group_end + 1))[:40]
            note = (f"cue \u201c{label}\u201d: drops \u201c{units[first].text[:70]}\u201d \u2192 retake "
                    f"\u201c{retake.text[:50]}\u201d")
            candidate = self.analysis.content_cut(units[first].first, retake.first, "cue", "cues", "high", note)
            if candidate:
                found.append(candidate)
            index = group_end + 1
        return found

    def attempt_start(self, index, retake):
        units = self.analysis.units
        cue = units[index]
        first = index
        earlier = index - 1
        while (earlier >= 0 and
                cue.start - units[earlier].start <= self.settings["lookbackSec"] and
                units[earlier].content and
                self.resaid(units[earlier], retake.start)):
            first = earlier
            earlier -= 1
        return first

    def cue_words_only(self, unit):
        return (0 < len(unit.tokens) <= self.max_words and
                set(unit.tokens) <= self.allowed)

    def standalone(self, unit):
        if not self.cue_words_only(unit):
            return False
        return (unit.tokens[0] in self.leaders or
                unit.tokens[-1] in self.enders)

    def trailing_cue(self, index):
        units = self.analysis.units
        unit = units[index]
        next_start = units[index + 1].start if index + 1 < len(units) else FAR_AWAY
        return (len(unit.tokens) > self.max_words and
                unit.tokens[-1] in self.trailing and
                next_start - unit.end >= self.settings["trailingGapSec"])

    def resaid(self, unit, after):
        analysis = self.analysis
        if any(analysis.units[target].start >= after for target, _ in analysis.matches[unit.index]):
            return True
        later = self.later_content(after, after + self.settings["resaidWindowSec"])
        shared = len(unit.content & later)
        if not unit.content:
            return False
        return (shared / len(unit.content) >= self.settings["resaidShare"] and
                (shared >= self.settings["resaidMinWords"] or len(unit.content) == 1))

    def later_content(self, start, end):
        content = set()
        for unit in self.analysis.units:
            if start <= unit.start <= end:
                content |= unit.content
        return content
