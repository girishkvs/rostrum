"""Retakes: a sentence said again within retakes.windowSec. Cut from the attempt to the start of the retake.

A sentence said again with swapped words ("before the change" -> "after the change") can be deliberate parallel
phrasing or a spoken correction, so it gets retakes.substitutionLevel (propose by default) instead of the group's
level. A restart or an extension, where the attempt is (nearly) the start of the retake, keeps the group's level.
"""
import difflib
from dataclasses import replace

from rostrum.detect.text import Stemmer

STRONG_KINDS = ("strong", "prefix", "same-script")
BRIDGE_KINDS = ("strong", "same-script")
NOT_APPLIED = "; not applied, would also remove unrepeated: "
WORD_EDGES = ".,!?;:\"'()[]\u201c\u201d\u2018\u2019"


class Substitutions:
    """Words swapped between two versions of one sentence, as opposed to a restart, an extension or a rewording.

    A swap replaces at most retakes.substitutionMaxWords words on each side and changes a content word. It counts
    when the sentences share a content word again after it, or when it ends both sentences and the attempt is
    complete. Different first words and an added ending do not matter; a longer change in between is a rewording,
    so the pair has no swaps. Two takes of the same script line have none either."""

    def __init__(self, analysis):
        self.analysis = analysis
        self.stemmer = Stemmer()
        self.max_words = analysis.retakes["substitutionMaxWords"]

    def find(self, attempt, retake):
        """[(old words, new words)] for each swap, or [] when the retake restarts, extends or rewords the attempt."""
        analysis = self.analysis
        if analysis.same_script_line(analysis.script_map[attempt.index], analysis.script_map[retake.index]):
            return []
        old = [self.stem(token) for token in attempt.tokens]
        new = [self.stem(token) for token in retake.tokens]
        opcodes = difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
        shared = [position for position, opcode in enumerate(opcodes) if opcode[0] == "equal"]
        if not shared:
            return []
        inner = [(position, opcodes[position]) for position in range(shared[0] + 1, shared[-1])
                 if opcodes[position][0] != "equal"]
        if not all(self.is_short(opcode) for _, opcode in inner):
            return []
        swaps = []
        for position, opcode in inner:
            meets_again = self.meets_again(attempt, opcodes[position + 1:])
            if (self.is_swap(attempt, retake, opcode) and
                    meets_again):
                swaps.append(opcode)
        tail = opcodes[shared[-1] + 1:]
        ends_with_swap = (len(tail) == 1 and
                          self.is_swap(attempt, retake, tail[0]) and
                          analysis.ends_sentence(attempt.text))
        if ends_with_swap:
            swaps.append(tail[0])
        return [(self.words(attempt, old_start, old_end), self.words(retake, new_start, new_end))
                for _, old_start, old_end, new_start, new_end in swaps]

    def stem(self, token):
        """Without apostrophes, so "agent's" and "agents" are one word here."""
        return self.stemmer.stem(token.replace("'", ""))

    def is_short(self, opcode):
        _, old_start, old_end, new_start, new_end = opcode
        return (old_end - old_start <= self.max_words and
                new_end - new_start <= self.max_words)

    def is_swap(self, attempt, retake, opcode):
        tag, old_start, old_end, new_start, new_end = opcode
        content = self.analysis.content_words
        changes_content = (bool(content.of(attempt.tokens[old_start:old_end])) or
                           bool(content.of(retake.tokens[new_start:new_end])))
        return (tag == "replace" and
                self.is_short(opcode) and
                changes_content)

    def meets_again(self, attempt, rest):
        shared = [token for tag, start, end, _, _ in rest if tag == "equal" for token in attempt.tokens[start:end]]
        return bool(self.analysis.content_words.of(shared))

    def words(self, unit, first_token, end_token):
        indexes = sorted(set(unit.token_words[first_token:end_token]))
        return " ".join(self.analysis.words[index].text.strip().strip(WORD_EDGES) for index in indexes)


class RetakeDetector:
    def __init__(self, analysis):
        self.analysis = analysis
        self.substitution_level = analysis.retakes["substitutionLevel"]
        self.substitutions = Substitutions(analysis)

    def find(self):
        found = []
        for unit in self.analysis.units:
            matches = self.analysis.matches[unit.index]
            if not matches:
                continue
            for retake_index, kind in self.picks(matches):
                candidate = self.candidate(unit, retake_index, kind)
                if candidate:
                    found.append(candidate)
        return found

    def picks(self, matches):
        """The first match, plus the first strong one when that is a different sentence."""
        picks = [matches[0]]
        strong = next((match for match in matches if match[1] in STRONG_KINDS), None)
        if (strong and
                strong[0] != matches[0][0]):
            picks.append(strong)
        return picks

    def candidate(self, attempt, retake_index, kind):
        analysis = self.analysis
        retake = analysis.units[retake_index]
        end_word = retake.first
        fails = analysis.guard(attempt.index, retake_index)
        before_retake = analysis.units[retake_index - 1] if retake_index > 0 else None
        if (before_retake is not None and
                before_retake.index >= attempt.index and
                before_retake.content and
                not analysis.ends_sentence(before_retake.text)):
            fails.append("retake starts mid-sentence after: " + before_retake.text[-50:])
        confident = (kind in STRONG_KINDS or
                     self.bridged(attempt.index, retake_index))
        is_prefix = kind == "prefix"
        cut_type = "repeat" if is_prefix else "retake"
        group = "repeats" if is_prefix else "retakes"
        note = (f"{cut_type} ({kind}): \u201c{analysis.text_span(attempt.first, end_word)}\u201d \u2192 retake "
                f"\u201c{analysis.text_span(end_word, end_word + 9)}\u201d")
        if fails:
            note += f"{NOT_APPLIED}\u201c{fails[0]}\u201d"
        applies = (confident and
                   not fails)
        confidence = "high" if applies else "low"
        candidate = analysis.content_cut(attempt.first, end_word, cut_type, group, confidence, note, kind, fails)
        return self.with_substitution(candidate, attempt, retake)

    def with_substitution(self, candidate, attempt, retake):
        """Said again with swapped words: retakes.substitutionLevel decides. auto keeps the plain retake."""
        if self.substitution_level == "auto":
            return candidate
        swaps = self.substitutions.find(attempt, retake)
        if not swaps:
            return candidate
        if self.substitution_level == "off":
            return None
        pairs = ", ".join(f"\u201c{old}\u201d \u2192 \u201c{new}\u201d" for old, new in swaps)
        words = "a different word" if len(swaps) == 1 else "different words"
        note = f"said again with {words} ({pairs}); keep if deliberate. {candidate.note}"
        return replace(candidate, level=self.substitution_level, note=note)

    def bridged(self, attempt_index, retake_index):
        """A loose match is trusted when a sentence in between is strongly said again at or after the retake."""
        for between in range(attempt_index + 1, retake_index):
            for target, kind in self.analysis.matches[between]:
                if (target >= retake_index and
                        kind in BRIDGE_KINDS):
                    return True
        return False


class RetakeNotes:
    def base(self, candidate):
        return candidate.note.split(NOT_APPLIED)[0]
