"""Words, sentence units and repeat matches over the talk timeline.

Times are seconds on the talk timeline. Word bounds are the transcript's, except that words longer than
transcript.longWordSec are trimmed to their loud frames.
"""
import difflib
from dataclasses import dataclass

from rostrum.detect.candidate import Candidate
from rostrum.detect.text import ContentWords, Tokenizer
from rostrum.detect.unclear import SpokenWord, UnclearSpans

ELLIPSIS = " \u2026"


@dataclass(eq=False)
class Word:
    index: int
    raw_start: float
    raw_end: float
    start: float
    end: float
    text: str
    probability: float
    tokens: list

    @property
    def middle(self):
        return (self.start + self.end) / 2


@dataclass(eq=False)
class Unit:
    """A sentence, or a run of words between pauses."""
    index: int
    first: int
    last: int
    start: float
    end: float
    tokens: list
    token_words: list
    content: set
    text: str
    probability: float
    echo: bool


@dataclass
class ScriptMatch:
    sentence: int
    score: float


class Analysis:
    def __init__(self, words, energy, total, script, failed, settings, segments=None):
        self.settings = settings
        self.energy = energy
        self.total = total
        self.script = script
        self.failed = failed
        self.retakes = settings["retakes"]
        self.offtalk = settings["offtalk"]
        self.offtalk_strong = frozenset(self.offtalk["strongWords"])
        self.offtalk_greetings = frozenset(self.offtalk["greetingWords"])
        self.tokenizer = Tokenizer()
        self.content_words = ContentWords(settings["text"]["stopWords"], settings["fillers"]["words"])
        self.fillers = frozenset(settings["fillers"]["words"])
        self.discourse = frozenset(settings["text"]["discourseWords"])
        self.sentence_end = settings["transcript"]["sentenceEnd"]
        self.script_sentences = [sentence.padded for sentence in script]
        self.script_padded = " " + " ".join(token for sentence in script for token in sentence.tokens) + " "
        self.words = [self.word(index, item) for index, item in enumerate(words)]
        self.unclear_spans = self.find_unclear(segments or [])
        self.word_untrusted, self.word_region = self.mark_unclear()
        self.units = self.build_units()
        self.script_map = [self.map_script(unit) for unit in self.units]
        size = self.retakes["ngram"]
        self.grams = [set(tuple(unit.tokens[i:i + size]) for i in range(len(unit.tokens) - size + 1)) for unit in self.units]
        self.matches = self.find_matches()

    def word(self, index, item):
        start, end = self.trim(item["start"], item["end"])
        return Word(index, item["start"], item["end"], start, end, item["word"], item["probability"],
                    self.tokenizer.tokens(item["word"]))

    def trim(self, start, end):
        transcript = self.settings["transcript"]
        if end - start <= transcript["longWordSec"]:
            return start, end
        frame = self.energy.frame
        first = int(start / frame)
        last = max(first + 1, int(end / frame))
        loud = (self.energy.db[first:last] > self.energy.speech).nonzero()[0]
        if len(loud) == 0:
            return start, end
        return (max(start, (first + loud[0]) * frame - transcript["longWordPadBeforeSec"]),
                min(end, (first + loud[-1] + 1) * frame + transcript["longWordPadAfterSec"]))

    def in_failed(self, time):
        pad = self.settings["transcript"]["failedPadSec"]
        return any(start - pad <= time <= end + pad for start, end in self.failed)

    def find_unclear(self, segments):
        """Unclear spans on the effective (loud-frame) word times, so a word stretched into silence does not count."""
        spoken = [SpokenWord(word.start, word.end, word.text, word.probability) for word in self.words]
        return UnclearSpans(self.settings["unclear"]).find(spoken, segments, self.failed, self.energy)

    def mark_unclear(self):
        """Per word: is its text untrusted, and which unclear stretch (if any) it sits in. With
        unclear.breakSentences off, neither is set and matching is exactly the original rules'."""
        if not self.settings["unclear"]["breakSentences"]:
            return [False] * len(self.words), [None] * len(self.words)
        untrusted_spans = [span for span in self.unclear_spans if span.untrusted]
        untrusted = [self.inside(word, untrusted_spans) is not None for word in self.words]
        regions = UnclearSpans(self.settings["unclear"]).merged(self.unclear_spans, 0.0)
        return untrusted, [self.inside(word, regions) for word in self.words]

    def inside(self, word, spans):
        middle = word.middle
        for index, span in enumerate(spans):
            if span.start <= middle <= span.end:
                return index
        return None

    def words_between(self, start, end):
        """(first word, one past the last word) whose middle lies in [start, end], or (None, None)."""
        inside = [word.index for word in self.words if start <= word.middle <= end]
        if not inside:
            return None, None
        return inside[0], inside[-1] + 1

    def ends_sentence(self, text):
        return text[-1:] in self.sentence_end

    def build_units(self):
        gap = self.settings["transcript"]["unitGapSec"]
        groups = []
        current = []
        for index, word in enumerate(self.words):
            current.append(index)
            following = self.words[index + 1] if index + 1 < len(self.words) else None
            if (self.ends_sentence(word.text.strip()) or
                    following is None or
                    following.start - word.end >= gap or
                    self.breaks_after(index)):
                groups.append(current)
                current = []
        return [self.unit(number, group) for number, group in enumerate(groups)]

    def breaks_after(self, index):
        """Unclear stretches end a sentence at their edges, and untrusted words stand alone."""
        changes_region = self.word_region[index] != self.word_region[index + 1]
        changes_trust = self.word_untrusted[index] != self.word_untrusted[index + 1]
        return (changes_region or
                changes_trust)

    def unit(self, number, indexes):
        tokens = []
        token_words = []
        for index in indexes:
            if self.word_untrusted[index]:
                continue
            for token in self.words[index].tokens:
                if token not in self.fillers:
                    tokens.append(token)
                    token_words.append(index)
        first = self.words[indexes[0]]
        last = self.words[indexes[-1]]
        return Unit(
            index=number,
            first=indexes[0],
            last=indexes[-1],
            start=first.start,
            end=last.end,
            tokens=tokens,
            token_words=token_words,
            content=self.content_words.of(tokens),
            text="".join(self.words[index].text for index in indexes).strip(),
            probability=sum(self.words[index].probability for index in indexes) / len(indexes),
            echo=self.in_failed((first.start + last.end) / 2),
        )

    def map_script(self, unit):
        if len(unit.content) < 2:
            return None
        best = None
        for index, sentence in enumerate(self.script):
            shared = len(unit.content & sentence.content)
            if not shared:
                continue
            score = 2 * shared / (len(unit.content) + len(sentence.content))
            if (best is None or
                    score > best.score):
                best = ScriptMatch(index, score)
        return best

    def is_scripted(self, unit):
        match = self.script_map[unit.index]
        return (match is not None and
                match.score >= self.retakes["scriptMapped"])

    def scripted_count(self, tokens):
        pattern = " " + " ".join(tokens) + " "
        return sum(1 for sentence in self.script_sentences if pattern in sentence)

    def in_script(self, tokens):
        return (" " + " ".join(tokens) + " ") in self.script_padded

    def longest_run(self, first, second):
        """Longest run of equal tokens: (length, start in first, start in second)."""
        best = 0
        best_i = 0
        best_j = 0
        previous = [0] * (len(second) + 1)
        for i in range(1, len(first) + 1):
            current = [0] * (len(second) + 1)
            for j in range(1, len(second) + 1):
                if first[i - 1] == second[j - 1]:
                    current[j] = previous[j - 1] + 1
                    if current[j] > best:
                        best, best_i, best_j = current[j], i - current[j], j - current[j]
            previous = current
        return best, best_i, best_j

    def strip_discourse(self, tokens):
        index = 0
        while (index < len(tokens) and
                tokens[index] in self.discourse):
            index += 1
        return tokens[index:]

    def find_matches(self):
        matches = [[] for _ in self.units]
        window = self.retakes["windowSec"]
        for unit in self.units:
            for later in range(unit.index + 1, len(self.units)):
                other = self.units[later]
                if other.start - unit.end > window:
                    break
                kind = self.pair_kind(unit, other)
                if kind:
                    matches[unit.index].append((other.index, kind))
        return matches

    def pair_kind(self, attempt, retake):
        """How `retake` repeats `attempt`: strong, prefix, same-script, fuzzy, or None."""
        if (not attempt.tokens or
                attempt.echo or
                retake.echo):
            return None
        first_match = self.script_map[attempt.index]
        second_match = self.script_map[retake.index]
        if self.scripted_apart(first_match, second_match):
            return None
        same = self.same_script_line(first_match, second_match)
        if self.strong_run(attempt, retake, same):
            return "strong"
        if self.short_similar(attempt, retake):
            return "strong"
        if self.prefix(attempt, retake):
            return "prefix"
        return self.loose(attempt, retake, same)

    def scripted_apart(self, first_match, second_match):
        conflict = self.retakes["scriptConflict"]
        return bool(first_match and
                    second_match and
                    first_match.sentence != second_match.sentence and
                    first_match.score >= conflict and
                    second_match.score >= conflict)

    def same_script_line(self, first_match, second_match):
        same = self.retakes["scriptSame"]
        return bool(first_match and
                    second_match and
                    first_match.sentence == second_match.sentence and
                    first_match.score >= same and
                    second_match.score >= same)

    def strong_run(self, attempt, retake, same):
        if not (self.grams[attempt.index] & self.grams[retake.index]):
            return False
        run, first, _ = self.longest_run(attempt.tokens, retake.tokens)
        short_but_most = (run >= self.retakes["shortRun"] and
                          run >= self.retakes["shortRunShare"] * len(attempt.tokens))
        long_enough = (run >= self.retakes["strongRun"] or
                       short_but_most)
        if not long_enough:
            return False
        shared = attempt.tokens[first:first + min(run, self.retakes["scriptedRunMax"])]
        return (same or
                self.scripted_count(shared) < self.retakes["scriptedLimit"])

    def short_similar(self, attempt, retake):
        settings = self.retakes
        sizes_fit = (settings["shortMinWords"] <= len(attempt.tokens) <= settings["shortMaxWords"] and
                     settings["shortMinWords"] <= len(retake.tokens) <= settings["shortRetakeMaxWords"])
        if not sizes_fit:
            return False
        return difflib.SequenceMatcher(None, attempt.tokens, retake.tokens).ratio() >= settings["shortRatio"]

    def prefix(self, attempt, retake):
        is_next_and_short = (retake.index == attempt.index + 1 and
                             len(attempt.tokens) <= self.retakes["prefixMaxWords"] and
                             retake.start - attempt.end <= self.retakes["prefixMaxGapSec"])
        if not is_next_and_short:
            return False
        head = self.strip_discourse(attempt.tokens)
        body = self.strip_discourse(retake.tokens)
        return bool(head and
                    len(head) < len(body) and
                    body[:len(head)] == head)

    def loose(self, attempt, retake, same):
        if not attempt.content:
            return None
        settings = self.retakes
        shared = len(attempt.content & retake.content)
        share = shared / len(attempt.content)
        near = retake.start - attempt.end <= settings["nearSec"]
        broad = (shared >= settings["sharedWords"] and
                 share >= settings["sharedShare"])
        close = (shared >= settings["nearSharedWords"] and
                 share >= settings["nearShare"] and
                 near)
        if not (broad or
                close):
            return None
        if (same and
                shared >= settings["sharedWords"]):
            return "same-script"
        return "fuzzy"

    def first_match(self, index):
        return self.matches[index][0][0] if self.matches[index] else None

    def survives(self, index, limit, region_end):
        """Following the chain of retakes from `index` past `limit`, is the sentence said again by `region_end`?"""
        while (index is not None and
                index < limit):
            index = self.first_match(index)
        return (index is not None and
                self.units[index].start <= region_end)

    def offtalk_like(self, unit):
        has_strong_word = any(token in self.offtalk_strong for token in unit.tokens)
        is_greeting = bool(self.offtalk_greetings & set(unit.tokens))
        return (len(unit.tokens) <= self.offtalk["maxWords"] and
                not self.is_scripted(unit) and
                has_strong_word and
                not is_greeting)

    def disposable(self, unit):
        """Can this sentence go with a retake even though it is not said again?"""
        tokens = unit.tokens
        if not unit.content:
            return True
        if self.offtalk_like(unit):
            return True
        short_aside_neighbour = (len(tokens) <= self.retakes["asideMaxWords"] and
                                 not self.is_scripted(unit))
        for neighbour in (unit.index - 1, unit.index + 1):
            if not (0 <= neighbour < len(self.units)):
                continue
            other = self.units[neighbour]
            if (short_aside_neighbour and
                    self.offtalk_like(other)):
                gap = unit.start - other.end if neighbour < unit.index else other.start - unit.end
                if gap < self.retakes["asideGapSec"]:
                    return True
        if (unit.index + 1 < len(self.units) and
                len(tokens) <= self.retakes["prefixMaxWords"]):
            head = self.strip_discourse(tokens)
            following = self.strip_discourse(self.units[unit.index + 1].tokens)
            if (head and
                    following[:len(head)] == head):
                return True
        return (len(tokens) <= self.retakes["unclearMaxWords"] and
                unit.probability < self.retakes["unclearBelow"])

    def guard(self, attempt_index, retake_index):
        """Reasons why cutting from the attempt to the retake would also drop something that is not said again."""
        settings = self.retakes
        fails = []
        attempt = self.units[attempt_index]
        retake = self.units[retake_index]
        region_end = retake.start + max(settings["guardRegionMinSec"],
                                        settings["guardRegionFactor"] * (retake.start - attempt.start))
        retake_content = set()
        for unit in self.units[retake_index:retake_index + settings["guardRetakeUnits"]]:
            retake_content |= unit.content
        before = self.units[attempt_index - 1] if attempt_index > 0 else None
        if (before is not None and
                before.content and
                not self.ends_sentence(before.text)):
            fails.append("abandoned attempt starts mid-sentence after: " + before.text[-50:])
        for index in range(attempt_index, retake_index):
            unit = self.units[index]
            if unit.echo:
                fails.append("[transcription failed here]")
            elif (index == attempt_index or
                    self.survives(index, retake_index, region_end) or
                    self.disposable(unit)):
                continue
            elif (unit.content and
                    len(unit.content & retake_content) / len(unit.content) >= settings["guardCoveredShare"]):
                continue
            else:
                fails.append(unit.text[:70])
        return fails

    def snap_edge(self, next_word, side):
        """A cut edge between word next_word - 1 and next_word, in the quietest frame of the gap."""
        edges = self.settings["edges"]
        lookback = self.words[max(0, next_word - edges["lookbackWords"]):next_word]
        previous_end = max((word.end for word in lookback), default=0.0)
        next_start = self.words[next_word].start if next_word < len(self.words) else self.total
        gap = next_start - previous_end
        tight = edges["tightGapSec"]
        if gap <= tight:
            return self.energy.quietest(previous_end - tight, next_start + tight, (previous_end + next_start) / 2,
                                        edges["tightRadiusSec"])
        offset = min(edges["maxOffsetSec"], gap / 2)
        target = previous_end + offset if side == "start" else next_start - offset
        return self.energy.quietest(previous_end + edges["marginSec"], next_start - edges["marginSec"], target,
                                    edges["radiusSec"])

    def content_cut(self, first_word, end_word, cut_type, group, confidence, note, kind="", fails=()):
        """A cut that removes words first_word .. end_word - 1."""
        if end_word <= first_word:
            return None
        return Candidate(self.snap_edge(first_word, "start"), self.snap_edge(end_word, "end"), cut_type, group,
                         confidence, note, first_word, end_word, kind, list(fails))

    def text_span(self, first_word, end_word, limit=9):
        shown = self.words[first_word:min(end_word, first_word + limit)]
        text = "".join(word.text for word in shown).strip()
        return text + (ELLIPSIS if end_word - first_word > limit else "")

    def unit_of_words(self):
        owner = {}
        for unit in self.units:
            for index in range(unit.first, unit.last + 1):
                owner[index] = unit.index
        return owner
