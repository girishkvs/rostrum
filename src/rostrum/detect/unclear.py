"""Unclear stretches: where the transcript cannot be trusted, so a cut there needs a listen first.

(a) a word whose sound lasts longer than unclear.maxWordSec: the engine stretched it over speech it dropped;
(b) a run of words below unclear.minProbability lasting at least unclear.minSec;
(c) a segment whose average log-probability is below unclear.maxAvgLogprob, or that contains a known
    hallucination phrase;
(d) a gap between words holding at least unclear.untranscribedSec of speech-level sound;
plus ranges where transcription failed (the engine echoed its prompt).

With unclear.breakSentences on (the default), matching never joins words across an unclear stretch: sentences end at
its edges, and the words of (a) and of hallucination segments stand alone and are not matched.
"""
import re
from dataclasses import dataclass, field

from rostrum.detect.candidate import Candidate

NOT_PLAIN = re.compile(r"[^a-z0-9' ]+")


@dataclass
class SpokenWord:
    start: float
    end: float
    text: str
    probability: float


@dataclass
class UnclearSpan:
    start: float
    end: float
    reasons: list = field(default_factory=list)
    untrusted: bool = False


class UnclearSpans:
    def __init__(self, unclear_settings):
        self.settings = unclear_settings
        self.phrases = [(phrase, self.plain(phrase)) for phrase in unclear_settings["hallucinationPhrases"]]

    def plain(self, text):
        return " ".join(NOT_PLAIN.sub(" ", text.lower()).split())

    def find(self, words, segments, failed=(), energy=None):
        """Every unclear span, in time order. Spans may overlap."""
        spans = self.overlong(words) + self.low_confidence(words) + self.doubtful(segments)
        if energy is not None:
            spans += self.untranscribed(words, energy)
        spans += [UnclearSpan(float(start), float(end), ["transcription failed here (the engine echoed its prompt)"])
                  for start, end in failed]
        return sorted(spans, key=lambda span: (span.start, span.end))

    def overlong(self, words):
        longest = self.settings["maxWordSec"]
        spans = []
        for word in words:
            length = word.end - word.start
            if length > longest:
                reason = f"one word, \u201c{word.text.strip()}\u201d, lasts {length:.1f} s"
                spans.append(UnclearSpan(word.start, word.end, [reason], untrusted=True))
        return spans

    def low_confidence(self, words):
        spans = []
        for run in self.low_runs(words):
            length = run[-1].end - run[0].start
            if length >= self.settings["minSec"]:
                spans.append(UnclearSpan(run[0].start, run[-1].end,
                                         [f"{len(run)} low-confidence words in a row ({length:.1f} s)"]))
        return spans

    def low_runs(self, words):
        """Consecutive low-probability words, split where one starts more than mergeGapSec after the one before."""
        runs = []
        previous = None
        for word in words:
            if word.probability >= self.settings["minProbability"]:
                previous = None
                continue
            starts_run = (previous is None or
                          word.start - previous.end > self.settings["mergeGapSec"])
            if starts_run:
                runs.append([])
            runs[-1].append(word)
            previous = word
        return runs

    def untranscribed(self, words, energy):
        """Gaps between words holding at least untranscribedSec of speech-level sound: speech the engine dropped."""
        least = self.settings["untranscribedSec"]
        total = len(energy.db) * energy.frame
        spans = []
        previous_end = 0.0
        for word in [*words, None]:
            next_start = word.start if word is not None else total
            if next_start - previous_end >= least:
                runs = energy.loud_runs(previous_end, next_start, 0.0)
                sound = sum(end - start for start, end in runs)
                if sound >= least:
                    spans.append(UnclearSpan(runs[0][0], runs[-1][1], [f"{sound:.1f} s of speech with no words"]))
            if word is not None:
                previous_end = max(previous_end, word.end)
        return spans

    def doubtful(self, segments):
        spans = []
        for segment in segments:
            reasons = []
            logprob = segment.get("avg_logprob")
            if (logprob is not None and
                    logprob < self.settings["maxAvgLogprob"]):
                reasons.append(f"the engine was unsure of this segment (average log-probability {logprob:.2f})")
            text = self.plain(segment.get("text", ""))
            phrase = next((original for original, plain in self.phrases if plain and plain in text), None)
            if phrase:
                reasons.append(f"it reads like a known hallucination (\u201c{phrase}\u201d)")
            if reasons:
                spans.append(UnclearSpan(float(segment["start"]), float(segment["end"]), reasons, untrusted=bool(phrase)))
        return spans

    def merged(self, spans, gap):
        """Spans that overlap or are closer than `gap` become one, with all their reasons."""
        merged = []
        for span in sorted(spans, key=lambda item: (item.start, item.end)):
            if (merged and
                    span.start - merged[-1].end <= gap):
                last = merged[-1]
                last.end = max(last.end, span.end)
                last.reasons += [reason for reason in span.reasons if reason not in last.reasons]
                last.untrusted = last.untrusted or span.untrusted
            else:
                merged.append(UnclearSpan(span.start, span.end, list(span.reasons), span.untrusted))
        return merged


class UnclearDetector:
    def __init__(self, analysis):
        self.analysis = analysis
        self.settings = analysis.settings["unclear"]

    def find(self):
        analysis = self.analysis
        spans = UnclearSpans(self.settings).merged(analysis.unclear_spans, self.settings["mergeGapSec"])
        found = []
        for span in spans:
            first, end = analysis.words_between(span.start, span.end)
            note = "unclear: " + "; ".join(span.reasons) + "; listen before deciding"
            found.append(Candidate(span.start, span.end, "unclear", "unclear", "high", note, first, end))
        return found
