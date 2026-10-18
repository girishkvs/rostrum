from rostrum.detect.analysis import Analysis
from rostrum.detect.candidate import Candidate, Levels
from rostrum.detect.cues import CueDetector, StopCueDetector
from rostrum.detect.fillers import FillerDetector
from rostrum.detect.offtalk import OffTalkDetector
from rostrum.detect.pauses import TrimDetector
from rostrum.detect.planner import KeepRanges, Merger, Painter, Planner
from rostrum.detect.repeats import RepeatDetector
from rostrum.detect.retakes import RetakeDetector, Substitutions
from rostrum.detect.script import ScriptText
from rostrum.detect.text import ContentWords, NumberWords, Stemmer, Tokenizer
from rostrum.reconcile import Reconciler


class TestText:
    def test_numbers_times_and_percents_are_spelled_out(self):
        tokenizer = Tokenizer()
        assert tokenizer.tokens("9:06,") == ["nine", "oh", "six"]
        assert tokenizer.tokens("42%") == ["forty", "two", "percent"]
        assert tokenizer.tokens("1,500") == ["one", "thousand", "five", "hundred"]
        assert tokenizer.tokens("Don\u2019t!") == ["don't"]
        assert NumberWords().say(2026) == "two thousand twenty six"

    def test_letters_outside_ascii_stay_in_their_word(self):
        tokenizer = Tokenizer()
        assert tokenizer.text_tokens("Caf\u00e9 \u2014 na\u00efve, Stra\u00dfe!") == ["caf\u00e9", "na\u00efve", "strasse"]
        assert tokenizer.tokens("Cafe\u0301") == ["caf\u00e9"]
        assert tokenizer.text_tokens("\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e") == [
            "\u0928\u092e\u0938\u094d\u0924\u0947", "\u0926\u0941\u0928\u093f\u092f\u093e"]
        assert tokenizer.tokens("\u00ab\u00bfQu\u00e9?\u00bb") == ["qu\u00e9"]

    def test_stemmer_and_content_words(self):
        stemmer = Stemmer()
        assert stemmer.stem("checks") == "check"
        assert stemmer.stem("tried") == "try"
        assert stemmer.stem("running") == "runn"
        content = ContentWords(["the", "a"], ["um"])
        assert content.of(["the", "um", "servers", "x", "decide"]) == {"server", "decid"}


class TestRetakes:
    def test_sentence_said_again_is_cut_back_to_the_retake(self, settings, talk):
        talk.say(1.0, "Welcome to the session.")
        talk.say(3.0, "The scheduler moves every job to the quietest node.")
        talk.say(8.0, "The scheduler moves every job to the quietest node in the pool.")
        analysis = talk.analysis(settings, total=14.0)
        found = RetakeDetector(analysis).find()
        assert len(found) == 1
        cut = found[0]
        assert (cut.type, cut.kind, cut.confidence, cut.group, cut.level) == ("retake", "strong", "high", "retakes", None)
        assert Levels(settings).applies(cut)
        assert 2.35 < cut.start < 3.0
        assert 6.1 < cut.end < 8.0

    def test_parallel_sentence_with_a_swapped_word_is_only_proposed(self, settings, talk):
        talk.say(1.0, "Let's look at an example.")
        talk.say(3.0, "Here is the checkout page before the change.")
        talk.say(6.5, "Here is the checkout page after the change.")
        talk.say(10.0, "Here is the checkout page after the change, with one clear button.")
        analysis = talk.analysis(settings, total=16.0)
        found = RetakeDetector(analysis).find()
        assert [(cut.kind, cut.level) for cut in found] == [("fuzzy", "propose"), ("strong", None)]
        assert found[0].note.startswith(
            "said again with a different word (\u201cbefore\u201d \u2192 \u201cafter\u201d); keep if deliberate. ")
        final = Reconciler(analysis).run(Planner(analysis).run())
        (first_start, first_end), (second_start, second_end) = final.keep
        assert first_start == 0.0
        assert 5.75 < first_end < 6.5
        assert 9.25 < second_start < 10.0
        assert [(cut.kind, cut.level) for cut in final.proposals] == [("fuzzy", "propose")]

    def test_substitution_level_auto_cuts_the_parallel_sentence_too(self, settings, talk):
        settings["retakes"]["substitutionLevel"] = "auto"
        talk.say(1.0, "Let's look at an example.")
        talk.say(3.0, "Here is the checkout page before the change.")
        talk.say(6.5, "Here is the checkout page after the change.")
        talk.say(10.0, "Here is the checkout page after the change, with one clear button.")
        analysis = talk.analysis(settings, total=16.0)
        final = Reconciler(analysis).run(Planner(analysis).run())
        (_, first_end), (second_start, _) = final.keep
        assert 2.7 < first_end < 3.0
        assert 9.25 < second_start < 10.0

    def test_spoken_correction_is_proposed(self, settings, talk):
        talk.say(1.0, "Let's look at an example.")
        talk.say(3.0, "Here is the checkout page after the change.")
        talk.say(6.5, "Here is the checkout page after the update.")
        found = RetakeDetector(talk.analysis(settings, total=12.0)).find()
        assert [(cut.kind, cut.confidence, cut.level) for cut in found] == [("strong", "high", "propose")]
        assert "(\u201cchange\u201d \u2192 \u201cupdate\u201d)" in found[0].note
        assert not Levels(settings).applies(found[0])
        settings["retakes"]["substitutionLevel"] = "auto"
        found = RetakeDetector(talk.analysis(settings, total=12.0)).find()
        assert [(cut.kind, cut.level) for cut in found] == [("strong", None)]
        assert Levels(settings).applies(found[0])
        settings["retakes"]["substitutionLevel"] = "off"
        assert RetakeDetector(talk.analysis(settings, total=12.0)).find() == []

    def test_restarts_and_extensions_have_no_swaps(self, settings, talk):
        talk.say(1.0, "The second part is to measure, to measure the")
        talk.say(5.5, "The second part is to measure what users actually do.")
        talk.say(10.0, "We ship small changes.")
        talk.say(12.0, "We ship small changes behind a switch.")
        talk.say(15.5, "Every answer needs facts.")
        talk.say(17.5, "Every answer needs a fact.")
        talk.say(20.0, "The second part is to measure the")
        talk.say(23.5, "The second part is to measure users.")
        talk.say(27.0, "Here is the checkout page before the change.")
        talk.say(30.5, "Here is the cart page after the change.")
        talk.say(34.0, "Every answer needs a source and a date.")
        talk.say(37.5, "Every answer needs a clear source plus the date and a reason.")
        talk.say(42.0, "The service writes each reply into a notebook.")
        talk.say(45.5, "The service writes every reply into the notebook now.")
        talk.say(50.0, "The agent's answer is cached.")
        talk.say(52.5, "The agents answer is cached.")
        analysis = talk.analysis(settings, total=55.0)
        units = analysis.units
        assert len(units) == 16
        swaps = Substitutions(analysis)
        assert swaps.find(units[0], units[1]) == []
        assert swaps.find(units[2], units[3]) == []
        assert swaps.find(units[4], units[5]) == []
        assert swaps.find(units[6], units[7]) == []
        assert swaps.find(units[8], units[9]) == [("checkout", "cart"), ("before", "after")]
        assert swaps.find(units[10], units[11]) == []
        assert swaps.find(units[12], units[13]) == []
        assert swaps.find(units[14], units[15]) == []

    def test_loose_match_is_only_proposed(self, settings, talk):
        talk.say(1.0, "Every answer needs a source and a date.")
        talk.say(6.0, "Every answer needs a clear source plus the date and a reason.")
        analysis = talk.analysis(settings, total=12.0)
        found = RetakeDetector(analysis).find()
        assert [(cut.kind, cut.confidence) for cut in found] == [("fuzzy", "low")]

    def test_attempt_that_starts_mid_sentence_is_not_applied(self, settings, talk):
        talk.say(1.0, "and the cache layer")
        talk.say(3.2, "The scheduler moves every job to the quietest node.")
        talk.say(7.5, "The scheduler moves every job to the quietest server.")
        analysis = talk.analysis(settings, total=12.0)
        found = RetakeDetector(analysis).find()
        assert found[0].confidence == "low"
        assert "mid-sentence" in found[0].fails[0]

    def test_script_lines_tell_retakes_from_planned_repeats(self, settings, talk):
        script_text = ("## Slide 1 \u2014 Intro\n### Notes\nignored here\n### Say\nThe cache keeps the last answer.\n[click]\n"
                       "## Slide 2 \u2014 More\n### Say\nThe cache keeps the last answer for a day.\n")
        reader = ScriptText(settings["script"], Tokenizer(), ContentWords(settings["text"]["stopWords"], []))
        sentences = reader.sentences(script_text)
        assert [(sentence.section, sentence.text) for sentence in sentences] == [
            (1, "The cache keeps the last answer."), (2, "The cache keeps the last answer for a day.")]
        talk.say(1.0, "The cache keeps the last answer.")
        talk.say(4.0, "The cache keeps the last answer for a day.")
        analysis = talk.analysis(settings, total=10.0, script=sentences)
        assert RetakeDetector(analysis).find() == []


class TestRepeats:
    def test_restarted_phrase_inside_a_sentence(self, settings, talk):
        talk.say(1.0, "We shipped it on, shipped it on a Friday with no rollback plan.")
        analysis = talk.analysis(settings, total=8.0)
        found = RepeatDetector(analysis).find()
        assert len(found) == 1
        cut = found[0]
        assert (cut.type, cut.group, cut.confidence) == ("repeat", "repeats", "high")
        words = analysis.words
        assert words[cut.first_word].text.strip() == "shipped"
        assert words[cut.end_word].text.strip() == "shipped"
        assert cut.end_word == cut.first_word + 3

    def test_repeats_are_proposed_by_default_and_cut_when_auto(self, settings, talk):
        talk.say(1.0, "We shipped it on, shipped it on a Friday with no rollback plan.")
        result = Planner(talk.analysis(settings, total=8.0)).run()
        assert [cut.type for cut in result.proposals] == ["repeat"]
        assert "repeat" not in [cut.type for cut in result.painted]
        settings["repeats"]["level"] = "auto"
        result = Planner(talk.analysis(settings, total=8.0)).run()
        assert "repeat" in [cut.type for cut in result.painted]

    def test_off_level_does_not_run(self, settings, talk):
        talk.say(1.0, "We shipped it on, shipped it on a Friday with no rollback plan.")
        settings["repeats"]["level"] = "off"
        result = Planner(talk.analysis(settings, total=8.0)).run()
        assert all(cut.type != "repeat" for cut in result.proposals + result.painted)


class TestCues:
    def test_cue_phrase_cuts_back_over_the_abandoned_attempt(self, settings, talk):
        talk.say(1.0, "Welcome everyone.")
        talk.say(3.0, "The cache keeps the last answer.")
        talk.say(6.0, "Let me redo that.")
        retake_start = 8.0
        talk.say(retake_start, "The cache keeps the latest answer.")
        analysis = talk.analysis(settings, total=12.0)
        found = CueDetector(analysis).find()
        assert found
        for cut in found:
            assert (cut.type, cut.confidence) == ("cue", "high")
            assert 1.65 < cut.start < 3.0
            assert 7.35 < cut.end < retake_start

    def test_low_confidence_phrase_alone_is_low(self, settings, talk):
        talk.say(1.0, "Welcome to the talk.")
        talk.say(3.0, "Oops, the slide is wrong.")
        found = CueDetector(talk.analysis(settings, total=6.0)).find()
        assert [(cut.type, cut.confidence) for cut in found] == [("cue", "low")]

    def test_standalone_stop_cuts_back_to_the_retake(self, settings, talk):
        talk.say(1.0, "Hello and welcome.")
        talk.say(3.0, "The service writes every request to the audit log.")
        talk.say(7.0, "Stop.")
        talk.say(9.0, "The service writes every request to the audit log first.")
        analysis = talk.analysis(settings, total=14.0)
        found = StopCueDetector(analysis).find()
        assert len(found) == 1
        cut = found[0]
        assert analysis.words[cut.first_word].text.strip() == "The"
        assert analysis.words[cut.first_word].start == 3.0
        assert analysis.words[cut.end_word].start == 9.0

    def test_cue_words_are_configurable(self, settings, talk):
        talk.say(1.0, "The service writes every request to the audit log.")
        talk.say(5.0, "Again.")
        talk.say(7.0, "The service writes every request to the audit log first.")
        assert StopCueDetector(talk.analysis(settings, total=12.0)).find() == []
        settings["cues"]["words"] = ["again"]
        assert len(StopCueDetector(talk.analysis(settings, total=12.0)).find()) == 1

    def test_a_cue_phrase_the_script_says_is_not_a_cue(self, settings, talk):
        talk.say(1.0, "Let us go over the cache rules.")
        talk.say(4.0, "Let us go over the cache rules one more time.")
        assert len(CueDetector(talk.analysis(settings, total=9.0)).find()) == 1
        reader = ScriptText(settings["script"], Tokenizer(), ContentWords(settings["text"]["stopWords"], []))
        script = reader.sentences("## Slide 1 \u2014 Recap\n### Say\nLet us go over the cache rules one more time.\n")
        assert CueDetector(talk.analysis(settings, total=9.0, script=script)).find() == []


class TestOffTalk:
    def test_isolated_aside_is_found_and_proposed_by_default(self, settings, talk):
        talk.say(1.0, "Welcome to the talk.")
        talk.say(4.0, "Sorry, one second please.")
        talk.say(7.0, "The first topic is caching.")
        analysis = talk.analysis(settings, total=10.0)
        found = OffTalkDetector(analysis).find()
        assert [(cut.type, cut.confidence) for cut in found] == [("offtalk", "high")]
        assert 3.7 < found[0].start < 4.0
        assert 5.3 < found[0].end < 5.7
        result = Planner(analysis).run()
        assert "offtalk" in [cut.type for cut in result.proposals]

    def test_greeting_is_not_an_aside(self, settings, talk):
        talk.say(1.0, "Hello everyone.")
        talk.say(4.0, "The first topic is caching.")
        assert OffTalkDetector(talk.analysis(settings, total=8.0)).find() == []


class TestTrims:
    def test_long_pause_is_trimmed_and_leads_are_cut(self, settings, talk):
        talk.say(3.0, "First part of the talk.")
        talk.say(10.0, "Second part of the talk.")
        analysis = talk.analysis(settings, total=15.0)
        found = TrimDetector(analysis, Levels(settings)).find([])
        spans = {cut.type: (round(cut.start, 3), round(cut.end, 3)) for cut in found}
        assert spans == {"pause": (5.2, 9.5), "lead-in": (0.0, 2.0), "lead-out": (13.2, 15.0)}

    def test_pause_settings_are_used(self, settings, talk):
        talk.say(1.0, "First part.")
        talk.say(5.0, "Second part.")
        settings["pauses"]["maxSec"] = 5.0
        found = TrimDetector(talk.analysis(settings, total=7.0), Levels(settings)).find([])
        assert "pause" not in [cut.type for cut in found]


class TestFillers:
    def test_fillers_are_off_by_default_and_cut_word_by_word_when_on(self, settings, talk):
        talk.say(1.0, "So um the server answers.")
        result = Planner(talk.analysis(settings, total=5.0)).run()
        assert "filler" not in [cut.type for cut in result.painted + result.proposals]
        settings["fillers"]["level"] = "auto"
        analysis = talk.analysis(settings, total=5.0)
        found = FillerDetector(analysis).find()
        assert len(found) == 1
        assert analysis.words[found[0].first_word].text.strip() == "um"


class TestUnclear:
    def proposals(self, settings, talk, total, segments=()):
        analysis = talk.analysis(settings, total=total, segments=segments)
        return analysis, [cut for cut in Planner(analysis).run().proposals if cut.type == "unclear"]

    def test_overlong_word_is_proposed(self, settings, talk):
        talk.say(1.0, "Welcome to the session.")
        talk.words.append({"start": 3.0, "end": 9.0, "word": " this.", "probability": 0.99})
        talk.say(10.0, "The next part.")
        analysis, found = self.proposals(settings, talk, 14.0)
        assert [(cut.start, cut.end) for cut in found] == [(3.0, 9.0)]
        assert "one word, \u201cthis.\u201d, lasts 6.0 s" in found[0].note
        assert analysis.word_untrusted[4] is True

    def test_word_stretched_into_silence_is_not_overlong(self, settings, talk):
        talk.say(1.0, "Welcome to the session.")
        talk.words.append({"start": 3.0, "end": 9.0, "word": " this.", "probability": 0.99})
        analysis = talk.analysis(settings, total=12.0)
        analysis.energy.db[int(3.4 / 0.02):] = -80.0
        quiet = type(analysis)(talk.words, analysis.energy, 12.0, [], [], settings)
        assert [span for span in quiet.unclear_spans if "lasts" in span.reasons[0]] == []

    def test_long_low_confidence_run_is_proposed_short_one_is_not(self, settings, talk):
        talk.say(1.0, "Welcome to the session.")
        talk.say(3.0, "mumble mumble mumble mumble", probability=0.2)
        talk.say(6.0, "Back to the talk.")
        talk.say(8.0, "mumble mumble", probability=0.2)
        _, found = self.proposals(settings, talk, 12.0)
        assert [(cut.start, round(cut.end, 2)) for cut in found] == [(3.0, 4.35)]
        assert "4 low-confidence words in a row" in found[0].note

    def test_a_long_pause_ends_a_low_confidence_run(self, settings, talk):
        talk.say(1.0, "Welcome to the session.")
        talk.say(3.0, "attempt.", probability=0.2)
        talk.say(7.5, "Both", probability=0.2)
        talk.say(7.85, "failure branches matter.")
        _, found = self.proposals(settings, talk, 10.0)
        assert found == []

    def test_doubtful_and_hallucinated_segments(self, settings, talk):
        talk.say(1.0, "Welcome to the session.")
        talk.say(3.0, "Something hard to hear here.")
        talk.say(6.0, "See you in the next video.")
        segments = [{"start": 3.0, "end": 4.7, "text": " Something hard to hear here.", "avg_logprob": -1.2},
                    {"start": 6.0, "end": 8.05, "text": " See you in the next video.", "avg_logprob": -0.3}]
        analysis, found = self.proposals(settings, talk, 10.0, segments)
        assert [(cut.start, cut.end) for cut in found] == [(3.0, 4.7), (6.0, 8.05)]
        assert "average log-probability -1.20" in found[0].note
        assert "known hallucination (\u201csee you in the next video\u201d)" in found[1].note
        hallucinated = [unit for unit in analysis.units if unit.start >= 6.0]
        assert [unit.tokens for unit in hallucinated] == [[]]

    def test_unclear_stretches_split_sentences(self, settings, talk):
        talk.say(1.0, "the quick brown")
        talk.say(2.05, "mumble mumble mumble mumble", probability=0.2)
        talk.say(3.45, "fox jumps over the dog.")
        analysis = talk.analysis(settings, total=6.0)
        assert [unit.tokens[0] for unit in analysis.units] == ["the", "mumble", "fox"]
        settings["unclear"]["breakSentences"] = False
        assert len(talk.analysis(settings, total=6.0).units) == 1

    def test_a_hallucinated_line_no_longer_blocks_a_retake(self, settings, talk):
        talk.say(1.0, "Welcome to the session.")
        talk.say(3.0, "The scheduler moves every job to the quietest node.")
        talk.say(6.8, "See you in the next video.")
        talk.say(9.5, "The scheduler moves every job to the quietest server.")
        segments = [{"start": 6.8, "end": 8.85, "text": " See you in the next video.", "avg_logprob": -0.4}]
        retakes = RetakeDetector(talk.analysis(settings, total=15.0, segments=segments)).find()
        assert [(cut.kind, cut.confidence) for cut in retakes] == [("strong", "high")]
        before = RetakeDetector(talk.analysis(settings, total=15.0)).find()
        assert [cut.confidence for cut in before] == ["low"]

    def test_speech_with_no_words_joins_the_overlong_word_after_it(self, settings, talk):
        talk.say(1.0, "Welcome to the session.")
        talk.words.append({"start": 12.0, "end": 20.0, "word": " this.", "probability": 0.99})
        energy = talk.energy(24.0, settings)
        energy.db[int(4.0 / 0.02):int(5.5 / 0.02)] = -20.0
        energy.db[int(7.0 / 0.02):int(11.6 / 0.02)] = -20.0
        analysis = Analysis(talk.words, energy, 24.0, [], [], settings, [])
        found = [cut for cut in Planner(analysis).run().proposals if cut.type == "unclear"]
        assert [(cut.start, cut.end) for cut in found] == [(4.0, 20.0)]
        assert "6.1 s of speech with no words" in found[0].note
        assert "one word, \u201cthis.\u201d, lasts 8.0 s" in found[0].note

    def test_level_and_failed_ranges(self, settings, talk):
        talk.say(1.0, "Welcome to the session.")
        talk.say(4.0, "Back again.")
        analysis = talk.analysis(settings, total=8.0, failed=[[2.5, 3.8]])
        found = [cut for cut in Planner(analysis).run().proposals if cut.type == "unclear"]
        assert ["transcription failed" in cut.note for cut in found] == [True]
        settings["unclear"]["level"] = "off"
        analysis = talk.analysis(settings, total=8.0, failed=[[2.5, 3.8]])
        assert all(cut.type != "unclear" for cut in Planner(analysis).run().proposals)


class TestPlanner:
    def test_overlapping_cuts_merge_and_pauses_split_around_content(self):
        merged = Merger().merge([Candidate(1.0, 3.0, "cue", "cues", "high", "a"),
                                 Candidate(2.0, 4.0, "retake", "retakes", "high", "b")])
        assert [(cut.start, cut.end, cut.type, cut.merged) for cut in merged] == [(1.0, 4.0, "retake", 2)]
        painted = Painter(0.05).paint(merged + [Candidate(0.5, 6.0, "pause", "pauses", "high", "p")])
        assert [(cut.start, cut.end, cut.type) for cut in painted] == [
            (0.5, 1.0, "pause"), (1.0, 4.0, "retake"), (4.0, 6.0, "pause")]

    def test_tiny_kept_sliver_without_words_becomes_a_gap(self):
        cuts = [Candidate(1.0, 2.0, "pause", "pauses", "high", ""), Candidate(2.1, 3.0, "pause", "pauses", "high", "")]
        keep, gaps = KeepRanges(5.0, [0.5, 4.0], 0.25).compute(cuts)
        assert keep == [[0.0, 1.0], [3.0, 5.0]]
        assert [(gap.start, gap.end, gap.type) for gap in gaps] == [(2.0, 2.1, "gap")]
        keep, gaps = KeepRanges(5.0, [0.5, 2.05, 4.0], 0.25).compute(cuts)
        assert keep == [[0.0, 1.0], [2.0, 2.1], [3.0, 5.0]]
        assert gaps == []

    def test_first_pass_cuts_the_retake_and_keeps_the_rest(self, settings, talk):
        talk.say(2.0, "Welcome to the session.")
        talk.say(4.0, "The scheduler moves every job to the quietest node.")
        talk.say(9.0, "The scheduler moves every job to the quietest node in the pool.")
        result = Planner(talk.analysis(settings, total=15.0)).run()
        types = [cut.type for cut in result.painted]
        assert types.count("retake") == 1
        assert "lead-in" in types
        assert "lead-out" in types
        kept_middles = [2.15, 4.15, 9.15]
        inside = [any(start <= middle < end for start, end in result.keep) for middle in kept_middles]
        assert inside == [True, False, True]
