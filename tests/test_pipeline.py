import json
import wave

import numpy as np
import pytest

from rostrum.audio import SAMPLE_RATE, AudioDecoder
from rostrum.cli import main
from rostrum.config import ConfigError
from rostrum.detect.candidate import Candidate
from rostrum.detect.planner import Planner
from rostrum.edl import Cut, Edl, Source
from rostrum.engines.base import Transcript
from rostrum.pipeline import CutFactory, DecisionsFile, EdlBuilder, SourceStage, TimelineWords
from rostrum.tools import Tools


def word(start, end, text, probability=0.9):
    return {"start": start, "end": end, "word": text, "probability": probability}


class TestTalkTimeline:
    def test_sources_are_joined_end_to_end(self):
        first = Transcript(words=[word(1.0, 1.5, " one"), word(9.8, 10.4, " two"), word(10.2, 10.5, " gone")],
                           segments=[{"start": 1.0, "end": 10.4, "text": " one two", "avg_logprob": -0.3, "words": []}],
                           failed=[[2.0, 3.0]])
        second = Transcript(words=[word(0.5, 0.9, " three")], segments=[{"start": 0.5, "end": 0.9, "text": " three"}],
                            failed=[[1.0, 2.0]])
        sources = [Source("a.mp4", 10.0, 0.0), Source("b.mp4", 8.0, 10.0)]
        joined = TimelineWords().join([first, second], sources)
        assert [(item["word"], item["start"], item["end"], item["source"]) for item in joined.words] == [
            (" one", 1.0, 1.5, 0), (" two", 9.8, 10.0, 0), (" three", 10.5, 10.9, 1)]
        assert joined.segments == [{"start": 1.0, "end": 10.0, "text": " one two", "avg_logprob": -0.3},
                                   {"start": 10.5, "end": 10.9, "text": " three"}]
        assert joined.failed == [[2.0, 3.0], [11.0, 12.0]]
        assert joined.estimated is False

    def test_retake_across_a_file_boundary_is_found_and_split_per_file(self, settings, make_talk):
        first = make_talk()
        first.say(1.0, "Welcome to the demo.")
        first.say(5.0, "The scheduler moves every job to the quietest node.")
        second = make_talk()
        second.say(0.5, "The scheduler moves every job to the quietest node now.")
        second.say(5.0, "That is all for today.")
        sources = [Source("a.mp4", 9.0, 0.0), Source("b.mp4", 9.0, 9.0)]
        words = TimelineWords().join([Transcript(first.words), Transcript(second.words)], sources).words
        timeline = make_talk()
        timeline.words = words
        result = Planner(timeline.analysis(settings, total=18.0)).run()
        retakes = [cut for cut in result.painted if cut.type == "retake"]
        assert len(retakes) == 1
        assert retakes[0].start < 9.0 < retakes[0].end
        segments = Edl("demo", sources, result.keep).segments()
        assert [segment.source_index for segment in segments] == [0, 1]
        assert segments[1].start == pytest.approx(retakes[0].end - 9.0, abs=0.001)


class TestDecisions:
    def cuts(self):
        return [Cut("retake-aaaa", 2.0, 5.0, "retake", "auto", "cut", "r", "e", 0.9, "auto"),
                Cut("pause-bbbb", 7.0, 8.0, "pause", "auto", "cut", "p", "", 1.0, "auto"),
                Cut("offtalk-cccc", 9.0, 9.5, "offtalk", "propose", "keep", "o", "", 0.9, "auto")]

    def builder(self):
        return EdlBuilder("t", [Source("a.mp4", 12.0)], "30/1", [1.0, 6.0, 9.2, 11.0], {"minKeepSec": 0.25}, "key")

    def test_keep_is_the_timeline_minus_cuts_decided_cut(self):
        result = self.builder().build(self.cuts(), {})
        assert result.edl.keep == [[0.0, 2.0], [5.0, 7.0], [8.0, 12.0]]
        assert result.stale == []

    def test_user_decisions_and_edges_apply_and_unknown_ids_are_stale(self):
        decisions = {"retake-aaaa": {"decision": "keep", "decided_by": "user"},
                     "offtalk-cccc": {"decision": "cut", "end": 9.8, "decided_by": "user"},
                     "gone-1234": {"decision": "cut", "decided_by": "user"}}
        result = self.builder().build(self.cuts(), decisions)
        assert result.edl.keep == [[0.0, 7.0], [8.0, 9.0], [9.8, 12.0]]
        assert [entry["id"] for entry in result.stale] == ["gone-1234"]
        decided = {cut.id: (cut.decision, cut.decided_by, cut.end) for cut in result.cuts}
        assert decided["retake-aaaa"] == ("keep", "user", 5.0)
        assert decided["offtalk-cccc"] == ("cut", "user", 9.8)

    def test_tiny_leftover_becomes_a_gap_that_can_be_kept(self):
        cuts = [Cut("pause-a", 1.0, 2.0, "pause"), Cut("pause-b", 2.1, 3.0, "pause")]
        builder = EdlBuilder("t", [Source("a.mp4", 5.0)], "", [0.5, 4.0], {"minKeepSec": 0.25}, "key")
        result = builder.build(cuts, {})
        gaps = [cut for cut in result.cuts if cut.type == "gap"]
        assert len(gaps) == 1
        assert result.edl.keep == [[0.0, 1.0], [3.0, 5.0]]
        result = builder.build(cuts, {gaps[0].id: {"decision": "keep", "decided_by": "user"}})
        assert result.edl.keep == [[0.0, 1.0], [2.0, 2.1], [3.0, 5.0]]

    def test_decisions_file_round_trip(self, tmp_path):
        decisions = DecisionsFile(tmp_path / "decisions.json")
        decisions.set("t", "retake-aaaa", "keep", start=None, end=6.5)
        entry = decisions.load()["retake-aaaa"]
        assert (entry["decision"], entry["decided_by"], entry["end"]) == ("keep", "user", 6.5)
        assert "start" not in entry
        with pytest.raises(ConfigError, match="keep or cut"):
            decisions.set("t", "retake-aaaa", "maybe")

    def test_review_and_change_times_come_from_the_file_or_its_time(self, tmp_path):
        path = tmp_path / "decisions.json"
        decisions = DecisionsFile(path)
        assert (decisions.review_time(), decisions.change_time()) == (None, None)
        decisions.save("t", {}, 1234.5, 1200.0)
        assert (decisions.review_time(), decisions.change_time()) == (1234.5, 1200.0)
        decisions.save("t", {}, None, None)
        assert (decisions.review_time(), decisions.change_time()) == (None, None)
        assert json.loads(path.read_text())["reviewedAt"] is None
        path.write_text(json.dumps({"schema": "rostrum.decisions/1", "talk": "t", "decisions": {}}))
        assert decisions.review_time() == path.stat().st_mtime
        assert decisions.change_time() == path.stat().st_mtime
        assert json.loads((tmp_path / "decisions.json.bak").read_text())["reviewedAt"] == 1234.5

    def test_ids_are_stable_and_unique(self):
        candidate = Candidate(2.0, 5.0, "retake", "retakes", "high", "note")
        first = CutFactory("key")
        second = CutFactory("key")
        one = first.cut(candidate, applied=True, evidence="so the service [checks] it")
        assert second.cut(candidate, applied=True, evidence="so the service [checks] it").id == one.id
        assert first.cut(candidate, applied=True, evidence="so the service [checks] it").id == one.id + "-2"
        assert (one.level, one.decision) == ("auto", "cut")
        proposal = CutFactory("key").cut(candidate, applied=False, evidence="x")
        assert (proposal.level, proposal.decision) == ("propose", "keep")


class TestTwoPartTalkEndToEnd:
    def run(self, capsys, *args):
        code = main([*args, "--json"])
        return code, json.loads(capsys.readouterr().out)

    def test_init_plan_decide_and_stale(self, tmp_path, media, make_talk, capsys):
        folder = tmp_path / "event"
        folder.mkdir()
        media.clip(folder / "demo-part1.mp4", 6)
        media.clip(folder / "demo-part2.mp4", 6, frequency=500)
        first = make_talk()
        first.say(0.5, "Welcome to the demo.")
        first.say(2.0, "The scheduler moves every job to the quietest node.")
        second = make_talk()
        second.say(0.5, "The scheduler moves every job to the quietest node now.")
        second.say(4.0, "That is all for today.")
        (folder / "demo-part1.words.json").write_text(json.dumps({"words": first.words}), encoding="utf-8")
        (folder / "demo-part2.words.json").write_text(json.dumps({"words": second.words}), encoding="utf-8")
        project = tmp_path / "rostrum.json"

        code, data = self.run(capsys, "init", str(folder), "--engine", "import", "-p", str(project))
        assert code == 0, data
        assert data["talks"][0]["id"] == "demo"
        written = json.loads(project.read_text(encoding="utf-8"))
        assert written["talks"][0]["sources"] == ["event/demo-part1.mp4", "event/demo-part2.mp4"]
        assert written["defaults"]["speech"]["engine"] == "import"

        candidates = json.loads((tmp_path / "demo.rostrum" / "candidates.json").read_text(encoding="utf-8"))
        boundary = candidates["sources"][1]["offset"]
        assert 5.9 < boundary < 6.1
        retakes = [cut for cut in candidates["cuts"] if cut["type"] == "retake"]
        assert len(retakes) == 1
        assert retakes[0]["start"] < boundary < retakes[0]["end"]

        code, plan = self.run(capsys, "plan", "-p", str(project))
        before = plan["talks"][0]["kept"]
        assert code == 0 and plan["talks"][0]["totals"]["retake"]["count"] == 1

        code, decided = self.run(capsys, "decide", retakes[0]["id"], "keep", "-p", str(project))
        assert code == 0
        assert decided["kept"] == pytest.approx(before + retakes[0]["end"] - retakes[0]["start"], abs=0.01)
        edl = json.loads((tmp_path / "demo.rostrum" / "edl.json").read_text(encoding="utf-8"))
        assert edl["schema"] == "rostrum.edl/1"
        assert next(cut for cut in edl["cuts"] if cut["id"] == retakes[0]["id"])["decided_by"] == "user"

        changed = make_talk()
        changed.say(0.5, "The scheduler moves each job to the quietest server.")
        changed.say(4.0, "That is all for today.")
        (folder / "demo-part2.words.json").write_text(json.dumps({"words": changed.words}), encoding="utf-8")
        code, _ = self.run(capsys, "init", "-p", str(project))
        assert code == 0
        code, plan = self.run(capsys, "plan", "-p", str(project))
        assert [entry["id"] for entry in plan["talks"][0]["stale"]] == [retakes[0]["id"]]
        assert all(cut["decided_by"] == "auto" for cut in plan["talks"][0]["cuts"])

        code, status = self.run(capsys, "status", "-p", str(project))
        assert code == 0
        assert status["talks"][0]["stale"] == 1
        assert status["talks"][0]["state"] == "ready for review"

    def test_rerun_uses_the_cache(self, tmp_path, media, make_talk, capsys, monkeypatch):
        media.clip(tmp_path / "solo.mp4", 4)
        speech = make_talk()
        speech.say(0.5, "One short talk.")
        (tmp_path / "solo.words.json").write_text(json.dumps({"words": speech.words}), encoding="utf-8")
        project = tmp_path / "rostrum.json"
        assert self.run(capsys, "init", str(tmp_path / "solo.mp4"), "--engine", "import", "-p", str(project))[0] == 0
        work = tmp_path / "solo.rostrum"
        stamps = {name: (work / name).stat().st_mtime_ns for name in ("probe.json", "transcript-1.json", "detect.json")}
        assert self.run(capsys, "init", "-p", str(project))[0] == 0
        assert {name: (work / name).stat().st_mtime_ns for name in stamps} == stamps
        assert (tmp_path / "rostrum.json.bak").exists() is False
        assert self.run(capsys, "init", str(tmp_path / "solo.mp4"), "-p", str(project))[0] == 0
        assert (tmp_path / "rostrum.json.bak").exists()
        monkeypatch.setattr("rostrum.pipeline.__version__", "0.0.0-test")
        assert self.run(capsys, "init", "-p", str(project))[0] == 0
        changed = {name for name in stamps if (work / name).stat().st_mtime_ns != stamps[name]}
        assert changed == {"probe.json", "detect.json"}

    def test_init_of_a_new_talk_keeps_the_review_of_an_unchanged_one(self, tmp_path, media, make_talk, capsys):
        project = tmp_path / "rostrum.json"
        for name in ("alpha", "beta"):
            media.clip(tmp_path / f"{name}.mp4", 8)
            speech = make_talk()
            speech.say(0.5, "The scheduler moves every job to the quietest node.")
            speech.say(4.0, "The scheduler moves every job to the quietest node now.")
            (tmp_path / f"{name}.words.json").write_text(json.dumps({"words": speech.words}), encoding="utf-8")
        assert self.run(capsys, "init", str(tmp_path / "alpha.mp4"), "--engine", "import", "-p", str(project))[0] == 0
        candidates = tmp_path / "alpha.rostrum" / "candidates.json"
        first = json.loads(candidates.read_text(encoding="utf-8"))
        assert first["cuts"]
        assert self.run(capsys, "decide", first["cuts"][0]["id"], "keep", "-p", str(project))[0] == 0

        assert self.run(capsys, "init", str(tmp_path / "beta.mp4"), "-p", str(project))[0] == 0
        assert json.loads(candidates.read_text(encoding="utf-8"))["analyzedAt"] == first["analyzedAt"]
        code, status = self.run(capsys, "status", "-p", str(project))
        assert {talk["id"]: talk["state"] for talk in status["talks"]} == {
            "alpha": "reviewed, ready to render", "beta": "ready for review"}
        code, data = self.run(capsys, "render", "-p", str(project))
        assert code == 2
        assert "not reviewed since the cuts last changed: beta." in data["error"]
        code, data = self.run(capsys, "render", "--talk", "alpha", "-p", str(project))
        assert code == 0, data


class TestSourceStreams:
    def test_late_audio_is_padded_onto_the_file_clock(self, media, tmp_path):
        clip = media.late_audio(tmp_path / "late.mp4", 4, 1.0)
        offset = SourceStage(None, None, None, Tools(), {}).describe(clip)["audioOffset"]
        assert offset == pytest.approx(1.0, abs=0.1)
        decoder = AudioDecoder(Tools())
        audio = decoder.decode(clip, offset)
        onset = np.nonzero(np.abs(audio) > 0.01)[0][0] / SAMPLE_RATE
        assert onset == pytest.approx(1.0, abs=0.005)
        quiet = np.sqrt(np.mean(audio[int(0.2 * SAMPLE_RATE):int(0.8 * SAMPLE_RATE)] ** 2))
        loud = np.sqrt(np.mean(audio[int(1.3 * SAMPLE_RATE):int(1.9 * SAMPLE_RATE)] ** 2))
        assert (quiet < 0.001, loud > 0.05) == (True, True)
        assert len(audio) - len(decoder.decode(clip)) == round(offset * SAMPLE_RATE)
        written = decoder.write_wav(clip, tmp_path / "late.wav", offset)
        with wave.open(str(written)) as handle:
            assert (handle.getframerate(), handle.getnchannels(), handle.getnframes()) == (SAMPLE_RATE, 1, len(audio))

    def test_a_source_without_video_or_audio_fails_init_early(self, media, tmp_path, capsys):
        tone = media.wav(tmp_path / "tone.wav", 2)
        silent = media.video_only(tmp_path / "silent.mp4", 2)
        project = tmp_path / "rostrum.json"
        code = main(["init", str(tone), str(silent), "--engine", "import", "-p", str(project), "--json"])
        data = json.loads(capsys.readouterr().out)
        assert code == 1
        errors = {talk["id"]: talk["error"] for talk in data["talks"]}
        assert "tone.wav has no video stream" in errors["tone"]
        assert "silent.mp4 has no audio stream" in errors["silent"]
