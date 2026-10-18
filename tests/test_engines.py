import json
import os
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from rostrum.audio import Energy
from rostrum.engines import EngineError, EngineRegistry, SourceInput, UnknownEngine
from rostrum.engines.base import WorkerCount
from rostrum.engines.faster_whisper import ChunkMerge, ChunkPlanner, CuePass, EchoCheck, FasterWhisperEngine
from rostrum.engines.formats import TranscriptParser
from rostrum.engines.importer import ImportEngine
from rostrum.tools import Tools


def source_input(path, tmp_path, transcript=None, duration=10.0):
    return SourceInput(index=0, path=Path(path), duration=duration, scratch=tmp_path / "scratch",
                       transcript=Path(transcript) if transcript else None)


class TestTranscriptFormats:
    def test_words_json_in_our_shape(self, settings):
        text = json.dumps({"duration": 10.0, "words": [{"start": 0.5, "end": 0.9, "word": " Hello", "probability": 0.9},
                                                       {"start": 1.0, "end": 1.3, "word": " there.", "probability": 0.8}],
                           "failed": [[4.0, 5.0]]})
        transcript = TranscriptParser(settings["speech"]).parse(text)
        assert [word["word"] for word in transcript.words] == [" Hello", " there."]
        assert transcript.duration == 10.0
        assert transcript.failed == [[4.0, 5.0]]

    def test_baseline_shape_with_duration_s_and_plain_segments(self, settings):
        text = json.dumps({"model": "m", "duration_s": 12.5,
                           "words": [{"start": 3.6, "end": 4.1, "word": " Hello,", "probability": 0.7}],
                           "segments": [{"start": 3.6, "end": 4.1, "text": " Hello,", "avg_logprob": -0.2}]})
        transcript = TranscriptParser(settings["speech"]).parse(text)
        assert transcript.duration == 12.5
        assert transcript.words[0]["probability"] == 0.7
        assert transcript.segments == [{"start": 3.6, "end": 4.1, "text": " Hello,", "avg_logprob": -0.2}]

    def test_words_nested_in_segments(self, settings):
        text = json.dumps({"segments": [{"start": 0, "end": 2, "text": "a b", "words": [
            {"start": 0.1, "end": 0.5, "word": "a"}, {"start": 0.6, "end": 1.0, "word": "b"}]}]})
        transcript = TranscriptParser(settings["speech"]).parse(text)
        assert [(word["word"], word["probability"]) for word in transcript.words] == [(" a", 1.0), (" b", 1.0)]

    def test_field_mapping_with_milliseconds(self, settings):
        settings["speech"]["fields"].update({"words": "transcription[]", "start": "offsets.from", "end": "offsets.to",
                                             "word": "text", "probability": "p", "timeScale": 0.001})
        text = json.dumps({"transcription": [{"offsets": {"from": 1000, "to": 1500}, "text": " stop", "p": 0.6},
                                             {"offsets": {"from": 1600, "to": 2000}, "text": " now", "p": 0.9}]})
        transcript = TranscriptParser(settings["speech"]).parse(text)
        assert transcript.words == [{"start": 1.0, "end": 1.5, "word": " stop", "probability": 0.6},
                                    {"start": 1.6, "end": 2.0, "word": " now", "probability": 0.9}]

    def test_top_level_word_list(self, settings):
        settings["speech"]["fields"]["words"] = ""
        text = json.dumps([{"start": 0.0, "end": 0.4, "word": "one"}, {"start": 0.5, "end": 0.9, "word": "two"}])
        assert len(TranscriptParser(settings["speech"]).parse(text).words) == 2

    def test_srt_words_are_spread_over_each_cue(self, settings):
        text = "1\n00:00:01,000 --> 00:00:03,000\nHello there everyone\n\n2\n00:00:04,500 --> 00:00:05,500\nNext part\n"
        transcript = TranscriptParser(settings["speech"]).parse(text)
        assert [word["word"] for word in transcript.words] == [" Hello", " there", " everyone", " Next", " part"]
        assert transcript.words[0]["start"] == 1.0
        assert transcript.words[2]["end"] == 3.0
        assert transcript.words[3]["start"] == 4.5
        assert transcript.meta["wordTimes"] == "estimated"

    def test_vtt_with_tags_and_short_times(self, settings):
        text = ("WEBVTT\n\n00:01.000 --> 00:03.000\n<v Speaker>Hello <b>there</b>\n\n"
                "00:00:04.000 --> 00:00:05.000 align:start\nAgain\n")
        transcript = TranscriptParser(settings["speech"]).parse(text)
        assert [word["word"] for word in transcript.words] == [" Hello", " there", " Again"]
        assert transcript.segments[0] == {"start": 1.0, "end": 3.0, "text": "Hello there"}
        assert transcript.meta["format"] == "vtt"

    def test_empty_inputs_are_clear_errors(self, settings):
        with pytest.raises(EngineError, match="no words found"):
            TranscriptParser(settings["speech"]).parse(json.dumps({"words": []}))
        with pytest.raises(EngineError, match="no SRT cues"):
            TranscriptParser(settings["speech"]).parse("nothing here")


class TestImportEngine:
    def test_finds_the_transcript_next_to_the_source(self, settings, tmp_path):
        media = tmp_path / "talk.mp4"
        media.write_bytes(b"")
        (tmp_path / "talk.words.json").write_text(json.dumps({"words": [{"start": 0, "end": 1, "word": " hi"}]}))
        engine = ImportEngine(settings, Tools())
        transcript = engine.transcribe(source_input(media, tmp_path))
        assert transcript.words[0]["word"] == " hi"
        assert transcript.meta["engine"] == "import"

    def test_listed_transcript_and_missing_one(self, settings, tmp_path):
        srt = tmp_path / "other.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi there\n")
        engine = ImportEngine(settings, Tools())
        assert len(engine.transcribe(source_input(tmp_path / "x.mp4", tmp_path, transcript=srt)).words) == 2
        with pytest.raises(EngineError, match="no transcript for x.mp4"):
            engine.transcribe(source_input(tmp_path / "x.mp4", tmp_path))


class TestCommandEngine:
    def engine(self, settings, command):
        settings["speech"]["engine"] = "command"
        settings["speech"]["command"] = command
        return EngineRegistry().create("command", settings, Tools())

    def test_words_json_on_stdout(self, settings, tmp_path):
        script = tmp_path / "fake_stt.py"
        script.write_text("import json, sys\n"
                          "print(json.dumps({'words': [{'start': 0.5, 'end': 0.9, 'word': 'hello', 'probability': 0.9},"
                          " {'start': 1.0, 'end': 1.4, 'word': sys.argv[1][-6:]}]}))\n")
        engine = self.engine(settings, [sys.executable, str(script), "{input}"])
        transcript = engine.transcribe(source_input(tmp_path / "talk.mp4", tmp_path))
        assert [word["word"] for word in transcript.words] == [" hello", " lk.mp4"]
        assert transcript.words[1]["probability"] == 1.0

    def test_output_file_with_an_added_extension(self, settings, tmp_path):
        script = tmp_path / "fake_stt.py"
        script.write_text("import sys\n"
                          "open(sys.argv[1] + '.srt', 'w').write('1\\n00:00:01,000 --> 00:00:02,000\\nstop here\\n')\n")
        engine = self.engine(settings, [sys.executable, str(script), "{output}"])
        transcript = engine.transcribe(source_input(tmp_path / "talk.mp4", tmp_path))
        assert [word["word"] for word in transcript.words] == [" stop", " here"]
        assert not list((tmp_path / "scratch").glob("command-output-*"))

    def test_failures_are_reported(self, settings, tmp_path):
        script = tmp_path / "broken.py"
        script.write_text("import sys\nsys.stderr.write('boom')\nsys.exit(3)\n")
        with pytest.raises(EngineError, match="exit code 3: boom"):
            self.engine(settings, [sys.executable, str(script)]).transcribe(source_input(tmp_path / "t.mp4", tmp_path))
        with pytest.raises(EngineError, match="speech.command is empty"):
            self.engine(settings, []).transcribe(source_input(tmp_path / "t.mp4", tmp_path))

    def test_audio_placeholder_gets_a_16k_mono_wav(self, settings, tmp_path, media):
        clip = media.wav(tmp_path / "talk.wav", 1)
        script = tmp_path / "fake_stt.py"
        script.write_text("import json, sys, wave\n"
                          "w = wave.open(sys.argv[1])\n"
                          "print(json.dumps({'words': [{'start': 0, 'end': 0.5, 'word': f'{w.getframerate()}-{w.getnchannels()}'}]}))\n")
        engine = self.engine(settings, [sys.executable, str(script), "{audio}"])
        transcript = engine.transcribe(source_input(clip, tmp_path))
        assert transcript.words[0]["word"] == " 16000-1"
        assert not list((tmp_path / "scratch").glob("*.wav"))

    @pytest.mark.skipif(os.name != "nt", reason="only Windows runs .bat programs through cmd.exe")
    def test_batch_program_never_gets_cmd_special_characters(self, settings, tmp_path):
        script = tmp_path / "fake_stt.bat"
        script.write_text('@echo {"words": [{"start": 0.5, "end": 0.9, "word": "hello"}]}\r\n', encoding="ascii")
        engine = self.engine(settings, [str(script), "{input}"])
        with pytest.raises(EngineError, match=r"fake_stt\.bat, a batch file, .* has &, which cmd\.exe"):
            engine.transcribe(source_input(tmp_path / "talk&calc&.mp4", tmp_path))
        transcript = engine.transcribe(source_input(tmp_path / "talk.mp4", tmp_path))
        assert [word["word"] for word in transcript.words] == [" hello"]

    @pytest.mark.skipif(os.name != "nt", reason="cmd.exe exists only on Windows")
    def test_cmd_itself_never_gets_cmd_special_characters(self, settings, tmp_path):
        engine = self.engine(settings, ["cmd", "/c", "type", "{input}"])
        with pytest.raises(EngineError, match=r"(?i)cmd\.exe, the Windows command shell, .* has &, which cmd\.exe"):
            engine.transcribe(source_input(tmp_path / "talk&calc&.mp4", tmp_path))


class TestRegistry:
    def test_unknown_engine_lists_the_available_ones(self, settings):
        registry = EngineRegistry()
        assert registry.names()[:3] == ["faster-whisper", "command", "import"]
        with pytest.raises(UnknownEngine, match="available: faster-whisper, command, import"):
            registry.create("nope", settings, Tools())


class FakeWord:
    def __init__(self, start, end, word, probability=0.9):
        self.start, self.end, self.word, self.probability = start, end, word, probability


class FakeSegment:
    def __init__(self, start, end, words):
        self.start, self.end, self.words = start, end, words
        self.text = "".join(word.word for word in words)
        self.avg_logprob, self.no_speech_prob, self.compression_ratio = -0.2, 0.1, 1.2


class FakeModel:
    """Says "hello" one second into each chunk; only without the voice filter does it also hear a quick "Stop."."""

    def transcribe(self, audio, **options):
        words = [FakeWord(1.0, 1.4, " hello")]
        if not options["vad_filter"]:
            words.append(FakeWord(1.5, 1.8, " Stop.", 0.8))
        return [FakeSegment(0.0, len(audio) / 16000, words)], None


class FakeFactory:
    def __init__(self):
        self.created = []

    def create(self, model, device, compute_type, threads):
        self.created.append((model, device, compute_type, threads))
        return FakeModel()


class TestFasterWhisperPieces:
    def test_chunks_end_in_silences_near_every_chunk_length(self, settings):
        db = np.full(int(300 / 0.02), -20.0, dtype=np.float32)
        db[int(118 / 0.02):int(119 / 0.02)] = -90.0
        db[int(241 / 0.02):int(242 / 0.02)] = -90.0
        energy = Energy(db, settings["audio"])
        chunks = ChunkPlanner(settings["speech"], energy).plan(300.0)
        bounds = [(round(own_start, 1), round(own_end, 1)) for _, own_start, own_end, _, _ in chunks]
        assert bounds == [(0.0, 118.5), (118.5, 241.5), (241.5, 300.0)]
        assert chunks[1][3] == pytest.approx(113.5, abs=0.05)
        assert chunks[1][4] == pytest.approx(246.5, abs=0.05)

    def test_overlapping_chunks_keep_each_word_once(self):
        words = [{"start": 9.0, "end": 9.4, "word": " a", "probability": 1}, {"start": 10.2, "end": 10.6, "word": " b", "probability": 1}]
        first = {"index": 0, "own": [0.0, 10.0], "segments": [{"start": 9.0, "end": 10.6, "text": " a b", "words": words}]}
        second = {"index": 1, "own": [10.0, 20.0], "segments": [{"start": 9.0, "end": 10.6, "text": " a b", "words": words}]}
        merged, segments = ChunkMerge().merge([second, first])
        assert [word["word"] for word in merged] == [" a", " b"]
        assert [segment["text"] for segment in segments] == [" a", " b"]

    def test_prompt_echo_is_spotted(self):
        check = EchoCheck("Kubernetes, webhook, telemetry, runbook, retry policy", 4)
        assert check.is_echo(" kubernetes webhook telemetry runbook")
        assert not check.is_echo(" the webhook sends telemetry to the runbook")
        assert not EchoCheck(None, 4).is_echo(" anything at all here")

    def test_cue_pass_windows_and_merge(self, settings):
        cue = CuePass(settings["speech"], settings["unclear"])
        assert cue.windows(130.0) == [(0.0, 60.0), (57.0, 117.0), (114.0, 130.0)]
        main = [{"start": 1.0, "end": 1.4, "word": " hello", "probability": 0.9}]
        found = [{"start": 1.5, "end": 1.8, "word": " Stop.", "probability": 0.8},
                 {"start": 1.1, "end": 1.3, "word": " stop", "probability": 0.8},
                 {"start": 1.52, "end": 1.78, "word": " stop", "probability": 0.6}]
        merged, added = cue.merge(main, found)
        assert [word["word"] for word in merged] == [" hello", " Stop."]
        assert added[0]["cue"] is True

    def test_targeted_cue_windows_cover_sounds_in_gaps_and_unclear_spans(self, settings):
        db = np.full(int(120 / 0.02), -80.0, dtype=np.float32)
        db[int(10 / 0.02):int(19.8 / 0.02)] = -20.0
        db[int(30.0 / 0.02):int(30.4 / 0.02)] = -25.0
        db[int(50 / 0.02):int(59.8 / 0.02)] = -20.0
        db[int(70 / 0.02):int(110 / 0.02)] = -20.0
        energy = Energy(db, settings["audio"])
        words = [{"start": 10.0 + index, "end": 10.8 + index, "word": " word", "probability": 0.95} for index in range(10)]
        words += [{"start": 50.0 + index, "end": 50.8 + index, "word": " word", "probability": 0.95} for index in range(10)]
        words += [{"start": 70.0, "end": 110.0, "word": " this.", "probability": 0.99}]
        cue = CuePass(settings["speech"], settings["unclear"])
        windows = cue.targeted(words, [], energy, 120.0)
        assert [(round(start, 1), round(end, 1)) for start, end in windows] == [
            (29.0, 31.4), (69.0, 99.0), (99.0, 111.0)]

    def burst_wav(self, path):
        """Four seconds of silence with two tone bursts: "hello" at 1.0-1.4 s and a lone word at 2.0-2.3 s."""
        rate = 16000
        times = np.arange(4 * rate) / rate
        signal = np.zeros_like(times)
        for start, end in [(1.0, 1.4), (2.0, 2.3)]:
            inside = (times >= start) & (times < end)
            signal[inside] = 0.3 * np.sin(2 * np.pi * 440 * times[inside])
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes((signal * 32767).astype("<i2").tobytes())
        return path

    def test_engine_with_a_fake_model(self, settings, tmp_path, media):
        clip = self.burst_wav(tmp_path / "talk.wav")
        settings["speech"].update({"workers": 1, "hints": ["Kubernetes"]})
        factory = FakeFactory()
        engine = FasterWhisperEngine(settings, Tools(), model_factory=factory)
        transcript = engine.transcribe(source_input(clip, tmp_path, duration=4.0))
        assert [word["word"] for word in transcript.words] == [" hello", " Stop."]
        assert (transcript.meta["cueScope"], transcript.meta["cueWindows"], transcript.meta["cueWordsAdded"]) == (
            "full", 1, 1)
        assert set(transcript.meta["timings"]) == {"decode", "mainPass", "cuePass"}
        threads = WorkerCount().threads("auto", 1)
        assert factory.created[0] == ("small.en", "cpu", "int8", threads) or factory.created[0][1] == "cuda"
        assert not list((tmp_path / "scratch").glob("*.npy"))

    def test_targeted_cue_scope_and_no_cue_pass(self, settings, tmp_path, media):
        clip = self.burst_wav(tmp_path / "talk.wav")
        settings["speech"].update({"workers": 1, "cueScope": "targeted"})
        transcript = FasterWhisperEngine(settings, Tools(), model_factory=FakeFactory()).transcribe(
            source_input(clip, tmp_path, duration=4.0))
        assert [word["word"] for word in transcript.words] == [" hello", " Stop."]
        assert (transcript.meta["cueWindows"], transcript.meta["cueAudioSec"], transcript.meta["cueWordsAdded"]) == (
            1, 2.3, 1)
        settings["speech"]["cuePass"] = False
        transcript = FasterWhisperEngine(settings, Tools(), model_factory=FakeFactory()).transcribe(
            source_input(clip, tmp_path, duration=4.0))
        assert [word["word"] for word in transcript.words] == [" hello"]
        assert transcript.meta["cueScope"] is None


class TestWorkerCount:
    def test_auto_uses_cpus_memory_and_a_cap(self, monkeypatch):
        monkeypatch.setattr("rostrum.engines.base.os.cpu_count", lambda: 16)
        monkeypatch.setattr(WorkerCount, "available_gb", lambda self: 11.5)
        assert WorkerCount().resolve("auto") == 4
        monkeypatch.setattr(WorkerCount, "available_gb", lambda self: 5.0)
        assert WorkerCount().resolve("auto") == 2
        monkeypatch.setattr("rostrum.engines.base.os.cpu_count", lambda: 64)
        monkeypatch.setattr(WorkerCount, "available_gb", lambda self: 100.0)
        assert WorkerCount().resolve("auto") == 6
        monkeypatch.setattr("rostrum.engines.base.os.cpu_count", lambda: 2)
        monkeypatch.setattr(WorkerCount, "available_gb", lambda self: 0.5)
        assert WorkerCount().resolve("auto") == 1
        assert "speech.workers auto: 2 CPUs, 0.5 GB available" in WorkerCount().describe("auto")

    def test_a_number_is_used_as_given(self):
        assert WorkerCount().resolve(3) == 3
        assert WorkerCount().describe(3, 2) == "3 worker(s) x 2 thread(s) (speech.workers 3)"

    def test_threads_auto_share_the_cpus(self, monkeypatch):
        monkeypatch.setattr("rostrum.engines.base.os.cpu_count", lambda: 16)
        assert WorkerCount().threads("auto", 4) == 4
        assert WorkerCount().threads("auto", 32) == 1
        assert WorkerCount().threads(2, 4) == 2
