"""faster-whisper engine: parallel chunks with overlap and word timestamps, a retry where
the model echoed its prompt, and a second pass without voice filtering that catches short spoken cues.

Audio is decoded by ffmpeg and passed to the model as arrays; faster_whisper.decode_audio is never used.
"""
import inspect
import multiprocessing
import os
import re
import time
from bisect import bisect_left

import numpy as np

from rostrum.audio import SAMPLE_RATE, AudioDecoder, Energy, EnergyMeter
from rostrum.detect.text import Tokenizer
from rostrum.detect.unclear import SpokenWord, UnclearSpans
from rostrum.engines.base import EngineError, SpeechEngine, Transcript, WorkerCount

# A last chunk shorter than this share of speech.chunkSec is joined to the one before.
LAST_CHUNK_SHARE = 0.4
# Chunk split score: silence length minus (distance from the target / this).
DISTANCE_WEIGHT = 10.0
# Prompt-echo retry: merge echo segments closer than this; re-transcribe with this much context.
ECHO_MERGE_GAP = 3.0
ECHO_CONTEXT = 2.0
ECHO_WORD_PAD = 0.3
# A cue-pass word is new when no main-pass word covers its middle within this margin.
CUE_OVERLAP_PAD = 0.15
NOT_LETTER = re.compile(r"[^a-z]")
# Unclear settings that do not change where the targeted cue pass looks.
CUE_TARGET_KEYS_IGNORED = ("level", "breakSentences")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
WORKER = {}


class ModelFactory:
    def create(self, model, device, compute_type, threads):
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise EngineError("faster-whisper is missing, but it ships with rostrum; reinstall rostrum to get it back") from error
        return WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=threads, num_workers=1)


class ModelOptions:
    """Speech settings as faster-whisper arguments."""

    def __init__(self, speech):
        self.speech = speech

    def device(self):
        device = self.speech["device"]
        if device != "auto":
            return device
        try:
            import ctranslate2

            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            return "cpu"

    def compute_type(self, device):
        compute_type = self.speech["computeType"]
        if compute_type != "auto":
            return compute_type
        return "float16" if device == "cuda" else "int8"

    def model(self, workers=1):
        device = self.device()
        return {"model": self.speech["model"], "device": device, "compute_type": self.compute_type(device),
                "threads": WorkerCount().threads(self.speech["threadsPerWorker"], workers)}

    def prompt(self):
        if self.speech.get("prompt"):
            return self.speech["prompt"]
        hints = self.speech.get("hints") or []
        return ", ".join(hints) if hints else None

    def transcribe(self, with_prompt=True, voice_filter=True):
        options = {"language": self.speech.get("language"), "beam_size": self.speech["beamSize"], "word_timestamps": True,
                   "vad_filter": voice_filter, "condition_on_previous_text": False}
        if voice_filter:
            options["vad_parameters"] = {"min_silence_duration_ms": self.speech["vadMinSilenceMs"]}
        prompt = self.prompt()
        if (with_prompt and
                prompt):
            options["initial_prompt"] = prompt
            options["hotwords"] = prompt
        return options

    def cue(self):
        """No voice filter and no prompt. Without fallback, each window is decoded once at temperature 0: on noise
        and silence the fallback re-decodes up to five more times, at random, and can invent words."""
        options = self.transcribe(with_prompt=False, voice_filter=False)
        if not self.speech["cueFallback"]:
            options["temperature"] = 0.0
        return options


class Segments:
    def of(self, segment, offset):
        return {"start": round(segment.start + offset, 3), "end": round(segment.end + offset, 3), "text": segment.text,
                "avg_logprob": round(segment.avg_logprob, 4), "no_speech_prob": round(segment.no_speech_prob, 4),
                "compression_ratio": round(segment.compression_ratio, 3),
                "words": [{"start": round(word.start + offset, 3), "end": round(word.end + offset, 3), "word": word.word,
                           "probability": round(word.probability, 4)} for word in (segment.words or [])]}


class ChunkWorker:
    """One model copy. Runs main-pass chunks (voice filter on) and cue-pass windows (voice filter off)."""

    def __init__(self, audio_path, model_settings, options, cue_options, factory=None):
        for name in THREAD_VARIABLES:
            os.environ[name] = str(model_settings["threads"])
        self.audio = np.load(audio_path, mmap_mode="r")
        self.model = (factory or ModelFactory()).create(**model_settings)
        self.options = self.supported(options)
        self.cue_options = self.supported(cue_options)

    def supported(self, options):
        parameters = inspect.signature(self.model.transcribe).parameters.values()
        if any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters):
            return dict(options)
        accepted = {parameter.name for parameter in parameters}
        return {key: value for key, value in options.items() if key in accepted}

    def run(self, task):
        began = time.time()
        start, end = task["range"]
        samples = np.ascontiguousarray(self.audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)], dtype=np.float32)
        options = self.options if task["voiceFilter"] else self.cue_options
        segments, _ = self.model.transcribe(samples, **options)
        return dict(task, segments=[Segments().of(segment, start) for segment in segments],
                    seconds=round(time.time() - began, 1), pid=os.getpid())


def start_worker(audio_path, model_settings, options, cue_options):
    WORKER["worker"] = ChunkWorker(audio_path, model_settings, options, cue_options)


def run_task(task):
    return WORKER["worker"].run(task)


class ChunkPlanner:
    def __init__(self, speech, energy):
        self.speech = speech
        self.energy = energy

    def plan(self, total):
        """[(index, own start, own end, audio start, audio end)]: chunks end in silences near every chunkSec."""
        chunk = float(self.speech["chunkSec"])
        overlap = float(self.speech["overlapSec"])
        if chunk <= 0:
            return [(0, 0.0, total, 0.0, total)]
        bounds = [0.0]
        target = chunk
        while target < total - chunk * LAST_CHUNK_SHARE:
            bounds.append(self.silence_near(target))
            target = bounds[-1] + chunk
        bounds.append(total)
        return [(index, bounds[index], bounds[index + 1], max(0.0, bounds[index] - overlap), min(total, bounds[index + 1] + overlap))
                for index in range(len(bounds) - 1)]

    def silence_near(self, target):
        db = self.energy.db
        frame = self.energy.frame
        span = self.speech["chunkSearchSec"]
        threshold = self.energy.silence
        low = max(0, int((target - span) / frame))
        high = min(len(db), int((target + span) / frame))
        best = None
        index = low
        while index < high:
            if db[index] >= threshold:
                index += 1
                continue
            stop = index
            while (stop < high and
                    db[stop] < threshold):
                stop += 1
            length = (stop - index) * frame
            middle = (index + stop) / 2 * frame
            score = length - abs(middle - target) / DISTANCE_WEIGHT
            if (length >= self.speech["chunkMinSilenceSec"] and
                    (best is None or score > best[0])):
                best = (score, middle)
            index = stop
        return best[1] if best else target


class ChunkMerge:
    def merge(self, results):
        """Each chunk owns the words whose middle falls in its own range; overlaps are dropped."""
        words = []
        segments = []
        for result in sorted(results, key=lambda item: item["index"]):
            own_start, own_end = result["own"]
            for segment in result["segments"]:
                owned = [word for word in segment["words"] if own_start <= (word["start"] + word["end"]) / 2 < own_end]
                middle = (segment["start"] + segment["end"]) / 2
                if (not owned and
                        not (own_start <= middle < own_end)):
                    continue
                kept = dict(segment)
                kept["words"] = owned
                if owned:
                    kept["start"] = owned[0]["start"]
                    kept["end"] = owned[-1]["end"]
                    kept["text"] = "".join(word["word"] for word in owned)
                segments.append(kept)
                words.extend(owned)
        words.sort(key=lambda word: (word["start"], word["end"]))
        segments.sort(key=lambda segment: segment["start"])
        return words, segments


class EchoCheck:
    """The model sometimes repeats its prompt instead of transcribing. Spot that by a long run of prompt words."""

    def __init__(self, prompt, min_words):
        self.tokenizer = Tokenizer()
        self.prompt_tokens = self.tokenizer.tokens(prompt.replace(",", " ").replace(":", " ")) if prompt else []
        self.min_words = min_words

    def is_echo(self, text):
        tokens = self.tokenizer.text_tokens(text)
        if (len(tokens) < self.min_words or
                not self.prompt_tokens):
            return False
        best = 0
        for i in range(len(tokens)):
            for j in range(len(self.prompt_tokens)):
                run = 0
                while (i + run < len(tokens) and
                        j + run < len(self.prompt_tokens) and
                        tokens[i + run] == self.prompt_tokens[j + run]):
                    run += 1
                best = max(best, run)
        return best >= self.min_words


class EchoRetry:
    def __init__(self, check, load_model, options, report):
        self.check = check
        self.load_model = load_model
        self.options = options
        self.report = report

    def ranges(self, segments):
        bad = []
        for segment in segments:
            echoed = (bool(segment["words"]) and
                      self.check.is_echo(segment["text"]))
            if not echoed:
                continue
            if (bad and
                    segment["start"] - bad[-1][1] < ECHO_MERGE_GAP):
                bad[-1][1] = segment["end"]
            else:
                bad.append([segment["start"], segment["end"]])
        return bad

    def run(self, audio, words, segments, total):
        bad = self.ranges(segments)
        if not bad:
            return words, segments, []
        self.report(f"  prompt echo in {len(bad)} place(s); transcribing them again without the prompt")
        model = self.load_model()
        failed = []
        for start, end in bad:
            first = max(0.0, start - ECHO_CONTEXT)
            last = min(total, end + ECHO_CONTEXT)
            samples = np.ascontiguousarray(audio[int(first * SAMPLE_RATE):int(last * SAMPLE_RATE)], dtype=np.float32)
            fresh = [Segments().of(segment, first) for segment in model.transcribe(samples, **self.options)[0]]
            inside = [word for segment in fresh for word in segment["words"] if self.near(word, start, end)]
            if (not inside or
                    self.check.is_echo("".join(word["word"] for word in inside))):
                failed.append([start, end])
                continue
            words = [word for word in words if not self.near(word, start, end)] + inside
            segments = [segment for segment in segments if not self.near(segment, start, end)]
            segments += [dict(segment, words=[word for word in segment["words"] if word in inside])
                         for segment in fresh if any(word in inside for word in segment["words"])]
        words.sort(key=lambda word: (word["start"], word["end"]))
        segments.sort(key=lambda segment: segment["start"])
        return words, segments, failed

    def near(self, item, start, end):
        return start - ECHO_WORD_PAD <= (item["start"] + item["end"]) / 2 <= end + ECHO_WORD_PAD


class CuePass:
    """The voice filter drops short isolated words, such as a quick "Stop." between two attempts. The cue pass
    transcribes again without it and adds the cue words (speech.cueWords) that the main pass missed.

    full (default): the whole timeline, in overlapping cueWindowSec windows.
    targeted: only where such a word can be. That is a burst of sound inside a gap between main-pass words, or a
    span the main pass got wrong: an overlong word, low-confidence words, a doubtful segment, speech with no words.
    """

    def __init__(self, speech, unclear_settings):
        self.speech = speech
        self.unclear = unclear_settings
        self.cue_words = {self.plain(word) for word in speech["cueWords"]}

    def plain(self, text):
        return NOT_LETTER.sub("", text.lower())

    def windows(self, total):
        window = float(self.speech["cueWindowSec"])
        step = float(self.speech["cueStepSec"])
        if (window <= 0 or
                step <= 0):
            return []
        ranges = []
        start = 0.0
        while start < total:
            ranges.append((start, min(total, start + window)))
            start += step
        return ranges

    def targeted(self, words, segments, energy, total):
        places = self.sounds_in_gaps(words, energy, total) + self.doubtful(words, segments, energy)
        pad = float(self.speech["cuePadSec"])
        padded = sorted((max(0.0, start - pad), min(total, end + pad)) for start, end in places)
        merged = []
        for start, end in padded:
            if (merged and
                    start - merged[-1][1] <= float(self.speech["cueMergeSec"])):
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        longest = float(self.speech["cueMaxWindowSec"])
        windows = []
        for start, end in merged:
            while (longest > 0 and
                   end - start > longest):
                windows.append((start, start + longest))
                start += longest
            windows.append((start, end))
        return windows

    def sounds_in_gaps(self, words, energy, total):
        """Bursts above the silence level inside gaps of at least cueGapSec between main-pass words."""
        gap = float(self.speech["cueGapSec"])
        shortest = float(self.speech["cueSoundSec"])
        places = []
        previous_end = 0.0
        for word in [*words, None]:
            next_start = word["start"] if word is not None else total
            if next_start - previous_end >= gap:
                places += energy.loud_runs(previous_end, next_start, shortest, energy.silence)
            if word is not None:
                previous_end = max(previous_end, word["end"])
        return places

    def doubtful(self, words, segments, energy):
        spoken = [SpokenWord(word["start"], word["end"], word["word"], word["probability"]) for word in words]
        return [(span.start, span.end) for span in UnclearSpans(self.unclear).find(spoken, segments, energy=energy)]

    def words(self, results, total, edge):
        found = []
        for result in sorted(results, key=lambda item: item["range"][0]):
            start, end = result["range"]
            low = start + (edge if start > 0 else 0.0)
            high = end - (edge if end < total else 0.0)
            for segment in result["segments"]:
                for word in segment["words"]:
                    middle = (word["start"] + word["end"]) / 2
                    if (low <= middle < high and
                            self.plain(word["word"]) in self.cue_words):
                        found.append(word)
        return found

    def merge(self, words, cue_words):
        starts = [word["start"] for word in words]
        added = []
        for cue in cue_words:
            middle = (cue["start"] + cue["end"]) / 2
            nearby = words[max(0, bisect_left(starts, middle - 5.0)):bisect_left(starts, middle + 5.0)]
            covered = any(word["start"] - CUE_OVERLAP_PAD <= middle <= word["end"] + CUE_OVERLAP_PAD for word in nearby)
            already = any(abs((other["start"] + other["end"]) / 2 - middle) <= CUE_OVERLAP_PAD for other in added)
            is_known = (covered or
                        already)
            if not is_known:
                added.append(dict(cue, cue=True))
        merged = sorted(words + added, key=lambda word: (word["start"], word["end"]))
        return merged, added


class WorkerPool:
    """Model copies for the main pass and the cue pass: in this process (one worker, or a test model factory),
    or in spawned processes that each load the model once for both passes."""

    def __init__(self, workers, audio_path, model_settings, options, cue_options, factory, report):
        self.workers = workers
        self.audio_path = audio_path
        self.model_settings = model_settings
        self.options = options
        self.cue_options = cue_options
        self.factory = factory
        self.report = report
        self.pool = None
        self.worker = None

    def __enter__(self):
        in_process = (self.workers == 1 or
                      self.factory is not None)
        if in_process:
            self.worker = ChunkWorker(self.audio_path, self.model_settings, self.options, self.cue_options, self.factory)
        else:
            context = multiprocessing.get_context("spawn")
            initargs = (str(self.audio_path), self.model_settings, self.options, self.cue_options)
            self.pool = context.Pool(self.workers, initializer=start_worker, initargs=initargs)
        return self

    def __exit__(self, *exc_info):
        self.worker = None
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
        return False

    def run(self, tasks):
        results = []
        if self.pool is None:
            for task in tasks:
                results.append(self.worker.run(task))
                self.progress(results[-1], len(results), len(tasks))
            return results
        for result in self.pool.imap_unordered(run_task, tasks):
            results.append(result)
            self.progress(result, len(results), len(tasks))
        return results

    def progress(self, result, done, count):
        kind = "chunk" if result["voiceFilter"] else "cue window"
        start, end = result["range"]
        self.report(f"  {kind} {start:.0f}-{end:.0f}s done in {result['seconds']:.0f}s [{done}/{count}]")


class FasterWhisperEngine(SpeechEngine):
    name = "faster-whisper"
    supports_cue_pass = True

    def __init__(self, settings, tools, report=None, model_factory=None):
        super().__init__(settings, tools, report)
        self.factory = model_factory
        self.options = ModelOptions(self.speech)

    def transcribe(self, source):
        began = time.time()
        decoder = AudioDecoder(self.tools)
        audio = decoder.decode(source.path, source.audio_offset)
        total = len(audio) / SAMPLE_RATE
        energy = Energy(EnergyMeter(self.settings["audio"]["frameSec"]).frame_db(audio), self.settings["audio"])
        chunks = ChunkPlanner(self.speech, energy).plan(total)
        tasks = [{"index": index, "own": [own_start, own_end], "range": [start, end], "voiceFilter": True}
                 for index, own_start, own_end, start, end in chunks]
        workers = WorkerCount().resolve(self.speech["workers"])
        self.report("  " + WorkerCount().describe(self.speech["workers"], self.speech["threadsPerWorker"]))
        source.scratch.mkdir(parents=True, exist_ok=True)
        audio_path = source.scratch / f"audio16k-{source.index + 1}.npy"
        np.save(audio_path, audio)
        timings = {"decode": round(time.time() - began, 1)}
        failed = []
        added = []
        windows = []
        try:
            pool = WorkerPool(workers, audio_path, self.options.model(workers), self.options.transcribe(),
                              self.options.cue(), self.factory, self.report)
            with pool:
                began = time.time()
                words, segments = ChunkMerge().merge(pool.run(tasks))
                timings["mainPass"] = round(time.time() - began, 1)
                words, segments, failed = self.retry_echo(audio, words, segments, total)
                if self.speech["cuePass"]:
                    began = time.time()
                    words, added, windows = self.cue_pass(pool, words, segments, energy, total, len(tasks))
                    timings["cuePass"] = round(time.time() - began, 1)
        finally:
            audio_path.unlink(missing_ok=True)
        meta = {"engine": self.name, **self.options.model(workers), "beamSize": self.speech["beamSize"],
                "workers": workers,
                "chunks": len(chunks), "cueScope": self.speech["cueScope"] if self.speech["cuePass"] else None,
                "cueWindows": len(windows), "cueAudioSec": round(sum(end - start for start, end in windows), 1),
                "cueWordsAdded": len(added), "timings": timings}
        return Transcript(words=words, segments=segments, duration=total, failed=failed, meta=meta)

    def retry_echo(self, audio, words, segments, total):
        prompt = self.options.prompt()
        should_retry = (bool(prompt) and
                        self.speech["retryEcho"])
        if not should_retry:
            return words, segments, []
        check = EchoCheck(prompt, self.speech["echoMinWords"])
        options = self.options.transcribe(with_prompt=False)
        return EchoRetry(check, self.model, options, self.report).run(audio, words, segments, total)

    def cue_pass(self, pool, words, segments, energy, total, first_index):
        cue = CuePass(self.speech, self.settings["unclear"])
        targeted = self.speech["cueScope"] == "targeted"
        windows = cue.targeted(words, segments, energy, total) if targeted else cue.windows(total)
        audio_seconds = sum(end - start for start, end in windows)
        self.report(f"  cue pass ({self.speech['cueScope']}): {len(windows)} window(s), {audio_seconds:.0f}s of audio")
        tasks = [{"index": first_index + number, "own": [start, end], "range": [start, end], "voiceFilter": False}
                 for number, (start, end) in enumerate(windows)]
        edge = 0.0 if targeted else float(self.speech["cueEdgeSec"])
        words, added = cue.merge(words, cue.words(pool.run(tasks), total, edge))
        if added:
            self.report(f"  cue pass added {len(added)} word(s): " + ", ".join(word["word"].strip() for word in added[:10]))
        return words, added, windows

    def model(self):
        settings = self.options.model()
        return (self.factory or ModelFactory()).create(**settings)

    def check(self):
        try:
            import faster_whisper
        except ImportError:
            return {"ok": False, "detail": "missing; it ships with rostrum, so reinstall rostrum"}
        detail = f"faster-whisper {faster_whisper.__version__}"
        try:
            import av

            detail += f", PyAV {av.__version__}"
            if int(av.__version__.split(".")[0]) >= 19:
                return {"ok": False, "detail": detail + " (PyAV 19+ breaks faster-whisper 1.2; install av<19)"}
        except ImportError:
            pass
        return {"ok": True, "detail": detail}

    def cache_inputs(self, source):
        """What decides the words. Workers and threads only change the speed."""
        model = {key: value for key, value in self.options.model().items() if key != "threads"}
        inputs = {"model": model}
        targeted = (self.speech["cuePass"] and
                    self.speech["cueScope"] == "targeted")
        if targeted:
            unclear = {key: value for key, value in self.settings["unclear"].items()
                       if key not in CUE_TARGET_KEYS_IGNORED}
            inputs["cueTargets"] = {"unclear": unclear, "audio": self.settings["audio"]}
        return inputs
