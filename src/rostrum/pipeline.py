"""The first pass for each talk: probe, measure loudness, transcribe, detect, reconcile; then candidates and the EDL.

Each stage caches its output in the talk's work folder with a key built from its inputs and settings, so a re-run
only redoes what changed. Decisions (decisions.json) are applied on top of the candidates to give edl.json.
"""
import bisect
import hashlib
import importlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rostrum import __version__
from rostrum.audio import AudioDecoder, Energy, EnergyMeter
from rostrum.config import ConfigError, ConfigLoader
from rostrum.detect.analysis import Analysis
from rostrum.detect.candidate import CandidateFile
from rostrum.detect.planner import KeepRanges, PassResult, Planner
from rostrum.detect.script import ScriptText
from rostrum.detect.text import ContentWords, Tokenizer
from rostrum.edl import Cut, CutIds, Edl, Source, SourceKey
from rostrum.engines import EngineRegistry, SourceInput
from rostrum.engines.base import TranscriptData
from rostrum.media import MEDIA_DEFAULTS, MediaError, Prober, Rational
from rostrum.project import ProjectFile, TalkPaths
from rostrum.reconcile import Reconciler
from rostrum.slides_timeline import SlideTimelineFiles
from rostrum.tools import Tools

CANDIDATES_SCHEMA = "rostrum.candidates/1"
DECISIONS_SCHEMA = "rostrum.decisions/1"
STAGE_SCHEMA = "rostrum.stage/1"
DETECT_GROUPS = ("audio", "transcript", "text", "cues", "retakes", "repeats", "pauses", "leads", "offtalk", "fillers",
                 "unclear", "edges", "plan", "script")
# The settings that decide the cuts; a change after init is reported until init runs again.
CUT_SETTING_GROUPS = DETECT_GROUPS + ("reconcile", "slides")
SPEECH_KEYS_IGNORED = ("workers", "timeoutSec")
# probe.json fields; bump when they change so older probe files are rebuilt.
PROBE_FORMAT = 2
EVIDENCE_INSIDE_MAX = 30
DECISIONS = ("keep", "cut")


class JsonWriter:
    def write(self, path, data):
        """Writes through a temporary file so a crash never leaves half a file behind."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=self.plain) + "\n", encoding="utf-8")
        os.replace(temporary, target)

    def plain(self, value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"cannot write {type(value).__name__} as JSON")

    def read(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))


class Fingerprint:
    def of(self, value):
        text = json.dumps(value, sort_keys=True, default=JsonWriter().plain)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def file(self, path):
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]


class StageClock:
    """Wall time per pipeline stage. Cached stages show up as near zero."""

    def __init__(self):
        self.started = time.monotonic()
        self.laps = {}

    def lap(self, name):
        now = time.monotonic()
        self.laps[name] = round(now - self.started, 2)
        self.started = now

    def text(self, engine_timings):
        parts = [f"{name} {seconds:.1f}s" for name, seconds in self.laps.items()]
        engine = {}
        for timings in engine_timings:
            for name, seconds in timings.items():
                engine[name] = engine.get(name, 0.0) + seconds
        if engine:
            parts.append("engine: " + ", ".join(f"{name} {seconds:.1f}s" for name, seconds in engine.items()))
        return ", ".join(parts)


class StageCache:
    def __init__(self, path):
        self.path = Path(path)

    def load(self, key):
        if not self.path.is_file():
            return None
        try:
            data = JsonWriter().read(self.path)
        except (OSError, json.JSONDecodeError):
            return None
        return data if data.get("key") == key else None

    def save(self, key, data):
        JsonWriter().write(self.path, {"schema": STAGE_SCHEMA, "key": key, "rostrum": __version__, **data})


class Clock:
    def text(self, seconds, decimals=1):
        seconds = max(0.0, float(seconds))
        hours = int(seconds // 3600)
        minutes = int(seconds % 3600 // 60)
        rest = seconds - hours * 3600 - minutes * 60
        width = 3 + decimals if decimals else 2
        second_text = f"{rest:0{width}.{decimals}f}"
        if hours:
            return f"{hours}:{minutes:02d}:{second_text}"
        return f"{minutes:02d}:{second_text}"


class Evidence:
    """The transcript around a cut: words before, [words inside], words after. Built from the whole transcript,
    so it does not change when decisions change (cut ids depend on it)."""

    def __init__(self, words, transcript_settings):
        self.words = words
        self.middles = [word.middle for word in words]
        self.context = transcript_settings["contextWords"]
        self.unclear = transcript_settings["unclearBelow"]

    def text(self, start, end):
        first = bisect.bisect_left(self.middles, start)
        last = bisect.bisect_right(self.middles, end)
        before = self.words[max(0, first - self.context):first]
        inside = self.words[first:last]
        after = self.words[last:last + self.context]
        if len(inside) > EVIDENCE_INSIDE_MAX:
            half = EVIDENCE_INSIDE_MAX // 2
            inner = self.show(inside[:half]) + " \u2026 " + self.show(inside[-half:])
        else:
            inner = self.show(inside)
        return f"{self.show(before)} [{inner}] {self.show(after)}".strip()

    def show(self, words):
        parts = []
        previous_unclear = False
        for word in words:
            unclear = word.probability < self.unclear
            if not unclear:
                parts.append(word.text)
            elif not previous_unclear:
                parts.append(" [unclear]")
            previous_unclear = unclear
        return "".join(parts).strip()


class CutFactory:
    """Candidates become edl.Cut objects with stable ids."""

    def __init__(self, source_key, evidence=None):
        self.ids = CutIds(source_key)
        self.evidence = evidence
        self.used = set()

    def cut(self, candidate, applied, evidence=None):
        if evidence is None:
            evidence = self.evidence.text(candidate.start, candidate.end) if self.evidence else ""
        cut_id = self.unique(self.ids.make(candidate.type, candidate.start, candidate.end, evidence))
        return Cut(
            id=cut_id,
            start=float(candidate.start),
            end=float(candidate.end),
            type=candidate.type,
            level="auto" if applied else "propose",
            decision="cut" if applied else "keep",
            reason=candidate.note,
            evidence=evidence,
            confidence=self.confidence(candidate),
            decided_by="auto",
        )

    def unique(self, cut_id):
        candidate = cut_id
        number = 2
        while candidate in self.used:
            candidate = f"{cut_id}-{number}"
            number += 1
        self.used.add(candidate)
        return candidate

    def confidence(self, candidate):
        if candidate.is_trim:
            return 1.0
        if candidate.type == "unclear":
            return 0.2
        return 0.9 if candidate.confidence == "high" else 0.5


class DecisionsFile:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        if not self.path.is_file():
            return {}
        data = JsonWriter().read(self.path)
        return dict(data.get("decisions", {}))

    def review_time(self):
        """When these decisions were last reviewed, on the clock of the candidates' analyzedAt: reviewedAt, or None
        when rostrum saved them without a review (reviewedAt null). A file without the field (a hand edit, or a page
        saved straight over decisions.json) counts from its file time. None when there is no file."""
        if not self.path.is_file():
            return None
        data = JsonWriter().read(self.path)
        if "reviewedAt" not in data:
            return self.path.stat().st_mtime
        value = data["reviewedAt"]
        is_number = (isinstance(value, (int, float)) and
                     not isinstance(value, bool))
        return float(value) if is_number else None

    def change_time(self):
        """When a decision last changed: changedAt, or None when rostrum saved without a change (changedAt null).
        A file without the field counts from its file time. None when there is no file."""
        if not self.path.is_file():
            return None
        data = JsonWriter().read(self.path)
        if "changedAt" not in data:
            return self.path.stat().st_mtime
        value = data["changedAt"]
        is_number = (isinstance(value, (int, float)) and
                     not isinstance(value, bool))
        return float(value) if is_number else None

    def save(self, talk_id, decisions, reviewed_at, changed_at):
        """The previous file is kept as decisions.json.bak first. reviewed_at None saves without a review;
        changed_at is when a decision last changed."""
        if self.path.is_file():
            shutil.copy2(self.path, self.path.with_name(self.path.name + ".bak"))
        JsonWriter().write(self.path, {"schema": DECISIONS_SCHEMA, "talk": talk_id, "reviewedAt": reviewed_at,
                                       "changedAt": changed_at, "decisions": decisions})

    def set(self, talk_id, cut_id, decision, start=None, end=None, by="user"):
        if decision not in DECISIONS:
            raise ConfigError(f"decision must be keep or cut, not {decision!r}")
        decisions = self.load()
        entry = {"decision": decision, "decided_by": by, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if start is not None:
            entry["start"] = float(start)
        if end is not None:
            entry["end"] = float(end)
        decisions[cut_id] = entry
        now = time.time()
        self.save(talk_id, decisions, now, now)
        return entry


@dataclass
class EdlResult:
    edl: Edl
    cuts: list
    stale: list


class EdlBuilder:
    """keep = the talk timeline minus every cut whose decision is cut; tiny kept slivers with no words become gaps."""

    def __init__(self, talk_id, sources, fps, word_middles, plan_settings, source_key):
        self.talk_id = talk_id
        self.sources = sources
        self.fps = fps
        self.middles = word_middles
        self.min_keep = plan_settings["minKeepSec"]
        self.source_key = source_key

    def build(self, cuts, decisions):
        known = set()
        decided = []
        for cut in cuts:
            known.add(cut.id)
            decided.append(self.apply(cut, decisions.get(cut.id)))
        total = sum(source.duration for source in self.sources)
        cut_set = [cut for cut in decided if cut.decision == "cut"]
        keep, gaps = KeepRanges(total, self.middles, self.min_keep).compute(cut_set)
        factory = CutFactory(self.source_key)
        factory.used = set(known)
        gap_cuts = []
        for gap in gaps:
            cut = factory.cut(gap, applied=True, evidence="")
            known.add(cut.id)
            cut = self.apply(cut, decisions.get(cut.id))
            if cut.decision == "keep":
                keep.append([round(gap.start, 3), round(gap.end, 3)])
            gap_cuts.append(cut)
        keep.sort()
        stale = [dict(entry, id=cut_id) for cut_id, entry in decisions.items() if cut_id not in known]
        everything = sorted(decided + gap_cuts, key=lambda cut: (cut.start, cut.end))
        edl = Edl(talk=self.talk_id, sources=self.sources, keep=keep, cuts=everything, fps=self.fps)
        return EdlResult(edl=edl, cuts=everything, stale=stale)

    def apply(self, cut, entry):
        if entry is None:
            return cut
        start = float(entry.get("start", cut.start))
        end = float(entry.get("end", cut.end))
        return replace(cut, decision=entry["decision"], start=start, end=end, decided_by=entry.get("decided_by", "user"))


class CandidatesFile:
    def __init__(self, path):
        self.path = Path(path)

    def save(self, data):
        JsonWriter().write(self.path, data)

    def load(self):
        if not self.path.is_file():
            raise ConfigError(f"no candidates yet for this talk ({self.path}); run `rostrum init` first")
        data = JsonWriter().read(self.path)
        if data.get("schema") != CANDIDATES_SCHEMA:
            raise ConfigError(f"{self.path}: expected schema {CANDIDATES_SCHEMA}")
        return data

    def cuts(self, data):
        return [Cut(**item) for item in data["cuts"]]

    def sources(self, data):
        return [Source(**item) for item in data["sources"]]


class AnalysisTime:
    """analyzedAt for new candidates. A re-run that changes nothing a review covered (the sources, the cut settings,
    and each cut's id, edges, level and decision) keeps the previous time, so the talk stays reviewed."""

    def __init__(self, path):
        self.path = Path(path)

    def of(self, source_key, cut_settings, cuts):
        previous = self.previous()
        unchanged = (previous is not None and
                     previous.get("sourceKey") == source_key and
                     previous.get("cutSettings") == cut_settings and
                     self.cut_list(previous.get("cuts", [])) == self.cut_list(cuts))
        if not unchanged:
            return time.time()
        recorded = previous.get("analyzedAt")
        return float(recorded) if recorded is not None else self.path.stat().st_mtime

    def previous(self):
        if not self.path.is_file():
            return None
        try:
            data = JsonWriter().read(self.path)
        except (OSError, ValueError):
            return None
        return data if data.get("schema") == CANDIDATES_SCHEMA else None

    def cut_list(self, cuts):
        fields = ("id", "start", "end", "level", "decision")
        return [tuple(cut.get(name) for name in fields) for cut in cuts if isinstance(cut, dict)]


@dataclass
class TalkContext:
    """Everything the first pass knows about one talk. Passed to rostrum.review.write_review(context)."""
    project: object
    talk: object
    settings: dict
    paths: TalkPaths
    sources: list = field(default_factory=list)
    media: list = field(default_factory=list)
    words: list = field(default_factory=list)
    cuts: list = field(default_factory=list)
    edl: Edl | None = None
    stale: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: dict = field(default_factory=dict)
    # The candidates' analyzedAt; the review page records it, so an import can tell which analysis it reviewed.
    analyzed_at: float | None = None

    @property
    def talk_id(self):
        return self.talk.id

    @property
    def title(self):
        return self.talk.title or self.talk.id

    @property
    def work_dir(self):
        return self.paths.root

    @property
    def decisions_path(self):
        return self.paths.decisions

    @property
    def candidates_path(self):
        return self.paths.candidates

    @property
    def edl_path(self):
        return self.paths.edl

    @property
    def review_path(self):
        return self.paths.review

    def summary(self):
        edl = self.edl
        applied = [cut for cut in self.cuts if cut.decision == "cut"]
        proposed = [cut for cut in self.cuts if cut.level == "propose"]
        return {
            "id": self.talk.id,
            "ok": True,
            "workDir": str(self.paths.root),
            "duration": round(edl.duration, 3) if edl else None,
            "kept": round(edl.kept, 3) if edl else None,
            "cuts": len(applied),
            "proposed": len(proposed),
            "stale": len(self.stale),
            "warnings": self.warnings,
            "files": {"candidates": str(self.paths.candidates), "edl": str(self.paths.edl),
                      "decisions": str(self.paths.decisions), "review": self.info.get("review")},
        }


class SourceStage:
    """Probes every source; offsets put them end to end on one talk timeline.

    A source's duration is where both its video and audio still have data (media.Media.usable_end); the container
    and per-stream durations are kept in probe.json and candidates.json."""

    def __init__(self, project, talk, paths, tools, media_settings):
        self.project = project
        self.talk = talk
        self.paths = paths
        self.tools = tools
        self.media_settings = media_settings

    def run(self):
        files = self.project.source_paths(self.talk)
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise ConfigError(f"talk {self.talk.id}: source not found: {', '.join(missing)}")
        stats = [{"path": str(path), "size": path.stat().st_size, "mtime": path.stat().st_mtime_ns} for path in files]
        key = Fingerprint().of({"files": stats, "media": self.media_settings, "format": PROBE_FORMAT,
                                "version": __version__})
        cache = StageCache(self.paths.probe)
        cached = cache.load(key)
        if cached is None:
            media = [self.describe(path) for path in files]
            cache.save(key, {"media": media})
        else:
            media = cached["media"]
        sources = []
        offset = 0.0
        for path, info in zip(files, media):
            self.check_streams(path, info)
            duration = float(info["usableEnd"])
            if duration <= 0:
                raise ConfigError(f"{path}: no usable duration (ffprobe found no audio or video length)")
            sources.append(Source(path=str(path), duration=duration, offset=offset))
            offset += duration
        return media, sources

    def check_streams(self, path, info):
        missing = [kind for kind in ("video", "audio") if kind not in info]
        if missing:
            raise MediaError(f"talk {self.talk.id}: {path.name} has no {' and no '.join(missing)} stream; "
                             "each source must be a video with an audio track")

    def describe(self, path):
        media = Prober(self.tools, self.media_settings).probe(path, check_vfr=True)
        return {**media.summary(), "usableEnd": media.usable_end, "fps": Rational().text(media.fps),
                "audioOffset": round(media.audio_offset, 6)}


class EnergyStage:
    def __init__(self, paths, tools, audio_settings, report):
        self.paths = paths
        self.tools = tools
        self.audio_settings = audio_settings
        self.report = report

    def run(self, sources, source_keys, offsets):
        key = Fingerprint().of({"sources": source_keys, "durations": [source.duration for source in sources],
                                "frameSec": self.audio_settings["frameSec"], "audioOffsets": offsets,
                                "version": __version__})
        path = self.paths.energy
        if path.is_file():
            with np.load(path) as data:
                if str(data["key"]) == key:
                    return Energy(data["db"], self.audio_settings), key
        self.report("  measuring loudness")
        audio = AudioDecoder(self.tools).timeline(sources, offsets)
        db = EnergyMeter(self.audio_settings["frameSec"]).frame_db(audio)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".tmp.npz")
        np.savez(temporary, db=db, key=np.array(key))
        os.replace(temporary, path)
        return Energy(db, self.audio_settings), key


class TranscribeStage:
    def __init__(self, project, talk, paths, settings, tools, report):
        self.project = project
        self.talk = talk
        self.paths = paths
        self.settings = settings
        self.tools = tools
        self.report = report

    def run(self, sources, source_keys, offsets):
        speech = self.settings["speech"]
        engine = EngineRegistry().create(speech["engine"], self.settings, self.tools, self.report)
        transcripts = []
        keys = []
        for index, source in enumerate(sources):
            source_input = SourceInput(index=index, path=Path(source.path), duration=source.duration,
                                       scratch=self.paths.scratch, transcript=self.transcript_file(index),
                                       audio_offset=offsets[index])
            speech_inputs = {key: value for key, value in speech.items() if key not in SPEECH_KEYS_IGNORED}
            key = Fingerprint().of({"source": source_keys[index], "duration": source.duration, "speech": speech_inputs,
                                    "engine": engine.cache_inputs(source_input), "adapter": engine.adapter_version,
                                    "audioOffset": offsets[index]})
            cache = StageCache(self.paths.transcript(index))
            cached = cache.load(key)
            if cached is None:
                self.report(f"  transcribing {Path(source.path).name} with {engine.name}")
                transcript = engine.transcribe(source_input)
                cache.save(key, {"transcript": transcript.to_dict()})
            else:
                transcript = TranscriptData().from_dict(cached["transcript"])
            transcripts.append(transcript)
            keys.append(key)
        return transcripts, keys

    def transcript_file(self, index):
        listed = self.settings["speech"]["transcripts"]
        if not listed:
            return None
        if len(listed) != len(self.talk.sources):
            raise ConfigError(f"talk {self.talk.id}: speech.transcripts has {len(listed)} entries for "
                              f"{len(self.talk.sources)} sources")
        return self.project.resolve(listed[index])


@dataclass
class TalkTranscript:
    """Every source's transcript moved onto the talk timeline."""
    words: list
    segments: list
    failed: list
    estimated: bool


class TimelineWords:
    def join(self, transcripts, sources):
        """Words, segments and failed ranges of every source, moved onto the talk timeline."""
        words = []
        segments = []
        failed = []
        estimated = False
        for index, (transcript, source) in enumerate(zip(transcripts, sources)):
            for word in transcript.words:
                if word["start"] >= source.duration:
                    continue
                words.append({"start": word["start"] + source.offset, "end": min(word["end"], source.duration) + source.offset,
                              "word": word["word"], "probability": word["probability"], "source": index})
            for segment in transcript.segments:
                if segment["start"] >= source.duration:
                    continue
                moved = {key: value for key, value in segment.items() if key != "words"}
                moved["start"] = segment["start"] + source.offset
                moved["end"] = min(segment["end"], source.duration) + source.offset
                segments.append(moved)
            failed += [[start + source.offset, end + source.offset] for start, end in transcript.failed]
            estimated = (estimated or
                         transcript.meta.get("wordTimes") == "estimated")
        words.sort(key=lambda word: (word["start"], word["end"]))
        segments.sort(key=lambda segment: (segment["start"], segment["end"]))
        return TalkTranscript(words, segments, failed, estimated)


class PassFile:
    def save(self, cache, key, result):
        cache.save(key, {"painted": [cut.to_dict() for cut in result.painted],
                         "proposals": [cut.to_dict() for cut in result.proposals],
                         "content": [cut.to_dict() for cut in result.content],
                         "keep": result.keep, "info": result.info})

    def load(self, data):
        reader = CandidateFile()
        return PassResult(painted=[reader.from_dict(item) for item in data["painted"]],
                          proposals=[reader.from_dict(item) for item in data["proposals"]],
                          keep=[list(pair) for pair in data["keep"]],
                          content=[reader.from_dict(item) for item in data["content"]],
                          info=data.get("info", {}))


class ReviewHook:
    def write(self, context):
        try:
            module = importlib.import_module("rostrum.review")
        except ModuleNotFoundError as error:
            if error.name == "rostrum.review":
                return None
            raise
        write = getattr(module, "write_review", None)
        if write is None:
            return None
        return write(context)


class TalkPipeline:
    def __init__(self, project, talk, config, output, tools=None):
        self.project = project
        self.talk = talk
        self.settings = config.values
        self.paths = TalkPaths(project, talk, self.settings)
        self.tools = tools or Tools(self.settings["tools"])
        self.output = output

    def report(self, message):
        self.output.info(message)

    def run(self):
        settings = self.settings
        clock = StageClock()
        context = TalkContext(self.project, self.talk, settings, self.paths)
        self.paths.root.mkdir(parents=True, exist_ok=True)
        media_settings = {key: settings["render"][key] for key in MEDIA_DEFAULTS if key in settings["render"]}
        context.media, context.sources = SourceStage(self.project, self.talk, self.paths, self.tools, media_settings).run()
        for info in context.media:
            if info.get("vfr"):
                context.warnings.append(f"{Path(info['path']).name} has a variable frame rate ({info.get('vfr_reason', '')}); "
                                        "render converts it to constant frame rate first")
        source_keys = [SourceKey().of([source.path]) for source in context.sources]
        offsets = [float(info.get("audioOffset", 0.0)) for info in context.media]
        clock.lap("probe")
        energy, energy_key = EnergyStage(self.paths, self.tools, settings["audio"], self.report).run(
            context.sources, source_keys, offsets)
        clock.lap("loudness")
        transcripts, transcript_keys = TranscribeStage(self.project, self.talk, self.paths, settings, self.tools,
                                                       self.report).run(context.sources, source_keys, offsets)
        clock.lap("transcribe")
        joined = TimelineWords().join(transcripts, context.sources)
        if joined.estimated:
            context.warnings.append("the transcript has no word times; they were estimated, so cut edges are rough")
        if not joined.words:
            context.warnings.append("the transcript has no words; only lead and pause trims are possible")
        context.words = joined.words
        total = sum(source.duration for source in context.sources)
        script_path = self.script_path()
        content_words = ContentWords(settings["text"]["stopWords"], settings["fillers"]["words"])
        script = ScriptText(settings["script"], Tokenizer(), content_words).load(script_path)
        analysis = Analysis(joined.words, energy, total, script, joined.failed, settings, joined.segments)
        clock.lap("analysis")
        detect_key = Fingerprint().of({
            "transcripts": transcript_keys, "energy": energy_key,
            "sources": [(source.offset, source.duration) for source in context.sources],
            "settings": {group: settings[group] for group in DETECT_GROUPS},
            "script": Fingerprint().file(script_path) if script_path else None,
            "version": __version__,
        })
        first = self.cached_pass(self.paths.detect, detect_key, lambda: Planner(analysis).run())
        clock.lap("detect")
        final = first
        if settings["reconcile"]["enabled"]:
            timeline_files = SlideTimelineFiles(self.project.dir)
            timeline = timeline_files.load(settings["slides"]["timeline"], context.sources)
            reconcile_key = Fingerprint().of({
                "detect": detect_key, "reconcile": settings["reconcile"], "slides": settings["slides"],
                "timeline": self.timeline_fingerprint(timeline_files, settings["slides"]["timeline"]),
                "version": __version__,
            })
            final = self.cached_pass(self.paths.reconcile, reconcile_key, lambda: Reconciler(analysis, timeline).run(first))
            clock.lap("reconcile")
        source_key = SourceKey().of([source.path for source in context.sources])
        factory = CutFactory(source_key, Evidence(analysis.words, settings["transcript"]))
        cuts = [factory.cut(candidate, applied=True) for candidate in final.applied]
        cuts += [factory.cut(candidate, applied=False) for candidate in final.proposals]
        cuts.sort(key=lambda cut: (cut.start, cut.end))
        fps = context.media[0]["fps"] if context.media else ""
        middles = [word.middle for word in analysis.words]
        cut_settings = CutSettings().fingerprint(settings)
        cut_list = [asdict(cut) for cut in cuts]
        context.analyzed_at = AnalysisTime(self.paths.candidates).of(source_key, cut_settings, cut_list)
        candidates = {
            "schema": CANDIDATES_SCHEMA,
            "talk": self.talk.id,
            "rostrum": __version__,
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "analyzedAt": context.analyzed_at,
            "sourceKey": source_key,
            "cutSettings": cut_settings,
            "duration": total,
            "fps": fps,
            "sources": [asdict(source) for source in context.sources],
            "media": context.media,
            "reconciled": settings["reconcile"]["enabled"],
            "info": final.info,
            "timings": dict(clock.laps),
            "engineTimings": [transcript.meta.get("timings", {}) for transcript in transcripts],
            "warnings": context.warnings,
            "wordMiddles": middles,
            "cuts": cut_list,
        }
        CandidatesFile(self.paths.candidates).save(candidates)
        result = TalkEdl(self.paths, settings).write(candidates)
        clock.lap("edl")
        context.cuts = result.cuts
        context.edl = result.edl
        context.stale = result.stale
        context.info = dict(final.info)
        context.info["timings"] = clock.laps
        context.info["engine"] = [transcript.meta.get("timings", {}) for transcript in transcripts]
        self.report("  timings: " + clock.text(context.info["engine"]))
        try:
            review = ReviewHook().write(context)
        except Exception as error:
            context.warnings.append(f"review page not written: {error}")
            review = None
        if review is not None:
            context.info["review"] = str(review)
        return context

    def cached_pass(self, path, key, compute):
        cache = StageCache(path)
        cached = cache.load(key)
        if cached is not None:
            return PassFile().load(cached)
        result = compute()
        PassFile().save(cache, key, result)
        return result

    def script_path(self):
        path = self.settings["script"].get("path")
        return self.project.resolve(path) if path else None

    def timeline_fingerprint(self, files, value):
        if not value:
            return None
        prefixes = [value] if isinstance(value, str) else list(value)
        return [Fingerprint().file(files.prefix(prefix) + "-timeline.csv") for prefix in prefixes]


class TalkEdl:
    """Regenerates edl.json from candidates.json and decisions.json."""

    def __init__(self, paths, settings):
        self.paths = paths
        self.settings = settings

    def build(self, candidates):
        files = CandidatesFile(self.paths.candidates)
        builder = EdlBuilder(candidates["talk"], files.sources(candidates), candidates.get("fps", ""),
                             candidates.get("wordMiddles", []), self.settings["plan"], candidates["sourceKey"])
        return builder.build(files.cuts(candidates), DecisionsFile(self.paths.decisions).load())

    def write(self, candidates):
        result = self.build(candidates)
        JsonWriter().write(self.paths.edl, result.edl.to_dict())
        return result


class CutSettings:
    """A fingerprint of the settings that decide the cuts, kept in candidates.json, so plan and render can tell that
    rostrum.json changed after init."""

    def fingerprint(self, settings):
        return Fingerprint().of({group: settings[group] for group in CUT_SETTING_GROUPS})

    def changed(self, settings, candidates):
        recorded = candidates.get("cutSettings")
        return (recorded is not None and
                recorded != self.fingerprint(settings))


class ReviewState:
    """Reviewed means the decisions were reviewed after the last analysis: decisions.json's reviewedAt (written by
    rostrum decide and decide --import) is later than the candidates' analyzedAt. A decisions.json without reviewedAt
    (a hand edit, or a page saved straight over it) counts from its file time."""

    def __init__(self, paths, candidates):
        self.paths = paths
        self.candidates = candidates

    def analyzed_at(self):
        recorded = self.candidates.get("analyzedAt")
        return float(recorded) if recorded is not None else self.paths.candidates.stat().st_mtime

    def decided_at(self):
        """The time of the last review, or None."""
        return DecisionsFile(self.paths.decisions).review_time()

    def reviewed(self):
        decided = self.decided_at()
        return (decided is not None and
                decided > self.analyzed_at())


class SourceDrift:
    """Whether a talk's sources changed after init: a file replaced or edited, or the talk's list of sources changed
    in the project file. The cut times belong to the files as they were at init."""

    def problem(self, project, talk, candidates):
        recorded = [Path(item["path"]).resolve() for item in candidates.get("sources", [])]
        current = project.source_paths(talk)
        if [path.resolve() for path in current] != recorded:
            return "the talk's sources in the project file changed since init"
        missing = [path.name for path in current if not path.is_file()]
        if missing:
            return "source not found: " + ", ".join(missing)
        if SourceKey().of([str(path) for path in current]) != candidates.get("sourceKey"):
            return "a source file changed since init"
        return None


class ReviewRefresh:
    """Rebuilds review.html after decisions change, so the page shows (and saves back) the current decisions."""

    def write(self, project, talk, settings, paths, candidates, result):
        sources = CandidatesFile(paths.candidates).sources(candidates)
        context = TalkContext(project, talk, settings, paths, sources=sources, media=candidates.get("media", []),
                              cuts=result.cuts, edl=result.edl, stale=result.stale,
                              analyzed_at=candidates.get("analyzedAt"))
        return ReviewHook().write(context)


class ProjectDefaults:
    """Settings when no talk is chosen: user config plus project defaults when the project file exists,
    otherwise user config plus the built-in defaults."""

    def __init__(self, loader=None):
        self.loader = loader or ConfigLoader()

    def values(self, project_path):
        project = self.project(project_path)
        if project is not None:
            return self.loader.effective(project).values
        return self.loader.effective().values

    def event(self, project_path):
        project = self.project(project_path)
        return project.event if project is not None else ""

    def project(self, project_path):
        path = Path(project_path) if project_path else None
        if (path is not None and
                path.is_file()):
            return ProjectFile(self.loader.schema).load(path)
        return None


class Workspace:
    """A loaded project plus settings per talk. Shared by the commands."""

    def __init__(self, project_path, flags=None, loader=None):
        self.loader = loader or ConfigLoader()
        self.project = ProjectFile(self.loader.schema).load(project_path)
        self.flags = flags

    def talks(self, talk_id=None):
        return self.project.select(talk_id)

    def config(self, talk):
        return self.loader.effective(self.project, talk, self.flags)

    def paths(self, talk):
        return TalkPaths(self.project, talk, self.config(talk).values)

    def state(self, talk):
        """Candidates with decisions applied, without writing anything."""
        settings = self.config(talk).values
        paths = TalkPaths(self.project, talk, settings)
        candidates = CandidatesFile(paths.candidates).load()
        return candidates, TalkEdl(paths, settings).build(candidates), paths

    def find_cut(self, cut_id, talk_id=None):
        matches = []
        for talk in self.talks(talk_id):
            try:
                _, result, _ = self.state(talk)
            except ConfigError:
                continue
            for cut in result.cuts:
                if cut.id == cut_id:
                    matches.append((talk, cut))
        return matches
