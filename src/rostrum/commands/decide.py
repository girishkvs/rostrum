"""rostrum decide: keep or cut one candidate (optionally moving its edges), or import a decisions file.
Either way decisions.json is updated, edl.json is rebuilt and review.html is rewritten with the new decisions."""
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from rostrum.config import ConfigError, JsonFile
from rostrum.output import Output
from rostrum.pipeline import (DECISIONS, DECISIONS_SCHEMA, CandidatesFile, Clock, DecisionsFile, ReviewRefresh,
                              ReviewState, TalkEdl, Workspace)
from rostrum.project import EXIT_INPUT, EXIT_OK

CLOCK_TEXT = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")
# A page's basedOn is the candidates' analyzedAt written again; this absorbs float rounding on the way.
ANALYSIS_TOLERANCE = 1e-6


class TimeValue:
    """Seconds ("754.2") or a clock ("12:34.2", "1:02:03")."""

    def seconds(self, text):
        if text is None:
            return None
        value = str(text).strip()
        match = CLOCK_TEXT.match(value)
        if match:
            hours = int(match.group(1) or 0)
            return hours * 3600 + int(match.group(2)) * 60 + float(match.group(3))
        try:
            return float(value)
        except ValueError as error:
            raise ConfigError(f"not a time: {text!r} (use seconds or mm:ss.s)") from error


class DecisionEntry:
    """One entry of an imported decisions file: {decision, start?, end?, decided_by?, at?}."""

    def __init__(self, source):
        self.source = source

    def normalized(self, cut_id, entry, cut):
        if not isinstance(entry, dict):
            raise ConfigError(f"{self.source}: decisions.{cut_id}: expected an object")
        if entry.get("decision") not in DECISIONS:
            raise ConfigError(f"{self.source}: decisions.{cut_id}.decision: expected keep or cut, "
                              f"not {entry.get('decision')!r}")
        result = {"decision": entry["decision"],
                  "decided_by": str(entry.get("decided_by") or "user"),
                  "at": str(entry.get("at") or datetime.now(timezone.utc).isoformat(timespec="seconds"))}
        for side in ("start", "end"):
            if entry.get(side) is not None:
                result[side] = self.number(cut_id, side, entry[side])
        if cut is not None:
            start = result.get("start", cut.start)
            end = result.get("end", cut.end)
            if end <= start:
                raise ConfigError(f"{self.source}: decisions.{cut_id}: end ({end:.3f}) must be after start ({start:.3f})")
        return result

    def number(self, cut_id, side, value):
        is_number = (isinstance(value, (int, float)) and
                     not isinstance(value, bool))
        if not is_number:
            raise ConfigError(f"{self.source}: decisions.{cut_id}.{side}: expected seconds, not {value!r}")
        return float(value)


class DecisionTime:
    """The "at" time of a decision entry (ISO 8601); a time without a zone is UTC."""

    def of(self, entry):
        try:
            moment = datetime.fromisoformat(str(entry.get("at")))
        except (TypeError, ValueError):
            return None
        return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)

    def newer(self, existing, incoming):
        """Is the entry already saved strictly newer than the imported one?"""
        existing_time = self.of(existing)
        incoming_time = self.of(incoming)
        return (existing_time is not None and
                incoming_time is not None and
                existing_time > incoming_time)


@dataclass
class ImportOutcome:
    talk: object
    applied: dict
    skipped: list
    kept_newer: list
    candidates: dict
    result: object
    paths: object
    basis: str
    saved: bool
    reviewed: bool

    @property
    def page_outdated(self):
        return self.basis in ("older", "newer")


class PageBasis:
    """How an imported file relates to the current analysis, from its basedOn (the candidates' analyzedAt that the
    review page was built from): "current", "older", "newer", or "none" when the file has no basedOn."""

    def of(self, path, data, analyzed_at):
        based_on = data.get("basedOn")
        if based_on is None:
            return "none"
        is_number = (isinstance(based_on, (int, float)) and
                     not isinstance(based_on, bool))
        if not is_number:
            raise ConfigError(f"{path}: basedOn must be a number (the analysis time the page was built from)")
        if abs(based_on - analyzed_at) <= ANALYSIS_TOLERANCE:
            return "current"
        return "older" if based_on < analyzed_at else "newer"


class DecisionImport:
    """Merges a decisions file (schema rostrum.decisions/1) into the talk's decisions.json.

    Entries for cut ids that no longer exist are skipped and reported as stale. When both files have a decision for
    the same cut, the one with the newer "at" stays, so an older page saved later never undoes a newer decision.
    Whether the import is a review depends on the file's basedOn: a page built from the current analysis is one,
    even when nothing changed; a page built from another analysis is not, and the previous review time stays; a
    file without basedOn is one when it applied something. A file that applies nothing, from anything but a current
    page, leaves decisions.json alone."""

    def __init__(self, workspace):
        self.workspace = workspace

    def run(self, path, talk_option):
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"decisions file not found: {path}")
        data = JsonFile().read(path)
        if (not isinstance(data, dict) or
                data.get("schema") != DECISIONS_SCHEMA):
            raise ConfigError(f"{path}: expected a JSON object with schema {DECISIONS_SCHEMA}")
        entries = data.get("decisions")
        if not isinstance(entries, dict):
            raise ConfigError(f"{path}: decisions must be an object keyed by cut id")
        talk = self.workspace.project.talk(self.talk_id(path, data.get("talk"), talk_option))
        candidates, current, paths = self.workspace.state(talk)
        basis = PageBasis().of(path, data, ReviewState(paths, candidates).analyzed_at())
        cuts = {cut.id: cut for cut in current.cuts}
        reader = DecisionEntry(path)
        accepted = {}
        skipped = []
        for cut_id, entry in entries.items():
            normalized = reader.normalized(cut_id, entry, cuts.get(cut_id))
            if cut_id in cuts:
                accepted[cut_id] = normalized
            else:
                skipped.append({"id": cut_id, **normalized})
        decisions = DecisionsFile(paths.decisions)
        previous_review = decisions.review_time()
        previous_change = decisions.change_time()
        merged, applied, kept_newer = self.merge(decisions.load(), accepted)
        reviewed = self.is_review(basis, applied)
        saved = (basis == "current" or
                 bool(applied))
        settings = self.workspace.config(talk).values
        if saved:
            now = time.time()
            decisions.save(talk.id, merged, now if reviewed else previous_review, now if applied else previous_change)
            result = TalkEdl(paths, settings).write(candidates)
        else:
            result = TalkEdl(paths, settings).build(candidates)
        return ImportOutcome(talk, applied, skipped, kept_newer, candidates, result, paths, basis, saved, reviewed)

    def merge(self, merged, accepted):
        """Imported entries replace saved ones unless the saved one is strictly newer. An entry that says what the
        saved one already says (same decision and edges) changes nothing, so it is not applied."""
        applied = {}
        kept_newer = []
        timing = DecisionTime()
        for cut_id, entry in accepted.items():
            existing = merged.get(cut_id)
            if existing is not None:
                if timing.newer(existing, entry):
                    kept_newer.append(cut_id)
                    continue
                if self.same_effect(existing, entry):
                    continue
            merged[cut_id] = entry
            applied[cut_id] = entry
        return merged, applied, kept_newer

    def same_effect(self, existing, entry):
        return all(existing.get(key) == entry.get(key) for key in ("decision", "start", "end"))

    def is_review(self, basis, applied):
        if basis == "current":
            return True
        if basis == "none":
            return bool(applied)
        return False

    def talk_id(self, path, from_file, from_option):
        if (from_file and
                from_option and
                from_file != from_option):
            raise ConfigError(f"{path} is for talk {from_file!r}, but --talk is {from_option!r}")
        talk_id = from_file or from_option
        if not talk_id:
            raise ConfigError(f"{path} names no talk; pass --talk")
        return talk_id


class DecideCommand:
    def run(self, args):
        output = Output(args.json)
        try:
            self.check_arguments(args)
            workspace = Workspace(args.project)
            if args.import_file:
                return self.run_import(workspace, args, output)
            return self.run_one(workspace, args, output)
        except ConfigError as error:
            return output.fail(str(error), EXIT_INPUT)

    def check_arguments(self, args):
        single = (args.id, args.decision, args.start, args.end)
        if args.import_file:
            if any(value is not None for value in single):
                raise ConfigError("with --import, leave out the cut id, keep|cut, --start and --end")
            return
        if (args.id is None or
                args.decision is None):
            raise ConfigError("give a cut id and keep or cut (rostrum decide <id> keep|cut), or --import FILE")

    def run_one(self, workspace, args, output):
        talk, cut = self.find(workspace, args.id, args.talk)
        start = TimeValue().seconds(args.start)
        end = TimeValue().seconds(args.end)
        new_start = cut.start if start is None else start
        new_end = cut.end if end is None else end
        if new_end <= new_start:
            raise ConfigError(f"end ({new_end:.3f}) must be after start ({new_start:.3f})")
        settings = workspace.config(talk).values
        paths = workspace.paths(talk)
        DecisionsFile(paths.decisions).set(talk.id, cut.id, args.decision, start, end)
        candidates = CandidatesFile(paths.candidates).load()
        result = TalkEdl(paths, settings).write(candidates)
        review = self.refresh_review(workspace, talk, paths, candidates, result, output)
        updated = next(item for item in result.cuts if item.id == cut.id)
        data = {"ok": True, "talk": talk.id, "cut": asdict(updated), "kept": round(result.edl.kept, 3),
                "edl": str(paths.edl), "review": review, "stale": result.stale}
        clock = Clock()
        text = (f"{talk.id}: {updated.id} -> {updated.decision} {clock.text(updated.start)}-{clock.text(updated.end)}; "
                f"expected output {clock.text(result.edl.kept)}")
        output.result(data, text)
        return EXIT_OK

    def run_import(self, workspace, args, output):
        outcome = DecisionImport(workspace).run(args.import_file, args.talk)
        talk, paths, result = outcome.talk, outcome.paths, outcome.result
        if outcome.saved:
            review = self.refresh_review(workspace, talk, paths, outcome.candidates, result, output)
        else:
            review = str(paths.review) if paths.review.is_file() else None
        data = {"ok": True, "talk": talk.id, "imported": len(outcome.applied), "stale": outcome.skipped,
                "keptNewer": outcome.kept_newer, "staleSaved": result.stale, "kept": round(result.edl.kept, 3),
                "decisions": str(paths.decisions), "edl": str(paths.edl), "review": review,
                "saved": outcome.saved, "reviewed": outcome.reviewed, "pageOutdated": outcome.page_outdated}
        output.result(data, "\n".join(self.import_lines(outcome)))
        return EXIT_OK

    def import_lines(self, outcome):
        talk_id = outcome.talk.id
        if outcome.saved:
            lines = [f"{talk_id}: imported {len(outcome.applied)} decision(s); "
                     f"expected output {Clock().text(outcome.result.edl.kept)}"]
        else:
            lines = [f"{talk_id}: nothing to import; decisions.json unchanged"]
        if outcome.kept_newer:
            lines.append(f"  kept {len(outcome.kept_newer)} newer decision(s) already saved: "
                         + ", ".join(outcome.kept_newer))
        if outcome.skipped:
            lines.append(f"  {len(outcome.skipped)} STALE (no cut has this id now, not applied): "
                         + ", ".join(entry["id"] for entry in outcome.skipped))
        if outcome.page_outdated:
            lines.append("  " + self.outdated_note(outcome))
        return lines

    def outdated_note(self, outcome):
        if outcome.basis == "older":
            built = "this page was built before the last init changed the cuts"
            cuts = "the new cuts"
        else:
            built = "this page was built from a different analysis than the current cuts"
            cuts = "the current cuts"
        if outcome.saved:
            return f"{built}; decisions applied, but {cuts} still need a review (open review.html again)"
        return f"{built}; open review.html again to review {cuts}"

    def refresh_review(self, workspace, talk, paths, candidates, result, output):
        """review.html is rewritten with the new decisions, so the page never saves an older set back."""
        try:
            page = ReviewRefresh().write(workspace.project, talk, workspace.config(talk).values, paths, candidates,
                                         result)
        except Exception as error:
            # The decision is saved either way; the page belongs to the review module.
            output.info(f"warning: review page not updated: {error}")
            return None
        return str(page) if page else None

    def find(self, workspace, cut_id, talk_id):
        matches = workspace.find_cut(cut_id, talk_id)
        if not matches:
            where = f" in talk {talk_id}" if talk_id else ""
            raise ConfigError(f"no cut {cut_id!r}{where}; list them with: rostrum cuts")
        if len(matches) > 1:
            talks = ", ".join(talk.id for talk, _ in matches)
            raise ConfigError(f"cut {cut_id!r} is in several talks ({talks}); pass --talk")
        return matches[0]


def register(subparsers, common):
    parser = subparsers.add_parser(
        "decide", parents=[common], help="keep or cut one candidate, or import a decisions file",
        description="Records your decision in decisions.json (decided_by: user; the previous file is kept as "
                    "decisions.json.bak), rewrites edl.json and refreshes review.html. A decision counts as a review "
                    "for rostrum render. --start and --end move the cut's edges (seconds or mm:ss.s, on the talk "
                    "timeline). --import merges a decisions file (schema rostrum.decisions/1, as saved by "
                    "review.html): for a cut decided in both files, the entry with the newer \"at\" time stays; "
                    "entries for cuts that no longer exist are skipped and reported as stale. A page built from the "
                    "current analysis counts as a review; a page built before the last init changed the cuts does "
                    "not; a file without basedOn counts when it applies a decision. A file that applies nothing "
                    "leaves decisions.json unchanged.")
    parser.add_argument("id", nargs="?", help="cut id from rostrum cuts or rostrum plan")
    parser.add_argument("decision", nargs="?", choices=list(DECISIONS))
    parser.add_argument("--start", help="new start of the cut")
    parser.add_argument("--end", help="new end of the cut")
    parser.add_argument("--talk", help="talk id, when the cut id alone is not enough, or for --import")
    parser.add_argument("--import", dest="import_file", metavar="FILE", help="decisions file to merge")
    parser.set_defaults(handler=DecideCommand().run)
