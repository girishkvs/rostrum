"""rostrum status (also bare `rostrum`): a summary of every talk in the project."""
from pathlib import Path

from rostrum.config import ConfigError
from rostrum.output import Output
from rostrum.pipeline import (CandidatesFile, Clock, CutSettings, DecisionsFile, ReviewState, SourceDrift, TalkEdl,
                              Workspace)
from rostrum.project import EXIT_INPUT, EXIT_OK, PROJECT_FILE
from rostrum.render import RENDER_JSON, RenderFolders


class TalkStatus:
    def of(self, workspace, talk):
        settings = workspace.config(talk).values
        paths = workspace.paths(talk)
        transcripts = sum(1 for index in range(len(talk.sources)) if paths.transcript(index).is_file())
        folders = RenderFolders(settings["render"])
        renders = folders.all(paths.renders)
        status = {
            "id": talk.id,
            "title": talk.title,
            "speaker": talk.speaker,
            "sources": len(talk.sources),
            "workDir": str(paths.root),
            "stages": {
                "transcribed": f"{transcripts}/{len(talk.sources)}",
                "detected": paths.detect.is_file(),
                "reconciled": paths.reconcile.is_file(),
                "candidates": paths.candidates.is_file(),
                "edl": paths.edl.is_file(),
            },
            "renders": [folder.name for _, folder, _ in reversed(renders)],
            "failedRenders": [folder.name for _, folder, ok in reversed(renders) if not ok],
            "notes": [],
        }
        if not paths.candidates.is_file():
            status["state"] = "not analyzed (run rostrum init)"
            return status
        candidates = CandidatesFile(paths.candidates).load()
        result = TalkEdl(paths, settings).build(candidates)
        decisions = DecisionsFile(paths.decisions).load()
        status.update({
            "state": self.state(paths, candidates, folders),
            "duration": round(result.edl.duration, 3),
            "kept": round(result.edl.kept, 3),
            "cuts": sum(1 for cut in result.cuts if cut.decision == "cut"),
            "proposed": sum(1 for cut in result.cuts if cut.level == "propose" and cut.decision == "keep"),
            "decisions": len(decisions),
            "stale": len(result.stale),
        })
        drift = SourceDrift().problem(workspace.project, talk, candidates)
        if drift:
            status["notes"].append(f"{drift}; run rostrum init")
        if CutSettings().changed(settings, candidates):
            status["notes"].append("cut settings changed since init; run rostrum init to apply them")
        return status

    def state(self, paths, candidates, folders):
        """rendered: a finished render newer than the analysis and the last decision change; reviewed: reviewed since
        the cuts last changed; otherwise ready for review."""
        review = ReviewState(paths, candidates)
        changed = DecisionsFile(paths.decisions).change_time()
        newest, _ = folders.newest(paths.renders)
        rendered_at = (newest / RENDER_JSON).stat().st_mtime if newest is not None else None
        rendered = (rendered_at is not None and
                    rendered_at > review.analyzed_at() and
                    (changed is None or rendered_at >= changed))
        if rendered:
            return f"rendered ({newest.name})"
        if review.reviewed():
            return "reviewed, ready to render"
        return "ready for review"


class StatusCommand:
    def run(self, args):
        output = Output(args.json)
        if not Path(args.project).is_file():
            message = (f"no {PROJECT_FILE} here. Start with: rostrum init <video files or a folder>"
                       if args.project == PROJECT_FILE else f"project file not found: {args.project}")
            return output.fail(message, EXIT_INPUT)
        try:
            workspace = Workspace(args.project)
            talks = [TalkStatus().of(workspace, talk) for talk in workspace.talks(args.talk)]
        except ConfigError as error:
            return output.fail(str(error), EXIT_INPUT)
        data = {"ok": True, "project": str(workspace.project.path), "event": workspace.project.event, "talks": talks}
        output.result(data, self.text(data))
        return EXIT_OK

    def text(self, data):
        clock = Clock()
        lines = [f"{data['event'] or 'project'}: {data['project']}"]
        for talk in data["talks"]:
            label = f"{talk['id']}" + (f" - {talk['title']}" if talk["title"] else "")
            lines.append(f"{label}: {talk['state']}")
            if "kept" in talk:
                lines.append(f"  {clock.text(talk['duration'])} -> {clock.text(talk['kept'])}; {talk['cuts']} cuts, "
                             f"{talk['proposed']} proposed, {talk['decisions']} decisions"
                             + (f", {talk['stale']} STALE" if talk["stale"] else ""))
            if talk["renders"]:
                failed = set(talk["failedRenders"])
                names = [name + (" (failed)" if name in failed else "") for name in talk["renders"]]
                lines.append(f"  renders: {', '.join(names)}")
            lines += [f"  note: {note}" for note in talk["notes"]]
        return "\n".join(lines)


def register(subparsers, common):
    parser = subparsers.add_parser("status", parents=[common], help="project summary (the default command)",
                                   description="Shows each talk: analysis stages, cuts, decisions and renders.")
    parser.add_argument("--talk", help="only this talk")
    parser.set_defaults(handler=StatusCommand().run)
