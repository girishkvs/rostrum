"""rostrum init: write the project file from media files, then run the first pass for every talk."""
import copy
from pathlib import Path

from rostrum.audio import AudioError
from rostrum.config import ConfigError, ConfigLoader
from rostrum.engines import EngineError, EngineRegistry
from rostrum.media import FfmpegError, MediaError
from rostrum.output import Output
from rostrum.pipeline import Clock, TalkPipeline, Workspace
from rostrum.project import EXIT_FAILED, EXIT_INPUT, EXIT_OK, Project, ProjectFile, TalkGrouper
from rostrum.tools import ToolNotFound

RUN_ERRORS = (ConfigError, EngineError, AudioError, MediaError, FfmpegError, ToolNotFound, OSError)


class ProjectSetup:
    """Adds talks for the given files to the project file (created if needed). Other settings are kept."""

    def __init__(self, loader):
        self.loader = loader
        self.files = ProjectFile(loader.schema)

    def write(self, project_path, paths, engine):
        existing = self.files.load(project_path) if project_path.is_file() else None
        settings = self.loader.effective(existing).values
        grouper = TalkGrouper(settings["init"])
        media = grouper.media_files(paths)
        if not media:
            raise ConfigError("no media files found in: " + ", ".join(str(path) for path in paths))
        data = copy.deepcopy(existing.data) if existing else self.files.new_data([])
        project = Project(project_path, data)
        talks = {talk["id"]: talk for talk in data["talks"]}
        for talk_id, group in grouper.group(media):
            sources = [project.relative(path) for path in group]
            if talk_id in talks:
                talks[talk_id]["sources"] = sources
                continue
            entry = {"id": talk_id, "sources": sources}
            data["talks"].append(entry)
            talks[talk_id] = entry
        if engine:
            data.setdefault("defaults", {}).setdefault("speech", {})["engine"] = engine
        backup = self.files.save(project_path, data)
        return backup


class InitCommand:
    def run(self, args):
        output = Output(args.json)
        project_path = Path(args.project)
        loader = ConfigLoader()
        backup = None
        try:
            if args.engine:
                EngineRegistry().engine_class(args.engine)
            if args.paths:
                backup = ProjectSetup(loader).write(project_path, args.paths, args.engine)
                note = f" (previous file kept as {backup.name})" if backup else ""
                output.info(f"wrote {project_path.resolve()}{note}")
            flags = {"speech": {"engine": args.engine}} if args.engine else None
            workspace = Workspace(project_path, flags, loader)
            talks = workspace.talks(args.talk)
        except (ConfigError, EngineError) as error:
            return output.fail(str(error), EXIT_INPUT)
        results = []
        for talk in talks:
            output.info(f"[{talk.id}] first pass")
            try:
                config = workspace.config(talk)
                context = TalkPipeline(workspace.project, talk, config, output).run()
                results.append(context.summary())
            except RUN_ERRORS as error:
                output.info(f"[{talk.id}] failed: {error}")
                results.append({"id": talk.id, "ok": False, "error": str(error)})
        ok = all(item["ok"] for item in results)
        data = {"ok": ok, "project": str(workspace.project.path), "backup": str(backup) if backup else None, "talks": results}
        output.result(data, self.text(data))
        return EXIT_OK if ok else EXIT_FAILED

    def text(self, data):
        clock = Clock()
        lines = []
        for talk in data["talks"]:
            if not talk["ok"]:
                lines.append(f"{talk['id']}: FAILED - {talk['error']}")
                continue
            removed = talk["duration"] - talk["kept"]
            lines.append(f"{talk['id']}: {clock.text(talk['duration'])} -> {clock.text(talk['kept'])} expected "
                         f"({clock.text(removed)} removed in {talk['cuts']} cuts; {talk['proposed']} proposed, kept for now)")
            for warning in talk["warnings"]:
                lines.append(f"  warning: {warning}")
            lines.append(f"  cuts: {talk['files']['candidates']}")
            lines.append(f"  edl:  {talk['files']['edl']}")
            if talk["files"].get("review"):
                lines.append(f"  review: {talk['files']['review']}")
            if talk["stale"]:
                lines.append(f"  {talk['stale']} earlier decision(s) no longer match a cut (STALE); see rostrum plan")
        lines.append("Next: rostrum plan, then rostrum decide <id> keep|cut for anything you disagree with.")
        return "\n".join(lines)


def register(subparsers, common):
    parser = subparsers.add_parser(
        "init", parents=[common], help="first pass: probe, transcribe and find cuts",
        description="Writes the project file for the given media files or folders (one talk per file; -part1, -part2 "
                    "files are joined), then transcribes each talk and finds cuts. Nothing is cut until you render. "
                    "With no paths, re-runs the talks already in the project; unchanged stages come from the cache.")
    parser.add_argument("paths", nargs="*", help="media files or folders")
    parser.add_argument("--engine", help="speech engine for this run: faster-whisper, command, import, or a plugin")
    parser.add_argument("--talk", help="only this talk")
    parser.set_defaults(handler=InitCommand().run)
