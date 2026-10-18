"""rostrum render: one video per talk (or per EDL file), with a cut map, in a new numbered folder."""
import argparse
from pathlib import Path

from rostrum.config import ConfigError
from rostrum.cutmap import TimeText
from rostrum.edl import EdlFile
from rostrum.media import FfmpegError, MediaError
from rostrum.output import Output
from rostrum.pipeline import CandidatesFile, CutSettings, ProjectDefaults, ReviewState, SourceDrift, TalkEdl, Workspace
from rostrum.project import EXIT_FAILED, EXIT_INPUT, EXIT_OK
from rostrum.render import RenderError, Renderer
from rostrum.tools import ToolNotFound, Tools

DEFAULT_OUT = "renders"
RENDER_ERRORS = (RenderError, MediaError, FfmpegError, ToolNotFound, OSError)
SETTINGS_CHANGED = "cut settings in the project changed since init; run rostrum init to apply them"


class InputError(ValueError):
    pass


class EdlInput:
    """--edl for rostrum EDLs. The hidden --baseline-edl (with --source and --duration) reads an older keep-list
    format from the tool rostrum grew out of."""

    def add_arguments(self, parser):
        parser.add_argument("--edl", help="EDL file (schema rostrum.edl/1)")
        parser.add_argument("--baseline-edl", help=argparse.SUPPRESS)
        parser.add_argument("--source", help=argparse.SUPPRESS)
        parser.add_argument("--duration", type=float, help=argparse.SUPPRESS)

    def given(self, args):
        return bool(args.edl or args.baseline_edl)

    def load(self, args):
        if (args.edl and
                args.baseline_edl):
            raise InputError("use either --edl or --baseline-edl, not both")
        if args.edl:
            path = Path(args.edl)
            return self.resolved(EdlFile().load(path), path.parent)
        if args.baseline_edl:
            if args.duration is None:
                raise InputError("--baseline-edl needs --duration (the source length in seconds)")
            path = Path(args.baseline_edl)
            source = str(Path(args.source).resolve()) if args.source else None
            return self.resolved(EdlFile().load_baseline(path, duration=args.duration, source_path=source), path.parent)
        raise InputError("pass --edl FILE")

    def resolved(self, edl, base):
        """Relative source paths in an EDL are relative to the EDL file."""
        for source in edl.sources:
            path = Path(source.path)
            if not path.is_absolute():
                source.path = str((base / path).resolve())
        return edl


class RenderReadiness:
    """What must hold before a talk renders: its sources are the files init analyzed, and its cuts were reviewed
    since they last changed (rostrum decide, or decide --import of a save from the current review.html), unless
    --yes."""

    def __init__(self, workspace):
        self.workspace = workspace

    def problems(self, talks, confirmed):
        problems = []
        waiting = []
        for talk in talks:
            paths = self.workspace.paths(talk)
            if not paths.candidates.is_file():
                continue
            candidates = CandidatesFile(paths.candidates).load()
            drift = SourceDrift().problem(self.workspace.project, talk, candidates)
            if drift:
                problems.append(f"talk {talk.id}: {drift}; run rostrum init")
                continue
            reviewed = (confirmed or
                        ReviewState(paths, candidates).reviewed())
            if not reviewed:
                waiting.append(talk.id)
        if waiting:
            problems.append(f"not reviewed since the cuts last changed: {', '.join(waiting)}. Review them first "
                            "(rostrum plan and rostrum decide, or save from review.html and run rostrum decide "
                            "--import), or pass --yes to render them as they are.")
        return problems


class RenderCommand:
    def __init__(self):
        self.text = TimeText()

    def run(self, args):
        output = Output(args.json)
        if EdlInput().given(args):
            return self.run_file(args, output)
        return self.run_project(args, output)

    def run_file(self, args, output):
        try:
            edl = EdlInput().load(args)
            defaults = ProjectDefaults()
            values = defaults.values(args.project)
            event = defaults.event(args.project)
            renderer = Renderer(values["render"], Tools(values["tools"]), report=output.info)
        except ConfigError as error:
            return output.fail(str(error), EXIT_INPUT)
        except (InputError, ValueError, KeyError, OSError) as error:
            return output.fail(str(error))
        fields = {"talk": args.talk, "speaker": args.speaker, "title": args.title, "event": event}
        try:
            result = renderer.render(edl, Path(args.out or DEFAULT_OUT), fields)
        except RENDER_ERRORS as error:
            return output.fail(str(error))
        output.result(result, self.describe(result))
        return EXIT_OK

    def run_project(self, args, output):
        try:
            workspace = Workspace(args.project)
            talks = workspace.talks(args.talk)
            refused = RenderReadiness(workspace).problems(talks, args.yes)
        except ConfigError as error:
            return output.fail(f"{error} (or render an EDL file with --edl FILE)", EXIT_INPUT)
        if refused:
            return output.fail("\n".join(refused), EXIT_INPUT)
        results = [self.render_talk(workspace, talk, args, output) for talk in talks]
        ok = all(item["ok"] for item in results)
        data = {"ok": ok, "project": str(workspace.project.path), "talks": results}
        output.result(data, "\n\n".join(self.describe_talk(item) for item in results))
        return EXIT_OK if ok else EXIT_FAILED

    def render_talk(self, workspace, talk, args, output):
        """Rebuilds edl.json first, so decisions saved since the last run (for example from review.html) count."""
        try:
            values = workspace.config(talk).values
            paths = workspace.paths(talk)
            candidates = CandidatesFile(paths.candidates).load()
            warnings = [SETTINGS_CHANGED] if CutSettings().changed(values, candidates) else []
            for warning in warnings:
                output.info(f"[{talk.id}] warning: {warning}")
            edl = TalkEdl(paths, values).write(candidates).edl
            renderer = Renderer(values["render"], Tools(values["tools"]), report=output.info)
            out_dir = Path(args.out) / talk.id if args.out else paths.renders
            fields = {"talk": talk.id, "speaker": args.speaker or talk.speaker, "title": args.title or talk.title,
                      "event": workspace.project.event}
            output.info(f"[{talk.id}] rendering")
            result = renderer.render(edl, out_dir, fields)
            result["warnings"] = warnings + result["warnings"]
            return {"talk": talk.id, **result}
        except (ConfigError, *RENDER_ERRORS) as error:
            output.info(f"[{talk.id}] failed: {error}")
            return {"talk": talk.id, "ok": False, "error": str(error)}

    def describe_talk(self, result):
        if not result["ok"]:
            return f"{result['talk']}: FAILED - {result['error']}"
        return f"{result['talk']}:\n{self.describe(result)}"

    def describe(self, result):
        speed = f" ({result['speed']:.1f}x real time)" if result.get("speed") else ""
        lines = [
            f"Rendered {Path(result['output']).name}",
            f"  folder   : {result['dir']}",
            f"  length   : {self.text.hms(result['duration'])} ({result['frames']} frames at {result['fps']} fps), "
            f"{result['segments']} segments, {result['joins']} joins",
            f"  time     : {self.text.hms(result['timings']['total'])} total, encode {self.text.hms(result['timings']['encode'])}{speed}",
            f"  cut map  : {result['cutmap']}",
            f"  next     : rostrum verify --render \"{result['dir']}\"",
        ]
        lines += [f"  warning  : {warning}" for warning in result["warnings"]]
        return "\n".join(lines)


def register(subparsers, common):
    parser = subparsers.add_parser(
        "render", parents=[common], help="render the kept segments into one video",
        description="Render the kept segments into one video, plus cutmap.json and render.json, in a new numbered "
                    "folder. Existing files are never overwritten. Without --edl, renders the talks of the project: "
                    "edl.json is rebuilt from the cuts and your decisions first, and the output goes to "
                    "<work dir>/renders. A talk renders only after its cuts were reviewed since they last changed "
                    "(rostrum decide, or decide --import of a save from review.html), or with --yes; never when a "
                    "source changed since init.",
    )
    EdlInput().add_arguments(parser)
    parser.add_argument("--out", help=f"folder for render folders (default: ./{DEFAULT_OUT} with --edl, "
                                      f"<work dir>/{DEFAULT_OUT} per talk; with a project, DIR/<talk id>)")
    parser.add_argument("--talk", help="with a project: render only this talk; with --edl: the talk name for the "
                                       "output name template")
    parser.add_argument("--speaker", help="speaker for the output name template")
    parser.add_argument("--title", help="title for the output name template")
    parser.add_argument("--yes", action="store_true",
                        help="render the cuts as they are, without a review since they last changed")
    parser.set_defaults(handler=RenderCommand().run)
