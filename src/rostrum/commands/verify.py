"""rostrum verify: check a rendered video against its EDL (fast or full)."""
import json
from pathlib import Path

from rostrum.commands.render import RENDER_ERRORS, EdlInput, InputError
from rostrum.config import ConfigError
from rostrum.cutmap import TimeText
from rostrum.edl import EdlFile
from rostrum.output import Output
from rostrum.pipeline import ProjectDefaults, Workspace
from rostrum.project import EXIT_FAILED, EXIT_INPUT, EXIT_OK
from rostrum.render import RENDER_JSON, RenderFolders
from rostrum.tools import Tools
from rostrum.verify import MODES, VerifyError, Verifier

TARGET_ERRORS = (InputError, ValueError, KeyError, OSError)


class VerifyCommand:
    def __init__(self):
        self.text = TimeText()

    def run(self, args):
        output = Output(args.json)
        if (args.render or
                args.output):
            return self.run_file(args, output)
        return self.run_project(args, output)

    def run_file(self, args, output):
        try:
            edl, video = self.target(args)
            values = ProjectDefaults().values(args.project)
        except ConfigError as error:
            return output.fail(str(error), EXIT_INPUT)
        except TARGET_ERRORS as error:
            return output.fail(str(error))
        try:
            result = self.check(edl, video, values, args.mode, output)
        except (VerifyError, *RENDER_ERRORS) as error:
            return output.fail(str(error))
        output.result(result, self.describe(result))
        return EXIT_OK if result["ok"] else EXIT_FAILED

    def run_project(self, args, output):
        try:
            workspace = Workspace(args.project)
            talks = workspace.talks(args.talk)
        except ConfigError as error:
            return output.fail(f"{error} (or check a render folder with --render DIR)", EXIT_INPUT)
        results = [self.check_talk(workspace, talk, args, output) for talk in talks]
        ok = all(item["ok"] for item in results)
        data = {"ok": ok, "project": str(workspace.project.path), "talks": results}
        output.result(data, "\n\n".join(self.describe_talk(item) for item in results))
        return EXIT_OK if ok else EXIT_FAILED

    def check_talk(self, workspace, talk, args, output):
        """The talk's newest render that worked, in <work dir>/renders. Planning settings come from its render.json."""
        try:
            values = workspace.config(talk).values
            folder, failed = RenderFolders(values["render"]).newest(workspace.paths(talk).renders)
            skipped = [path.name for path in failed]
            if folder is None:
                reason = f" ({', '.join(skipped)} failed)" if skipped else ""
                raise InputError(f"no finished render for talk {talk.id}{reason}; run: rostrum render --talk {talk.id}")
            edl, video = self.render_folder(folder)
            note = f"; skipped newer failed render(s): {', '.join(skipped)}" if skipped else ""
            output.info(f"[{talk.id}] verifying {folder.name}{note}")
            result = self.check(edl, video, values, args.mode, output)
            return {"talk": talk.id, "render": str(folder), "skipped": skipped, **result}
        except (ConfigError, VerifyError, *TARGET_ERRORS, *RENDER_ERRORS) as error:
            output.info(f"[{talk.id}] failed: {error}")
            return {"talk": talk.id, "ok": False, "error": str(error)}

    def check(self, edl, video, values, mode, output):
        verify_settings = values["verify"]
        verifier = Verifier(verify_settings, Tools(values["tools"]), render_settings=values["render"], report=output.info)
        return verifier.verify(edl, video, mode or verify_settings.get("mode") or "fast")

    def target(self, args):
        if args.render:
            return self.render_folder(Path(args.render))
        return EdlInput().load(args), Path(args.output)

    def render_folder(self, folder):
        render_json = folder / RENDER_JSON
        if not render_json.is_file():
            raise InputError(f"no {RENDER_JSON} in {folder}")
        data = json.loads(render_json.read_text(encoding="utf-8"))
        if not data.get("ok", True):
            raise InputError(f"the render in {folder} failed: {data.get('error')}")
        return EdlFile().load(folder / data.get("edl", "edl.json")), folder / data["output"]

    def describe_talk(self, result):
        not_checked = (not result["ok"] and
                       "checks" not in result)
        if not_checked:
            return f"{result['talk']}: FAILED - {result['error']}"
        return f"{result['talk']} ({Path(result['render']).name}):\n{self.describe(result)}"

    def describe(self, result):
        summary = result["summary"]
        lines = [f"verify {result['mode']}: {'PASS' if result['ok'] else 'FAIL'} in {result['seconds']:.1f} s  "
                 f"({Path(result['output']).name})"]
        width = max((len(check["name"]) for check in result["checks"]), default=10)
        for check in result["checks"]:
            lines.append(f"  {'PASS' if check['ok'] else 'FAIL'}  {check['name']:<{width}}  {check['detail']}")
        shown = [row for row in result["joins"] if self.worth_showing(row, result["mode"])]
        if shown:
            lines.append("")
            lines.append(f"Joins ({len(shown)} of {summary.get('joins', len(result['joins']))} shown):")
            lines.append(" join  output        kind  cut on the talk timeline     v step  a step  before ms  after ms  result")
            lines += [self.join_line(row) for row in shown]
        lines.append("")
        lines.append(f"report     : {result['verify_json']}")
        if summary.get("png_dir"):
            lines.append(f"PNG pairs  : {summary['png_dir']}")
        return "\n".join(lines)

    def worth_showing(self, row, mode):
        sampled = (row["offset_before"] != "not sampled" or
                   row["offset_after"] != "not sampled")
        return (mode == "full" or
                sampled or
                bool(row["notes"]))

    def join_line(self, row):
        cut = f"{self.text.hms(row['cut_start'])}-{self.text.hms(row['cut_end'])}"
        result = "PASS" if not row["notes"] else "FAIL: " + "; ".join(row["notes"])
        return (f" {row['join']:>4}  {self.text.hms(row['output_time']):<12}  {row['kind']:<4}  {cut:<27}  "
                f"{self.number(row['video_step_ms']):>6}  {self.number(row['audio_step_error_ms']):>6}  "
                f"{self.offset(row, 'before'):>9}  {self.offset(row, 'after'):>8}  {result}")

    def offset(self, row, side):
        value = row[f"offset_{side}_ms"]
        return self.number(value) if value is not None else row[f"offset_{side}"]

    def number(self, value):
        return "-" if value is None else f"{value:.2f}"


def register(subparsers, common):
    parser = subparsers.add_parser(
        "verify", parents=[common], help="check a rendered video against its EDL",
        description="Check duration, frame and audio continuity, a full decode, timestamps at every join and the "
                    "A/V offset at joins. fast samples about 12 joins for A/V offset; full checks every join "
                    "and writes before/after PNG pairs. The report is verify-fast.json or verify-full.json next to "
                    "the video. Without --render or --output, checks the newest finished render of each talk in "
                    "the project (failed renders are skipped).",
    )
    parser.add_argument("--render", help="render folder (has render.json, edl.json and the output)")
    parser.add_argument("--output", help="rendered video; needs --edl")
    EdlInput().add_arguments(parser)
    parser.add_argument("--mode", choices=MODES, help="fast (default) or full")
    parser.add_argument("--talk", help="with a project: only this talk")
    parser.set_defaults(handler=VerifyCommand().run)
