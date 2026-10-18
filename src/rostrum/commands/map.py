"""rostrum map: convert times between a render and its sources, using the render's cutmap.json."""
from pathlib import Path

from rostrum.config import ConfigError
from rostrum.cutmap import CutMapFile, MapError, TimeText
from rostrum.output import Output
from rostrum.pipeline import Workspace


class RenderLookup:
    """--from is a render folder, its cutmap.json, or just a render folder name (render-4) under a talk's renders
    folder in the project."""

    def path(self, value, project, talk_id):
        given = Path(value)
        if (given.exists() or
                len(given.parts) > 1):
            return given
        try:
            workspace = Workspace(project)
            found = []
            for talk in workspace.talks(talk_id):
                folder = workspace.paths(talk).renders / value
                if folder.is_dir():
                    found.append((talk, folder))
        except ConfigError as error:
            raise MapError(f"no cut map at {value}, and no project to look it up in ({error})") from error
        if not found:
            raise MapError(f"no render folder {value!r} in the renders folder of any talk; give the folder path")
        if len(found) > 1:
            names = ", ".join(talk.id for talk, _ in found)
            raise MapError(f"{value} exists for several talks ({names}); pass --talk")
        return found[0][1]


class MapCommand:
    def __init__(self):
        self.text = TimeText()

    def run(self, args):
        output = Output(args.json)
        try:
            cutmap = CutMapFile().load(RenderLookup().path(args.map_from, args.project, args.talk))
            seconds = self.text.parse(args.time)
            if args.to == "source":
                result = cutmap.to_source(seconds)
            else:
                result = cutmap.to_output(seconds, args.source_index)
        except (MapError, OSError, ValueError, KeyError) as error:
            return output.fail(str(error))
        result = {"ok": True, **result}
        output.result(result, self.describe(result, cutmap))
        return 0

    def describe(self, result, cutmap):
        count = len(cutmap.segments)
        if result["direction"] == "output-to-source":
            return (f"output {self.text.hms(result['output'])} -> talk {self.text.hms(result['timeline'])}; "
                    f"{self.source(cutmap, result['source'])} at {self.text.hms(result['local'])} "
                    f"(segment {result['segment'] + 1} of {count})")
        asked = self.asked(result, cutmap)
        if not result["in_cut"]:
            return (f"{asked} -> output {self.text.hms(result['output'])} "
                    f"(segment {result['segment'] + 1} of {count})")
        start, end = result["cut"]
        reasons = f" ({'; '.join(result['reasons'])})" if result["reasons"] else ""
        lines = [f"{asked} is inside a cut: talk {self.text.hms(start)}-{self.text.hms(end)}{reasons}.",
                 "Nearest kept edges:"]
        for label, edge in (("before", result["previous_edge"]), ("after", result["next_edge"])):
            if edge is None:
                lines.append(f"  {label}: none (the cut runs to the {'start' if label == 'before' else 'end'} of the talk)")
                continue
            lines.append(f"  {label}: talk {self.text.hms(edge['timeline'])} = output {self.text.hms(edge['output'])} "
                         f"({self.source(cutmap, edge['source'])} at {self.text.hms(edge['local'])})")
        return "\n".join(lines)

    def asked(self, result, cutmap):
        if result["input_source"] is None:
            return f"talk {self.text.hms(result['timeline'])}"
        return f"{self.source(cutmap, result['input_source'])} at {self.text.hms(result['input'])}"

    def source(self, cutmap, index):
        return f"source {index} ({Path(cutmap.sources[index]['path']).name})"


def register(subparsers, common):
    parser = subparsers.add_parser(
        "map", parents=[common], help="convert a time between a render and its sources",
        description="Convert an output time to the talk timeline and source file time (--to source), or a talk "
                    "or source file time to the output time (--to output). A time inside a cut is reported with "
                    "the nearest kept edges.",
    )
    parser.add_argument("time", help="time as ss, mm:ss or hh:mm:ss, decimals allowed (90.5, 23:30, 1:02:03.5)")
    parser.add_argument("--from", dest="map_from", required=True, metavar="RENDER",
                        help="render folder, its cutmap.json, or a render folder name such as render-4 (looked up in "
                             "each talk's renders folder)")
    parser.add_argument("--talk", help="with a render folder name: the talk it belongs to")
    parser.add_argument("--to", choices=["source", "output"], default="source",
                        help="source: output time to source time (default); output: source time to output time")
    parser.add_argument("--source-index", type=int,
                        help="with --to output: the time is inside this source file (0-based); default: talk timeline")
    parser.set_defaults(handler=MapCommand().run)
