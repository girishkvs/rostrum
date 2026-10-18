"""rostrum cuts: the candidate cuts with their current decisions (use --json for agents)."""
from dataclasses import asdict

from rostrum.config import ConfigError
from rostrum.output import Output
from rostrum.pipeline import Clock, Workspace
from rostrum.project import EXIT_FAILED, EXIT_INPUT, EXIT_OK


class CutFilter:
    def __init__(self, args):
        self.types = set(args.type or [])
        self.level = args.level
        self.decision = args.decision

    def keeps(self, cut):
        if (self.types and
                cut.type not in self.types):
            return False
        if (self.level and
                cut.level != self.level):
            return False
        return not (self.decision and cut.decision != self.decision)


class CutsCommand:
    def run(self, args):
        output = Output(args.json)
        try:
            workspace = Workspace(args.project)
            talks = workspace.talks(args.talk)
        except ConfigError as error:
            return output.fail(str(error), EXIT_INPUT)
        chosen = CutFilter(args)
        listed = []
        for talk in talks:
            try:
                _, result, _ = workspace.state(talk)
            except ConfigError as error:
                listed.append({"id": talk.id, "ok": False, "error": str(error)})
                continue
            cuts = [asdict(cut) for cut in result.cuts if chosen.keeps(cut)]
            listed.append({"id": talk.id, "ok": True, "kept": round(result.edl.kept, 3), "stale": result.stale, "cuts": cuts})
        ok = all(item["ok"] for item in listed)
        data = {"ok": ok, "talks": listed}
        output.result(data, self.text(data))
        return EXIT_OK if ok else EXIT_FAILED

    def text(self, data):
        clock = Clock()
        lines = []
        for talk in data["talks"]:
            if not talk["ok"]:
                lines.append(f"{talk['id']}: {talk['error']}")
                continue
            lines.append(f"{talk['id']}: {len(talk['cuts'])} cuts")
            for cut in talk["cuts"]:
                lines.append(f"  {cut['id']:<18} {cut['decision']:<4} {cut['type']:<9} "
                             f"{clock.text(cut['start'])}-{clock.text(cut['end'])}  {cut['reason'][:70]}")
            for entry in talk["stale"]:
                lines.append(f"  STALE {entry['id']} ({entry['decision']})")
        return "\n".join(lines)


def register(subparsers, common):
    parser = subparsers.add_parser("cuts", parents=[common], help="list candidate cuts and their decisions",
                                   description="Lists candidate cuts with id, range, type, level, decision, reason, "
                                               "evidence and confidence. Use --json for agents.")
    parser.add_argument("--talk", help="only this talk")
    parser.add_argument("--type", action="append", help="only this cut type (repeatable)")
    parser.add_argument("--level", choices=["auto", "propose"], help="only this level")
    parser.add_argument("--decision", choices=["cut", "keep"], help="only this decision")
    parser.set_defaults(handler=CutsCommand().run)
