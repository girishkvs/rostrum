"""rostrum plan: every cut, its reason and the expected output length. A dry run; it never renders."""
from dataclasses import asdict

from rostrum.config import ConfigError
from rostrum.output import Output
from rostrum.pipeline import Clock, CutSettings, SourceDrift, Workspace
from rostrum.project import EXIT_FAILED, EXIT_INPUT, EXIT_OK

EVIDENCE_WIDTH = 60


class TalkPlan:
    def of(self, talk, result):
        edl = result.edl
        totals = {}
        for cut in result.cuts:
            if cut.decision != "cut":
                continue
            entry = totals.setdefault(cut.type, {"count": 0, "seconds": 0.0})
            entry["count"] += 1
            entry["seconds"] += cut.end - cut.start
        for entry in totals.values():
            entry["seconds"] = round(entry["seconds"], 3)
        return {
            "id": talk.id,
            "ok": True,
            "duration": round(edl.duration, 3),
            "kept": round(edl.kept, 3),
            "removed": round(edl.duration - edl.kept, 3),
            "cutCount": sum(1 for cut in result.cuts if cut.decision == "cut"),
            "proposed": sum(1 for cut in result.cuts if cut.level == "propose" and cut.decision == "keep"),
            "userDecisions": sum(1 for cut in result.cuts if cut.decided_by == "user"),
            "totals": totals,
            "stale": result.stale,
            "cuts": [self.row(cut) for cut in result.cuts],
        }

    def row(self, cut):
        data = asdict(cut)
        data["length"] = round(cut.end - cut.start, 3)
        data["short"] = self.short(cut.evidence)
        return data

    def short(self, evidence):
        text = " ".join(evidence.split())
        return text if len(text) <= EVIDENCE_WIDTH else text[:EVIDENCE_WIDTH - 1] + "\u2026"


class PlanText:
    def __init__(self):
        self.clock = Clock()

    def talk(self, plan):
        if not plan["ok"]:
            return [f"{plan['id']}: {plan['error']}"]
        clock = self.clock
        lines = [f"{plan['id']}: {clock.text(plan['duration'])} -> {clock.text(plan['kept'])} expected output "
                 f"({clock.text(plan['removed'])} removed in {plan['cutCount']} cuts; {plan['proposed']} proposed, kept)",
                 f"  {'ID':<18} {'TYPE':<9} {'LEVEL':<8} {'DECISION':<8} {'RANGE':<23} {'LENGTH':>7}  EVIDENCE"]
        for cut in plan["cuts"]:
            marker = "*" if cut["decided_by"] == "user" else " "
            span = f"{clock.text(cut['start'])}-{clock.text(cut['end'])}"
            lines.append(f" {marker}{cut['id']:<18} {cut['type']:<9} {cut['level']:<8} {cut['decision']:<8} {span:<23} "
                         f"{cut['length']:>6.1f}s  {cut['short']}")
        totals = ", ".join(f"{name} {entry['count']} ({clock.text(entry['seconds'])})"
                           for name, entry in sorted(plan["totals"].items(), key=lambda item: -item[1]["seconds"]))
        lines.append(f"  totals: {totals or 'no cuts'}")
        if plan["userDecisions"]:
            lines.append(f"  * = your decision ({plan['userDecisions']})")
        for entry in plan["stale"]:
            lines.append(f"  STALE decision {entry['id']} ({entry['decision']}): no cut has this id after the last run; "
                         "it is not applied. Decide again on the current cuts.")
        lines += [f"  warning: {note}" for note in plan.get("notes", [])]
        return lines

    def all(self, data):
        lines = []
        for plan in data["talks"]:
            lines.extend(self.talk(plan))
            lines.append("")
        lines.append("Nothing was rendered. Change a cut with: rostrum decide <id> keep|cut [--start S] [--end E]")
        return "\n".join(lines)


class PlanCommand:
    def run(self, args):
        output = Output(args.json)
        try:
            workspace = Workspace(args.project)
            talks = workspace.talks(args.talk)
        except ConfigError as error:
            return output.fail(str(error), EXIT_INPUT)
        plans = []
        for talk in talks:
            try:
                candidates, result, _ = workspace.state(talk)
            except ConfigError as error:
                plans.append({"id": talk.id, "ok": False, "error": str(error)})
                continue
            plan = TalkPlan().of(talk, result)
            plan["notes"] = self.notes(workspace, talk, candidates)
            plans.append(plan)
        ok = all(plan["ok"] for plan in plans)
        data = {"ok": ok, "project": str(workspace.project.path), "talks": plans}
        output.result(data, PlanText().all(data))
        return EXIT_OK if ok else EXIT_FAILED

    def notes(self, workspace, talk, candidates):
        """Why this plan may not be what the next render or init gives."""
        notes = []
        if CutSettings().changed(workspace.config(talk).values, candidates):
            notes.append("cut settings changed since init; run rostrum init to apply them")
        drift = SourceDrift().problem(workspace.project, talk, candidates)
        if drift:
            notes.append(f"{drift}; run rostrum init (render refuses until then)")
        return notes


def register(subparsers, common):
    parser = subparsers.add_parser(
        "plan", parents=[common], help="list every cut and the expected output length (dry run)",
        description="Lists every cut with its id, type, level, decision, range and evidence, plus totals and the "
                    "expected output length. Never renders.")
    parser.add_argument("--talk", help="only this talk")
    parser.set_defaults(handler=PlanCommand().run)
