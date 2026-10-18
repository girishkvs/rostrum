"""rostrum config: the effective settings for each talk and where each value came from."""
import json
from pathlib import Path

from rostrum.config import ConfigError, ConfigLoader
from rostrum.output import Output
from rostrum.pipeline import Workspace
from rostrum.project import EXIT_INPUT, EXIT_OK


class ConfigCommand:
    def run(self, args):
        output = Output(args.json)
        loader = ConfigLoader()
        project = None
        try:
            if Path(args.project).is_file():
                workspace = Workspace(args.project, loader=loader)
                project = str(workspace.project.path)
                entries = [(talk.id, workspace.config(talk)) for talk in workspace.talks(args.talk)]
            else:
                entries = [(None, loader.effective())]
        except ConfigError as error:
            return output.fail(str(error), EXIT_INPUT)
        talks = []
        for talk_id, config in entries:
            leaves = [(key, value, origin) for key, value, origin in config.leaves() if self.wanted(key, args.key)]
            talks.append({"id": talk_id, "settings": self.subset(config.values, args.key),
                          "origins": {key: origin for key, _, origin in leaves},
                          "lines": leaves})
        data = {"ok": True, "project": project, "userConfig": str(loader.user_path()),
                "talks": [{key: value for key, value in talk.items() if key != "lines"} for talk in talks]}
        output.result(data, self.text(talks, project, loader))
        return EXIT_OK

    def wanted(self, key, prefix):
        if not prefix:
            return True
        return (key == prefix or
                key.startswith(prefix + "."))

    def subset(self, values, prefix):
        if not prefix:
            return values
        node = values
        for part in prefix.split("."):
            if (not isinstance(node, dict) or
                    part not in node):
                return None
            node = node[part]
        return node

    def text(self, talks, project, loader):
        lines = [f"project: {project or '(none)'}", f"user config: {loader.user_path()}",
                 "layers: default < user < project < talk < flag", ""]
        for talk in talks:
            lines.append(f"talk {talk['id']}" if talk["id"] else "settings (no project)")
            for key, value, origin in talk["lines"]:
                lines.append(f"  {key} = {json.dumps(value, ensure_ascii=False)}  ({origin})")
            lines.append("")
        return "\n".join(lines).rstrip()


def register(subparsers, common):
    parser = subparsers.add_parser(
        "config", parents=[common], help="show the effective settings and where each value came from",
        description="Shows the effective settings per talk. Each value is marked with the layer that set it: default, "
                    "user (~/.rostrum/config.json), project (defaults), talk, or flag.")
    parser.add_argument("--talk", help="only this talk")
    parser.add_argument("--key", help="only this key or group, for example pauses or speech.engine")
    parser.set_defaults(handler=ConfigCommand().run)
