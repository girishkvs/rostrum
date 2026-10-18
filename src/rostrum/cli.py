import argparse
import importlib
import sys

from rostrum import __version__

# Each module exposes register(subparsers, common) and sets a `handler` default
# that returns an exit code. Missing modules are skipped so features can land
# independently.
COMMAND_MODULES = [
    "rostrum.commands.init",
    "rostrum.commands.plan",
    "rostrum.commands.cuts",
    "rostrum.commands.decide",
    "rostrum.commands.render",
    "rostrum.commands.verify",
    "rostrum.commands.map",
    "rostrum.commands.config",
    "rostrum.commands.doctor",
    "rostrum.commands.status",
    "rostrum.commands.mcp",
]


class Cli:
    def __init__(self):
        self.common = argparse.ArgumentParser(add_help=False)
        self.common.add_argument("--json", action="store_true", help="print machine-readable JSON on stdout")
        self.common.add_argument("-p", "--project", default="rostrum.json", help="project file (default: ./rostrum.json)")
        self.parser = argparse.ArgumentParser(
            prog="rostrum",
            description="Clean up talk recordings: find redos, repeats and dead air, review every cut, render and verify.",
        )
        self.parser.add_argument("--version", action="version", version=f"rostrum {__version__}")
        self.subparsers = self.parser.add_subparsers(dest="command", metavar="<command>")
        self.registered = []
        for module_name in COMMAND_MODULES:
            self.register(module_name)

    def register(self, module_name):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name == module_name:
                return
            raise
        module.register(self.subparsers, self.common)
        self.registered.append(module_name.rsplit(".", 1)[-1])

    def run(self, argv):
        if (not argv and
                "status" in self.registered):
            argv = ["status"]
        args = self.parser.parse_args(argv)
        if args.command is None:
            self.parser.print_help()
            return 0
        return args.handler(args)


def main(argv=None):
    # Redirected output (status > out.txt) must not crash on non-Latin titles on Windows.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    cli = Cli()
    try:
        return cli.run(sys.argv[1:] if argv is None else argv)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
