import sys


class McpCommand:
    def run(self, args):
        try:
            from rostrum.mcp_server import RostrumServer
        except ImportError:
            print("The MCP server package is missing from this install. Reinstall rostrum, for example:\n"
                  "  uv tool install --reinstall rostrum",
                  file=sys.stderr)
            return 2
        RostrumServer().serve()
        return 0


def register(subparsers, common):
    parser = subparsers.add_parser(
        "mcp",
        help="run the MCP server over stdio for agents (Copilot CLI, Claude Code, Codex, Gemini)",
        description="Run the rostrum MCP server over stdio. Your agent host starts this; it opens no network port.",
    )
    parser.set_defaults(handler=McpCommand().run)
