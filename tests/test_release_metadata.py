import asyncio
import json
import tomllib
from pathlib import Path

from rostrum import __version__
from rostrum.mcp_server import RostrumServer

ROOT = Path(__file__).resolve().parents[1]


class TestReleaseMetadata:
    """Every file a release publishes names the same version, so PyPI, the MCP Registry, the plugins and the
    Claude Desktop bundle never disagree. Bump them together (CONTRIBUTING.md, Releasing)."""

    def setup_method(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.version = project["version"]

    def read(self, name):
        return json.loads((ROOT / name).read_text(encoding="utf-8"))

    def test_package_version_matches_pyproject(self):
        assert __version__ == self.version

    def test_server_json_names_this_version(self):
        server = self.read("server.json")
        assert server["version"] == self.version
        assert [package["version"] for package in server["packages"]] == [self.version]
        assert len(server["description"]) <= 100

    def test_plugins_pin_this_version(self):
        for name in (".claude-plugin/plugin.json", "gemini-extension.json"):
            manifest = self.read(name)
            assert manifest["version"] == self.version, name
            assert manifest["mcpServers"]["rostrum"]["args"] == [f"rostrum@{self.version}", "mcp"], name

    def test_readme_keeps_the_registry_ownership_line(self):
        assert "mcp-name: io.github.girishkvs/rostrum" in (ROOT / "README.md").read_text(encoding="utf-8")

    def test_bundle_lists_every_mcp_tool(self):
        manifest = self.read("packaging/mcpb/manifest.json")
        server = RostrumServer()
        try:
            tools = asyncio.run(server.mcp.list_tools())
        finally:
            server.close()
        assert manifest["version"] == "__VERSION__"
        assert {tool["name"] for tool in manifest["tools"]} == {tool.name for tool in tools}
