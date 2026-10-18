# Changelog

## 0.1.0 — 2026-10-18

First public version.

- One talk from one or many files, or a whole event of talks, configured in plain JSON.
- Pluggable speech-to-text: `faster-whisper` (local, default), `command` (any program that prints
  words, SRT or VTT) and `import` (an existing transcript).
- Detectors for spoken cues, retakes, repeats, pauses, lead-in/out, off-talk and fillers, each with
  its own level (`auto`, `propose`, `off`) and thresholds.
- A sentence said again with one word swapped ("before" → "after") is only proposed, since it may
  be on purpose (`retakes.substitutionLevel`).
- Unclear stretches (an overlong word, low-confidence words, a doubtful or hallucinated segment,
  speech with no words) are proposed for a listen, and matching never joins words across them.
- Words in any script are matched; the default model and word lists are English.
- Static review page with join preview, keep/cut toggles and edge nudges. It saves
  `<talk>-decisions.json` for `decide --import`, which keeps the newer decision for each cut.
  `decide` and `decide --import` keep the page current.
- `render` waits until the cuts were reviewed since they last changed (`decide`, or importing a
  save from the current page); `--yes` renders them as they are. It refuses when a source
  changed after `init`.
- ffmpeg render with frame-accurate joins and short audio fades; optional EBU R128 loudness.
- `verify` (fast or full): duration, frame and audio continuity, full decode, A/V offset at joins.
- `map` converts times between any render (by folder or name, such as `render-4`) and the source.
- `--json` on every command except `mcp`, and an MCP server for agents. The speech engine and the MCP server are
  in the default install.
- Install from PyPI (`uv tool install rostrum`, or `uvx rostrum mcp` for agents), as a plugin for Claude Code
  and GitHub Copilot CLI, as a Gemini CLI extension, or as a Claude Desktop bundle (`.mcpb`). Listed in the
  official MCP Registry as `io.github.girishkvs/rostrum`.
