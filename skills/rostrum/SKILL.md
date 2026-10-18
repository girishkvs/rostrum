---
name: rostrum
description: Clean up talk, course or demo recordings with rostrum. Use when the user wants to remove redos, retakes, repeated sentences, long pauses or dead air from one or more videos, review proposed cuts, render a cleaned video, check a render, or map a time in an edited video back to the source.
---

# rostrum

rostrum finds redos ("redo", "stop", "restart"), retakes, repeats and dead air in recordings,
lets the user review every proposed cut, then renders with ffmpeg and verifies the result.
Everything runs locally.

## Prefer the MCP tools

If the `rostrum` MCP server is available (`rostrum mcp`), use its tools. Otherwise run the CLI with
`--json` and read stdout; progress goes to stderr.

| Step | MCP tool | CLI |
|---|---|---|
| Analyze files or a folder | `analyze` → poll `job_status` | `rostrum init <files or folder> --json` |
| Project summary | `status` | `rostrum status --json` |
| Show proposed cuts | `plan`, `list_cuts` | `rostrum plan --json`, `rostrum cuts --json` |
| Record a decision | `decide` | `rostrum decide <id> keep|cut [--start S] [--end E] --json` |
| Apply decisions saved from review.html | `import_decisions` | `rostrum decide --import <talk>-decisions.json --json` |
| Render (only after the user says so) | `render` with `confirmed: true` → poll `job_status` | `rostrum render --yes --json` |
| Verify a render | `verify` → poll `job_status` | `rostrum verify --mode fast|full --json` (newest render) |
| Map a time | `map_time` | `rostrum map <time> --from <render folder or name, e.g. render-4> --json` |
| Diagnose setup | `doctor` | `rostrum doctor --json` |

## Rules

1. **Show before you cut.** After analysis, show the user the plan: each cut's time, length,
   reason and evidence, plus the expected output length.
2. **Ask about uncertain cuts.** Cuts with level `propose` default to keep. Ask the user about
   them, and about any cut with low confidence, before rendering.
3. **Never render without a go-ahead.** rostrum refuses to render a talk nobody has reviewed.
   Pass `confirmed: true` (MCP) or `--yes` (CLI) only after the user explicitly says to render.
4. **Only record what the user decided.** Use `decide` for the user's answers. Do not flip other
   cuts on your own. When the user saved decisions from review.html, apply that file with
   `import_decisions`. If the result says `"reviewed": false`, the page was built before the cuts
   last changed: ask the user to review the refreshed page.
5. **Report progress on long jobs.** Analysis and rendering take minutes; relay `job_status`
   progress instead of waiting silently.
6. **Use source time for corrections.** When the user quotes a time from an edited video
   ("23:30 in the last render"), convert it with `map_time` before acting on it.
7. **Never touch the sources.** rostrum never modifies them and never overwrites outputs; do not
   work around that.

## Setup problems

Run `doctor` first. The usual fixes:
- install ffmpeg (`winget install Gyan.FFmpeg`, `brew install ffmpeg`, `apt install ffmpeg`);
- reinstall rostrum if the speech engine or MCP package is missing;
- set `ROSTRUM_FFMPEG` / `ROSTRUM_FFPROBE`, or `defaults.tools.ffmpeg` / `defaults.tools.ffprobe` in
  `rostrum.json` (a top-level `tools` key there is rejected), or `tools` in `~/.rostrum/config.json`.
