# AGENTS.md

Guidance for coding agents working **on** this repository. For agents **using** rostrum to edit
recordings, see [`skills/rostrum/SKILL.md`](skills/rostrum/SKILL.md).

## Build and test

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

ffmpeg-dependent tests skip when ffmpeg/ffprobe are missing. Set `ROSTRUM_FFMPEG` and
`ROSTRUM_FFPROBE` to point at specific binaries.

## Layout

| Path | What |
|---|---|
| `src/rostrum/cli.py` | Command registry; each `commands/<name>.py` exposes `register(subparsers, common)` |
| `src/rostrum/config.py` + `src/rostrum/schema/rostrum.schema.json` | Layered JSON config and its schema |
| `src/rostrum/engines/` | Speech-to-text adapters (`faster-whisper`, `command`, `import`) |
| `src/rostrum/detect/`, `reconcile.py` | Cut detectors and evidence merging |
| `src/rostrum/edl.py` | Timeline, cut and EDL model shared by every stage |
| `src/rostrum/render.py`, `verify.py`, `cutmap.py`, `media.py` | ffmpeg render, checks, time mapping, probing |
| `src/rostrum/review.py` + `review_template.html` | Static review page |

## Rules

- Never commit media, transcripts or real recordings. Use synthetic fixtures.
- Every new constant becomes a config key with a schema entry and a default.
- Detectors propose; they never cut without a reason and evidence on the `Cut`.
- Keep results on stdout (JSON with `--json`) and progress on stderr.
- Follow the code style in [CONTRIBUTING.md](CONTRIBUTING.md).
