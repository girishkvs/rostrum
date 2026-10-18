# Contributing

Thanks for taking a look. rostrum is small on purpose: a CLI, a few detectors, ffmpeg for media,
and a static review page.

## Getting set up

```bash
git clone https://github.com/girishkvs/rostrum
cd rostrum
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"     # Windows: .venv\Scripts\python.exe
.venv/bin/python -m pytest -q
```

Python 3.11 or newer. Tests that need ffmpeg and ffprobe skip cleanly when they are not found;
install ffmpeg (or set `ROSTRUM_FFMPEG` / `ROSTRUM_FFPROBE`) to run them.

The speech engine and the MCP server install with rostrum. Unit tests never download a model.

## Ground rules

- **No media, transcripts or real recordings in the repo.** Tests generate their media at run time
  with ffmpeg and use hand-written word lists. CI rejects committed media files.
- **No private package mirrors.** Don't commit pip config, lockfiles or URLs that point at an
  internal index. CI rejects them.
- **Everything is configurable.** A new threshold or word list is a config key with a default and
  a schema entry, not a constant.
- **Never cut silently.** New detectors produce candidates with a reason and evidence; the user (or
  their agent) decides.

## Code style

- Readability first.
- Small classes with instance methods; no `@staticmethod` helpers.
- Multi-condition `if`: one condition per line, operator at the end of the line:

  ```python
  if (cut.decision == "cut" and
          cut.length > min_length):
  ```

- Comments only where the code needs them.

## Pull requests

Keep each PR focused, include tests, and run `pytest -q` before pushing. By contributing you agree
that your contribution is licensed under the MIT License.

## Releasing

1. Bump the version in `pyproject.toml`, `src/rostrum/__init__.py`, `server.json` (both `version`
   fields), and `.claude-plugin/plugin.json` and `gemini-extension.json` (`version` and the
   `rostrum@X.Y.Z` pin). `tests/test_release_metadata.py` fails until they all agree.
2. Add the CHANGELOG entry, then commit and push to `main`.
3. Once CI is green, publish the GitHub release `vX.Y.Z` from your own account. A release made with a
   workflow token doesn't start other workflows.
4. `.github/workflows/release.yml` builds once, then:
   - publishes to PyPI through trusted publishing;
   - attaches `rostrum-X.Y.Z.mcpb` and `SHA256SUMS.txt` to the release;
   - publishes `server.json` to the MCP Registry.
