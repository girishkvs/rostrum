# Security Policy

## Supported versions

Fixes land on the latest released minor version only.

| Version | Supported |
| ------- | --------- |
| Latest released 0.x minor | yes |
| Earlier releases | no |

## Reporting a vulnerability

Please report security issues privately through
[GitHub Security Advisories](https://github.com/girishkvs/rostrum/security/advisories/new).
Do not open a public issue for a vulnerability.

Include the version, your platform, what an attacker can achieve, and steps to reproduce.
You can expect an acknowledgement within 7 days and an assessment within 30.

## Threat model

rostrum runs locally on your own recordings. It uploads nothing and collects no telemetry.
A few properties still matter:

- **A project file can run programs.** The `command` speech engine and the `tools.ffmpeg` /
  `tools.ffprobe` paths in `rostrum.json` (or `~/.rostrum/config.json`) name programs that rostrum
  executes, and `render.extraOutputArgs` passes arguments straight to ffmpeg. Treat a project
  file from someone else like a script: read it before you run it. On Windows, a `.bat` or `.cmd`
  program, or `cmd` itself, never gets an argument containing `& ^ % ! < > |` or a quote, so a file
  name can't add a command.
- **Agents follow the project file too.** If an agent runs `analyze` (`rostrum init`) in a folder
  you downloaded, any `rostrum.json` there is used as-is, including the settings above. Read
  project files from others before you let an agent work in that folder.
- **Cloud speech engines send audio off your machine.** The built-in engines are local. If you
  configure a command or adapter that calls a hosted service, that service receives your audio.
- **The review page is a local file.** `review.html` embeds your transcript excerpts and points at
  your source videos. It loads no external resources, but anyone you send it to sees that text.
- **The MCP server speaks stdio only.** `rostrum mcp` is started by your agent host and does not
  open a network port. Its tools can read and write files in your project's work folder and render
  videos, so only wire it into hosts you trust.
- **Sources are never modified and outputs are never overwritten.** Renders go to new files and
  folders; decisions and config writes keep a `.bak` of the previous version.
