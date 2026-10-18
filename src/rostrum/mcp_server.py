import json
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import psutil
from mcp.server.mcpserver import MCPServer

from rostrum import __version__

INSTRUCTIONS = """rostrum cleans up talk recordings: it finds redos, retakes, repeats and dead air,
lets the user review every cut, then renders and verifies the result.

Rules for agents:
- Run `analyze` first; it is long-running and returns a job id. Poll `job_status` until done.
- Show the user the proposed cuts (`plan` or `list_cuts`) before rendering, and ask about every
  cut whose level is "propose" or whose confidence is low.
- Never call `render` until the user has said to render; then pass `confirmed=true`. Without it,
  rostrum refuses to render a talk whose cuts nobody has reviewed.
- Record the user's answers with `decide`; never change decisions the user did not ask for.
- When the user saved decisions from review.html, apply that file with `import_decisions`.
- `render` and `verify` are long-running jobs; report progress from `job_status`.
- Use `map_time` when the user quotes a time from a rendered video."""


class CliRunner:
    """Every tool runs the same CLI the user runs, with --json, so behaviour never diverges."""

    def __init__(self, executable=None):
        self.executable = executable or sys.executable

    def command(self, args):
        return [self.executable, "-m", "rostrum", *args, "--json"]

    def parse(self, exit_code, stdout, stderr=""):
        text = stdout.strip()
        try:
            data = json.loads(text) if text else {}
        except json.JSONDecodeError:
            data = {"output": text[-4000:]}
        failed = exit_code != 0
        if (failed and
                isinstance(data, dict)):
            data.setdefault("ok", False)
            if stderr.strip():
                data.setdefault("stderr", stderr.strip()[-2000:])
        return {"exitCode": exit_code, "result": data}

    def run(self, args, cwd, timeout=300):
        completed = subprocess.run(
            self.command(args),
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return self.parse(completed.returncode, completed.stdout, completed.stderr)


class ProcessTree:
    """A job's CLI process and everything it started (ffmpeg, speech workers)."""

    def stop(self, pid, timeout=5):
        try:
            root = psutil.Process(pid)
            # Children are listed before anything stops: once the CLI exits, they can't be found from it.
            family = root.children(recursive=True) + [root]
        except psutil.NoSuchProcess:
            return
        for process in family:
            self.send(process.terminate)
        _, alive = psutil.wait_procs(family, timeout=timeout)
        for process in alive:
            self.send(process.kill)

    def send(self, signal):
        try:
            signal()
        except psutil.NoSuchProcess:
            pass


class Job:
    def __init__(self, job_id, command, cwd, log_dir, runner):
        self.id = job_id
        self.command = command
        self.cwd = cwd
        self.log_path = Path(log_dir) / f"{job_id}.log"
        self.runner = runner
        self.stdout = ""
        self.exit_code = None
        self.started_at = time.time()
        self.finished_at = None

    def start(self):
        self.log_file = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            stdout=subprocess.PIPE,
            stderr=self.log_file,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.collector = threading.Thread(target=self.collect, daemon=True)
        self.collector.start()
        return self

    def collect(self):
        self.stdout = self.process.stdout.read()
        self.exit_code = self.process.wait()
        self.finished_at = time.time()
        self.log_file.close()

    def stop(self):
        if self.exit_code is not None:
            return
        ProcessTree().stop(self.process.pid)
        # The collector closes the log file; wait for it so the log folder can be removed.
        self.collector.join(timeout=5)

    def progress(self, lines=15):
        if not self.log_path.exists():
            return []
        return self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]

    def snapshot(self):
        running = self.exit_code is None
        end = self.finished_at or time.time()
        state = {
            "job": self.id,
            "state": "running" if running else ("done" if self.exit_code == 0 else "failed"),
            "elapsedSec": round(end - self.started_at, 1),
            "progress": self.progress(),
        }
        if not running:
            state.update(self.runner.parse(self.exit_code, self.stdout))
        return state


class RostrumServer:
    def __init__(self, runner=None):
        self.runner = runner or CliRunner()
        self.jobs = {}
        self.temp = tempfile.TemporaryDirectory(prefix="rostrum-mcp-", ignore_cleanup_errors=True)
        self.log_dir = Path(self.temp.name)
        self.mcp = MCPServer("rostrum", instructions=INSTRUCTIONS, version=__version__)
        for tool in (
            self.analyze,
            self.job_status,
            self.status,
            self.plan,
            self.list_cuts,
            self.decide,
            self.import_decisions,
            self.render,
            self.verify,
            self.map_time,
            self.doctor,
        ):
            self.mcp.add_tool(tool)

    def serve(self):
        try:
            self.mcp.run("stdio")
        finally:
            self.close()

    def close(self):
        """Stops running jobs and removes their logs; nothing is left behind in the temp folder."""
        for job in self.jobs.values():
            job.stop()
        self.temp.cleanup()

    def start_job(self, args, cwd):
        job = Job(uuid.uuid4().hex[:8], self.runner.command(args), cwd, self.log_dir, self.runner).start()
        self.jobs[job.id] = job
        return {"job": job.id, "state": "running", "next": "Poll job_status with this job id."}

    def project_args(self, project):
        return ["-p", project] if project else []

    def analyze(self, paths: list[str], project_dir: str, engine: str | None = None) -> dict:
        """Transcribe recordings and find proposed cuts. `paths` are video files or folders. Files named like
        talk-part1, talk-part2 are joined into one talk; every other file is a talk of its own. Writes rostrum.json in
        project_dir. Long-running: returns a job id."""
        args = ["init", *paths]
        if engine:
            args += ["--engine", engine]
        return self.start_job(args, project_dir)

    def job_status(self, job_id: str) -> dict:
        """Progress and, when finished, the result of a long-running job."""
        job = self.jobs.get(job_id)
        if job is None:
            return {"ok": False, "error": f"unknown job {job_id}", "jobs": sorted(self.jobs)}
        return job.snapshot()

    def status(self, project_dir: str, project: str | None = None) -> dict:
        """Summary of every talk in the project: stage, cut counts, expected output length."""
        return self.runner.run(["status", *self.project_args(project)], project_dir)

    def plan(self, project_dir: str, project: str | None = None) -> dict:
        """Every cut per talk with reason, level, decision and the expected output length. Never renders."""
        return self.runner.run(["plan", *self.project_args(project)], project_dir)

    def list_cuts(self, project_dir: str, talk: str | None = None, project: str | None = None) -> dict:
        """All cut candidates with evidence, for one talk or all talks."""
        args = ["cuts", *self.project_args(project)]
        if talk:
            args += ["--talk", talk]
        return self.runner.run(args, project_dir)

    def decide(self, project_dir: str, cut_id: str, decision: str, talk: str | None = None,
               start: float | None = None, end: float | None = None, project: str | None = None) -> dict:
        """Record the user's decision for one cut: decision is "keep" or "cut"; start/end adjust its edges."""
        args = ["decide", cut_id, decision, *self.project_args(project)]
        if talk:
            args += ["--talk", talk]
        if start is not None:
            args += ["--start", str(start)]
        if end is not None:
            args += ["--end", str(end)]
        return self.runner.run(args, project_dir)

    def import_decisions(self, project_dir: str, path: str, talk: str | None = None,
                         project: str | None = None) -> dict:
        """Apply a decisions file saved from review.html. For each cut the newer decision wins, so decisions
        made after the page was built are kept. The result says whether it counts as a review."""
        args = ["decide", "--import", path, *self.project_args(project)]
        if talk:
            args += ["--talk", talk]
        return self.runner.run(args, project_dir)

    def render(self, project_dir: str, talk: str | None = None, confirmed: bool = False,
               project: str | None = None) -> dict:
        """Render the reviewed cut list with ffmpeg. Long-running: returns a job id.
        Set confirmed=true only after the user explicitly said to render; without it, rostrum refuses
        to render a talk whose cuts nobody has reviewed yet."""
        args = ["render", *self.project_args(project)]
        if talk:
            args += ["--talk", talk]
        if confirmed:
            args.append("--yes")
        return self.start_job(args, project_dir)

    def verify(self, project_dir: str, render_dir: str | None = None, talk: str | None = None,
               mode: str = "fast", project: str | None = None) -> dict:
        """Check a render: duration, frames, audio continuity, decode, A/V sync at joins.
        Without render_dir, checks the newest render of the talk. mode is "fast" or "full".
        Long-running: returns a job id."""
        args = ["verify", "--mode", mode, *self.project_args(project)]
        if render_dir:
            args += ["--render", render_dir]
        if talk:
            args += ["--talk", talk]
        return self.start_job(args, project_dir)

    def map_time(self, project_dir: str, time: str, render_dir: str, to: str = "source", talk: str | None = None,
                 project: str | None = None) -> dict:
        """Convert a time quoted from a rendered video (e.g. "23:30") to source time, or back with to="output".
        render_dir is a render folder, its cutmap.json, or a render name such as render-4. With a name, pass
        talk when more than one talk has a render of that name."""
        args = ["map", time, "--from", render_dir, "--to", to, *self.project_args(project)]
        if talk:
            args += ["--talk", talk]
        return self.runner.run(args, project_dir)

    def doctor(self, project_dir: str = ".") -> dict:
        """Check ffmpeg, the speech engine, model cache, disk and memory."""
        return self.runner.run(["doctor"], project_dir)
