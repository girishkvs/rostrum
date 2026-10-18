"""rostrum doctor: checks Python, ffmpeg, speech engines, the model cache, disk and memory."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rostrum.config import ConfigError, ConfigLoader
from rostrum.engines import EngineRegistry
from rostrum.engines.base import WorkerCount
from rostrum.output import Output
from rostrum.pipeline import Workspace
from rostrum.project import EXIT_FAILED, EXIT_INPUT, EXIT_OK
from rostrum.tools import ToolNotFound, Tools

MIN_PYTHON = (3, 11)
MIN_FREE_DISK_GB = 5.0
MEMORY_PER_WORKER_GB = 2.0
GB = 1024 ** 3


class Check:
    def __init__(self, name, status, detail, hint=""):
        self.name = name
        self.status = status
        self.detail = detail
        self.hint = hint

    def to_dict(self):
        data = {"name": self.name, "status": self.status, "detail": self.detail}
        if self.hint:
            data["hint"] = self.hint
        return data


class ModelCache:
    """Where faster-whisper models land: the Hugging Face hub cache."""

    def __init__(self, model):
        self.model = model

    def repo(self):
        if "/" in self.model:
            return self.model
        try:
            from faster_whisper.utils import _MODELS

            return _MODELS.get(self.model, f"Systran/faster-whisper-{self.model}")
        except ImportError:
            return f"Systran/faster-whisper-{self.model}"

    def hub(self):
        if os.environ.get("HF_HUB_CACHE"):
            return Path(os.environ["HF_HUB_CACHE"])
        if os.environ.get("HF_HOME"):
            return Path(os.environ["HF_HOME"]) / "hub"
        return Path.home() / ".cache" / "huggingface" / "hub"

    def find(self):
        local = Path(self.model).expanduser()
        if (local / "model.bin").is_file():
            return local
        folder = self.hub() / ("models--" + self.repo().replace("/", "--")) / "snapshots"
        if not folder.is_dir():
            return None
        for snapshot in sorted(folder.iterdir()):
            if (snapshot / "model.bin").is_file():
                return snapshot
        return None


class Doctor:
    def __init__(self, settings, work_dir):
        self.settings = settings
        self.work_dir = Path(work_dir)
        self.tools = Tools(settings["tools"])

    def checks(self):
        found = [self.python()]
        found.extend(self.ffmpeg())
        found.extend(self.engines())
        found.append(self.model())
        found.append(self.disk())
        found.append(self.memory())
        return [check for check in found if check is not None]

    def python(self):
        version = ".".join(str(part) for part in sys.version_info[:3])
        if sys.version_info[:2] < MIN_PYTHON:
            return Check("python", "fail", version, "rostrum needs Python 3.11 or newer")
        return Check("python", "ok", version)

    def run(self, command):
        return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")

    def ffmpeg(self):
        checks = []
        for name in ("ffmpeg", "ffprobe"):
            try:
                path = self.tools.find(name)
            except ToolNotFound as error:
                checks.append(Check(name, "fail", str(error)))
                continue
            result = self.run([path, "-hide_banner", "-version"])
            first = (result.stdout.splitlines() or ["unknown version"])[0]
            checks.append(Check(name, "ok" if result.returncode == 0 else "fail", f"{first} ({path})"))
            if name == "ffmpeg":
                checks.append(self.encoders(path))
        return checks

    def encoders(self, path):
        result = self.run([path, "-hide_banner", "-encoders"])
        names = {line.split()[1] for line in result.stdout.splitlines() if len(line.split()) > 1}
        missing = [name for name in ("libx264", "aac") if name not in names]
        if missing:
            return Check("encoders", "fail", "missing: " + ", ".join(missing),
                         "use an ffmpeg build with libx264 (for example the gyan.dev or BtbN builds)")
        return Check("encoders", "ok", "libx264, aac")

    def engines(self):
        registry = EngineRegistry()
        configured = self.settings["speech"]["engine"]
        checks = []
        names = registry.names()
        if configured not in names:
            checks.append(Check("engine", "fail", f"speech.engine {configured!r} is unknown; available: {', '.join(names)}"))
        for name in names:
            try:
                result = registry.create(name, self.settings, self.tools).check()
            except Exception as error:
                result = {"ok": False, "detail": f"cannot load: {error}"}
            in_use = name == configured
            status = "ok" if result["ok"] else ("fail" if in_use else "warn")
            label = f"engine {name}" + (" (configured)" if in_use else "")
            checks.append(Check(label, status, result["detail"]))
        return checks

    def model(self):
        speech = self.settings["speech"]
        if speech["engine"] != "faster-whisper":
            return None
        found = ModelCache(speech["model"]).find()
        if found is None:
            return Check("model", "warn", f"{speech['model']} is not downloaded yet",
                         "the first transcription downloads it from Hugging Face (small.en is about 480 MB)")
        return Check("model", "ok", f"{speech['model']} at {found}")

    def disk(self):
        folder = self.work_dir
        while not folder.exists():
            folder = folder.parent
        free = shutil.disk_usage(folder).free / GB
        status = "ok" if free >= MIN_FREE_DISK_GB else "warn"
        return Check("disk", status, f"{free:.1f} GB free at {folder}",
                     "" if status == "ok" else "renders and work files need a few GB per talk")

    def memory(self):
        try:
            import psutil
        except ImportError:
            return Check("memory", "warn", "unknown (psutil is not installed)")
        memory = psutil.virtual_memory()
        setting = self.settings["speech"]["workers"]
        workers = WorkerCount().resolve(setting)
        available = memory.available / GB
        detail = (f"{available:.1f} GB available of {memory.total / GB:.1f} GB; "
                  f"{WorkerCount().describe(setting, self.settings['speech']['threadsPerWorker'])} "
                  f"need about {workers * MEMORY_PER_WORKER_GB:.0f} GB")
        if available < workers * MEMORY_PER_WORKER_GB:
            fit = max(1, int(available // MEMORY_PER_WORKER_GB))
            return Check("memory", "warn", detail, f"set speech.workers to {fit}")
        return Check("memory", "ok", detail)


class DoctorCommand:
    def run(self, args):
        output = Output(args.json)
        try:
            if Path(args.project).is_file():
                workspace = Workspace(args.project)
                talks = workspace.talks()
                settings = workspace.config(talks[0]).values if talks else ConfigLoader().effective(workspace.project).values
                work_dir = workspace.project.dir
            else:
                settings = ConfigLoader().effective().values
                work_dir = Path.cwd()
        except ConfigError as error:
            return output.fail(str(error), EXIT_INPUT)
        checks = Doctor(settings, work_dir).checks()
        failed = [check for check in checks if check.status == "fail"]
        data = {"ok": not failed, "checks": [check.to_dict() for check in checks]}
        output.result(data, self.text(checks))
        return EXIT_FAILED if failed else EXIT_OK

    def text(self, checks):
        marks = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}
        lines = []
        for check in checks:
            lines.append(f"[{marks[check.status]}] {check.name}: {check.detail}")
            if check.hint:
                lines.append(f"       {check.hint}")
        return "\n".join(lines)


def register(subparsers, common):
    parser = subparsers.add_parser("doctor", parents=[common], help="check the setup",
                                   description="Checks Python, ffmpeg and ffprobe (with the libx264 encoder), the "
                                               "speech engines, the model cache, free disk and memory.")
    parser.set_defaults(handler=DoctorCommand().run)
