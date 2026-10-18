import os
import shutil
from pathlib import Path

INSTALL_HINT = "Install ffmpeg (winget install Gyan.FFmpeg, brew install ffmpeg, apt install ffmpeg) or set tools.{name} in your config."


class ToolNotFound(RuntimeError):
    pass


class Tools:
    """Finds ffmpeg and ffprobe: config `tools.<name>`, then ROSTRUM_<NAME>, then PATH."""

    def __init__(self, settings=None):
        self.settings = settings or {}

    def ffmpeg(self):
        return self.find("ffmpeg")

    def ffprobe(self):
        return self.find("ffprobe")

    def find(self, name):
        configured = self.settings.get(name) or os.environ.get(f"ROSTRUM_{name.upper()}")
        if configured:
            if not Path(configured).is_file():
                raise ToolNotFound(f"{name} not found at {configured}")
            return configured
        found = shutil.which(name)
        if found is None:
            raise ToolNotFound(f"{name} is not on PATH. " + INSTALL_HINT.format(name=name))
        return found
