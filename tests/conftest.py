import copy
import json
import subprocess

import numpy as np
import pytest

from rostrum.audio import Energy
from rostrum.config import ConfigLoader
from rostrum.detect.analysis import Analysis
from rostrum.tools import ToolNotFound, Tools

WORD_SECONDS = 0.3
WORD_GAP = 0.05
FRAME = 0.02
LOUD_DB = -20.0
QUIET_DB = -80.0


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never read the developer's ~/.rostrum during tests."""
    home = tmp_path / "rostrum-home"
    home.mkdir()
    monkeypatch.setenv("ROSTRUM_HOME", str(home))
    return home


@pytest.fixture
def settings(isolated_home):
    return copy.deepcopy(ConfigLoader(home=isolated_home).effective().values)


class Talk:
    """Hand-written synthetic speech: sentences at given times, with loud frames under every word."""

    def __init__(self):
        self.words = []

    def say(self, start, text, probability=0.95, word_seconds=WORD_SECONDS):
        cursor = start
        for token in text.split():
            self.words.append({"start": round(cursor, 3), "end": round(cursor + word_seconds, 3), "word": " " + token,
                               "probability": probability})
            cursor += word_seconds + WORD_GAP
        return cursor

    def energy(self, total, settings):
        frames = int(total / FRAME)
        db = np.full(frames, QUIET_DB, dtype=np.float32)
        for word in self.words:
            db[int(word["start"] / FRAME):int(word["end"] / FRAME)] = LOUD_DB
        return Energy(db, settings["audio"])

    def analysis(self, settings, total, script=(), failed=(), segments=()):
        return Analysis(self.words, self.energy(total, settings), total, list(script), list(failed), settings,
                        list(segments))


@pytest.fixture
def talk():
    return Talk()


@pytest.fixture
def make_talk():
    return Talk


class Media:
    """Tiny real media files made with ffmpeg at test time."""

    def __init__(self, ffmpeg):
        self.ffmpeg = ffmpeg

    def clip(self, path, seconds, frequency=440):
        command = [self.ffmpeg, "-v", "error", "-nostdin", "-y",
                   "-f", "lavfi", "-i", f"color=c=black:s=64x36:r=10:d={seconds}",
                   "-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=16000:duration={seconds}",
                   "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)]
        subprocess.run(command, check=True, capture_output=True)
        return path

    def wav(self, path, seconds):
        command = [self.ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "lavfi",
                   "-i", f"sine=frequency=300:sample_rate=16000:duration={seconds}", str(path)]
        subprocess.run(command, check=True, capture_output=True)
        return path

    def late_audio(self, path, seconds, offset):
        """Video from 0 s; a tone that starts `offset` seconds later and runs to the end."""
        command = [self.ffmpeg, "-v", "error", "-nostdin", "-y",
                   "-f", "lavfi", "-i", f"color=c=black:s=64x36:r=10:d={seconds}",
                   "-itsoffset", str(offset),
                   "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=16000:duration={seconds - offset}",
                   "-map", "0:v", "-map", "1:a",
                   "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)]
        subprocess.run(command, check=True, capture_output=True)
        return path

    def video_only(self, path, seconds):
        command = [self.ffmpeg, "-v", "error", "-nostdin", "-y",
                   "-f", "lavfi", "-i", f"color=c=black:s=64x36:r=10:d={seconds}",
                   "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)]
        subprocess.run(command, check=True, capture_output=True)
        return path


@pytest.fixture
def media():
    try:
        tools = Tools()
        tools.ffprobe()
        return Media(tools.ffmpeg())
    except ToolNotFound:
        pytest.skip("ffmpeg/ffprobe not found (set ROSTRUM_FFMPEG and ROSTRUM_FFPROBE, or put them on PATH)")


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path
