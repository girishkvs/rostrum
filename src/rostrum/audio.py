"""Audio for analysis. ffmpeg decodes straight to 16 kHz mono float32; no PyAV needed.

Times are file times: when a file's audio stream starts late (media.Media.audio_offset), its first samples are
silence, so transcript and loudness times line up with render, review and verify."""
import subprocess
import wave
from pathlib import Path

import numpy as np

# Whisper-family models need 16 kHz input; the loudness analysis uses the same samples.
SAMPLE_RATE = 16000
SILENT_DB = -120.0
ERROR_TAIL = 600


class AudioError(RuntimeError):
    pass


class AudioDecoder:
    def __init__(self, tools):
        self.tools = tools

    def pcm(self, path, offset=0.0):
        """16-bit mono 16 kHz samples, after `offset` seconds of silence (an audio stream that starts late)."""
        command = [self.tools.ffmpeg(), "-v", "error", "-nostdin", "-i", str(path),
                   "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
        result = subprocess.run(command, capture_output=True)
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", "replace").strip()[-ERROR_TAIL:]
            raise AudioError(f"ffmpeg cannot decode audio from {path}: {message}")
        samples = np.frombuffer(result.stdout, "<i2")
        padding = int(round(max(0.0, offset) * SAMPLE_RATE))
        if padding == 0:
            return samples
        return np.concatenate([np.zeros(padding, dtype="<i2"), samples])

    def decode(self, path, offset=0.0):
        """Float32 samples in [-1, 1), mono, 16 kHz, on the file's clock."""
        return self.pcm(path, offset).astype(np.float32) / 32768.0

    def write_wav(self, path, target, offset=0.0):
        samples = self.pcm(path, offset)
        try:
            with wave.open(str(target), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(SAMPLE_RATE)
                out.writeframes(samples.tobytes())
        except OSError as error:
            raise AudioError(f"cannot write {target}: {error}") from error
        return Path(target)

    def fit(self, audio, seconds):
        """Trims or pads with silence so the samples cover exactly `seconds` of timeline."""
        wanted = int(round(seconds * SAMPLE_RATE))
        if len(audio) >= wanted:
            return audio[:wanted]
        return np.concatenate([audio, np.zeros(wanted - len(audio), dtype=np.float32)])

    def timeline(self, sources, offsets=None):
        """All sources of a talk, each fitted to its duration, as one array. offsets: each source's audio start."""
        offsets = offsets or [0.0] * len(sources)
        parts = [self.fit(self.decode(source.path, offset), source.duration) for source, offset in zip(sources, offsets)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


class EnergyMeter:
    def __init__(self, frame_sec):
        self.frame_sec = frame_sec

    def frame_db(self, audio):
        """Mean power per frame in dB. Computed in blocks to keep memory flat on long recordings."""
        frame = int(self.frame_sec * SAMPLE_RATE)
        count = len(audio) // frame
        out = np.empty(count, dtype=np.float32)
        block = 100000
        for first in range(0, count, block):
            last = min(count, first + block)
            samples = np.asarray(audio[first * frame:last * frame], dtype=np.float32).reshape(-1, frame)
            out[first:first + len(samples)] = 10 * np.log10(np.mean(samples * samples, axis=1) + 1e-12)
        return out


class Energy:
    """Frame loudness over the talk timeline, with the noise floor, the speech peak and the thresholds between them."""

    def __init__(self, db, audio_settings):
        self.db = db
        self.frame = float(audio_settings["frameSec"])
        has_frames = len(db) > 0
        self.floor = float(np.percentile(db, audio_settings["floorPercentile"])) if has_frames else SILENT_DB
        self.peak = float(np.percentile(db, audio_settings["peakPercentile"])) if has_frames else SILENT_DB
        self.silence = self.floor + audio_settings["silenceLevel"] * (self.peak - self.floor)
        self.speech = self.floor + audio_settings["speechLevel"] * (self.peak - self.floor)

    def quietest(self, low, high, target, radius):
        """Middle of the quietest frame within `radius` of `target`, inside [low, high]."""
        start = max(low, target - radius)
        end = min(high, target + radius)
        if end - start < self.frame:
            return target
        first = int(start / self.frame)
        last = max(int(start / self.frame) + 1, int(end / self.frame))
        frames = self.db[first:last]
        return (first + int(frames.argmin()) + 0.5) * self.frame if len(frames) else target

    def loud_runs(self, start, end, min_length, level=None):
        """Runs of frames above `level` (default: the speech threshold) lasting at least min_length."""
        if end <= start:
            return []
        threshold = self.speech if level is None else level
        first = int(start / self.frame)
        last = int(end / self.frame)
        loud = self.db[first:last] > threshold
        runs = []
        index = 0
        while index < len(loud):
            if not loud[index]:
                index += 1
                continue
            stop = index
            while (stop < len(loud) and
                    loud[stop]):
                stop += 1
            if (stop - index) * self.frame >= min_length:
                runs.append(((first + index) * self.frame, (first + stop) * self.frame))
            index = stop
        return runs

    def summary(self):
        return {"frames": int(len(self.db)), "frameSec": self.frame, "floorDb": round(self.floor, 3),
                "peakDb": round(self.peak, 3), "silenceDb": round(self.silence, 3), "speechDb": round(self.speech, 3)}
