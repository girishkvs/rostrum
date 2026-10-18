"""Small test media made with ffmpeg lavfi at test time. No media is checked in."""
import subprocess
from pathlib import Path

from rostrum.tools import Tools

# A tone plus pink noise: the noise gives the A/V offset check a clear correlation peak.
AUDIO = (
    "sine=frequency=440:sample_rate=48000:duration={seconds}[tone];"
    "anoisesrc=color=pink:amplitude=0.3:seed={seed}:sample_rate=48000:duration={seconds}[noise];"
    "[tone][noise]amix=inputs=2:normalize=0[out0]"
)


class MediaFactory:
    """Creates each file once per folder. Raises ToolNotFound when ffmpeg or ffprobe is missing."""

    def __init__(self, folder):
        tools = Tools()
        self.ffmpeg = tools.ffmpeg()
        tools.ffprobe()
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def cfr(self, name, rate, seconds, size="160x120", seed=1):
        path = self.folder / name
        if not path.exists():
            self.make(path, [
                "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={seconds}",
                "-f", "lavfi", "-i", AUDIO.format(seconds=seconds, seed=seed),
                "-c:v", "libx264", "-preset", "ultrafast", "-g", "15", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", "-ac", "2",
            ])
        return path

    def vfr(self, name, seconds, seed=3):
        """30 fps frames with alternating 1.4 and 0.6 frame steps."""
        path = self.folder / name
        if not path.exists():
            self.make(path, [
                "-f", "lavfi", "-i", f"testsrc2=size=160x120:rate=30:duration={seconds}",
                "-f", "lavfi", "-i", AUDIO.format(seconds=seconds, seed=seed),
                "-vf", "settb=1/30000,setpts='(N+0.4*mod(N,2))*1000'",
                "-fps_mode", "passthrough", "-enc_time_base:v", "1:30000", "-video_track_timescale", "30000",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", "-ac", "2",
            ])
        return path

    def make(self, path, args):
        partial = path.with_name(f"{path.stem}.partial{path.suffix}")
        command = [self.ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-y", *args, str(partial)]
        subprocess.run(command, check=True, capture_output=True)
        partial.replace(path)
