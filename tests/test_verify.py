import json
import subprocess

import numpy as np
import pytest

from media_factory import MediaFactory
from rostrum.cli import main
from rostrum.edl import Edl, Source
from rostrum.render import Renderer
from rostrum.tools import ToolNotFound, Tools
from rostrum.verify import VERIFY_DEFAULTS, Correlator, JoinSample, Verifier

FAST = {"preset": "ultrafast"}


class FrameGrabber:
    """One frame as gray pixels, from a PNG or from a source frame number."""

    def __init__(self, ffmpeg):
        self.ffmpeg = ffmpeg

    def png(self, path):
        return self.gray(["-i", str(path)])

    def source(self, path, frame, fps):
        return self.gray(["-ss", f"{(frame - 0.5) / fps:.6f}", "-i", str(path)])

    def gray(self, args):
        command = [self.ffmpeg, "-v", "error", "-nostdin", *args, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
        pixels = subprocess.run(command, capture_output=True, check=True).stdout
        return np.frombuffer(pixels, dtype=np.uint8).astype(np.float64)

    def closest(self, image, path, frames, fps):
        errors = {frame: float(np.mean((self.source(path, frame, fps) - image) ** 2)) for frame in frames}
        return min(errors, key=errors.get)


@pytest.fixture(scope="module")
def factory(tmp_path_factory):
    try:
        return MediaFactory(tmp_path_factory.getbasetemp() / "media")
    except ToolNotFound as error:
        pytest.skip(str(error))


class TestCorrelator:
    def setup_method(self):
        self.signal = np.random.default_rng(5).standard_normal(4000) * 0.1

    def test_finds_a_known_shift(self):
        max_lag = 800
        shift = 40
        reference = self.signal
        window = reference[max_lag + shift: max_lag + shift + 2000]
        index, correlation, rms = Correlator().best_lag(window, reference[: 2000 + 2 * max_lag], max_lag)
        assert index - max_lag == shift
        assert correlation > 0.99
        assert rms > 0.05

    def test_silence_has_no_correlation(self):
        index, correlation, rms = Correlator().best_lag(np.zeros(1000), self.signal[:1200], 100)
        assert correlation == 0.0
        assert rms == 0.0


class TestJoinSample:
    def test_spreads_picks_from_first_to_last(self):
        picks = JoinSample().pick(162, 12)
        assert len(picks) == 12
        assert min(picks) == 0
        assert max(picks) == 161

    def test_few_joins_are_all_picked(self):
        assert JoinSample().pick(3, 12) == {0, 1, 2}
        assert JoinSample().pick(0, 12) == set()


class TestVerify:
    def setup_method(self):
        self.quiet = {"report": lambda message: None}

    def edl(self, source, keep):
        return Edl("check", [Source(str(source), 4.0)], keep)

    def test_audio_from_the_wrong_place_fails_the_sync_check(self, factory, tmp_path):
        source = factory.cfr("pal.mp4", 25, 4)
        expected = self.edl(source, [[0.2, 1.32], [1.92, 2.72], [3.08, 3.88]])
        shifted = self.edl(source, [[0.2, 1.32], [1.96, 2.76], [3.08, 3.88]])
        rendered = Renderer(FAST, Tools(), **self.quiet).render(shifted, tmp_path, {})
        result = Verifier({}, Tools(), **self.quiet).verify(expected, rendered["output"], "full")
        failed = {check["name"] for check in result["checks"] if not check["ok"]}
        assert not result["ok"]
        assert failed == {"A/V in sync at every join"}
        first, second = result["joins"]
        assert first["offset_before_ms"] == pytest.approx(0, abs=0.5)
        assert first["offset_after_ms"] == pytest.approx(40, abs=1)
        assert second["offset_before_ms"] == pytest.approx(40, abs=1)

    def test_png_pair_shows_the_frames_on_each_side_of_the_join(self, factory, tmp_path):
        source = factory.cfr("pal.mp4", 25, 4)
        edl = self.edl(source, [[0.2, 1.32], [1.92, 2.72]])
        rendered = Renderer(FAST, Tools(), **self.quiet).render(edl, tmp_path, {})
        result = Verifier({}, Tools(), **self.quiet).verify(edl, rendered["output"], "full")
        join = result["joins"][0]
        frames = FrameGrabber(factory.ffmpeg)
        assert frames.closest(frames.png(join["png_before"]), source, [31, 32, 33], 25) == 32
        assert frames.closest(frames.png(join["png_after"]), source, [47, 48, 49], 25) == 48

    def test_missing_segment_fails_duration_and_frame_count(self, factory, tmp_path):
        source = factory.cfr("ntsc.mp4", "30000/1001", 4)
        expected = self.edl(source, [[0.2, 1.3], [1.9, 2.75], [3.1, 3.9]])
        shorter = self.edl(source, [[0.2, 1.3], [1.9, 2.75]])
        rendered = Renderer(FAST, Tools(), **self.quiet).render(shorter, tmp_path, {})
        result = Verifier({"fastDecode": False}, Tools(), **self.quiet).verify(expected, rendered["output"], "fast")
        failed = {check["name"] for check in result["checks"] if not check["ok"]}
        assert "Duration = kept time" in failed
        assert "Video frames continuous, 30000/1001 fps" in failed

    def test_cli_verifies_a_render_folder(self, factory, tmp_path, capsys):
        source = factory.cfr("ntsc.mp4", "30000/1001", 4)
        edl_path = tmp_path / "talk.edl.json"
        self.edl(source, [[0.2, 1.3], [1.9, 2.75], [3.1, 3.9]]).save(edl_path)
        assert main(["render", "--edl", str(edl_path), "--out", str(tmp_path / "renders"), "--json"]) == 0
        rendered = json.loads(capsys.readouterr().out)
        assert main(["verify", "--render", rendered["dir"], "--mode", "fast", "--json"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["ok"]
        assert result["summary"]["offset_joins"] == 2
        assert (tmp_path / "renders" / "render-1" / "verify-fast.json").is_file()
        assert not (tmp_path / "renders" / "render-1" / "verify-full.json").exists()

    def test_defaults_match_the_baseline_limits(self):
        assert VERIFY_DEFAULTS["avOffsetLimitMs"] == 5.0
        assert VERIFY_DEFAULTS["avLengthToleranceMs"] == 50
        assert VERIFY_DEFAULTS["durationToleranceFrames"] == 1
        assert VERIFY_DEFAULTS["fastSampleJoins"] == 12
