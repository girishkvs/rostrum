from fractions import Fraction

import pytest

from media_factory import MediaFactory
from rostrum.media import MEDIA_DEFAULTS, MediaError, Prober, Rational, Runner, VfrDetector
from rostrum.tools import ToolNotFound, Tools


@pytest.fixture(scope="module")
def factory(tmp_path_factory):
    try:
        return MediaFactory(tmp_path_factory.getbasetemp() / "media")
    except ToolNotFound as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def prober(factory):
    return Prober(Tools())


class TestRational:
    def setup_method(self):
        self.rational = Rational()

    def test_ntsc_rate_is_exact(self):
        assert self.rational.parse("30000/1001") == Fraction(30000, 1001)

    def test_missing_and_zero_rates_are_none(self):
        assert self.rational.parse("0/0") is None
        assert self.rational.parse("N/A") is None
        assert self.rational.parse(None) is None

    def test_plain_number(self):
        assert self.rational.parse("25") == Fraction(25)
        assert self.rational.text(Fraction(30000, 1001)) == "30000/1001"


class TestFilterScriptOption:
    """ffmpeg before 7 only knows -filter_complex_script (Ubuntu's apt ffmpeg 6.x); 7 and later take -/filter_complex."""

    @pytest.mark.parametrize("version_line, option", [
        ("ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers", "-filter_complex_script"),
        ("ffmpeg version n6.0 Copyright (c) 2000-2023 the FFmpeg developers", "-filter_complex_script"),
        ("ffmpeg version n7.1-12-g1234abcd Copyright (c) 2000-2024 the FFmpeg developers", "-/filter_complex"),
        ("ffmpeg version 9.0.2-essentials_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers",
         "-/filter_complex"),
        ("ffmpeg version N-12345-g0123abcd Copyright (c) 2000-2026 the FFmpeg developers", "-/filter_complex"),
    ])
    def test_option_follows_the_major_version(self, version_line, option):
        runner = Runner(tools=None)
        runner.version_line = version_line
        assert runner.filter_script_option() == option


class TestJitter:
    def setup_method(self):
        self.detector = VfrDetector(runner=None, settings=MEDIA_DEFAULTS)

    def test_constant_steps_are_even(self):
        assert self.detector.jitter([1001] * 100)["uneven"] == 0

    def test_millisecond_rounding_is_not_jitter(self):
        steps = [33, 34, 33] * 40
        assert self.detector.jitter(steps)["ratio"] == 0.0

    def test_alternating_steps_are_jitter(self):
        result = self.detector.jitter([1400, 600] * 50)
        assert result["ratio"] > MEDIA_DEFAULTS["vfrJitterRatio"]


class TestProbe:
    def test_ntsc_rate_stays_an_exact_fraction(self, factory, prober):
        media = prober.probe(factory.cfr("ntsc.mp4", "30000/1001", 4))
        assert media.fps == Fraction(30000, 1001)
        assert not media.vfr
        assert (media.video.width, media.video.height) == (160, 120)
        assert media.video.codec == "h264"
        assert (media.audio.codec, media.audio.sample_rate, media.audio.channels) == ("aac", 48000, 2)
        assert media.video_offset == 0.0
        assert 3.9 < media.usable_end <= 4.0

    def test_pal_rate(self, factory, prober):
        media = prober.probe(factory.cfr("pal.mp4", 25, 4))
        assert media.fps == Fraction(25)
        assert not media.vfr

    def test_variable_frame_rate_is_detected(self, factory, prober):
        media = prober.probe(factory.vfr("vfr.mp4", 4))
        assert media.vfr
        assert "uneven" in media.vfr_reason
        assert media.timing["uneven"] > 0

    def test_missing_file_is_a_media_error(self, prober, tmp_path):
        with pytest.raises(MediaError):
            prober.probe(tmp_path / "missing.mp4")
