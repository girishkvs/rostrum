import json
from fractions import Fraction

import pytest

from media_factory import MediaFactory
from rostrum.edl import Edl, Source
from rostrum.media import AudioStream, Media, Prober, VideoStream
from rostrum.render import (RENDER_DEFAULTS, FilterGraph, OutputName, Planner, RenderError, Renderer, RunDirs,
                            SourcePolicy, UniquePath)
from rostrum.tools import ToolNotFound, Tools
from rostrum.verify import Verifier

FAST = {"preset": "ultrafast"}


class FakeMedia:
    """Probe results without ffmpeg, for planning tests."""

    def make(self, path, fps, seconds, width=160, height=120, vfr=False, average=None):
        rate = Fraction(fps)
        video = VideoStream(
            index=0, codec="h264", width=width, height=height, pix_fmt="yuv420p", time_base=Fraction(1, 90000),
            r_frame_rate=rate, avg_frame_rate=average or rate, fps=average or rate, start=0.0, duration=seconds,
            frames=None,
        )
        audio = AudioStream(
            index=1, codec="aac", sample_rate=48000, channels=2, channel_layout="stereo",
            time_base=Fraction(1, 48000), start=0.0, duration=seconds,
        )
        return Media(path=path, format="mov", start=0.0, duration=seconds, size=0, video=video, audio=audio, vfr=vfr,
                     vfr_reason="uneven steps" if vfr else "")


@pytest.fixture(scope="module")
def factory(tmp_path_factory):
    try:
        return MediaFactory(tmp_path_factory.getbasetemp() / "media")
    except ToolNotFound as error:
        pytest.skip(str(error))


class TestPlanner:
    def setup_method(self):
        self.planner = Planner(RENDER_DEFAULTS)
        self.fake = FakeMedia()

    def plan(self, keep, media, sources=None):
        sources = sources or [Source(item.path, item.duration) for item in media]
        return self.planner.plan(Edl(talk="t", sources=sources, keep=keep), media)

    def test_snaps_keep_points_to_ntsc_frames(self):
        plan = self.plan([[1.0, 2.0]], [self.fake.make("a.mp4", Fraction(30000, 1001), 10)])
        segment = plan.segments[0]
        assert (segment.start_frame, segment.end_frame) == (30, 60)
        assert segment.local_start == pytest.approx(30 * 1001 / 30000, abs=1e-12)
        assert plan.duration == pytest.approx(30 * 1001 / 30000, abs=1e-12)

    def test_never_assumes_30_fps(self):
        plan = self.plan([[1.01, 2.0]], [self.fake.make("a.mp4", 25, 10)])
        assert (plan.segments[0].start_frame, plan.segments[0].end_frame) == (25, 50)

    def test_segments_touching_after_snapping_merge(self):
        plan = self.plan([[1.0, 2.01], [2.015, 3.0]], [self.fake.make("a.mp4", 30, 10)])
        assert [(s.start_frame, s.end_frame) for s in plan.segments] == [(30, 90)]
        assert plan.joins() == []

    def test_keep_shorter_than_a_frame_is_skipped_with_a_warning(self):
        plan = self.plan([[1.0, 1.01], [2.0, 3.0]], [self.fake.make("a.mp4", 30, 10)])
        assert [(s.start_frame, s.end_frame) for s in plan.segments] == [(60, 90)]
        assert "shorter than one frame" in plan.warnings[0]

    def test_keep_range_across_files_splits_at_the_boundary(self):
        media = [self.fake.make("a.mp4", 25, 10), self.fake.make("b.mp4", 25, 5)]
        sources = [Source("a.mp4", 10.0, 0.0), Source("b.mp4", 5.0, 10.0)]
        plan = self.plan([[2.0, 12.0], [13.0, 15.0]], media, sources)
        assert [(s.source_index, s.start_frame, s.end_frame, s.output_frame) for s in plan.segments] == [
            (0, 50, 250, 0), (1, 0, 50, 200), (1, 75, 125, 250)]
        assert [s.timeline_start for s in plan.segments] == [2.0, 10.0, 13.0]
        assert [join.kind for join in plan.joins()] == ["file", "cut"]
        assert plan.frames == 300

    def test_end_is_clamped_to_the_last_full_frame(self):
        media = self.fake.make("a.mp4", Fraction(30000, 1001), 5.972633)
        plan = self.plan([[5.0, 6.0]], [media], [Source("a.mp4", 6.0)])
        assert plan.segments[0].end_frame == 179

    def test_keep_far_past_the_end_is_rejected(self):
        with pytest.raises(RenderError, match="only has video and audio"):
            self.plan([[1.0, 9.0]], [self.fake.make("a.mp4", 30, 5)], [Source("a.mp4", 10.0)])

    def test_mixed_frame_rates_are_rejected(self):
        media = [self.fake.make("a.mp4", 30, 5), self.fake.make("b.mp4", 25, 5)]
        with pytest.raises(RenderError, match=r"a\.mp4 30/1 fps, b\.mp4 25/1 fps"):
            self.plan([[0, 10]], media, [Source("a.mp4", 5.0, 0.0), Source("b.mp4", 5.0, 5.0)])

    def test_mixed_sizes_are_rejected(self):
        media = [self.fake.make("a.mp4", 30, 5), self.fake.make("b.mp4", 30, 5, width=320, height=240)]
        with pytest.raises(RenderError, match="different video sizes"):
            self.plan([[0, 10]], media, [Source("a.mp4", 5.0, 0.0), Source("b.mp4", 5.0, 5.0)])


class TestSourcePolicy:
    def setup_method(self):
        self.vfr = FakeMedia().make("phone.mp4", 150, 5, vfr=True, average=Fraction(13500, 451))

    def test_vfr_is_rejected_when_normalizing_is_off(self):
        policy = SourcePolicy({**RENDER_DEFAULTS, "normalizeVfr": False})
        with pytest.raises(RenderError, match="variable frame rate"):
            policy.planning_media([self.vfr])

    def test_vfr_is_planned_at_the_nearest_standard_rate(self):
        planned = SourcePolicy(RENDER_DEFAULTS).planning_media([self.vfr])[0]
        assert planned.fps == Fraction(30000, 1001)
        assert not planned.vfr
        assert (planned.video.index, planned.audio.index) == (0, 1)


class TestFilterGraph:
    def test_split_trim_fades_and_concat(self):
        fake = FakeMedia()
        media = [fake.make("a.mp4", 25, 10), fake.make("b.mp4", 25, 5)]
        sources = [Source("a.mp4", 10.0, 0.0), Source("b.mp4", 5.0, 10.0)]
        plan = Planner(RENDER_DEFAULTS).plan(Edl("t", sources, [[0.0, 4.0], [6.0, 12.0], [13.0, 15.0]]), media)
        graph = FilterGraph(RENDER_DEFAULTS).build(plan, media, "anull")
        lines = graph.splitlines()
        assert lines[0] == "[0:0]split=2[v0_0][v0_1];"
        assert "[1:1]asplit=2[a1_0][a1_1];" in lines
        assert "[v0_0]trim=start=0.000000:end=4.000000,setpts=PTS-STARTPTS[v0];" in lines
        first_audio = next(line for line in lines if line.startswith("[a0_0]"))
        last_audio = next(line for line in lines if line.endswith("[a3];"))
        assert "afade=t=in" not in first_audio
        assert "afade=t=out:st=3.985000:d=0.015000" in first_audio
        assert "afade=t=in:st=0:d=0.015000" in last_audio
        assert "afade=t=out" not in last_audio
        assert "[v0][a0][v1][a1][v2][a2][v3][a3]concat=n=4:v=1:a=1[vc][ac];" in lines
        assert lines[-2] == "[vc]fps=25/1,format=yuv420p[vout];"


class TestNaming:
    def test_name_template_and_cleanup(self):
        names = OutputName(RENDER_DEFAULTS)
        assert names.make({"speaker": "Jane Doe", "title": "Keynote: Part 1"}, "talk") == "Jane_Doe-Keynote_Part_1"
        assert names.make({"title": "Keynote"}, "talk") == "Keynote"
        assert names.make({}, "my talk") == "my_talk"

    def test_unknown_template_field_is_an_error(self):
        with pytest.raises(RenderError, match="unknown field"):
            OutputName({**RENDER_DEFAULTS, "name": "{venue}-{title}"}).make({"title": "x"}, "t")

    def test_event_in_the_name(self):
        names = OutputName({**RENDER_DEFAULTS, "name": "{event}-{talk}"})
        assert names.make({"event": "Example Conf"}, "keynote") == "Example_Conf-keynote"
        assert names.make({}, "keynote") == "keynote"

    def test_existing_files_get_a_number(self, tmp_path):
        (tmp_path / "talk.mp4").write_bytes(b"x")
        (tmp_path / "talk-2.mp4").write_bytes(b"x")
        assert UniquePath().make(tmp_path / "talk.mp4").name == "talk-3.mp4"

    def test_render_folders_are_numbered(self, tmp_path):
        folders = RunDirs(RENDER_DEFAULTS)
        assert [folders.create(tmp_path).name for _ in range(3)] == ["render-1", "render-2", "render-3"]


class TestRender:
    def test_render_and_verify_ntsc_with_several_joins(self, factory, tmp_path):
        source = factory.cfr("ntsc.mp4", "30000/1001", 4)
        edl = Edl("ntsc", [Source(str(source), 4.0)], [[0.2, 1.1], [1.5, 2.2], [2.5, 3.2], [3.4, 3.9]])
        result = Renderer(FAST, Tools(), report=lambda message: None).render(edl, tmp_path, {"title": "Demo"})
        assert result["fps"] == "30000/1001"
        assert result["joins"] == 3
        assert result["frames"] == sum(round(end * 30000 / 1001) - round(start * 30000 / 1001)
                                       for start, end in edl.keep)
        cutmap = json.loads((tmp_path / "render-1" / "cutmap.json").read_text(encoding="utf-8"))
        assert [segment["output_frame"] for segment in cutmap["segments"]] == [0, 27, 48, 69]
        assert json.loads((tmp_path / "render-1" / "render.json").read_text(encoding="utf-8"))["ok"]
        for mode in ("fast", "full"):
            verified = Verifier({}, Tools(), report=lambda message: None).verify(edl, result["output"], mode)
            assert verified["ok"], [check for check in verified["checks"] if not check["ok"]]
        assert len(list((tmp_path / "render-1" / "verify-png").glob("*.png"))) == 6

    def test_multi_file_talk_renders_across_the_boundary(self, factory, tmp_path):
        first = factory.cfr("part1.mp4", 25, 3, seed=11)
        second = factory.cfr("part2.mp4", 25, 3, seed=12)
        prober = Prober(Tools())
        durations = [prober.probe(path).duration for path in (first, second)]
        sources = [Source(str(first), durations[0], 0.0), Source(str(second), durations[1], durations[0])]
        edl = Edl("parts", sources, [[0.3, 1.2], [1.8, 4.1], [4.6, 5.6]])
        result = Renderer(FAST, Tools(), report=lambda message: None).render(edl, tmp_path, {})
        assert result["segments"] == 4
        verified = Verifier({}, Tools(), report=lambda message: None).verify(edl, result["output"], "full")
        assert verified["ok"], [check for check in verified["checks"] if not check["ok"]]
        assert [row["kind"] for row in verified["joins"]] == ["cut", "file", "cut"]

    def test_vfr_source_is_normalized_first(self, factory, tmp_path):
        source = factory.vfr("vfr.mp4", 4)
        edl = Edl("vfr", [Source(str(source), 4.0)], [[0.3, 1.5], [2.0, 3.6]])
        messages = []
        result = Renderer(FAST, Tools(), report=messages.append).render(edl, tmp_path, {})
        assert result["fps"] == "30000/1001"
        assert any("variable frame rate" in warning for warning in result["warnings"])
        assert any(message.startswith("warning:") for message in messages)
        verified = Verifier({}, Tools(), report=lambda message: None).verify(edl, result["output"], "fast")
        assert verified["ok"], [check for check in verified["checks"] if not check["ok"]]

    def test_second_render_never_overwrites_and_source_is_untouched(self, factory, tmp_path):
        source = factory.cfr("pal.mp4", 25, 4)
        before = (source.stat().st_size, source.stat().st_mtime_ns)
        edl = Edl("again", [Source(str(source), 4.0)], [[0.5, 2.0], [2.5, 3.5]])
        renderer = Renderer({**FAST, "loudnessLufs": -20}, Tools(), report=lambda message: None)
        first = renderer.render(edl, tmp_path, {})
        second = renderer.render(edl, tmp_path, {})
        assert first["dir"] != second["dir"]
        assert first["loudness"]["input_i"]
        assert (source.stat().st_size, source.stat().st_mtime_ns) == before
        assert not (tmp_path / ".work").exists()
