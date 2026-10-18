import json
from fractions import Fraction

import pytest

from rostrum.cli import main
from rostrum.cutmap import CutMap, CutMapBuilder, CutMapFile, MapError, TimeText
from rostrum.edl import Cut, Edl, Source
from rostrum.media import AudioStream, Media, VideoStream
from rostrum.render import RENDER_DEFAULTS, Planner


class TwoPartTalk:
    """A 10 s + 5 s talk at 25 fps with three cuts, as a cut map built from a real render plan."""

    def media(self, path, seconds):
        video = VideoStream(index=0, codec="h264", width=160, height=120, pix_fmt="yuv420p",
                            time_base=Fraction(1, 12800), r_frame_rate=Fraction(25), avg_frame_rate=Fraction(25),
                            fps=Fraction(25), start=0.0, duration=seconds, frames=None)
        audio = AudioStream(index=1, codec="aac", sample_rate=48000, channels=2, channel_layout="stereo",
                            time_base=Fraction(1, 48000), start=0.0, duration=seconds)
        return Media(path=path, format="mov", start=0.0, duration=seconds, size=0, video=video, audio=audio)

    def data(self):
        sources = [Source("part1.mp4", 10.0, 0.0), Source("part2.mp4", 5.0, 10.0)]
        cuts = [Cut("lead-1", 0.0, 1.0, "lead-in", reason="lead-in"),
                Cut("redo-1", 4.0, 6.0, "redo", reason="redo: said again")]
        edl = Edl("demo", sources, [[1.0, 4.0], [6.0, 12.0], [13.0, 15.0]], cuts)
        plan = Planner(RENDER_DEFAULTS).plan(edl, [self.media("part1.mp4", 10.0), self.media("part2.mp4", 5.0)])
        return CutMapBuilder().build(plan, edl, "demo.mp4")


class TestTimeText:
    def setup_method(self):
        self.text = TimeText()

    def test_accepted_forms(self):
        assert self.text.parse("90") == 90
        assert self.text.parse("90.25") == 90.25
        assert self.text.parse(".5") == 0.5
        assert self.text.parse("23:30") == 1410
        assert self.text.parse("1:02:03.5") == 3723.5
        assert self.text.parse("75:30") == 4530

    def test_rejected_forms(self):
        for bad in ["", "-5", "1:2:3:4", "1:75", "1:60:00", "1.5:30", "abc", "1:"]:
            with pytest.raises(MapError):
                self.text.parse(bad)

    def test_format(self):
        assert self.text.hms(3723.5) == "1:02:03.500"
        assert self.text.hms(0.0004) == "0:00:00.000"


class TestCutMap:
    def setup_method(self):
        self.cutmap = CutMap(TwoPartTalk().data())

    def test_segments_and_cuts(self):
        data = self.cutmap.data
        assert [(s["source"], s["local_start"], s["local_end"]) for s in data["segments"]] == [
            (0, 1.0, 4.0), (0, 6.0, 10.0), (1, 0.0, 2.0), (1, 3.0, 5.0)]
        assert [(s["output_start"], s["output_end"]) for s in data["segments"]] == [
            (0.0, 3.0), (3.0, 7.0), (7.0, 9.0), (9.0, 11.0)]
        assert [(c["timeline_start"], c["timeline_end"], c["reasons"]) for c in data["cuts"]] == [
            (0.0, 1.0, ["lead-in"]), (4.0, 6.0, ["redo: said again"]), (12.0, 13.0, [])]

    def test_output_to_source_crosses_into_the_second_file(self):
        result = self.cutmap.to_source(8.5)
        assert (result["timeline"], result["source"], result["local"], result["segment"]) == (11.5, 1, 1.5, 2)

    def test_output_time_at_a_join_belongs_to_the_next_segment(self):
        assert self.cutmap.to_source(3.0)["timeline"] == 6.0

    def test_source_to_output_and_back(self):
        result = self.cutmap.to_output(7.25)
        assert (result["output"], result["in_cut"]) == (4.25, False)
        assert self.cutmap.to_source(result["output"])["timeline"] == 7.25

    def test_local_time_in_a_source_file(self):
        assert self.cutmap.to_output(3.5, source_index=1)["output"] == 9.5

    def test_time_inside_a_cut_reports_the_nearest_kept_edges(self):
        result = self.cutmap.to_output(5.0)
        assert result["in_cut"]
        assert result["cut"] == [4.0, 6.0]
        assert result["reasons"] == ["redo: said again"]
        assert (result["previous_edge"]["timeline"], result["previous_edge"]["output"]) == (4.0, 3.0)
        assert (result["next_edge"]["timeline"], result["next_edge"]["output"]) == (6.0, 3.0)

    def test_lead_in_cut_has_no_previous_edge(self):
        result = self.cutmap.to_output(0.5)
        assert result["previous_edge"] is None
        assert result["next_edge"]["output"] == 0.0

    def test_times_outside_are_errors(self):
        with pytest.raises(MapError):
            self.cutmap.to_source(11.5)
        with pytest.raises(MapError):
            self.cutmap.to_output(16.0)
        with pytest.raises(MapError):
            self.cutmap.to_output(1.0, source_index=2)


class TestMapCommand:
    def write(self, folder):
        CutMapFile().save(TwoPartTalk().data(), folder / "cutmap.json")
        return folder

    def test_output_to_source_as_json(self, tmp_path, capsys):
        assert main(["map", "0:08.5", "--from", str(self.write(tmp_path)), "--json"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert (result["ok"], result["timeline"], result["source"], result["local"]) == (True, 11.5, 1, 1.5)

    def test_inside_a_cut_as_text(self, tmp_path, capsys):
        cutmap = self.write(tmp_path) / "cutmap.json"
        assert main(["map", "5", "--from", str(cutmap), "--to", "output"]) == 0
        text = capsys.readouterr().out
        assert "inside a cut" in text
        assert "redo: said again" in text
        assert "output 0:00:03.000" in text

    def test_bad_time_fails(self, tmp_path, capsys):
        assert main(["map", "1:75", "--from", str(self.write(tmp_path))]) == 1
        assert "seconds must be below 60" in capsys.readouterr().err
