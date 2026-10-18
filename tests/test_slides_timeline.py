import csv
import json

import pytest

from rostrum.config import ConfigError
from rostrum.edl import Source
from rostrum.slides_timeline import SlideName, SlideTimelineFiles


def table(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class TestSlideTimelineFiles:
    def outputs(self, folder, label):
        table(folder / f"{label}-timeline.csv", ["start", "end", "slide", "score", "entry", "final_pass"],
              [["0.000", "4.000", "S01", "0.99", "start", "yes"], ["4.000", "5.000", "OFF", "0.20", "off", "no"],
               ["5.000", "9.000", "S02", "0.98", "forward", "yes"]])
        table(folder / f"{label}-rewinds.csv", ["time", "slide", "magnitude"], [["6.500", "S02", "0.3"]])
        (folder / f"{label}-summary.json").write_text(json.dumps({"segments": 3}), encoding="utf-8")
        return folder / label

    def test_prefix_loads_visits_and_extra_tables(self, tmp_path):
        prefix = self.outputs(tmp_path, "part")
        timeline = SlideTimelineFiles(tmp_path).load("part", [Source("a.mp4", 9.0)])
        assert [(visit.start, visit.end, visit.slide, visit.final) for visit in timeline.visits] == [
            (0.0, 4.0, 1, True), (4.0, 5.0, None, False), (5.0, 9.0, 2, True)]
        assert timeline.tables["rewinds"][0]["time"] == 6.5
        assert timeline.summaries == [{"segments": 3}]
        assert timeline.slides_present() == [1, 2]
        same = SlideTimelineFiles(tmp_path).load(str(prefix) + "-timeline.csv", [Source("a.mp4", 9.0)])
        assert len(same.visits) == 3

    def test_one_timeline_per_source_is_moved_onto_the_talk_timeline(self, tmp_path):
        self.outputs(tmp_path, "one")
        self.outputs(tmp_path, "two")
        sources = [Source("a.mp4", 9.0, 0.0), Source("b.mp4", 9.0, 9.0)]
        timeline = SlideTimelineFiles(tmp_path).load(["one", "two"], sources)
        assert [visit.start for visit in timeline.visits] == [0.0, 4.0, 5.0, 9.0, 13.0, 14.0]
        assert [row["time"] for row in timeline.tables["rewinds"]] == [6.5, 15.5]

    def test_missing_outputs_and_count_mismatch_are_clear_errors(self, tmp_path):
        with pytest.raises(ConfigError, match="slide timeline not found"):
            SlideTimelineFiles(tmp_path).load("nothing", [Source("a.mp4", 9.0)])
        with pytest.raises(ConfigError, match="2 entries for 3 sources"):
            SlideTimelineFiles(tmp_path).load(["a", "b"], [Source("a", 1.0), Source("b", 1.0), Source("c", 1.0)])

    def test_slide_names(self):
        names = SlideName()
        assert names.number("S07") == 7
        assert names.number("OFF") is None
        assert names.text(3) == "S03"
        assert names.text(None) == "OFF"
