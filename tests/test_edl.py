from fractions import Fraction

from rostrum.edl import CutIds, Edl, FrameGrid, Ranges, Source


class TestRanges:
    def setup_method(self):
        self.ranges = Ranges()

    def test_merge_joins_overlaps_and_drops_empty(self):
        assert self.ranges.merge([[5, 6], [0, 2], [1, 3], [7, 7]]) == [[0.0, 3.0], [5.0, 6.0]]

    def test_complement_inside_bounds(self):
        assert self.ranges.complement([[1, 2], [4, 5]], 0, 6) == [[0, 1.0], [2.0, 4.0], [5.0, 6]]

    def test_intersect_length(self):
        assert self.ranges.intersect_length([[0, 10]], [[2, 3], [9, 12]]) == 2.0


class TestEdlSegments:
    def test_keep_range_spanning_two_files_splits_at_boundary(self):
        edl = Edl(
            talk="t",
            sources=[Source("a.mp4", 10.0, 0.0), Source("b.mp4", 5.0, 10.0)],
            keep=[[2.0, 12.0], [13.0, 15.0]],
        )
        segments = [(s.source_index, s.start, s.end, s.timeline_start) for s in edl.segments()]
        assert segments == [(0, 2.0, 10.0, 0.0), (1, 0.0, 2.0, 8.0), (1, 3.0, 5.0, 10.0)]
        assert edl.kept == 12.0
        assert edl.removed() == [[0.0, 2.0], [12.0, 13.0]]


class TestFrameGrid:
    def test_ntsc_rate_snaps_to_exact_frames(self):
        grid = FrameGrid(Fraction(30000, 1001))
        assert abs(grid.snap(1.0) - 30 * 1001 / 30000) < 1e-9

    def test_integer_rate(self):
        assert FrameGrid(30).snap(0.51) == 0.5


class TestCutIds:
    def test_same_inputs_give_same_id_and_small_jitter_is_ignored(self):
        ids = CutIds("abc")
        first = ids.make("redo", 10.02, 14.9, "so the agent  — Redo")
        second = ids.make("redo", 10.1, 14.95, "so the agent — redo")
        assert first == second
        assert first.startswith("redo-")

    def test_different_evidence_gives_different_id(self):
        ids = CutIds("abc")
        assert ids.make("redo", 10, 15, "one") != ids.make("redo", 10, 15, "two")
