import json
import re

from rostrum.edl import Cut, Source
from rostrum.review import ReviewPage, write_review


class TestReviewPage:
    def setup_method(self):
        self.sources = [Source("part1.mp4", 60.0, 0.0), Source("part 2.mp4", 30.0, 60.0)]
        self.cuts = [
            Cut("pause-1", 20.0, 22.5, "pause", reason="Pause 4.5 s trimmed"),
            Cut("redo-2", 55.0, 63.0, "redo", level="propose", decision="keep",
                reason="Spoken cue", evidence="so the result </script><b>is</b> — redo"),
        ]

    def embedded(self, page_html):
        match = re.search(r'<script id="data" type="application/json">(.*?)</script>', page_html, re.S)
        assert match, "data block missing"
        return json.loads(match.group(1).replace("<\\/", "</"))

    def test_writes_sorted_cuts_and_sources_with_offsets(self, tmp_path):
        (tmp_path / "part1.mp4").write_bytes(b"")
        (tmp_path / "part 2.mp4").write_bytes(b"")
        sources = [Source(str(tmp_path / s.path), s.duration, s.offset) for s in self.sources]
        page = ReviewPage("talk-1", sources, list(reversed(self.cuts)), title="Talk One")

        path = page.write(tmp_path / "work" / "review.html")
        data = self.embedded(path.read_text(encoding="utf-8"))

        assert data["talk"] == "talk-1"
        assert data["duration"] == 90.0
        assert [cut["id"] for cut in data["cuts"]] == ["pause-1", "redo-2"]
        assert data["sources"][1]["offset"] == 60.0
        assert data["sources"][1]["url"] == "../part%202.mp4"
        assert data["saveName"] == "talk-1-decisions.json"
        assert data["basedOn"] is None
        assert str(tmp_path) not in path.read_text(encoding="utf-8")

    def test_evidence_cannot_close_the_data_script(self, tmp_path):
        page = ReviewPage("t", [Source(str(tmp_path / "a.mp4"), 90.0)], self.cuts)
        page_html = page.html(tmp_path)

        assert page_html.count("</script>") == 2
        assert "<b>is</b>" in self.embedded(page_html)["cuts"][1]["evidence"]

    def test_title_is_escaped(self, tmp_path):
        page = ReviewPage("t", [Source(str(tmp_path / "a.mp4"), 1.0)], [], title="<i>x</i>")
        assert "&lt;i&gt;x&lt;/i&gt;" in page.html(tmp_path)


class FakeTalkContext:
    def __init__(self, root, sources, cuts, analyzed_at=None):
        self.talk_id = "talk-1"
        self.title = "Talk One"
        self.sources = sources
        self.cuts = cuts
        self.work_dir = root
        self.decisions_path = root / "decisions.json"
        self.review_path = root / "review.html"
        self.analyzed_at = analyzed_at


class TestWriteReviewHook:
    def test_embeds_existing_decisions_and_writes_to_the_talk_review_path(self, tmp_path):
        existing = {"redo-2": {"decision": "cut", "decided_by": "user", "end": 62.0}}
        (tmp_path / "decisions.json").write_text(
            json.dumps({"schema": "rostrum.decisions/1", "talk": "talk-1", "decisions": existing}), encoding="utf-8")
        cuts = [Cut("redo-2", 55.0, 62.0, "redo", decision="cut", decided_by="user")]
        context = FakeTalkContext(tmp_path, [Source(str(tmp_path / "a.mp4"), 90.0)], cuts, analyzed_at=1760745600.125)

        path = write_review(context)
        data = TestReviewPage().embedded(path.read_text(encoding="utf-8"))

        assert path == tmp_path / "review.html"
        assert data["decisions"] == existing
        assert data["saveName"] == "talk-1-decisions.json"
        assert data["basedOn"] == 1760745600.125

    def test_missing_decisions_file_means_none_yet(self, tmp_path):
        context = FakeTalkContext(tmp_path, [Source(str(tmp_path / "a.mp4"), 10.0)], [])
        data = TestReviewPage().embedded(write_review(context).read_text(encoding="utf-8"))
        assert data["decisions"] == {}
