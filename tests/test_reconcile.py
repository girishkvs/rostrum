import csv

from rostrum.detect.planner import Planner
from rostrum.edl import Source
from rostrum.reconcile import Reconciler
from rostrum.slides_timeline import SlideTimelineFiles

TIMELINE_COLUMNS = ["start", "end", "slide", "score", "duration", "entry", "final_pass"]


def write_timeline(folder, label, rows):
    path = folder / f"{label}-timeline.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TIMELINE_COLUMNS)
        for start, end, slide, entry, final in rows:
            writer.writerow([f"{start:.3f}", f"{end:.3f}", slide, "0.990", f"{end - start:.3f}", entry, "yes" if final else "no"])
    return folder / label


class TestRetakesSaidAgain:
    def test_loose_retake_is_cut_when_everything_is_said_again(self, settings, talk):
        talk.say(2.0, "Welcome to the session.")
        talk.say(4.0, "Every answer needs a source and a date.")
        talk.say(9.0, "Every answer needs a clear source plus the date and a reason.")
        analysis = talk.analysis(settings, total=16.0)
        first = Planner(analysis).run()
        assert "retake" not in [cut.type for cut in first.painted]
        assert "retake" in [cut.type for cut in first.proposals]
        second = Reconciler(analysis).run(first)
        retakes = [cut for cut in second.painted if cut.type == "retake"]
        assert len(retakes) == 1
        assert retakes[0].note.startswith("retake said again later")
        assert second.info["promoted"] == 1

    def test_loose_match_without_same_opening_stays_a_proposal(self, settings, talk):
        talk.say(2.0, "Welcome to the session.")
        talk.say(4.0, "Caching keeps every answer close to the user.")
        talk.say(10.0, "Caching keeps each answer close to the user and fast.")
        analysis = talk.analysis(settings, total=16.0)
        second = Reconciler(analysis).run(Planner(analysis).run())
        assert "retake" not in [cut.type for cut in second.painted]
        proposals = [cut for cut in second.proposals if cut.type == "retake"]
        assert proposals and "loose match only" in proposals[0].note

    def test_standalone_stop_is_cut_in_the_reconcile_pass(self, settings, talk):
        talk.say(2.0, "Hello and welcome.")
        talk.say(4.0, "The service writes each reply into a notebook.")
        talk.say(8.0, "Stop.")
        talk.say(10.0, "The service writes every reply into the notebook now.")
        analysis = talk.analysis(settings, total=16.0)
        second = Reconciler(analysis).run(Planner(analysis).run())
        cues = [cut for cut in second.painted if cut.type in ("cue", "retake")]
        assert cues
        assert min(cut.start for cut in cues) < 4.0
        assert max(cut.end for cut in cues) <= 10.0

    def test_reconcile_can_be_kept_to_proposals_by_level(self, settings, talk):
        talk.say(2.0, "Welcome to the session.")
        talk.say(4.0, "Every answer needs a source and a date.")
        talk.say(9.0, "Every answer needs a clear source plus the date and a reason.")
        settings["retakes"]["level"] = "propose"
        analysis = talk.analysis(settings, total=16.0)
        second = Reconciler(analysis).run(Planner(analysis).run())
        assert "retake" not in [cut.type for cut in second.painted]
        assert "retake" in [cut.type for cut in second.proposals]


class TestSlideEvidence:
    def scene(self, talk):
        talk.say(1.0, "Welcome to the session.")
        talk.say(7.0, "Caching keeps every answer close to the user.")
        talk.say(13.0, "Caching keeps each answer close to the user and fast.")

    def timeline(self, tmp_path):
        prefix = write_timeline(tmp_path, "talk", [(0, 6, "S01", "start", True), (6, 12, "S02", "forward", False),
                                                   (12, 30, "S02", "resume", True)])
        return SlideTimelineFiles(tmp_path).load(str(prefix), [Source("a.mp4", 30.0, 0.0)])

    def test_abandoned_slide_visit_is_cut_when_slides_are_auto(self, settings, talk, tmp_path):
        self.scene(talk)
        settings["slides"]["level"] = "auto"
        analysis = talk.analysis(settings, total=30.0)
        second = Reconciler(analysis, self.timeline(tmp_path)).run(Planner(analysis).run())
        slides = [cut for cut in second.painted if cut.type == "slide"]
        assert len(slides) == 1
        assert 5.8 < slides[0].start < 6.3
        assert 11.7 < slides[0].end < 12.9
        assert "aborted start" in slides[0].note
        assert second.info["regions"][0]["slide"] == 2

    def test_slide_cuts_are_proposed_by_default(self, settings, talk, tmp_path):
        self.scene(talk)
        analysis = talk.analysis(settings, total=30.0)
        second = Reconciler(analysis, self.timeline(tmp_path)).run(Planner(analysis).run())
        assert "slide" not in [cut.type for cut in second.painted]
        assert "slide" in [cut.type for cut in second.proposals]

    def test_without_a_timeline_reconcile_runs_without_slides(self, settings, talk):
        self.scene(talk)
        analysis = talk.analysis(settings, total=30.0)
        second = Reconciler(analysis, None).run(Planner(analysis).run())
        assert "regions" not in second.info
        assert all(cut.type != "slide" for cut in second.painted + second.proposals)
