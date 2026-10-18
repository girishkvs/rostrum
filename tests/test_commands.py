import json
import os
import time
from dataclasses import asdict

import pytest

from rostrum.cli import main
from rostrum.edl import Cut, SourceKey
from rostrum.pipeline import CANDIDATES_SCHEMA, CutSettings, Workspace

ANALYZED = 1000.0


def run(capsys, *args):
    code = main([*args, "--json"])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def project(tmp_path):
    """A project whose first pass is already done (candidates.json written by hand)."""
    media = tmp_path / "talk.mp4"
    media.write_bytes(b"0" * 64)
    path = tmp_path / "rostrum.json"
    path.write_text(json.dumps({"defaults": {"pauses": {"keepSec": 0.4}},
                                "talks": [{"id": "talk", "title": "A Talk", "sources": ["talk.mp4"]}]}), encoding="utf-8")
    work = tmp_path / "talk.rostrum"
    work.mkdir()
    cuts = [Cut("retake-aaaa", 2.0, 5.0, "retake", "auto", "cut", "retake (strong)", "a [b] c", 0.9, "auto"),
            Cut("pause-bbbb", 7.0, 8.0, "pause", "auto", "cut", "pause 3.0s", "", 1.0, "auto"),
            Cut("offtalk-cccc", 9.0, 9.5, "offtalk", "propose", "keep", "offtalk", "[hold on]", 0.9, "auto")]
    candidates = {"schema": CANDIDATES_SCHEMA, "talk": "talk", "sourceKey": SourceKey().of([str(media)]), "duration": 12.0,
                  "fps": "30/1", "sources": [{"path": str(media), "duration": 12.0, "offset": 0.0}],
                  "wordMiddles": [1.0, 6.0, 9.2, 11.0], "cuts": [asdict(cut) for cut in cuts]}
    (work / "candidates.json").write_text(json.dumps(candidates), encoding="utf-8")
    return path


class TestPlanCutsDecide:
    def test_plan_lists_cuts_totals_and_expected_length(self, capsys, project):
        code, data = run(capsys, "plan", "-p", str(project))
        talk = data["talks"][0]
        assert code == 0
        assert (talk["duration"], talk["kept"], talk["cutCount"], talk["proposed"]) == (12.0, 8.0, 2, 1)
        assert talk["totals"] == {"retake": {"count": 1, "seconds": 3.0}, "pause": {"count": 1, "seconds": 1.0}}
        assert talk["cuts"][0]["short"] == "a [b] c"
        assert not (project.parent / "talk.rostrum" / "edl.json").exists()

    def test_plan_text_output(self, capsys, project):
        assert main(["plan", "-p", str(project)]) == 0
        text = capsys.readouterr().out
        assert "retake-aaaa" in text
        assert "00:12.0 -> 00:08.0 expected output" in text
        assert "Nothing was rendered" in text

    def test_cuts_filters(self, capsys, project):
        code, data = run(capsys, "cuts", "-p", str(project), "--decision", "keep")
        assert code == 0
        assert [cut["id"] for cut in data["talks"][0]["cuts"]] == ["offtalk-cccc"]
        code, data = run(capsys, "cuts", "-p", str(project), "--type", "pause", "--type", "retake")
        assert [cut["id"] for cut in data["talks"][0]["cuts"]] == ["retake-aaaa", "pause-bbbb"]

    def test_decide_writes_decisions_and_regenerates_the_edl(self, capsys, project):
        code, data = run(capsys, "decide", "retake-aaaa", "keep", "-p", str(project))
        assert code == 0
        assert (data["cut"]["decision"], data["cut"]["decided_by"], data["kept"]) == ("keep", "user", 11.0)
        work = project.parent / "talk.rostrum"
        assert json.loads((work / "decisions.json").read_text())["decisions"]["retake-aaaa"]["decision"] == "keep"
        edl = json.loads((work / "edl.json").read_text())
        assert edl["keep"] == [[0.0, 7.0], [8.0, 12.0]]
        code, data = run(capsys, "decide", "offtalk-cccc", "cut", "--start", "8.9", "--end", "0:09.7", "-p", str(project))
        assert (code, data["cut"]["start"], data["cut"]["end"], data["kept"]) == (0, 8.9, 9.7, 10.2)

    def test_decide_rejects_unknown_ids_and_bad_edges(self, capsys, project):
        code, data = run(capsys, "decide", "nope-1234", "keep", "-p", str(project))
        assert code == 2
        assert "no cut 'nope-1234'" in data["error"]
        code, data = run(capsys, "decide", "pause-bbbb", "cut", "--start", "8.5", "-p", str(project))
        assert code == 2
        assert "must be after start" in data["error"]

    def test_decisions_for_ids_that_are_gone_are_stale_and_not_applied(self, capsys, project):
        run(capsys, "decide", "retake-aaaa", "keep", "-p", str(project))
        candidates_path = project.parent / "talk.rostrum" / "candidates.json"
        candidates = json.loads(candidates_path.read_text())
        candidates["cuts"][0]["id"] = "retake-ffff"
        candidates_path.write_text(json.dumps(candidates))
        code, data = run(capsys, "plan", "-p", str(project))
        talk = data["talks"][0]
        assert [entry["id"] for entry in talk["stale"]] == ["retake-aaaa"]
        assert talk["kept"] == 8.0
        main(["plan", "-p", str(project)])
        assert "STALE decision retake-aaaa" in capsys.readouterr().out


class TestDecisionImport:
    def decisions_file(self, folder, decisions, talk="talk", schema="rostrum.decisions/1"):
        path = folder / "saved-decisions.json"
        data = {"schema": schema, "decisions": decisions}
        if talk:
            data["talk"] = talk
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_import_merges_skips_stale_ids_and_rebuilds_the_edl(self, capsys, project):
        run(capsys, "decide", "offtalk-cccc", "cut", "-p", str(project))
        saved = self.decisions_file(project.parent, {
            "retake-aaaa": {"decision": "keep", "at": "2026-10-17T20:00:00+00:00"},
            "pause-bbbb": {"decision": "cut", "end": 8.5, "decided_by": "review"},
            "gone-0000": {"decision": "cut"},
        })
        code, data = run(capsys, "decide", "--import", str(saved), "-p", str(project))
        assert code == 0
        assert (data["talk"], data["imported"], [entry["id"] for entry in data["stale"]]) == ("talk", 2, ["gone-0000"])
        assert data["kept"] == 10.0
        work = project.parent / "talk.rostrum"
        stored = json.loads((work / "decisions.json").read_text())["decisions"]
        assert sorted(stored) == ["offtalk-cccc", "pause-bbbb", "retake-aaaa"]
        assert stored["retake-aaaa"]["at"] == "2026-10-17T20:00:00+00:00"
        assert (stored["pause-bbbb"]["decided_by"], stored["pause-bbbb"]["end"]) == ("review", 8.5)
        assert json.loads((work / "edl.json").read_text())["keep"] == [[0.0, 7.0], [8.5, 9.0], [9.5, 12.0]]

    def test_talk_comes_from_the_file_or_the_option(self, capsys, project):
        saved = self.decisions_file(project.parent, {"retake-aaaa": {"decision": "keep"}}, talk=None)
        code, data = run(capsys, "decide", "--import", str(saved), "-p", str(project))
        assert code == 2 and "pass --talk" in data["error"]
        code, data = run(capsys, "decide", "--import", str(saved), "--talk", "talk", "-p", str(project))
        assert (code, data["imported"]) == (0, 1)
        saved = self.decisions_file(project.parent, {"retake-aaaa": {"decision": "keep"}}, talk="other")
        code, data = run(capsys, "decide", "--import", str(saved), "--talk", "talk", "-p", str(project))
        assert code == 2 and "is for talk 'other'" in data["error"]

    def test_bad_files_change_nothing(self, capsys, project):
        work = project.parent / "talk.rostrum"
        saved = self.decisions_file(project.parent, {"retake-aaaa": {"decision": "maybe"}})
        code, data = run(capsys, "decide", "--import", str(saved), "-p", str(project))
        assert code == 2 and "expected keep or cut" in data["error"]
        saved = self.decisions_file(project.parent, {"pause-bbbb": {"decision": "cut", "start": 9.0}})
        code, data = run(capsys, "decide", "--import", str(saved), "-p", str(project))
        assert code == 2 and "must be after start" in data["error"]
        saved = self.decisions_file(project.parent, {}, schema="something/1")
        code, data = run(capsys, "decide", "--import", str(saved), "-p", str(project))
        assert code == 2 and "rostrum.decisions/1" in data["error"]
        assert not (work / "decisions.json").exists()

    def test_positionals_are_optional_only_with_import(self, capsys, project):
        code, data = run(capsys, "decide", "-p", str(project))
        assert code == 2 and "--import FILE" in data["error"]
        code, data = run(capsys, "decide", "retake-aaaa", "-p", str(project))
        assert code == 2
        saved = self.decisions_file(project.parent, {})
        code, data = run(capsys, "decide", "retake-aaaa", "keep", "--import", str(saved), "-p", str(project))
        assert code == 2 and "leave out" in data["error"]


class TestReviewGate:
    def work(self, project):
        return project.parent / "talk.rostrum"

    def test_render_waits_for_a_review_or_yes(self, capsys, project):
        code, data = run(capsys, "render", "-p", str(project))
        assert code == 2
        assert "not reviewed since the cuts last changed: talk" in data["error"]
        assert "--yes" in data["error"]
        code, data = run(capsys, "render", "--yes", "-p", str(project))
        assert code == 1
        assert "not reviewed" not in json.dumps(data)

    def test_a_decision_after_init_counts_and_a_new_init_resets_it(self, capsys, project):
        run(capsys, "decide", "retake-aaaa", "keep", "-p", str(project))
        code, data = run(capsys, "render", "-p", str(project))
        assert code == 1
        assert "not reviewed" not in json.dumps(data)
        path = self.work(project) / "candidates.json"
        candidates = json.loads(path.read_text())
        candidates["analyzedAt"] = time.time() + 60
        path.write_text(json.dumps(candidates))
        code, data = run(capsys, "render", "-p", str(project))
        assert code == 2

    def test_saving_the_page_over_decisions_json_counts(self, capsys, project):
        decisions = self.work(project) / "decisions.json"
        decisions.write_text(json.dumps({"schema": "rostrum.decisions/1", "talk": "talk", "decisions": {}}))
        # Written right after candidates.json, both files can get the same file time, which proves nothing about
        # order; a real save from the page comes seconds later.
        analyzed = (self.work(project) / "candidates.json").stat().st_mtime
        os.utime(decisions, (analyzed + 60, analyzed + 60))
        code, data = run(capsys, "render", "-p", str(project))
        assert code == 1
        assert "not reviewed" not in json.dumps(data)

    def test_a_changed_source_is_refused_even_with_yes(self, capsys, project):
        (project.parent / "talk.mp4").write_bytes(b"1" * 64)
        code, data = run(capsys, "render", "--yes", "-p", str(project))
        assert code == 2
        assert "talk talk: a source file changed since init; run rostrum init" in data["error"]
        code, data = run(capsys, "plan", "-p", str(project))
        assert any("a source file changed since init" in note for note in data["talks"][0]["notes"])

    def test_plan_and_render_warn_when_cut_settings_changed(self, capsys, project):
        workspace = Workspace(project)
        talk = workspace.talks()[0]
        path = self.work(project) / "candidates.json"
        candidates = json.loads(path.read_text())
        candidates["cutSettings"] = CutSettings().fingerprint(workspace.config(talk).values)
        path.write_text(json.dumps(candidates))
        code, data = run(capsys, "plan", "-p", str(project))
        assert data["talks"][0]["notes"] == []
        settings = json.loads(project.read_text())
        settings["defaults"]["pauses"]["keepSec"] = 0.6
        project.write_text(json.dumps(settings))
        code, data = run(capsys, "plan", "-p", str(project))
        assert data["talks"][0]["notes"] == ["cut settings changed since init; run rostrum init to apply them"]
        main(["render", "--yes", "-p", str(project), "--json"])
        assert "warning: cut settings in the project changed since init" in capsys.readouterr().err


class TestDecisionFiles:
    def work(self, project):
        return project.parent / "talk.rostrum"

    def decisions_file(self, folder, decisions):
        path = folder / "saved-decisions.json"
        path.write_text(json.dumps({"schema": "rostrum.decisions/1", "talk": "talk", "decisions": decisions}),
                        encoding="utf-8")
        return path

    def test_review_page_follows_every_decision(self, capsys, project):
        page = self.work(project) / "review.html"
        code, data = run(capsys, "decide", "retake-aaaa", "keep", "-p", str(project))
        assert (code, data["review"]) == (0, str(page))
        assert '"retake-aaaa": {"decision": "keep"' in page.read_text(encoding="utf-8")
        run(capsys, "decide", "retake-aaaa", "cut", "-p", str(project))
        assert '"retake-aaaa": {"decision": "cut"' in page.read_text(encoding="utf-8")
        saved = self.decisions_file(project.parent, {"pause-bbbb": {"decision": "keep"}})
        code, data = run(capsys, "decide", "--import", str(saved), "-p", str(project))
        assert '"pause-bbbb": {"decision": "keep"' in page.read_text(encoding="utf-8")

    def test_import_keeps_the_newer_decision(self, capsys, project):
        run(capsys, "decide", "retake-aaaa", "keep", "-p", str(project))
        saved = self.decisions_file(project.parent, {
            "retake-aaaa": {"decision": "cut", "at": "2020-01-01T00:00:00+00:00"},
            "pause-bbbb": {"decision": "keep", "at": "2099-01-01T00:00:00Z"},
        })
        code, data = run(capsys, "decide", "--import", str(saved), "-p", str(project))
        assert (code, data["keptNewer"], data["imported"]) == (0, ["retake-aaaa"], 1)
        stored = json.loads((self.work(project) / "decisions.json").read_text())["decisions"]
        assert (stored["retake-aaaa"]["decision"], stored["pause-bbbb"]["decision"]) == ("keep", "keep")
        saved = self.decisions_file(project.parent, {"retake-aaaa": {"decision": "cut", "at": "2099-01-02T00:00:00Z"}})
        code, data = run(capsys, "decide", "--import", str(saved), "-p", str(project))
        stored = json.loads((self.work(project) / "decisions.json").read_text())["decisions"]
        assert (data["keptNewer"], stored["retake-aaaa"]["decision"]) == ([], "cut")

    def test_every_save_keeps_a_backup(self, capsys, project):
        run(capsys, "decide", "retake-aaaa", "keep", "-p", str(project))
        assert not (self.work(project) / "decisions.json.bak").exists()
        run(capsys, "decide", "retake-aaaa", "cut", "-p", str(project))
        backup = json.loads((self.work(project) / "decisions.json.bak").read_text())
        assert backup["decisions"]["retake-aaaa"]["decision"] == "keep"

    def test_status_shows_renders_and_the_rendered_state(self, capsys, project):
        run(capsys, "decide", "retake-aaaa", "keep", "-p", str(project))
        renders = self.work(project) / "renders"
        for name, ok in (("render-1", True), ("render-2", False)):
            (renders / name).mkdir(parents=True)
            (renders / name / "render.json").write_text(json.dumps({"ok": ok}))
            # A file written right after decide can get a file time just before decide's clock reading.
            later = time.time() + 60
            os.utime(renders / name / "render.json", (later, later))
        code, data = run(capsys, "status", "-p", str(project))
        talk = data["talks"][0]
        assert (talk["state"], talk["renders"], talk["failedRenders"]) == (
            "rendered (render-1)", ["render-1", "render-2"], ["render-2"])
        main(["status", "-p", str(project)])
        assert "renders: render-1, render-2 (failed)" in capsys.readouterr().out


class TestImportReview:
    """decide --import: which imports count as a review for rostrum render, and which leave decisions.json alone."""

    def work(self, project):
        return project.parent / "talk.rostrum"

    def analyzed(self, project):
        path = self.work(project) / "candidates.json"
        candidates = json.loads(path.read_text())
        candidates["analyzedAt"] = ANALYZED
        path.write_text(json.dumps(candidates))

    def page(self, project, decisions, based_on=None):
        data = {"schema": "rostrum.decisions/1", "talk": "talk", "decisions": decisions}
        if based_on is not None:
            data["basedOn"] = based_on
        path = project.parent / "talk-decisions.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def stored(self, project):
        return json.loads((self.work(project) / "decisions.json").read_text())

    def state(self, capsys, project):
        return run(capsys, "status", "-p", str(project))[1]["talks"][0]["state"]

    def test_a_page_from_the_current_analysis_is_a_review_even_unchanged(self, capsys, project):
        self.analyzed(project)
        page = self.page(project, {}, based_on=ANALYZED)
        code, data = run(capsys, "decide", "--import", str(page), "-p", str(project))
        assert (code, data["imported"], data["saved"], data["reviewed"], data["pageOutdated"]) == (0, 0, True, True, False)
        assert self.stored(project)["reviewedAt"] > ANALYZED
        assert self.state(capsys, project) == "reviewed, ready to render"

    def test_an_older_page_applies_what_matches_but_is_not_a_review(self, capsys, project):
        self.analyzed(project)
        page = self.page(project, {"pause-bbbb": {"decision": "keep"}, "gone-0000": {"decision": "cut"}},
                         based_on=ANALYZED - 100)
        code, data = run(capsys, "decide", "--import", str(page), "-p", str(project))
        assert (data["imported"], data["saved"], data["reviewed"], data["pageOutdated"]) == (1, True, False, True)
        stored = self.stored(project)
        assert (stored["reviewedAt"], stored["decisions"]["pause-bbbb"]["decision"]) == (None, "keep")
        assert self.state(capsys, project) == "ready for review"
        assert run(capsys, "render", "-p", str(project))[0] == 2
        changed = self.page(project, {"pause-bbbb": {"decision": "cut"}}, based_on=ANALYZED - 100)
        main(["decide", "--import", str(changed), "-p", str(project)])
        assert ("this page was built before the last init changed the cuts; decisions applied, but the new cuts "
                "still need a review (open review.html again)") in capsys.readouterr().out

    def test_an_older_page_keeps_the_previous_review(self, capsys, project):
        self.analyzed(project)
        run(capsys, "decide", "retake-aaaa", "keep", "-p", str(project))
        before = self.stored(project)["reviewedAt"]
        page = self.page(project, {"pause-bbbb": {"decision": "keep"}}, based_on=ANALYZED - 100)
        code, data = run(capsys, "decide", "--import", str(page), "-p", str(project))
        assert (data["reviewed"], self.stored(project)["reviewedAt"]) == (False, before)
        assert self.state(capsys, project) == "reviewed, ready to render"

    def test_nothing_applied_leaves_decisions_json_alone(self, capsys, project):
        self.analyzed(project)
        for based_on in (None, ANALYZED - 100):
            page = self.page(project, {"gone-0000": {"decision": "cut"}}, based_on=based_on)
            code, data = run(capsys, "decide", "--import", str(page), "-p", str(project))
            assert (code, data["imported"], data["saved"], data["reviewed"]) == (0, 0, False, False)
        assert not (self.work(project) / "decisions.json").exists()
        assert not (self.work(project) / "decisions.json.bak").exists()
        main(["decide", "--import", str(self.page(project, {"gone-0000": {"decision": "cut"}})), "-p", str(project)])
        assert "talk: nothing to import; decisions.json unchanged" in capsys.readouterr().out
        assert run(capsys, "render", "-p", str(project))[0] == 2

    def test_a_file_without_based_on_is_a_review_when_it_applies_something(self, capsys, project):
        self.analyzed(project)
        page = self.page(project, {"pause-bbbb": {"decision": "keep"}})
        code, data = run(capsys, "decide", "--import", str(page), "-p", str(project))
        assert (data["imported"], data["saved"], data["reviewed"], data["pageOutdated"]) == (1, True, True, False)
        assert self.state(capsys, project) == "reviewed, ready to render"

    def test_an_agent_decision_survives_a_page_saved_later(self, capsys, project):
        self.analyzed(project)
        run(capsys, "decide", "retake-aaaa", "keep", "-p", str(project))
        page = self.page(project, {"retake-aaaa": {"decision": "cut", "at": "2020-01-01T00:00:00+00:00"}},
                         based_on=ANALYZED)
        code, data = run(capsys, "decide", "--import", str(page), "-p", str(project))
        assert (code, data["keptNewer"], data["reviewed"]) == (0, ["retake-aaaa"], True)
        assert self.stored(project)["decisions"]["retake-aaaa"]["decision"] == "keep"

    def test_repeating_the_saved_decisions_applies_nothing(self, capsys, project):
        self.analyzed(project)
        run(capsys, "decide", "retake-aaaa", "keep", "-p", str(project))
        path = self.work(project) / "candidates.json"
        candidates = json.loads(path.read_text())
        candidates["analyzedAt"] = time.time() + 60
        path.write_text(json.dumps(candidates))
        page = self.page(project, self.stored(project)["decisions"])
        code, data = run(capsys, "decide", "--import", str(page), "-p", str(project))
        assert (code, data["imported"], data["saved"], data["reviewed"]) == (0, 0, False, False)
        assert run(capsys, "render", "-p", str(project))[0] == 2

    def test_a_changed_decision_after_a_render_is_not_rendered(self, capsys, project):
        self.analyzed(project)
        run(capsys, "decide", "retake-aaaa", "keep", "-p", str(project))
        stored = self.stored(project)
        stored["reviewedAt"] = stored["changedAt"] = ANALYZED + 1000
        (self.work(project) / "decisions.json").write_text(json.dumps(stored))
        render_json = self.work(project) / "renders" / "render-1" / "render.json"
        render_json.parent.mkdir(parents=True)
        render_json.write_text(json.dumps({"ok": True}))
        os.utime(render_json, (ANALYZED + 2000, ANALYZED + 2000))
        assert self.state(capsys, project) == "rendered (render-1)"
        page = self.page(project, {"pause-bbbb": {"decision": "keep"}}, based_on=ANALYZED - 100)
        code, data = run(capsys, "decide", "--import", str(page), "-p", str(project))
        assert (data["imported"], data["reviewed"]) == (1, False)
        assert self.state(capsys, project) == "reviewed, ready to render"

    def test_based_on_must_be_a_number(self, capsys, project):
        page = self.page(project, {}, based_on="yesterday")
        code, data = run(capsys, "decide", "--import", str(page), "-p", str(project))
        assert code == 2
        assert "basedOn must be a number" in data["error"]

    def test_the_review_page_gets_the_analysis_time(self, capsys, project, monkeypatch):
        self.analyzed(project)
        seen = []
        monkeypatch.setattr("rostrum.pipeline.ReviewHook.write", lambda hook, context: seen.append(context.analyzed_at))
        run(capsys, "decide", "retake-aaaa", "keep", "-p", str(project))
        assert seen == [ANALYZED]


class TestMapRenderNames:
    def test_a_render_folder_name_is_looked_up_per_talk(self, capsys, tmp_path):
        project = tmp_path / "rostrum.json"
        project.write_text(json.dumps({"talks": [{"id": "a", "sources": ["a.mp4"]}, {"id": "b", "sources": ["b.mp4"]}]}))
        for talk in ("a", "b"):
            (tmp_path / f"{talk}.rostrum" / "renders" / "render-1").mkdir(parents=True)
        code, data = run(capsys, "map", "1.0", "--from", "render-1", "-p", str(project))
        assert code != 0
        assert "render-1 exists for several talks (a, b); pass --talk" in data["error"]
        code, data = run(capsys, "map", "1.0", "--from", "render-1", "--talk", "a", "-p", str(project))
        assert "no cut map at" in data["error"]
        assert "a.rostrum" in data["error"]
        code, data = run(capsys, "map", "1.0", "--from", "render-9", "-p", str(project))
        assert "no render folder 'render-9'" in data["error"]


class TestRenderAndVerifyProjectMode:
    def test_render_needs_the_first_pass(self, capsys, tmp_path):
        (tmp_path / "talk.mp4").write_bytes(b"")
        project = tmp_path / "rostrum.json"
        project.write_text(json.dumps({"talks": [{"id": "talk", "sources": ["talk.mp4"]}]}), encoding="utf-8")
        code, data = run(capsys, "render", "-p", str(project))
        assert code == 1
        assert "run `rostrum init` first" in data["talks"][0]["error"]

    def test_verify_needs_a_render(self, capsys, project):
        code, data = run(capsys, "verify", "-p", str(project), "--talk", "talk")
        assert code == 1
        assert "no finished render for talk talk" in data["talks"][0]["error"]

    def test_without_a_project_the_file_options_are_suggested(self, capsys, tmp_path):
        code, data = run(capsys, "render", "-p", str(tmp_path / "none.json"))
        assert code == 2 and "--edl FILE" in data["error"]
        code, data = run(capsys, "verify", "-p", str(tmp_path / "none.json"))
        assert code == 2 and "--render DIR" in data["error"]

    def test_newest_finished_render_folder(self, tmp_path):
        from rostrum.render import RenderFolders

        for name in ("render-1", "render-2", "render-10", "render-11", "render-12", "notes"):
            (tmp_path / name).mkdir()
        for name in ("render-1", "render-2", "render-10", "notes"):
            (tmp_path / name / "render.json").write_text('{"ok": true}')
        (tmp_path / "render-11" / "render.json").write_text('{"ok": false, "error": "disk full"}')
        folders = RenderFolders({"runDir": "render-{n}"})
        newest, failed = folders.newest(tmp_path)
        assert (newest.name, [path.name for path in failed]) == ("render-10", ["render-11"])
        assert folders.newest(tmp_path / "missing") == (None, [])


class TestStatusConfigInitDoctor:
    def test_status_summary(self, capsys, project):
        code, data = run(capsys, "status", "-p", str(project))
        assert data["talks"][0]["state"] == "ready for review"
        run(capsys, "decide", "retake-aaaa", "keep", "-p", str(project))
        code, data = run(capsys, "status", "-p", str(project))
        talk = data["talks"][0]
        assert code == 0
        assert (talk["title"], talk["cuts"], talk["decisions"], talk["state"]) == (
            "A Talk", 1, 1, "reviewed, ready to render")
        assert talk["notes"] == []

    def test_bare_command_without_a_project(self, capsys, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert main([]) == 2
        assert "rostrum init" in capsys.readouterr().err

    def test_config_shows_values_and_origins(self, capsys, project):
        code, data = run(capsys, "config", "-p", str(project), "--key", "pauses")
        talk = data["talks"][0]
        assert code == 0
        assert talk["settings"]["keepSec"] == 0.4
        assert talk["origins"]["pauses.keepSec"] == "project"
        assert talk["origins"]["pauses.maxSec"] == "default"
        assert main(["config", "-p", str(project), "--key", "pauses.keepSec"]) == 0
        assert "pauses.keepSec = 0.4  (project)" in capsys.readouterr().out

    def test_init_errors(self, capsys, tmp_path):
        code, data = run(capsys, "init", "-p", str(tmp_path / "missing.json"))
        assert code == 2 and "project file not found" in data["error"]
        (tmp_path / "a.mp4").write_bytes(b"")
        code, data = run(capsys, "init", str(tmp_path / "a.mp4"), "--engine", "bogus", "-p", str(tmp_path / "p.json"))
        assert code == 2 and "available: faster-whisper, command, import" in data["error"]

    def test_doctor_reports_every_check(self, capsys):
        code, data = run(capsys, "doctor")
        names = [check["name"] for check in data["checks"]]
        assert code in (0, 1)
        assert names[0] == "python" and data["checks"][0]["status"] == "ok"
        assert {"ffmpeg", "ffprobe", "disk", "memory"} <= set(names)
        assert any(name.startswith("engine import") for name in names)
