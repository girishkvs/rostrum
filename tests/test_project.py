import json

import pytest

from rostrum.config import ConfigError
from rostrum.project import Project, ProjectFile, TalkGrouper, TalkPaths


class TestTalkGrouper:
    def test_one_talk_per_file_and_parts_joined_in_part_order(self, settings, tmp_path):
        for name in ("intro.mp4", "deep-dive-part2.mp4", "deep-dive-part1.mp4", "notes.txt", "Deep-Dive_Part10.mp4"):
            (tmp_path / name).write_bytes(b"")
        grouper = TalkGrouper(settings["init"])
        talks = grouper.group(grouper.media_files([tmp_path]))
        assert [(talk_id, [path.name for path in files]) for talk_id, files in talks] == [
            ("deep-dive", ["deep-dive-part1.mp4", "deep-dive-part2.mp4", "Deep-Dive_Part10.mp4"]),
            ("intro", ["intro.mp4"]),
        ]

    def test_same_name_in_two_folders_gets_unique_ids(self, settings, tmp_path):
        for folder in ("day1", "day2"):
            (tmp_path / folder).mkdir()
            (tmp_path / folder / "My Talk.mov").write_bytes(b"")
        grouper = TalkGrouper(settings["init"])
        talks = grouper.group(grouper.media_files([tmp_path / "day1", tmp_path / "day2"]))
        assert [talk_id for talk_id, _ in talks] == ["my-talk", "my-talk-2"]

    def test_part_pattern_is_configurable(self, settings, tmp_path):
        for name in ("talk.take2.mp4", "talk.take1.mp4"):
            (tmp_path / name).write_bytes(b"")
        settings["init"]["partPattern"] = r"\.take(\d+)$"
        grouper = TalkGrouper(settings["init"])
        talks = grouper.group(grouper.media_files([tmp_path]))
        assert [(talk_id, [path.name for path in files]) for talk_id, files in talks] == [
            ("talk", ["talk.take1.mp4", "talk.take2.mp4"])]

    def test_only_part_files_merge_and_same_stem_files_stay_apart(self, settings, tmp_path):
        for name in ("rec.mkv", "rec.mp4", "demo-part2.mp4", "demo-part1.mp4", "demo-part1.mkv", "talk.wav"):
            (tmp_path / name).write_bytes(b"")
        grouper = TalkGrouper(settings["init"])
        talks = grouper.group(grouper.media_files([tmp_path]))
        assert [(talk_id, [path.name for path in files]) for talk_id, files in talks] == [
            ("demo", ["demo-part1.mkv"]),
            ("demo-2", ["demo-part1.mp4", "demo-part2.mp4"]),
            ("rec-mkv", ["rec.mkv"]),
            ("rec-mp4", ["rec.mp4"]),
        ]

    def test_a_part_number_must_be_digits(self, settings, tmp_path):
        settings["init"]["partPattern"] = r"-part(\w+)$"
        with pytest.raises(ConfigError, match="group 1 must capture the part number"):
            TalkGrouper(settings["init"]).group([tmp_path / "demo-partA.mp4"])

    def test_missing_path_is_an_error(self, settings, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            TalkGrouper(settings["init"]).media_files([tmp_path / "nope.mp4"])


class TestProjectFile:
    def test_save_keeps_a_backup_and_validates(self, tmp_path):
        files = ProjectFile()
        path = tmp_path / "rostrum.json"
        data = files.new_data([{"id": "a", "sources": ["a.mp4"]}], event="Demo Day")
        assert files.save(path, data) is None
        backup = files.save(path, data)
        assert backup == tmp_path / "rostrum.json.bak"
        project = files.load(path)
        assert project.event == "Demo Day"
        assert project.talk("a").sources == ["a.mp4"]
        with pytest.raises(ConfigError, match=r"talks\[0\]\.pauses\.maxSec"):
            files.save(path, {"talks": [{"id": "a", "sources": ["a.mp4"], "pauses": {"maxSec": "long"}}]})

    def test_duplicate_ids_and_unknown_talks_are_errors(self, tmp_path):
        path = tmp_path / "rostrum.json"
        path.write_text(json.dumps({"talks": [{"id": "a", "sources": ["a.mp4"]}, {"id": "a", "sources": ["b.mp4"]}]}))
        with pytest.raises(ConfigError, match="duplicate talk id"):
            ProjectFile().load(path)
        path.write_text(json.dumps({"talks": [{"id": "a", "sources": ["a.mp4"]}]}))
        with pytest.raises(ConfigError, match="no talk 'b'"):
            ProjectFile().load(path).talk("b")

    def test_paths_resolve_against_the_project_folder(self, settings, tmp_path):
        project = Project(tmp_path / "rostrum.json", {"talks": [{"id": "a", "sources": ["media/a.mp4"]}]})
        talk = project.talk("a")
        assert project.source_paths(talk) == [(tmp_path / "media" / "a.mp4").resolve()]
        assert project.relative(tmp_path / "media" / "a.mp4") == "media/a.mp4"
        settings["workDir"] = "work/{talk}"
        assert TalkPaths(project, talk, settings).edl == (tmp_path / "work" / "a" / "edl.json").resolve()
