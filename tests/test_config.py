import json
from pathlib import Path

import pytest

from rostrum.config import ConfigError, ConfigLoader, PassThroughSections, Schema, SchemaFile
from rostrum.project import ProjectFile
from rostrum.render import RENDER_DEFAULTS
from rostrum.verify import VERIFY_DEFAULTS

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "rostrum.example.json"


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestDefaults:
    def test_baseline_values_are_the_defaults(self, settings):
        assert settings["pauses"] == {"level": "auto", "maxSec": 2.0, "keepSec": 0.5, "islandMarginSec": 0.2,
                                      "islandMinSec": 0.1, "islandRemovedShare": 0.5}
        assert (settings["leads"]["inSec"], settings["leads"]["outSec"]) == (1.0, 1.5)
        assert settings["retakes"]["windowSec"] == 180
        assert (settings["retakes"]["substitutionLevel"], settings["retakes"]["substitutionMaxWords"]) == ("propose", 2)
        assert settings["speech"]["workers"] == "auto"
        assert settings["speech"]["cuePass"] is True
        assert (settings["speech"]["cueScope"], settings["speech"]["cueFallback"]) == ("full", False)
        assert settings["speech"]["threadsPerWorker"] == "auto"
        assert settings["fillers"]["level"] == "off"
        assert (settings["unclear"]["level"], settings["unclear"]["maxWordSec"], settings["unclear"]["minProbability"],
                settings["unclear"]["minSec"], settings["unclear"]["maxAvgLogprob"]) == ("propose", 2.0, 0.35, 1.0, -0.8)
        assert settings["workDir"] == "{talk}.rostrum"
        assert settings["script"]["path"] is None
        assert settings["slides"]["timeline"] is None
        assert settings["speech"]["hints"] == []

    def test_render_and_verify_follow_their_modules(self, settings):
        assert settings["render"] == RENDER_DEFAULTS
        assert settings["verify"] == VERIFY_DEFAULTS

    def test_published_schema_file_is_in_sync(self):
        document = json.loads(SchemaFile().path().read_text(encoding="utf-8"))
        published = document["$defs"]["settings"]["properties"]
        generated = PassThroughSections().sections()
        assert published["render"] == generated["render"], "run: python -m rostrum.config"
        assert published["verify"] == generated["verify"], "run: python -m rostrum.config"

    def test_every_setting_has_a_default(self):
        schema = Schema()
        missing = []
        self.walk(schema, schema.document["$defs"]["settings"], "", missing)
        assert missing == []

    def walk(self, schema, node, prefix, missing):
        for key, child in node.get("properties", {}).items():
            resolved = self.resolve(schema, child)
            dotted = f"{prefix}{key}"
            has_default = ("default" in child or
                           "default" in resolved)
            if has_default:
                continue
            branches = [self.resolve(schema, branch) for branch in resolved.get("oneOf", [])]
            objects = [branch for branch in [resolved, *branches] if "properties" in branch]
            if objects:
                self.walk(schema, objects[0], dotted + ".", missing)
            else:
                missing.append(dotted)

    def resolve(self, schema, node):
        while "$ref" in node:
            target = schema.document
            for part in node["$ref"].lstrip("#/").split("/"):
                target = target[part]
            node = target
        return node


class TestLayers:
    def test_later_layers_win_and_origins_are_reported(self, isolated_home, tmp_path):
        write(isolated_home / "config.json", {"pauses": {"maxSec": 3.0}, "speech": {"workers": 3}})
        project_path = write(tmp_path / "rostrum.json", {
            "defaults": {"pauses": {"keepSec": 0.4}, "speech": {"workers": 4}},
            "talks": [{"id": "a", "sources": ["a.mp4"], "pauses": {"maxSec": 4.0}, "script": "notes.md"}],
        })
        project = ProjectFile().load(project_path)
        config = ConfigLoader().effective(project, project.talk("a"), {"speech": {"engine": "import"}})
        expected = {
            "pauses.maxSec": (4.0, "talk"),
            "pauses.keepSec": (0.4, "project"),
            "pauses.level": ("auto", "default"),
            "speech.workers": (4, "project"),
            "speech.engine": ("import", "flag"),
            "script.path": ("notes.md", "talk"),
            "script.spokenHeading": ("### Say", "default"),
        }
        assert {key: (config.get(key), config.origin(key)) for key in expected} == expected
        assert ConfigLoader().effective().origin("pauses.maxSec") == "user"
        leaves = {key: origin for key, _, origin in config.leaves()}
        assert leaves["speech.engine"] == "flag"

    def test_errors_name_the_file_and_the_key(self, isolated_home):
        write(isolated_home / "config.json", {"pauses": {"maxSec": "long"}})
        with pytest.raises(ConfigError) as error:
            ConfigLoader().effective()
        message = str(error.value)
        assert "config.json: pauses.maxSec:" in message
        assert "'long' is not of type 'number'" in message

    def test_unknown_keys_and_bad_values_are_rejected(self, isolated_home):
        write(isolated_home / "config.json", {"pause": {"maxSec": 1}, "cues": {"level": "sometimes"}})
        with pytest.raises(ConfigError) as error:
            ConfigLoader().effective()
        message = str(error.value)
        assert "'pause' was unexpected" in message
        assert "cues.level: 'sometimes' is not one of ['auto', 'propose', 'off']" in message

    def test_broken_json_reports_line_and_column(self, isolated_home):
        (isolated_home / "config.json").write_text('{"pauses": {"maxSec": 2,}}', encoding="utf-8")
        with pytest.raises(ConfigError, match="line 1 column"):
            ConfigLoader().effective()

    def test_flags_are_validated_too(self):
        with pytest.raises(ConfigError, match="speech.workers"):
            ConfigLoader().effective(flags={"speech": {"workers": 0}})


class TestRules:
    def test_cue_words_are_stored_like_transcript_tokens(self):
        config = ConfigLoader().effective(flags={"cues": {"words": ["Stop.", "REDO"], "endWords": ["Okay!"]}})
        assert (config.get("cues.words"), config.get("cues.endWords")) == (["stop", "redo"], ["okay"])

    def test_a_cue_phrase_is_an_error(self, isolated_home):
        write(isolated_home / "config.json", {"cues": {"words": ["stop", "scratch that"]}})
        with pytest.raises(ConfigError) as error:
            ConfigLoader().effective()
        message = str(error.value)
        assert "cues.words: 'scratch that' is not a single word (from the user settings)" in message
        assert "cues.patterns" in message

    def test_part_pattern_needs_a_capture_group(self):
        with pytest.raises(ConfigError, match="init.partPattern: needs a capture group for the part number"):
            ConfigLoader().effective(flags={"init": {"partPattern": r"-part\d+$"}})
        with pytest.raises(ConfigError, match="init.partPattern: not a valid regular expression"):
            ConfigLoader().effective(flags={"init": {"partPattern": r"-part(\d+$"}})


class TestExample:
    def test_example_project_is_valid(self):
        project = ProjectFile().load(EXAMPLE)
        assert [talk.id for talk in project.talks()] == ["keynote", "talk-2", "talk-3"]
        config = ConfigLoader().effective(project, project.talk("talk-3"))
        assert config.get("pauses.maxSec") == 3.0
        assert config.get("speech.engine") == "import"
        assert config.get("render.loudnessLufs") == -16
