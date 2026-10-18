"""Layered settings. Later layers win: built-in defaults, ~/.rostrum/config.json, project defaults, talk, CLI flags.

The JSON schema is the single source of truth for keys, defaults and descriptions.
"""
import copy
import json
import os
import re
from importlib import resources
from pathlib import Path

import jsonschema
from jsonschema.exceptions import best_match

SCHEMA_URL = "https://raw.githubusercontent.com/girishkvs/rostrum/main/src/rostrum/schema/rostrum.schema.json"
SCHEMA_RESOURCE = "schema/rostrum.schema.json"
LAYERS = ("default", "user", "project", "talk", "flag")
TALK_FIELDS = ("id", "sources", "speaker", "title")
# Cue settings compared with single transcript tokens.
CUE_WORD_LISTS = ("words", "endWords", "companionWords", "trailingWords")
MISSING = object()
# Renderer options that take null to leave the ffmpeg option out (or fall back to a built-in value).
NULLABLE_RENDER_KEYS = frozenset(("crf", "preset", "profile", "scThreshold", "audioBitrate", "runDir"))
PASS_THROUGH_DESCRIPTIONS = {
    "crf": "x264 quality (lower is better); null leaves it to the encoder.",
    "preset": "x264 speed preset.",
    "audioFadeMs": "Audio fade at every join.",
    "audioBitrate": "Audio bitrate.",
    "audioRate": "Audio sample rate of the output.",
    "loudnessLufs": "EBU R128 loudness target; null leaves loudness alone.",
    "name": "Output name template: {talk}, {speaker}, {title}, {event}, {date}.",
    "container": "Output container: mp4, mkv or mov.",
    "normalizeVfr": "Convert variable frame rate sources to constant frame rate first.",
    "mode": "fast checks timestamps at every join and samples A/V offset; full also checks A/V offset at every join and writes before/after PNGs.",
}


class ConfigError(ValueError):
    pass


class JsonPath:
    def text(self, parts):
        text = ""
        for part in parts:
            if isinstance(part, int):
                text += f"[{part}]"
            else:
                text += f".{part}" if text else str(part)
        return text or "(top level)"


class Schema:
    def __init__(self):
        text = resources.files("rostrum").joinpath(SCHEMA_RESOURCE).read_text(encoding="utf-8")
        self.document = json.loads(text)
        self.document["$defs"]["settings"]["properties"].update(PassThroughSections().sections())
        self.project_validator = jsonschema.Draft202012Validator(self.document)
        settings_schema = {"$ref": "#/$defs/settingsFile", "$defs": self.document["$defs"]}
        self.settings_validator = jsonschema.Draft202012Validator(settings_schema)

    def defaults(self):
        return SchemaDefaults(self.document).settings()

    def check_project(self, data, where):
        self.raise_errors(self.project_validator.iter_errors(data), where)

    def check_settings(self, data, where):
        self.raise_errors(self.settings_validator.iter_errors(data), where)

    def raise_errors(self, errors, where):
        problems = sorted(errors, key=lambda error: [str(part) for part in error.absolute_path])
        if not problems:
            return
        lines = [f"{where}: {JsonPath().text(error.absolute_path)}: {self.message(error)}" for error in problems]
        raise ConfigError("\n".join(lines))

    def message(self, error):
        if error.context:
            deepest = best_match(error.context)
            return f"{deepest.message} (at {JsonPath().text(deepest.absolute_path)})"
        return error.message


class PassThroughSections:
    """The `render` and `verify` settings, generated from the renderer's and verifier's own defaults so the schema
    cannot drift from them. Run `python -m rostrum.config` to write them into the published schema file."""

    def sections(self):
        from rostrum.render import RENDER_DEFAULTS
        from rostrum.verify import VERIFY_DEFAULTS

        return {
            "render": self.section(RENDER_DEFAULTS, "Render settings (rostrum render)."),
            "verify": self.section(VERIFY_DEFAULTS, "Render checks (rostrum verify)."),
        }

    def section(self, defaults, description):
        return {"type": "object", "description": description,
                "properties": {key: self.property(key, value) for key, value in defaults.items()},
                "additionalProperties": False}

    def property(self, key, value):
        schema = {"type": self.json_type(key, value), "default": copy.deepcopy(value)}
        if isinstance(value, list):
            schema["items"] = {"type": "string"}
        if key in PASS_THROUGH_DESCRIPTIONS:
            schema["description"] = PASS_THROUGH_DESCRIPTIONS[key]
        return schema

    def json_type(self, key, value):
        if value is None:
            return ["string", "number", "null"]
        if isinstance(value, bool):
            kind = "boolean"
        elif isinstance(value, (int, float)):
            kind = "number"
        elif isinstance(value, str):
            kind = "string"
        elif isinstance(value, list):
            kind = "array"
        else:
            kind = "object"
        return [kind, "null"] if key in NULLABLE_RENDER_KEYS else kind


class SchemaFile:
    """Rewrites the published schema with the generated render and verify sections."""

    def path(self):
        return Path(str(resources.files("rostrum").joinpath(SCHEMA_RESOURCE)))

    def write(self):
        path = self.path()
        document = json.loads(path.read_text(encoding="utf-8"))
        document["$defs"]["settings"]["properties"].update(PassThroughSections().sections())
        path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path


class SchemaDefaults:
    """Builds the default settings object from the `default` values in the schema."""

    def __init__(self, document):
        self.document = document

    def settings(self):
        return self.value({"$ref": "#/$defs/settings"})

    def value(self, schema):
        if "default" in schema:
            return copy.deepcopy(schema["default"])
        target = self.resolve(schema)
        if "default" in target:
            return copy.deepcopy(target["default"])
        if "properties" in target:
            return self.object_defaults(target)
        for branch in target.get("oneOf", []) + target.get("anyOf", []):
            resolved = self.resolve(branch)
            if "properties" in resolved:
                return self.object_defaults(resolved)
        return MISSING

    def object_defaults(self, schema):
        values = {}
        for key, child in schema["properties"].items():
            value = self.value(child)
            if value is not MISSING:
                values[key] = value
        return values

    def resolve(self, schema):
        while "$ref" in schema:
            node = self.document
            for part in schema["$ref"].lstrip("#/").split("/"):
                node = node[part]
            schema = node
        return schema


class UserHome:
    """~/.rostrum, or ROSTRUM_HOME when set (tests and portable installs)."""

    def path(self):
        configured = os.environ.get("ROSTRUM_HOME")
        return Path(configured) if configured else Path.home() / ".rostrum"


class JsonFile:
    def read(self, path):
        file_path = Path(path)
        try:
            return json.loads(file_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as error:
            raise ConfigError(f"{file_path}: line {error.lineno} column {error.colno}: {error.msg}") from error


class SettingsShape:
    """`script` may be given as a plain path; everything downstream sees {"path": ...}."""

    def normalize(self, data):
        if not isinstance(data, dict):
            return data
        result = dict(data)
        if isinstance(result.get("script"), str):
            result["script"] = {"path": result["script"]}
        return result


class EffectiveConfig:
    def __init__(self, values, origins):
        self.values = values
        self.origins = origins

    def __getitem__(self, key):
        return self.values[key]

    def get(self, dotted, default=None):
        node = self.values
        for part in dotted.split("."):
            if (not isinstance(node, dict) or
                    part not in node):
                return default
            node = node[part]
        return node

    def origin(self, dotted):
        return self.origins.get(dotted, "default")

    def leaves(self):
        """(dotted key, value, origin) for every setting, in a stable order."""
        found = []
        self.collect(self.values, "", found)
        return found

    def collect(self, node, prefix, found):
        for key in sorted(node):
            value = node[key]
            dotted = f"{prefix}{key}"
            if (isinstance(value, dict) and
                    value):
                self.collect(value, dotted + ".", found)
            else:
                found.append((dotted, value, self.origin(dotted)))


class Layers:
    def __init__(self):
        self.values = {}
        self.origins = {}

    def add(self, name, data):
        self.merge(self.values, SettingsShape().normalize(data or {}), name, "")

    def merge(self, target, data, name, prefix):
        for key, value in data.items():
            dotted = f"{prefix}{key}"
            both_objects = (isinstance(value, dict) and
                            isinstance(target.get(key), dict))
            if both_objects:
                self.merge(target[key], value, name, dotted + ".")
                continue
            target[key] = copy.deepcopy(value)
            self.mark(dotted, value, name)

    def mark(self, dotted, value, name):
        if isinstance(value, dict):
            for key, child in value.items():
                self.mark(f"{dotted}.{key}", child, name)
            if not value:
                self.origins[dotted] = name
            return
        self.origins[dotted] = name


class SettingsRules:
    """Checks the schema cannot express, run on the merged settings.

    Cue word lists are compared with single transcript tokens, so each entry is stored the way a token looks (lower
    case, no punctuation) and a phrase is an error: phrases belong in cues.patterns. init.partPattern must compile
    and have a capture group for the part number."""

    def apply(self, values, origins):
        self.cue_words(values, origins)
        self.part_pattern(values, origins)

    def cue_words(self, values, origins):
        from rostrum.detect.text import Tokenizer

        tokenizer = Tokenizer()
        cues = values.get("cues", {})
        for key in CUE_WORD_LISTS:
            words = []
            for entry in cues.get(key, []):
                tokens = tokenizer.tokens(str(entry))
                if len(tokens) != 1:
                    where = self.where(origins, f"cues.{key}")
                    raise ConfigError(f"cues.{key}: {entry!r} is not a single word{where}; "
                                      "put phrases in cues.patterns")
                words.append(tokens[0])
            cues[key] = words

    def part_pattern(self, values, origins):
        text = values.get("init", {}).get("partPattern")
        if text is None:
            return
        where = self.where(origins, "init.partPattern")
        try:
            pattern = re.compile(text)
        except re.error as error:
            raise ConfigError(f"init.partPattern: not a valid regular expression{where}: {error}") from error
        if pattern.groups < 1:
            raise ConfigError(f"init.partPattern: needs a capture group for the part number{where}, "
                              "for example (?i)[-_ ]part\\s*(\\d+)$")

    def where(self, origins, dotted):
        origin = origins.get(dotted, "default")
        return "" if origin == "default" else f" (from the {origin} settings)"


class ConfigLoader:
    def __init__(self, home=None, schema=None):
        self.home = Path(home) if home else UserHome().path()
        self.schema = schema or Schema()
        self.user_cache = None

    def user_path(self):
        return self.home / "config.json"

    def user_settings(self):
        if self.user_cache is not None:
            return self.user_cache
        path = self.user_path()
        data = {}
        if path.is_file():
            data = JsonFile().read(path)
            if not isinstance(data, dict):
                raise ConfigError(f"{path}: expected a JSON object")
            data.pop("$schema", None)
            data.pop("$comment", None)
            self.schema.check_settings(data, str(path))
        self.user_cache = data
        return data

    def effective(self, project=None, talk=None, flags=None):
        layers = Layers()
        layers.add("default", self.schema.defaults())
        layers.add("user", self.user_settings())
        if project is not None:
            layers.add("project", project.defaults)
        if talk is not None:
            layers.add("talk", talk.overrides)
        if flags:
            layers.add("flag", flags)
        self.schema.check_settings(layers.values, "effective settings")
        SettingsRules().apply(layers.values, layers.origins)
        return EffectiveConfig(layers.values, layers.origins)


if __name__ == "__main__":
    print(SchemaFile().write())
