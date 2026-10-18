"""The project file (rostrum.json): an event or a single talk, and the talks' work folders."""
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rostrum.config import SCHEMA_URL, TALK_FIELDS, ConfigError, JsonFile, Schema

PROJECT_FILE = "rostrum.json"
RENDERS_DIR = "renders"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INPUT = 2


@dataclass
class Talk:
    id: str
    sources: list
    speaker: str = ""
    title: str = ""
    overrides: dict = field(default_factory=dict)

    @property
    def label(self):
        return self.title or self.id


class Project:
    def __init__(self, path, data):
        self.path = Path(path).resolve()
        self.data = data

    @property
    def dir(self):
        return self.path.parent

    @property
    def event(self):
        return self.data.get("event", "")

    @property
    def defaults(self):
        return self.data.get("defaults", {})

    def talks(self):
        return [self.talk_from(item) for item in self.data.get("talks", [])]

    def talk_from(self, item):
        overrides = {key: value for key, value in item.items() if key not in TALK_FIELDS}
        return Talk(
            id=item["id"],
            sources=list(item["sources"]),
            speaker=item.get("speaker", ""),
            title=item.get("title", ""),
            overrides=overrides,
        )

    def talk(self, talk_id):
        for talk in self.talks():
            if talk.id == talk_id:
                return talk
        known = ", ".join(talk.id for talk in self.talks()) or "none"
        raise ConfigError(f"no talk {talk_id!r} in {self.path} (talks: {known})")

    def select(self, talk_id=None):
        return [self.talk(talk_id)] if talk_id else self.talks()

    def resolve(self, value):
        path = Path(os.path.expanduser(str(value)))
        return path if path.is_absolute() else (self.dir / path).resolve()

    def source_paths(self, talk):
        return [self.resolve(source) for source in talk.sources]

    def relative(self, path):
        absolute = Path(path).resolve()
        try:
            return Path(os.path.relpath(absolute, self.dir)).as_posix()
        except ValueError:
            return str(absolute)


class ProjectFile:
    def __init__(self, schema=None):
        self.schema = schema or Schema()

    def load(self, path):
        file_path = Path(path)
        if not file_path.is_file():
            raise ConfigError(f"project file not found: {file_path}. Run `rostrum init <files or folder>` first.")
        data = JsonFile().read(file_path)
        if not isinstance(data, dict):
            raise ConfigError(f"{file_path}: expected a JSON object")
        self.schema.check_project(data, str(file_path))
        self.check_unique_ids(data, file_path)
        return Project(file_path, data)

    def check_unique_ids(self, data, file_path):
        seen = set()
        for index, talk in enumerate(data.get("talks", [])):
            if talk["id"] in seen:
                raise ConfigError(f"{file_path}: talks[{index}].id: duplicate talk id {talk['id']!r}")
            seen.add(talk["id"])

    def new_data(self, talks, event=None, defaults=None):
        data = {"$schema": SCHEMA_URL}
        if event:
            data["event"] = event
        data["defaults"] = defaults or {}
        data["talks"] = talks
        return data

    def save(self, path, data):
        """Writes the project; an existing file is kept as <name>.bak first."""
        file_path = Path(path)
        self.schema.check_project(data, str(file_path))
        backup = None
        if file_path.exists():
            backup = file_path.with_name(file_path.name + ".bak")
            shutil.copy2(file_path, backup)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return backup


class TalkGrouper:
    """Turns media files into talks. Files whose stems match init.partPattern (`talk-part1`, `talk-part2`) are parts
    of one talk, in part order, when they share a folder, name and extension. Every other file is a talk of its
    own, so rec.mkv next to rec.mp4 never becomes two parts of one talk."""

    def __init__(self, init_settings):
        self.part_pattern = re.compile(init_settings["partPattern"])
        self.extensions = {extension.lower() for extension in init_settings["mediaExtensions"]}

    def media_files(self, paths):
        files = []
        for path in paths:
            item = Path(path)
            if item.is_dir():
                files.extend(sorted(child for child in item.iterdir() if self.is_media(child)))
            elif item.is_file():
                files.append(item)
            else:
                raise ConfigError(f"not found: {item}")
        return files

    def is_media(self, path):
        return (path.is_file() and
                path.suffix.lower() in self.extensions)

    def group(self, files):
        whole_stems = Counter(self.stem_key(path) for path in files if self.split(path.stem)[1] is None)
        groups = {}
        order = []
        for file_path in files:
            key, name, part = self.place(file_path, whole_stems)
            if key not in groups:
                groups[key] = (name, [])
                order.append(key)
            groups[key][1].append((part, file_path))
        talks = []
        used = set()
        for key in order:
            name, parts = groups[key]
            parts.sort(key=lambda item: item[0])
            talk_id = self.unique(self.slug(name), used)
            used.add(talk_id)
            talks.append((talk_id, [path for _, path in parts]))
        return talks

    def place(self, file_path, whole_stems):
        """(group key, talk name, part number) for one file."""
        folder = str(file_path.parent.resolve())
        name, part = self.split(file_path.stem)
        if part is not None:
            return (folder, name.lower(), file_path.suffix.lower()), name, part
        name = file_path.stem
        if whole_stems[self.stem_key(file_path)] > 1:
            name = f"{name}-{file_path.suffix.lstrip('.')}"
        return (folder, file_path.name.lower()), name, 0

    def stem_key(self, path):
        return str(path.parent.resolve()), path.stem.lower()

    def split(self, stem):
        """(talk name, part number), or (stem, None) when the stem is not a part."""
        match = self.part_pattern.search(stem)
        if match is None:
            return stem, None
        number = match.group(1) or ""
        if not number.isdecimal():
            raise ConfigError(f"init.partPattern: group 1 must capture the part number, but in {stem!r} it "
                              f"captured {number!r}")
        name = (stem[:match.start()] + stem[match.end():]).strip(" -_") or stem
        return name, int(number)

    def slug(self, name):
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._").lower()
        return slug or "talk"

    def unique(self, talk_id, used):
        candidate = talk_id
        number = 2
        while candidate in used:
            candidate = f"{talk_id}-{number}"
            number += 1
        return candidate


class TalkPaths:
    """Files in a talk's work folder."""

    def __init__(self, project, talk, settings):
        self.root = project.resolve(str(settings["workDir"]).replace("{talk}", talk.id))

    @property
    def probe(self):
        return self.root / "probe.json"

    @property
    def energy(self):
        return self.root / "energy.npz"

    def transcript(self, index):
        return self.root / f"transcript-{index + 1}.json"

    @property
    def detect(self):
        return self.root / "detect.json"

    @property
    def reconcile(self):
        return self.root / "reconcile.json"

    @property
    def candidates(self):
        return self.root / "candidates.json"

    @property
    def decisions(self):
        return self.root / "decisions.json"

    @property
    def edl(self):
        return self.root / "edl.json"

    @property
    def review(self):
        return self.root / "review.html"

    @property
    def renders(self):
        return self.root / RENDERS_DIR

    @property
    def scratch(self):
        return self.root / "scratch"
