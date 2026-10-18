"""Speech engines by name: the built-ins plus plugins from the `rostrum.engines` entry-point group."""
import importlib
from importlib.metadata import entry_points

from rostrum.engines.base import EngineError

ENTRY_POINT_GROUP = "rostrum.engines"
BUILTIN = {
    "faster-whisper": ("rostrum.engines.faster_whisper", "FasterWhisperEngine"),
    "command": ("rostrum.engines.command", "CommandEngine"),
    "import": ("rostrum.engines.importer", "ImportEngine"),
}


class UnknownEngine(EngineError):
    pass


class EngineRegistry:
    def plugins(self):
        return {entry.name: entry for entry in entry_points(group=ENTRY_POINT_GROUP) if entry.name not in BUILTIN}

    def names(self):
        return list(BUILTIN) + sorted(self.plugins())

    def engine_class(self, name):
        if name in BUILTIN:
            module_name, class_name = BUILTIN[name]
            return getattr(importlib.import_module(module_name), class_name)
        plugins = self.plugins()
        if name in plugins:
            return plugins[name].load()
        raise UnknownEngine(f"unknown speech engine {name!r}; available: {', '.join(self.names())}")

    def create(self, name, settings, tools, report=None):
        return self.engine_class(name)(settings, tools, report)
