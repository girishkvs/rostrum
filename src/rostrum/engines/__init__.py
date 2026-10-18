"""Pluggable speech-to-text. See base.SpeechEngine for the contract and registry.EngineRegistry for lookup."""
from rostrum.engines.base import EngineError, SourceInput, SpeechEngine, Transcript
from rostrum.engines.registry import EngineRegistry, UnknownEngine

__all__ = ["EngineError", "EngineRegistry", "SourceInput", "SpeechEngine", "Transcript", "UnknownEngine"]
