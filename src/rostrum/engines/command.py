"""command engine: run any program that prints words JSON, SRT or VTT (whisper.cpp, Parakeet, a vendor CLI, ...)."""
import os
import shutil
import subprocess
from pathlib import Path

from rostrum.audio import AudioDecoder
from rostrum.engines.base import EngineError, SpeechEngine
from rostrum.engines.formats import TranscriptParser

OUTPUT_SUFFIXES = ("", ".json", ".srt", ".vtt", ".txt")
ERROR_TAIL = 800
BATCH_SUFFIXES = (".bat", ".cmd")
CMD_SHELLS = ("cmd", "cmd.exe")
# cmd.exe reads these in the arguments of a batch file as commands, even inside quotes.
CMD_SPECIAL = frozenset('&^%!<>|"')


class BatchGuard:
    """On Windows a .bat or .cmd program, or cmd itself, runs the command line through cmd.exe, which reads & | %
    and the like in its arguments as commands, so a file named "talk&calc&.mp4" could run calc. Such a program only
    gets plain arguments."""

    def check(self, arguments):
        if os.name != "nt":
            return
        program = Path(shutil.which(arguments[0]) or arguments[0])
        is_batch = program.suffix.lower() in BATCH_SUFFIXES
        is_shell = program.name.lower() in CMD_SHELLS
        if not (is_batch or
                is_shell):
            return
        kind = "a batch file" if is_batch else "the Windows command shell"
        for argument in arguments:
            special = sorted(set(argument) & CMD_SPECIAL)
            if special:
                raise EngineError(
                    f"speech.command runs {program.name}, {kind}, and the argument {argument!r} has "
                    f"{' '.join(special)}, which cmd.exe would run as commands. Rename that file or folder, or call "
                    "the program directly, without cmd or a batch file.")


class CommandEngine(SpeechEngine):
    name = "command"

    def transcribe(self, source):
        command = list(self.speech["command"])
        if not command:
            raise EngineError("speech.command is empty: give the program and its arguments, for example "
                              "[\"my-stt\", \"--json\", \"{audio}\"]")
        source.scratch.mkdir(parents=True, exist_ok=True)
        audio = source.scratch / f"audio16k-{source.index + 1}.wav"
        output = source.scratch / f"command-output-{source.index + 1}"
        uses_audio = any("{audio}" in argument for argument in command)
        uses_output = any("{output}" in argument for argument in command)
        self.clear(output)
        try:
            values = {"{audio}": str(audio), "{input}": str(source.path), "{output}": str(output),
                      "{language}": self.speech.get("language") or "", "{model}": self.speech.get("model") or ""}
            arguments = [self.fill(argument, values) for argument in command]
            BatchGuard().check(arguments)
            if uses_audio:
                AudioDecoder(self.tools).write_wav(source.path, audio, source.audio_offset)
            self.report(f"  running {Path(arguments[0]).name} on {Path(source.path).name}")
            text = self.run(arguments, output if uses_output else None)
        finally:
            audio.unlink(missing_ok=True)
        transcript = TranscriptParser(self.speech).parse(text, Path(arguments[0]).name)
        transcript.meta.update({"engine": self.name, "program": Path(arguments[0]).name})
        self.clear(output)
        return transcript

    def fill(self, argument, values):
        for key, value in values.items():
            argument = argument.replace(key, value)
        return argument

    def run(self, arguments, output):
        timeout = self.speech.get("timeoutSec")
        try:
            result = subprocess.run(arguments, capture_output=True, timeout=timeout)
        except FileNotFoundError as error:
            raise EngineError(f"speech.command program not found: {arguments[0]}") from error
        except subprocess.TimeoutExpired as error:
            raise EngineError(f"speech.command ran longer than {timeout} s") from error
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", "replace").strip()[-ERROR_TAIL:]
            raise EngineError(f"speech.command failed with exit code {result.returncode}: {message}")
        if output is not None:
            written = self.written(output)
            if written is not None:
                return written.read_text(encoding="utf-8-sig")
        text = result.stdout.decode("utf-8", "replace")
        if not text.strip():
            raise EngineError("speech.command printed nothing and wrote no {output} file")
        return text

    def written(self, output):
        """The {output} file, or {output} plus the extension some tools add."""
        for suffix in OUTPUT_SUFFIXES:
            candidate = Path(str(output) + suffix)
            if candidate.is_file():
                return candidate
        return None

    def clear(self, output):
        for suffix in OUTPUT_SUFFIXES:
            Path(str(output) + suffix).unlink(missing_ok=True)

    def check(self):
        command = self.speech.get("command") or []
        if not command:
            return {"ok": False, "detail": "speech.command is not set"}
        program = command[0]
        found = shutil.which(program) or (Path(program).is_file() and program)
        if not found:
            return {"ok": False, "detail": f"{program} not found"}
        return {"ok": True, "detail": str(found)}

    def cache_inputs(self, source):
        return {"command": list(self.speech.get("command") or [])}
