"""Optional talk script. Sentences said on the same script line are retakes; a phrase the script repeats is not."""
import re
from dataclasses import dataclass
from pathlib import Path

from rostrum.config import ConfigError

SENTENCE_SPLIT = r"(?<=[.!?])\s+(?=[A-Z\u201c\"])"


@dataclass
class ScriptSentence:
    section: int
    text: str
    tokens: list
    content: set

    @property
    def padded(self):
        return " " + " ".join(self.tokens) + " "


class ScriptText:
    def __init__(self, script_settings, tokenizer, content_words):
        pattern = script_settings.get("sectionPattern")
        self.section = re.compile(pattern, re.M) if pattern else None
        self.spoken_heading = script_settings.get("spokenHeading") or ""
        self.skip = [re.compile(item) for item in script_settings.get("skipLines", [])]
        self.tokenizer = tokenizer
        self.content_words = content_words

    def load(self, path):
        if not path:
            return []
        file_path = Path(path)
        if not file_path.is_file():
            raise ConfigError(f"script not found: {file_path}")
        return self.sentences(file_path.read_text(encoding="utf-8-sig"))

    def sentences(self, text):
        found = []
        for number, body in self.sections(text):
            spoken = self.spoken_part(body)
            lines = [line.strip() for line in spoken.splitlines()]
            joined = " ".join(line for line in lines if self.is_spoken(line))
            for sentence in re.split(SENTENCE_SPLIT, joined):
                sentence = sentence.strip()
                if not sentence:
                    continue
                tokens = self.tokenizer.text_tokens(sentence)
                found.append(ScriptSentence(number, sentence, tokens, self.content_words.of(tokens)))
        return found

    def sections(self, text):
        if self.section is None:
            return [(0, text)]
        headings = list(self.section.finditer(text))
        sections = []
        for index, heading in enumerate(headings):
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            sections.append((self.section_number(heading, index), text[heading.end():end]))
        return sections or [(0, text)]

    def section_number(self, heading, index):
        if (heading.re.groups >= 1 and
                (heading.group(1) or "").isdigit()):
            return int(heading.group(1))
        return index + 1

    def spoken_part(self, body):
        if (self.spoken_heading and
                self.spoken_heading in body):
            return body.split(self.spoken_heading, 1)[1]
        return body

    def is_spoken(self, line):
        if not line:
            return False
        return not any(pattern.search(line) for pattern in self.skip)
