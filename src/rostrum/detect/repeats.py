"""Repeats: a phrase restarted inside one sentence ("we shipped it on, shipped it on Friday")."""


class RepeatDetector:
    def __init__(self, analysis):
        self.analysis = analysis
        self.settings = analysis.settings["repeats"]

    def find(self):
        found = []
        for unit in self.analysis.units:
            found.extend(self.in_unit(unit))
        return found

    def in_unit(self, unit):
        analysis = self.analysis
        tokens = unit.tokens
        size = self.settings["ngram"]
        found = []
        position = size
        last = 0
        while position <= len(tokens) - size:
            start = self.restart_at(tokens, position, last)
            if start is None:
                position += 1
                continue
            first_word = unit.token_words[start]
            end_word = unit.token_words[position]
            note = (f"repeat: \u201c{analysis.text_span(first_word, end_word)}\u201d \u2192 "
                    f"\u201c{analysis.text_span(end_word, end_word + 9)}\u201d")
            candidate = analysis.content_cut(first_word, end_word, "repeat", "repeats", "high", note)
            if candidate:
                found.append(candidate)
            last = position
            position += size
        return found

    def restart_at(self, tokens, position, last):
        """Where the phrase restarted at `position` first began, if everything dropped is said again."""
        size = self.settings["ngram"]
        first = max(last, position - self.settings["lookbackWords"])
        for start in range(first, position - size + 1):
            if tokens[start:start + size] != tokens[position:position + size]:
                continue
            said_after = self.analysis.content_words.of(tokens[position:position + (position - start) + self.settings["extraWords"]])
            dropped = self.analysis.content_words.of(tokens[start:position])
            missing = [word for word in dropped if not self.covered(word, said_after)]
            if (not missing and
                    not self.analysis.in_script(tokens[start:position + size])):
                return start
        return None

    def covered(self, word, said_after):
        if word in said_after:
            return True
        if len(word) < self.settings["prefixMinLength"]:
            return False
        return any(other.startswith(word) for other in said_after)
