"""Fillers: "um", "uh" and the like, cut word by word. Off by default."""


class FillerDetector:
    def __init__(self, analysis):
        self.analysis = analysis
        self.settings = analysis.settings["fillers"]

    def find(self):
        analysis = self.analysis
        found = []
        for word in analysis.words:
            is_filler = (bool(word.tokens) and
                         all(token in analysis.fillers for token in word.tokens))
            if (not is_filler or
                    word.end - word.start < self.settings["minSec"]):
                continue
            candidate = analysis.content_cut(word.index, word.index + 1, "filler", "fillers", "high",
                                             f"filler \u201c{word.text.strip()}\u201d")
            if candidate:
                found.append(candidate)
        return found
