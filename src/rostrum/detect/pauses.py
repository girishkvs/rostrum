"""Dead air: long pauses are trimmed to pauses.keepSec on each side; silence before the first and after the last word goes."""
from rostrum.detect.candidate import Candidate

MIDDLE_TOLERANCE = 0.001


class RemovedTime:
    def __init__(self, cuts):
        self.ranges = [(cut.start, cut.end) for cut in cuts]

    def within(self, start, end):
        return sum(max(0.0, min(end, high) - max(start, low)) for low, high in self.ranges)


class TrimDetector:
    def __init__(self, analysis, levels):
        self.analysis = analysis
        self.levels = levels

    def find(self, content):
        """Pauses and leads around the words that the content cuts keep."""
        removed = RemovedTime(content)
        words = [word for word in self.analysis.words if self.kept(word, removed)]
        if not words:
            return []
        found = []
        if self.levels.enabled("pauses"):
            found.extend(self.pauses(words, removed))
        if self.levels.enabled("leads"):
            found.extend(self.leads(words))
        return found

    def kept(self, word, removed):
        middle = (word.start + word.end) / 2
        return removed.within(middle - MIDDLE_TOLERANCE, middle + MIDDLE_TOLERANCE) == 0

    def pauses(self, words, removed):
        settings = self.analysis.settings["pauses"]
        longest = settings["maxSec"]
        keep = settings["keepSec"]
        margin = settings["islandMarginSec"]
        found = []
        end_so_far = words[0].end
        for index in range(1, len(words)):
            gap_start = end_so_far
            gap_end = words[index].start
            if gap_end - gap_start > longest:
                islands = [run for run in self.analysis.energy.loud_runs(gap_start + margin, gap_end - margin, settings["islandMinSec"])
                           if removed.within(*run) < (run[1] - run[0]) * settings["islandRemovedShare"]]
                points = [(gap_start, gap_start)] + islands + [(gap_end, gap_end)]
                for (_, left_end), (right_start, _) in zip(points, points[1:]):
                    silent = (right_start - left_end) - removed.within(left_end, right_start)
                    if silent > longest:
                        found.append(Candidate(left_end + keep, right_start - keep, "pause", "pauses", "high",
                                               f"pause {silent:.1f}s \u2192 {2 * keep:.1f}s"))
            end_so_far = max(end_so_far, words[index].end)
        return found

    def leads(self, words):
        settings = self.analysis.settings["leads"]
        clear = settings["minProbability"]
        first = next((word for word in words if word.probability >= clear), words[0])
        last = next((word for word in reversed(words) if word.probability >= clear), words[-1])
        found = []
        if first.start - settings["inSec"] > settings["minSec"]:
            found.append(Candidate(0.0, first.start - settings["inSec"], "lead-in", "leads", "high", "lead-in"))
        total = self.analysis.total
        if total - (last.end + settings["outSec"]) > settings["minSec"]:
            found.append(Candidate(last.end + settings["outSec"], total, "lead-out", "leads", "high", "lead-out"))
        return found
