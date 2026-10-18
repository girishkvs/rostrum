import html
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

TEMPLATE_NAME = "review_template.html"
PLACEHOLDER = re.compile(r"__(TITLE|DATA)__")


class ReviewPage:
    """One static HTML file per talk: no server, no external resources."""

    def __init__(self, talk_id, sources, cuts, title=None, decisions=None, based_on=None):
        self.talk_id = talk_id
        self.sources = sources
        self.cuts = cuts
        self.title = title or talk_id
        self.decisions = decisions or {}
        self.based_on = based_on

    def source_url(self, path, base_dir):
        source = Path(path).resolve()
        try:
            relative = os.path.relpath(source, base_dir)
        except ValueError:
            return source.as_uri()
        return quote(Path(relative).as_posix())

    def data(self, base_dir):
        ordered = sorted(self.cuts, key=lambda cut: cut.start)
        return {
            "talk": self.talk_id,
            "title": self.title,
            # Saves go through rostrum decide --import, which merges by time; never straight over decisions.json.
            "saveName": f"{self.talk_id}-decisions.json",
            "basedOn": self.based_on,
            "decisions": self.decisions,
            "duration": sum(source.duration for source in self.sources),
            "sources": [
                {
                    "url": self.source_url(source.path, base_dir),
                    "name": Path(source.path).name,
                    "offset": source.offset,
                    "duration": source.duration,
                }
                for source in self.sources
            ],
            "cuts": [
                {
                    "id": cut.id,
                    "start": cut.start,
                    "end": cut.end,
                    "type": cut.type,
                    "level": cut.level,
                    "decision": cut.decision,
                    "reason": cut.reason,
                    "evidence": cut.evidence,
                    "confidence": cut.confidence,
                }
                for cut in ordered
            ],
        }

    def payload(self, base_dir):
        # Escaped so transcript text can never end the script element or open a comment.
        text = json.dumps(self.data(base_dir))
        return text.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")

    def html(self, base_dir):
        template = (Path(__file__).parent / TEMPLATE_NAME).read_text(encoding="utf-8")
        values = {"TITLE": html.escape(self.title), "DATA": self.payload(base_dir)}
        return PLACEHOLDER.sub(lambda match: values[match.group(1)], template)

    def write(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.html(target.parent.resolve()), encoding="utf-8")
        return target


class ExistingDecisions:
    def load(self, path):
        decisions_path = Path(path)
        if not decisions_path.is_file():
            return {}
        data = json.loads(decisions_path.read_text(encoding="utf-8"))
        decisions = data.get("decisions", {})
        return decisions if isinstance(decisions, dict) else {}


def write_review(context):
    """Pipeline hook: called after the first pass and after every decide, with a pipeline.TalkContext.
    basedOn is the analysis the page shows, so an import can tell a current page from an outdated one."""
    page = ReviewPage(
        context.talk_id,
        context.sources,
        context.cuts,
        title=context.title,
        decisions=ExistingDecisions().load(context.decisions_path),
        based_on=getattr(context, "analyzed_at", None),
    )
    return page.write(context.review_path)
