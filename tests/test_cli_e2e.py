"""End to end through the CLI on synthetic media: init, plan, render refused before a review, decide, render, status,
verify (full) and map (by render folder name).

A two-file talk. Part 1 ends with an attempt and a spoken "Redo."; part 2 starts with the retake, so the
cut crosses the file boundary. Part 2 also has a long pause. The audio is loud only while a sentence is spoken.
"""
import json
from pathlib import Path

import pytest
from media_factory import AUDIO, MediaFactory

from rostrum.cli import main
from rostrum.tools import ToolNotFound

FPS = 30
PART_SECONDS = 10
WORD_SECONDS = 0.3
WORD_GAP = 0.05
ROOM_LEVEL = 0.05
PARTS = [
    ("demo-part1", 1, [(0.5, "Welcome to the demo."),
                       (3.0, "The scheduler moves every job to the quietest node."),
                       (7.0, "Redo.")]),
    ("demo-part2", 2, [(0.6, "The scheduler moves every job to the quietest server."),
                       (7.0, "That is all for today.")]),
]


class SpokenPart:
    """Word times for the sentences, and the matching media: speech is loud, the room between sentences is quiet."""

    def __init__(self, sentences):
        self.words = []
        self.spans = []
        for start, text in sentences:
            cursor = start
            for token in text.split():
                self.words.append({"start": round(cursor, 3), "end": round(cursor + WORD_SECONDS, 3),
                                   "word": " " + token, "probability": 0.95})
                cursor += WORD_SECONDS + WORD_GAP
            self.spans.append((start, round(cursor - WORD_GAP, 3)))

    def audio_graph(self, seed):
        loud = "+".join(f"between(t,{start},{end})" for start, end in self.spans)
        graph = AUDIO.format(seconds=PART_SECONDS, seed=seed)
        return graph.replace("[out0]", f",volume='if({loud},1,{ROOM_LEVEL})':eval=frame[out0]")

    def write(self, factory, stem, seed):
        media = factory.folder / f"{stem}.mp4"
        factory.make(media, [
            "-f", "lavfi", "-i", f"testsrc2=size=160x120:rate={FPS}:duration={PART_SECONDS}",
            "-f", "lavfi", "-i", self.audio_graph(seed),
            "-c:v", "libx264", "-preset", "ultrafast", "-g", "15", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        ])
        (factory.folder / f"{stem}.words.json").write_text(json.dumps({"words": self.words}), encoding="utf-8")
        return media


def run(capsys, *args):
    code = main([*args, "--json"])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def factory(tmp_path):
    try:
        return MediaFactory(tmp_path / "event")
    except ToolNotFound as error:
        pytest.skip(str(error))


def test_two_part_talk_from_init_to_map(tmp_path, factory, capsys):
    for stem, seed, sentences in PARTS:
        SpokenPart(sentences).write(factory, stem, seed)
    project = tmp_path / "rostrum.json"

    code, init = run(capsys, "init", str(factory.folder), "--engine", "import", "-p", str(project))
    assert code == 0, init
    candidates = json.loads((tmp_path / "demo.rostrum" / "candidates.json").read_text(encoding="utf-8"))
    boundary = candidates["sources"][1]["offset"]
    assert abs(boundary - PART_SECONDS) < 0.05

    code, plan = run(capsys, "plan", "-p", str(project))
    assert code == 0
    applied = [cut for cut in plan["talks"][0]["cuts"] if cut["decision"] == "cut"]
    crossing = [cut for cut in applied if cut["start"] < boundary < cut["end"]]
    assert len(crossing) == 1
    crossing = crossing[0]
    assert crossing["type"] == "cue"
    assert 1.85 < crossing["start"] < 3.0
    assert boundary < crossing["end"] < boundary + 0.6
    pauses = [cut for cut in applied if cut["type"] == "pause"]
    assert [(round(cut["start"], 2), round(cut["end"], 2)) for cut in pauses] == [
        (round(boundary + 3.7 + 0.5, 2), round(boundary + 7.0 - 0.5, 2))]

    code, refused = run(capsys, "render", "-p", str(project))
    assert code == 2
    assert "not reviewed since the cuts last changed: demo" in refused["error"]

    code, decided = run(capsys, "decide", pauses[0]["id"], "keep", "-p", str(project))
    assert code == 0
    assert decided["kept"] == pytest.approx(candidates["duration"] - (crossing["end"] - crossing["start"]), abs=0.01)

    code, rendered = run(capsys, "render", "-p", str(project))
    assert code == 0, rendered
    render = rendered["talks"][0]
    render_dir = Path(render["dir"])
    assert render_dir.parent == tmp_path / "demo.rostrum" / "renders"
    assert (render["segments"], render["joins"]) == (2, 1)
    frame = 1.0 / FPS
    cutmap = json.loads((render_dir / "cutmap.json").read_text(encoding="utf-8"))
    first, second = cutmap["segments"]
    assert (first["source"], second["source"]) == (0, 1)
    assert first["timeline_end"] == pytest.approx(crossing["start"], abs=frame)
    assert second["timeline_start"] == pytest.approx(crossing["end"], abs=frame)
    code, status = run(capsys, "status", "-p", str(project))
    assert (status["talks"][0]["state"], status["talks"][0]["renders"]) == ("rendered (render-1)", ["render-1"])

    code, verified = run(capsys, "verify", "--mode", "full", "-p", str(project))
    assert code == 0, verified
    check = verified["talks"][0]
    assert (check["ok"], check["mode"], Path(check["render"])) == (True, "full", render_dir)
    assert Path(check["verify_json"]) == render_dir / "verify-full.json"

    output_time = first["output_end"] + 1.0
    code, mapped = run(capsys, "map", f"{output_time:.3f}", "--from", "render-1", "-p", str(project))
    assert code == 0, mapped
    assert mapped["source"] == 1
    assert mapped["local"] == pytest.approx(second["local_start"] + 1.0, abs=0.002)
    assert mapped["local"] == pytest.approx(crossing["end"] - boundary + 1.0, abs=frame + 0.002)

    inside = (crossing["start"] + crossing["end"]) / 2
    code, mapped = run(capsys, "map", f"{inside:.3f}", "--from", str(render_dir), "--to", "output")
    assert code == 0
    assert mapped["in_cut"] is True
