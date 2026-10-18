import asyncio
import json
import sys
import time

import psutil
import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from rostrum.mcp_server import CliRunner, RostrumServer

EXPECTED_TOOLS = {
    "analyze", "job_status", "status", "plan", "list_cuts",
    "decide", "import_decisions", "render", "verify", "map_time", "doctor",
}


class FakeRunner(CliRunner):
    def __init__(self, script):
        super().__init__()
        self.script = script

    def command(self, args):
        return [sys.executable, "-c", self.script]


class TestRostrumServer:
    def wait_for(self, server, job_id):
        deadline = time.time() + 20
        snapshot = server.job_status(job_id)
        while (snapshot["state"] == "running" and
               time.time() < deadline):
            time.sleep(0.1)
            snapshot = server.job_status(job_id)
        return snapshot

    def test_registers_every_tool_with_a_description(self):
        tools = asyncio.run(RostrumServer().mcp.list_tools())
        assert {tool.name for tool in tools} == EXPECTED_TOOLS
        assert all(tool.description for tool in tools)

    def test_job_reports_progress_then_parsed_result(self, tmp_path):
        script = (
            "import sys, json, time\n"
            "print('transcribing part 1', file=sys.stderr, flush=True)\n"
            "time.sleep(0.3)\n"
            "print(json.dumps({'ok': True, 'talks': 1}))\n"
        )
        server = RostrumServer(runner=FakeRunner(script))
        job_id = server.start_job(["init", "talk.mp4"], str(tmp_path))["job"]

        snapshot = self.wait_for(server, job_id)

        assert snapshot["state"] == "done"
        assert snapshot["result"] == {"ok": True, "talks": 1}
        assert "transcribing part 1" in snapshot["progress"]

    def test_failed_job_keeps_exit_code_and_marks_not_ok(self, tmp_path):
        server = RostrumServer(runner=FakeRunner("import sys; print('{\"error\": \"no ffmpeg\"}'); sys.exit(3)"))
        job_id = server.start_job(["render"], str(tmp_path))["job"]
        snapshot = self.wait_for(server, job_id)
        assert snapshot["state"] == "failed"
        assert snapshot["exitCode"] == 3
        assert snapshot["result"]["ok"] is False

    def test_unknown_job(self):
        assert RostrumServer().job_status("nope")["ok"] is False

    def test_render_passes_yes_only_when_confirmed(self, tmp_path):
        recorded = []

        class RecordingServer(RostrumServer):
            def start_job(self, args, cwd):
                recorded.append(args)
                return {"job": "x"}

        server = RecordingServer()
        server.render(str(tmp_path), talk="t1")
        server.render(str(tmp_path), talk="t1", confirmed=True)
        assert "--yes" not in recorded[0]
        assert recorded[1][-1] == "--yes"

    def test_map_time_passes_talk_and_project(self, tmp_path):
        recorded = []

        class RecordingRunner(CliRunner):
            def run(self, args, cwd, timeout=300):
                recorded.append(args)
                return {"ok": True}

        server = RostrumServer(runner=RecordingRunner())
        server.map_time(str(tmp_path), "23:30", "render-4", talk="t1", project="event.json")
        server.import_decisions(str(tmp_path), "t1-decisions.json", talk="t1")
        assert recorded == [
            ["map", "23:30", "--from", "render-4", "--to", "source", "-p", "event.json", "--talk", "t1"],
            ["decide", "--import", "t1-decisions.json", "--talk", "t1"],
        ]

    def test_close_also_stops_what_the_job_started(self, tmp_path):
        # Like a render: the CLI process starts ffmpeg; stopping only the CLI would leave ffmpeg encoding.
        script = (
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print(child.pid, file=sys.stderr, flush=True)\n"
            "time.sleep(60)\n"
        )
        server = RostrumServer(runner=FakeRunner(script))
        job_id = server.start_job(["render"], str(tmp_path))["job"]
        deadline = time.time() + 20
        while (not server.job_status(job_id)["progress"] and
               time.time() < deadline):
            time.sleep(0.1)
        child = psutil.Process(int(server.job_status(job_id)["progress"][0]))

        server.close()

        assert server.jobs[job_id].process.poll() is not None
        _, alive = psutil.wait_procs([child], timeout=5)
        assert not alive

    def test_close_stops_running_jobs_and_removes_logs(self, tmp_path):
        server = RostrumServer(runner=FakeRunner("import time; time.sleep(60)"))
        job_id = server.start_job(["render"], str(tmp_path))["job"]
        log_dir = server.log_dir
        server.close()
        assert server.jobs[job_id].process.poll() is not None
        assert not log_dir.exists()


class TestStdioTransport:
    def test_real_client_lists_tools_and_calls_one(self):
        async def session_roundtrip():
            parameters = StdioServerParameters(command=sys.executable, args=["-m", "rostrum", "mcp"])
            async with stdio_client(parameters) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    reply = await session.call_tool("job_status", {"job_id": "missing"})
                    return {tool.name for tool in tools.tools}, reply

        names, reply = asyncio.run(asyncio.wait_for(session_roundtrip(), timeout=60))
        assert names == EXPECTED_TOOLS
        payload = json.loads(reply.content[0].text)
        assert payload["ok"] is False
