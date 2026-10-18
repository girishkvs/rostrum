import json
import sys


class Output:
    """Results go to stdout (JSON with --json); progress and warnings go to stderr."""

    def __init__(self, as_json=False, stream=None, errors=None):
        self.as_json = as_json
        self.stream = stream or sys.stdout
        self.errors = errors or sys.stderr

    def result(self, data, text=None):
        if self.as_json:
            json.dump(data, self.stream, indent=2, default=str)
            self.stream.write("\n")
            return
        if text is None:
            text = json.dumps(data, indent=2, default=str)
        self.stream.write(text + "\n")

    def info(self, message):
        self.errors.write(message + "\n")
        self.errors.flush()

    def fail(self, message, code=1, data=None):
        payload = {"ok": False, "error": message}
        if data:
            payload.update(data)
        if self.as_json:
            self.result(payload)
        else:
            self.errors.write(f"error: {message}\n")
        return code
