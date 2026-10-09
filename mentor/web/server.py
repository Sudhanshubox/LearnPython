"""A small local web server for the study screen. Run with `python -m mentor web`.

It only listens on 127.0.0.1 by default and only ever writes to modules'
exercises.py files, progress.json and the saved chats in .mentor/.
"""

import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .. import client, config, curriculum, progress
from ..chat import GENERAL, Chat

STATIC = Path(__file__).parent / "static"


def module_list():
    completed = set(progress.load()["completed"])
    return [
        {"id": m.id, "name": m.path.name, "phase": m.phase, "title": m.title, "completed": m.id in completed}
        for m in curriculum.all_modules()
    ]


class Handler(BaseHTTPRequestHandler):
    server_version = "LearnPythonMentor"

    def log_message(self, format, *args):
        pass  # keep the terminal quiet

    # ---- helpers -------------------------------------------------------

    def _send_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._send_json({"error": message}, status)

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def _parts(self):
        return [p for p in self.path.split("?", 1)[0].split("/") if p]

    def _module(self, module_id):
        mod = curriculum.get(module_id)
        if mod is None:
            self._error(HTTPStatus.NOT_FOUND, f"Unknown module {module_id!r}")
        return mod

    def _chat(self, key):
        if key == GENERAL:
            return Chat(GENERAL)
        mod = self._module(key)
        return Chat(mod.id) if mod else None

    # ---- routes --------------------------------------------------------

    def do_GET(self):
        parts = self._parts()
        if not parts:
            return self._static("index.html")
        if parts[0] == "static":
            return self._static("/".join(parts[1:]))
        if parts == ["api", "modules"]:
            return self._send_json({"modules": module_list(), "profile": progress.load()["profile"]})
        if len(parts) == 3 and parts[:2] == ["api", "module"]:
            mod = self._module(parts[2])
            if mod:
                self._send_json({"id": mod.id, "title": mod.title, "lesson": mod.lesson, "code": mod.exercises})
            return
        if len(parts) == 3 and parts[:2] == ["api", "chat"]:
            chat = self._chat(parts[2])
            if chat:
                self._send_json({"messages": chat.display()})
            return
        self._error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        parts = self._parts()
        try:
            body = self._body()
        except json.JSONDecodeError:
            return self._error(HTTPStatus.BAD_REQUEST, "Invalid JSON")

        if len(parts) == 4 and parts[:2] == ["api", "module"]:
            mod = self._module(parts[2])
            if not mod:
                return
            if parts[3] == "code":
                mod.save_exercises(body.get("code", ""))
                return self._send_json({"ok": True})
            if parts[3] == "test":
                if "code" in body:
                    mod.save_exercises(body["code"])
                passed, output = curriculum.run_tests(mod)
                progress.record_attempt(mod.id, passed)
                Chat(mod.id).record_tests(passed, output)
                return self._send_json({"passed": passed, "output": output})

        if len(parts) == 4 and parts[:2] == ["api", "chat"]:
            chat = self._chat(parts[2])
            if not chat:
                return
            if parts[3] == "reset":
                chat.reset()
                return self._send_json({"ok": True})
            if parts[3] == "send":
                return self._stream_reply(chat, (body.get("message") or "").strip())

        self._error(HTTPStatus.NOT_FOUND, "Not found")

    def _stream_reply(self, chat, message):
        if not message:
            return self._error(HTTPStatus.BAD_REQUEST, "Empty message")
        # Stream plain text; the connection closing marks the end of the reply.
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def write(text):
            self.wfile.write(text.encode())
            self.wfile.flush()

        try:
            chat.send(message, write)
        except client.MentorError as e:
            write(f"\n\n[error] {e}")
        except (BrokenPipeError, ConnectionResetError):
            pass  # the learner closed the tab mid-reply

    def _static(self, rel):
        path = (STATIC / rel).resolve()
        if not path.is_relative_to(STATIC.resolve()) or not path.is_file():
            return self._error(HTTPStatus.NOT_FOUND, "Not found")
        body = path.read_bytes()
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(open_browser=True):
    httpd = ThreadingHTTPServer((config.HOST, config.PORT), Handler)
    url = f"http://{config.HOST}:{config.PORT}"
    print(f"Study screen running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nBye! Keep going tomorrow.")
    finally:
        httpd.server_close()
