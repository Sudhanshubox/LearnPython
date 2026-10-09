"""Saved per-module conversations for the web study screen.

Each module gets its own conversation with the mentor, stored in
.mentor/chats/<module>.json so it survives restarts. Every message the learner
sends automatically carries what the mentor needs to see: the lesson (once, at
the start), their current exercises.py whenever it changed, and the result of
their latest test run.
"""

import hashlib
import json
import threading

import anthropic

from . import client, config, curriculum, prompts

GENERAL = "general"
_locks = {}
_locks_guard = threading.Lock()


def _lock(key):
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


class Chat:
    def __init__(self, key):
        if key != GENERAL and curriculum.get(key) is None:
            raise KeyError(key)
        self.key = key
        self.path = config.CHATS_DIR / f"{key}.json"
        self.state = {"api": [], "display": [], "code_hash": None, "pending_tests": None}
        if self.path.exists():
            self.state.update(json.loads(self.path.read_text(encoding="utf-8")))

    @property
    def module(self):
        return None if self.key == GENERAL else curriculum.get(self.key)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    def display(self):
        return self.state["display"]

    def reset(self):
        if self.path.exists():
            self.path.unlink()
        self.__init__(self.key)

    def record_tests(self, passed, output):
        self.state["pending_tests"] = {"passed": passed, "output": output}
        self.save()

    def _user_content(self, text):
        parts = []
        mod = self.module
        if not self.state["api"] and mod:
            parts.append(prompts.MODULE_CONTEXT.format(
                module_id=mod.id, phase=mod.phase, lesson=mod.lesson,
            ))
        if mod:
            code = mod.code_snapshot()
            if _hash(code) != self.state["code_hash"]:
                parts.append(f"--- My current code ---\n{code}")
        tests = self.state["pending_tests"]
        if tests:
            verdict = "all passed" if tests["passed"] else "some failed"
            parts.append(f"--- My latest test run ({verdict}) ---\n```\n{tests['output'][-6000:]}\n```")
        parts.append(text)
        return "\n\n".join(parts)

    def send(self, text, on_text, api=None):
        """Send a learner message and stream the reply through on_text. Returns the reply text."""
        with _lock(self.key):
            api = api or anthropic.Anthropic()
            content = self._user_content(text)
            messages = self.state["api"] + [{"role": "user", "content": content}]
            # Raises MentorError before anything is saved, so a failed send leaves history valid.
            reply = client.ask(api, client.system_prompt(), messages, on_text)
            reply_text = client.text_of_blocks(reply.content)
            # Store the reply's blocks unchanged so the history stays append-only.
            blocks = [b.model_dump(mode="json", exclude_none=True) for b in reply.content]
            self.state["api"] = messages + [{"role": "assistant", "content": blocks}]
            self.state["display"] += [
                {"role": "user", "text": text},
                {"role": "assistant", "text": reply_text},
            ]
            mod = self.module
            if mod:
                self.state["code_hash"] = _hash(mod.code_snapshot())
            self.state["pending_tests"] = None
            self.save()
            return reply_text
