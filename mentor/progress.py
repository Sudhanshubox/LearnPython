"""Stores the learner's profile and progress in a local JSON file."""

import json
from datetime import date

from . import config

DEFAULT = {
    "profile": {},
    "completed": [],
    "attempts": {},
    "notes": [],
}


def load():
    if not config.PROGRESS_FILE.exists():
        return json.loads(json.dumps(DEFAULT))
    data = json.loads(config.PROGRESS_FILE.read_text(encoding="utf-8"))
    for key, value in DEFAULT.items():
        data.setdefault(key, type(value)())
    return data


def save(data):
    config.PROGRESS_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def record_attempt(module_id, passed):
    data = load()
    stats = data["attempts"].setdefault(module_id, {"runs": 0, "passed": False})
    stats["runs"] += 1
    if passed:
        stats["passed"] = True
        if module_id not in data["completed"]:
            data["completed"].append(module_id)
    save(data)
    return data


def add_note(text):
    data = load()
    data["notes"].append(f"{date.today().isoformat()}: {text}")
    save(data)


def summary(data):
    """Short text description of the learner, included in the mentor's context."""
    profile = data["profile"]
    lines = []
    if profile:
        lines.append("Learner profile:")
        lines.extend(f"- {k}: {v}" for k, v in profile.items())
    else:
        lines.append("Learner profile: not set yet (they have not run `mentor init`).")
    done = ", ".join(data["completed"]) or "none yet"
    lines.append(f"Completed modules: {done}")
    struggling = [m for m, s in data["attempts"].items() if not s["passed"] and s["runs"] >= 3]
    if struggling:
        lines.append(f"Modules with 3+ failed test runs: {', '.join(struggling)}")
    if data["notes"]:
        lines.append("Mentor notes (most recent last):")
        lines.extend(f"- {n}" for n in data["notes"][-10:])
    return "\n".join(lines)
