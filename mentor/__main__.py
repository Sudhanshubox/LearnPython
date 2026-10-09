"""Command-line entry point: `python -m mentor <command>`."""

import argparse
import re
from pathlib import Path

from . import client, curriculum, progress, prompts


def cmd_init(args):
    messages = client.session(prompts.INIT)
    for msg in reversed(messages):
        if msg["role"] != "assistant":
            continue
        match = re.search(r"^PROFILE:\s*(.+)$", client.text_of_blocks(msg["content"]), re.M)
        if match:
            data = progress.load()
            for part in match.group(1).split(";"):
                if "=" in part:
                    key, value = part.split("=", 1)
                    data["profile"][key.strip()] = value.strip()
            progress.save(data)
            print("\nProfile saved. Next: python -m mentor next")
            return
    print("\nProfile not saved (the interview didn't finish). Run `python -m mentor init` again.")


def cmd_status(args):
    data = progress.load()
    print(progress.summary(data), "\n")
    current_phase = None
    for mod in curriculum.all_modules():
        if mod.phase != current_phase:
            current_phase = mod.phase
            print(current_phase)
        mark = "x" if mod.id in data["completed"] else " "
        print(f"  [{mark}] {mod.path.name}")
    nxt = curriculum.next_module(data["completed"])
    if nxt:
        print(f"\nNext up: {nxt.id}  ->  python -m mentor learn {nxt.id}")


def cmd_next(args):
    nxt = curriculum.next_module(progress.load()["completed"])
    if nxt is None:
        print("You've finished every module that exists so far. Ask the mentor what to build next!")
        return
    args.module = nxt.id
    cmd_learn(args)


def cmd_learn(args):
    mod = curriculum.find(args.module)
    client.session(prompts.LEARN.format(
        module_id=mod.id, phase=mod.phase, lesson=mod.lesson, exercises=mod.exercises,
    ))


def cmd_check(args):
    mod = curriculum.find(args.module)
    passed, output = curriculum.run_tests(mod)
    print(output)
    progress.record_attempt(mod.id, passed)
    if args.no_ai:
        print("All tests passed!" if passed else "Some tests failed.")
        return
    template = prompts.CHECK_PASSED if passed else prompts.CHECK_FAILED
    client.session(template.format(module_id=mod.id, code=mod.exercises, output=output))


def cmd_review(args):
    path = Path(args.file)
    client.session(prompts.REVIEW.format(filename=path.name, code=path.read_text()))


def cmd_quiz(args):
    mod = curriculum.find(args.module)
    client.session(prompts.QUIZ.format(module_id=mod.id, lesson=mod.lesson))


def cmd_tip(args):
    client.session(prompts.TIP, interactive=False)


def cmd_chat(args):
    client.session(" ".join(args.question) or "Hi! Where am I in the program, and what should I focus on today?")


def cmd_note(args):
    progress.add_note(" ".join(args.text))
    print("Noted. The mentor will see this in future sessions.")


def main():
    parser = argparse.ArgumentParser(prog="python -m mentor", description="Your AI Python & AI mentor.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="first-time setup: the mentor interviews you").set_defaults(func=cmd_init)
    sub.add_parser("status", help="show your progress (no API call)").set_defaults(func=cmd_status)
    sub.add_parser("next", help="start the next unfinished module").set_defaults(func=cmd_next)

    p = sub.add_parser("learn", help="interactive lesson for a module")
    p.add_argument("module", help="module id, e.g. m01")
    p.set_defaults(func=cmd_learn)

    p = sub.add_parser("check", help="run a module's tests and get feedback")
    p.add_argument("module")
    p.add_argument("--no-ai", action="store_true", help="only run the tests")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("review", help="get a code review of any file")
    p.add_argument("file")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("quiz", help="get quizzed on a module")
    p.add_argument("module")
    p.set_defaults(func=cmd_quiz)

    sub.add_parser("tip", help="a practical tip for your level").set_defaults(func=cmd_tip)

    p = sub.add_parser("chat", help="ask the mentor anything")
    p.add_argument("question", nargs="*")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("note", help="save a note the mentor will remember")
    p.add_argument("text", nargs="+")
    p.set_defaults(func=cmd_note)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
