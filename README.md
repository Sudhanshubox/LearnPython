# LearnPython: from zero to research-level AI, with an AI mentor

A self-paced program that takes you from your first line of Python to building and researching AI systems. Every lesson comes with auto-graded exercises, and an AI mentor (powered by Claude) teaches you interactively, gives hints without spoiling answers, reviews your code and quizzes you.

See **[ROADMAP.md](ROADMAP.md)** for the full curriculum: 9 phases across three tracks (build, solve, research).

## Setup (5 minutes)

1. Install Python 3.11 or newer.
2. Clone this repo and install the dependencies:

   ```bash
   git clone https://github.com/Sudhanshubox/LearnPython.git
   cd LearnPython
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   This includes PyTorch for Phase 6 onwards, which is a large download on Linux (several GB with GPU libraries). Without an NVIDIA GPU you can install the much smaller CPU-only build first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

3. Get an API key from the [Anthropic Console](https://console.anthropic.com) and set it:

   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...   # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
   ```

4. Open the study screen:

   ```bash
   python -m mentor web
   ```

   Your browser opens at http://127.0.0.1:8765 (you can also open that address yourself).

## The study screen

```
┌──────────────────────────────────────┬─────────────────────────┐
│ Lesson | Exercises                   │ Mentor                  │
│                                      │                         │
│ The lesson, or a Python editor with  │ Ask anything, any time. │
│ a "Run tests" button                 │ It sees your code and   │
│                                      │ your test results.      │
│                                      │ [Teach me] [Hint]       │
│                                      │ [Review]   [Quiz me]    │
└──────────────────────────────────────┴─────────────────────────┘
```

- **Read** the lesson on the left. Select any sentence and press **Ask mentor about this** for a different explanation.
- **Solve** the exercises in the Exercises tab. `Ctrl+S` saves, `Ctrl+Enter` runs the tests.
- **Get unstuck** with **Hint**: the mentor reads your code and failing test and gives the smallest hint that helps, never the full answer unless you ask.
- Each module keeps its own conversation, saved between sessions. **New chat** starts it fresh.
- Drag the divider to resize the panels.

You can also edit `exercises.py` in your own editor (VS Code, PyCharm); the study screen and mentor always use the file on disk.

## Terminal commands

Everything also works from the terminal:

```bash
python -m mentor status          # your progress (no API call)
python -m mentor init            # the mentor interviews you to set up your profile
python -m mentor learn m01       # interactive lesson
python -m mentor check m01       # run tests + feedback
python -m mentor quiz m01
python -m mentor review my_script.py
python -m mentor chat "why does -7 // 2 give -4?"
python -m mentor note "I keep mixing up / and //"   # the mentor remembers this
```

Your profile and progress live in `progress.json`, and conversations in `.mentor/`. Both stay on your machine.

## How each module works

```
curriculum/phase01_foundations/m01_hello_python/
├── README.md            the lesson
├── exercises.py         functions for you to implement
└── test_exercises.py    tests that check your answers
```

Each lesson ends with **"Go deeper"** questions. They're optional, but they're where research-level understanding comes from.

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `MENTOR_MODEL` | `claude-opus-5-5` | Which Claude model teaches you |
| `MENTOR_EFFORT` | `medium` | `low` is faster and cheaper; `high` thinks harder on tough questions |
| `MENTOR_MAX_TOKENS` | `32000` | Maximum length of one reply |
| `MENTOR_PORT` | `8765` | Port for the study screen |

## For contributors

`pytest` runs the mentor's own tests. Module exercises are excluded, since they're meant to fail until you solve them.
