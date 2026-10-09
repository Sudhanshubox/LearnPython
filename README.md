# LearnPython: from zero to research-level AI, with an AI mentor

A self-paced program that takes you from your first line of Python to building and researching AI systems. Every lesson comes with auto-graded exercises, and an AI mentor (powered by Claude) teaches you interactively, gives hints without spoiling answers, reviews your code and quizzes you.

See **[ROADMAP.md](ROADMAP.md)** for the full curriculum: 9 phases across three tracks (build, solve, research).

## Setup (5 minutes)

1. Install Python 3.10 or newer.
2. Clone this repo and install the dependencies:

   ```bash
   git clone https://github.com/Sudhanshubox/LearnPython.git
   cd LearnPython
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Get an API key from the [Anthropic Console](https://console.anthropic.com) and set it:

   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...   # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
   ```

4. Meet your mentor:

   ```bash
   python -m mentor init
   ```

## Daily workflow

```bash
python -m mentor status          # where am I? (no API call)
python -m mentor next            # start the next module's interactive lesson
# ...solve curriculum/.../exercises.py in your editor...
python -m mentor check m01       # run the tests; get a hint if something fails, a code review if it passes
python -m mentor quiz m01        # check you really understood it
```

Any time:

```bash
python -m mentor chat "why does -7 // 2 give -4?"
python -m mentor review my_script.py
python -m mentor tip
python -m mentor note "I keep mixing up / and //"   # the mentor remembers this
```

Type `exit` (or press Ctrl+D) to end a conversation. Your profile and progress live in `progress.json`, which stays on your machine.

## How each module works

```
curriculum/phase01_foundations/m01_hello_python/
├── README.md            the lesson (readable on its own, or taught by the mentor)
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

## For contributors

`pytest` runs the mentor's own tests. Module exercises are excluded, since they're meant to fail until you solve them.
