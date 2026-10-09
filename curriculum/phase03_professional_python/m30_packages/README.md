# m30 · Modules, packages and environments

**By the end you can:** split code across files and packages, understand exactly how `import` finds code, avoid circular imports, isolate each project in a virtual environment, manage dependencies with pip, and describe a project with `pyproject.toml` so others can install it.

**Why it matters for AI:** real AI projects are dozens of files, depend on fragile combinations of library versions (PyTorch, CUDA, transformers), and must be reproducible on another machine or a cloud GPU. "It works on my laptop" isn't good enough when a training run costs hundreds of dollars. Packaging and environments are what make your work usable by others, and how you'll publish your own libraries.

**This module has several files.** Use the file picker at the top of the Exercises tab (or open the folder in VS Code). You'll build a small package called `textstats`.

---

## 1. Modules

Every `.py` file is a **module**. Importing it runs the file once and gives you its names:

```python
import math                    # the whole module: math.sqrt(2)
from math import sqrt, pi      # specific names
import numpy as np             # an alias
```

Python caches imported modules in `sys.modules`, so importing the same module twice doesn't run it twice.

## 2. `if __name__ == "__main__":`

When a file is run directly, its `__name__` is `"__main__"`; when it's imported, `__name__` is the module's name. So:

```python
def main():
    ...

if __name__ == "__main__":     # only when run as a script, not when imported
    main()
```

This lets one file be both an importable library and a runnable script.

## 3. Where does `import` look?

Python searches the folders in `sys.path` in order:

1. the folder of the script you ran (or the current folder, for `python -m` and the REPL),
2. folders from the `PYTHONPATH` environment variable,
3. the standard library,
4. `site-packages`: where pip installs packages.

**Classic bug:** naming your own file `random.py`, `json.py` or `torch.py`. It's found *first*, so `import random` imports your file instead of the real module. (This is also why the environment notes in this repo say not to run untrusted scripts from a downloads folder.)

## 4. Packages

A **package** is a folder of modules with an `__init__.py`:

```
textstats/
├── __init__.py      # runs on "import textstats"; defines the public API
├── tokens.py
├── stats.py
└── __main__.py      # runs on "python -m textstats"
```

```python
# textstats/stats.py
from .tokens import words          # RELATIVE import: from this same package

# textstats/__init__.py
from .stats import word_count, top_words
__all__ = ["word_count", "top_words"]   # what "from textstats import *" exports
__version__ = "0.1.0"
```

Users can then write `from textstats import word_count`, without knowing which file it lives in. That freedom to reorganize the internal files later is why packages re-export their public API from `__init__.py`.

Inside a package, prefer **relative** imports (`from .tokens import words`) for sibling modules; use **absolute** imports (`import json`) for everything else.

## 5. Circular imports

If `a.py` imports `b.py` and `b.py` imports `a.py`, one of them sees the other half-initialized and you get `ImportError: cannot import name ...` (partially initialized module). Fixes, best first: move the shared code into a third module both can import; or import inside the function that needs it.

## 6. Virtual environments

Different projects need different library versions. A **virtual environment** is a private folder of installed packages for one project:

```bash
python3 -m venv .venv                # create
source .venv/bin/activate            # macOS/Linux
.venv\Scripts\activate               # Windows
pip install numpy                    # installs into .venv only
deactivate
```

Never `pip install` into your system Python. Never commit `.venv/` to Git (add it to `.gitignore`); commit the list of dependencies instead.

## 7. Dependencies

```bash
pip install "numpy>=1.26,<3"        # a version range
pip freeze > requirements.txt        # exact versions of everything installed
pip install -r requirements.txt      # recreate the environment elsewhere
```

A **version specifier** like `>=1.26,<3` means "at least 1.26 and below 3". **Pinning** exact versions (`numpy==1.26.4`) makes experiments reproducible; ranges make libraries easier to combine. Applications and research code usually pin; libraries usually use ranges. Faster modern tools like `uv` do the same jobs.

## 8. pyproject.toml: describing a project

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "textstats"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []

[project.scripts]
textstats = "textstats.__main__:main"    # installs a "textstats" command that calls main()
```

`pip install -e .` installs your project in **editable** mode: the package is importable from anywhere, and code changes take effect without reinstalling. That's how you work on your own libraries.

---

## Problem-solving habit #28: design the public API

Before splitting code into files, decide what users should import (`from textstats import word_count`). Keep that surface small and stable, and treat everything else as internal (prefix with `_`). You can then refactor internals freely without breaking anyone's code.

## Go deeper (optional, research-level)

1. Print `sys.path` inside and outside an activated virtual environment. What exactly does "activating" change? (Look at the `activate` script and at `sys.prefix`.)
2. What's inside a **wheel** (`.whl`) file? Build one with `pip wheel .` in this module's folder and unzip it.
3. Why is installing PyTorch with GPU support harder than installing a pure-Python package? Read about compiled extensions, CUDA versions and platform-specific wheels.

## Your turn

1. Finish the package: `textstats/tokens.py`, `textstats/stats.py`, `textstats/__init__.py` and `textstats/__main__.py`.
2. Complete `pyproject.toml`.
3. Solve the functions in `exercises.py`.

Then press **Run tests**. When everything passes, try it for real: from this folder, run `python -m textstats README.md`, or `pip install -e .` and then just `textstats README.md`.
