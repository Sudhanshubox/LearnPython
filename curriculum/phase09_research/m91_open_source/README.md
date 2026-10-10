# m91 · Contributing to open-source AI

**By the end you can:** find a good first contribution in an open-source AI project, write a bug report maintainers can act on (with an automatically minimized reproduction), follow the conventions projects use for commits, versions and changelogs, and make a clean contribution on a branch, ready for a pull request.

**Why it matters for AI:** almost all of AI runs on open source: PyTorch, Hugging Face Transformers, scikit-learn, vLLM, llama.cpp. Contributing is the fastest way to learn how production-grade AI code is written, to get feedback from expert engineers, and to build a public track record. Many AI engineers were hired because of their open-source work.

---

## 1. Finding where to start

- Use the libraries you know (this course used NumPy, pandas, scikit-learn, PyTorch, FastAPI and the Anthropic SDK). You understand their purpose, which is half the battle.
- Look for issues labelled **good first issue** or **help wanted**, and documentation problems. Docs fixes and better error messages are real, valued contributions, and an easy way to learn the workflow.
- Read **CONTRIBUTING.md** and the **code of conduct** first. They explain how to set up, test and format code, and how the project wants changes proposed.
- For anything bigger than a small fix, **comment on the issue first** and ask whether the maintainers want it, so you don't spend a week on something they'll decline.

## 2. Bug reports that get fixed

Maintainers are volunteers with long queues. Make your report easy to act on:

- **Search first:** it may already be reported (or fixed in the latest version).
- A clear **title** ("`train_test_split` crashes with stratify on a single-class subset", not "Bug!!").
- **Steps to reproduce**, **expected** vs **actual** behaviour (with the full error message), and your **environment** (library version, Python version, OS).
- A **minimal reproducible example**: the smallest self-contained code that shows the bug. Not your whole project.

Shrinking a failing input by hand is tedious. **Delta debugging** (`ddmin`, Zeller & Hildebrandt, 2002) automates it: split the input into chunks, check whether a chunk alone (or everything except a chunk) still fails, keep the smaller failing version, and refine the split until no single element can be removed. You'll implement it; it works on lists of data rows, lines of code, or tokens of a prompt.

## 3. Conventions: commits, versions, changelogs

Many projects use **Conventional Commits**: `type(scope): description`, for example

```
feat(tokenizer): support special tokens in encode
fix(rag): ignore citations to documents that don't exist
docs: explain the context budget
refactor!: rename Run.log to Run.log_metrics      <- "!" marks a breaking change
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`. A footer line `BREAKING CHANGE: ...` also marks a breaking change.

Structured messages make **semantic versioning** automatic. A version is MAJOR.MINOR.PATCH: a breaking change bumps MAJOR (and resets the others), a new feature bumps MINOR, a fix bumps PATCH. They also make it possible to generate the **changelog**, grouped into breaking changes, features and bug fixes.

## 4. The contribution workflow

1. **Fork** the repository on GitHub and clone your fork (m32).
2. Create a **branch** from the latest main for each change: `git switch -c fix/empty-input`.
3. Make a **small, focused** change. One pull request, one purpose. Add or update **tests** (m31) and docs.
4. Run the project's tests and linters locally before pushing.
5. Commit with a clear (conventional) message and push your branch to your fork.
6. Open a **pull request**: explain what and why, link the issue ("Fixes #123"), show before/after, and say how you tested it.
7. **Review:** respond to every comment, push follow-up commits, stay polite and patient. Reviews are how you learn; a request for changes is normal, not a rejection.

## 5. Licenses, briefly

Open source is defined by its license. **MIT** and **BSD** are permissive (do almost anything, keep the notice); **Apache-2.0** is permissive with an explicit patent grant (common in AI: PyTorch is BSD, Transformers is Apache-2.0); **GPL** requires derived works to be open source under the GPL too. Model weights often come with their own licenses and usage policies: read them before building on a model. Some projects require signing a CLA or adding a DCO sign-off (`git commit -s`).

---

## Problem-solving habit #91: make your problem easy to help with

Whether you're asking a maintainer, a colleague or the mentor for help: minimize the failing example, state what you expected and what happened, say what you already tried. Half the time, preparing a good question reveals the answer yourself (rubber-duck debugging, m10). The other half, you get a fast, precise reply.

## Common mistakes

- A huge first pull request that touches 30 files.
- Opening a PR without running the tests.
- "It doesn't work" bug reports with no version, no error message and no example.
- Arguing with reviewers instead of asking why.
- Copying code from a GPL project into an MIT one.

## Go deeper (optional)

1. Pick one library from this course, find three "good first issue" items, and read the full discussion on each. What made the accepted PRs succeed?
2. Read the scikit-learn contributing guide. It's one of the most thorough. What does it require for a new feature to be accepted?
3. Make one contribution, however small (a docs typo counts). Then write a short blog post about the process (m90).

## Your turn

Open the **Exercises** tab. The last exercise makes a contribution branch in a real temporary git repository.
