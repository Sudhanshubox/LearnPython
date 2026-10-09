# m32 · Git and GitHub

**By the end you can:** explain what Git actually stores, use the everyday workflow (status, add, commit, log, diff), branch and merge (including resolving conflicts), undo mistakes safely, collaborate through GitHub pull requests, and keep a machine-learning repository clean.

**Why it matters for AI:** every AI team, open-source library and research codebase lives on Git and GitHub. Recording the exact commit used for every experiment is how you make results reproducible ("which code produced this model?"). Your GitHub profile is also your portfolio when applying for AI roles, and this very course lives in a Git repository.

---

## 1. The mental model

Git stores **snapshots** of your project, called **commits**. Each commit records the full state of every tracked file, a message, an author, a time, and a pointer to its **parent** commit. That chain of commits is your history.

```
A ← B ← C        main
         ↑
        HEAD     (where you are now)
```

- A **branch** is just a movable name pointing at a commit. Committing moves the current branch forward.
- **HEAD** points to the branch (or commit) you're on.
- Commits are identified by a hash like `3f9a2c1`.

Files move through three places: the **working directory** (what's on disk) → the **staging area** (`git add`: what goes into the next commit) → the **repository** (`git commit`).

## 2. The everyday loop

```bash
git status                     # what changed?
git diff                       # exactly how (unstaged changes)
git add file.py                # stage a file   (git add -A: everything)
git diff --staged              # review what you're about to commit
git commit -m "Add tokenizer for Hindi text"
git log --oneline              # history
```

**Good commit messages** say *what and why* in the imperative ("Fix off-by-one in batch slicing", not "fixed stuff"). Small, focused commits are easier to review, revert and bisect (m10).

## 3. .gitignore: what NOT to commit

```gitignore
.venv/
__pycache__/
.env                 # API keys and passwords: never commit secrets!
data/                # datasets: too big, and often private
*.pt                 # model checkpoints
wandb/  mlruns/      # experiment logs
```

If you commit an API key by accident, deleting it in a later commit does **not** remove it from history. Revoke the key immediately. For large data and models, use Git LFS, DVC or the Hugging Face Hub instead of plain Git.

## 4. Branches and merging

```bash
git switch -c feature/augmentation     # create a branch and switch to it
# ...edit, add, commit...
git switch main
git merge feature/augmentation         # bring the branch's commits into main
git branch -d feature/augmentation     # delete it when done
```

Work on a branch so `main` always stays working.

## 5. Merge conflicts

If both branches changed the same lines, Git can't decide and marks the file:

```
<<<<<<< HEAD
learning_rate = 0.001
=======
learning_rate = 0.01
>>>>>>> experiment
```

Edit the file to the version you want (deleting the markers), then `git add` it and `git commit`. Or give up with `git merge --abort`. Conflicts are normal, not a disaster.

## 6. Undoing things

| Situation | Command |
|---|---|
| Discard unstaged changes to a file | `git restore file.py` |
| Unstage a file | `git restore --staged file.py` |
| Fix the last commit's message or add a forgotten file | `git commit --amend` (only if not pushed yet) |
| Undo the last commit but keep its changes | `git reset --soft HEAD~1` |
| Undo a commit that's already shared | `git revert <hash>` (adds a new commit that reverses it) |
| See an old version of a file | `git show <hash>:path/to/file.py` |
| Find when some text was added or removed | `git log -S "text"` |

**Rule:** never rewrite history (`reset`, `--amend`, `rebase`) on commits you've already pushed to a shared branch. Use `revert` instead.

## 7. GitHub and collaboration

```bash
git clone https://github.com/user/repo.git     # copy a repository
git pull                                        # fetch and merge others' changes
git push -u origin feature/augmentation         # upload your branch
```

The **pull request** (PR) workflow: push a branch, open a PR on GitHub, teammates review the diff and comment, CI runs the tests, and then it's merged. To contribute to someone else's project, **fork** it (your own copy on GitHub), push to your fork, and open a PR to the original.

## 8. Git for machine-learning projects

- Log the commit hash with every experiment's results (`git rev-parse HEAD`).
- Make sure the working directory is clean before a long run (`git status --porcelain` prints nothing), so the hash really describes the code that ran.
- Keep notebooks small or strip their outputs before committing; they make diffs unreadable.

---

## Problem-solving habit #30: commit before you experiment

Before trying a risky change, commit (or branch). Then you can always get back to a working state with one command, so you can experiment boldly. Combined with `git bisect` (m10), a clean history turns "when did this break?" into a 5-minute question.

## Go deeper (optional, research-level)

1. Git is a **content-addressable** store. Run `git cat-file -p HEAD`, then follow the `tree` hash with `git cat-file -p`. What are blobs, trees and commits, and why does changing one file change the commit hash?
2. Read the first section of the free *Pro Git* book chapter "Git Internals". How does Git avoid storing a full copy of every file in every commit?
3. Compare merge and rebase. Why do some teams require a "linear history"? What are the risks?

## Your turn

The exercises drive Git from Python with `subprocess`, against real temporary repositories that the tests create. (You need `git` installed: run `git --version` to check.) Then do the **real-world task** in the box below.

> **Real-world task (not auto-graded):** create a GitHub account if you don't have one, push your own fork or copy of this course repository, and make a pull request to *yourself* from a branch containing your solutions. Ask the mentor to review the PR description.
