"""m32 exercises: driving Git from Python.

`repo` is always a pathlib.Path (or str) of a folder. Run git with
subprocess.run(["git", ...], cwd=repo, capture_output=True, text=True).
"""

import subprocess
from pathlib import Path


# 1. Run a git command in `repo` and return its stdout with surrounding whitespace
#    stripped. If git fails (non-zero return code), raise RuntimeError containing git's
#    stderr. run_git(repo, "status", "--short")
def run_git(repo, *args):
    raise NotImplementedError


# 2. Create a new repository in `path` (create the folder if needed) with the default
#    branch "main", and set user.name and user.email in THIS repository's config only.
def init_repo(path, name, email):
    raise NotImplementedError


# 3. Write `content` to the file `filename` (relative to repo), creating folders as needed.
def write_file(repo, filename, content):
    raise NotImplementedError


# 4. Stage EVERYTHING (new, changed and deleted files) and commit with `message`.
#    Return the new commit's full hash (git rev-parse HEAD).
def commit_all(repo, message):
    raise NotImplementedError


# 5. Commit messages (the subject line) of the current branch, NEWEST first.
#    Hint: git log --format=%s
def history(repo):
    raise NotImplementedError


# 6. Is the working directory clean (no staged, unstaged or untracked changes)?
#    Hint: git status --porcelain prints nothing when clean.
def is_clean(repo):
    raise NotImplementedError


# 7. Sorted list of paths with uncommitted changes (including untracked files).
#    Parse `git status --porcelain`: each line is 2 status characters, a space, then
#    the path. (Assume no renames and no spaces in filenames.)
def changed_files(repo):
    raise NotImplementedError


# 8. On a NEW branch called `branch` (started from the current commit), write a file,
#    commit it, then switch back to the branch you started on.
#    Hint: git branch --show-current tells you where you are; git switch -c creates.
def commit_on_branch(repo, branch, filename, content, message):
    raise NotImplementedError


# 9. Merge `branch` into the current branch (use --no-edit). Return True if it merged
#    cleanly. If there's a conflict, abort the merge (git merge --abort) so the repo
#    is back to how it was, and return False.
def try_merge(repo, branch):
    raise NotImplementedError


# 10. The contents of `path` as it was in commit/branch `rev`. (git show rev:path)
def file_at(repo, rev, path):
    raise NotImplementedError


# 11. Undo the last commit but KEEP its changes staged (soft reset).
def undo_last_commit(repo):
    raise NotImplementedError


# 12. The subject of the OLDEST commit that added or removed `text` anywhere in the
#     repository (git log -S text --format=%s lists matches, newest first), or None.
def first_commit_with(repo, text):
    raise NotImplementedError
