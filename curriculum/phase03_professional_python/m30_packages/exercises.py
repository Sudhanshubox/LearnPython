"""m30 exercises: modules and dependencies.

The main work of this module is the textstats/ package and pyproject.toml
(use the file picker above the editor). These functions practise reading
dependency specifications, the way pip does.
"""

import importlib


# 1. Return the public names of a module: its __all__ if it defines one,
#    otherwise every name that doesn't start with "_", sorted.
#    public_names(importlib.import_module("textstats")) uses __all__.
def public_names(module):
    raise NotImplementedError


# 2. Parse one line of a requirements.txt file into (name, specifier).
#    - Ignore comments (from "#" to the end) and surrounding whitespace.
#    - The name is lowercased; the specifier is the rest with spaces removed.
#    - Blank or comment-only lines return None.
#    parse_requirement("NumPy >= 1.26, <3  # arrays") -> ("numpy", ">=1.26,<3")
#    parse_requirement("requests") -> ("requests", "")
#    Specifiers start with one of: == != >= <= > < ~=
def parse_requirement(line):
    raise NotImplementedError


# 3. Does `version` (like "1.26.4") satisfy `specifier` (like ">=1.26,<3")?
#    Support the operators == != >= <= > < separated by commas (all must hold).
#    Compare versions as tuples of ints, padding with zeros: "2" == "2.0.0".
#    An empty specifier is satisfied by every version.
def satisfies(version, specifier):
    raise NotImplementedError


# 4. Given a dict of installed versions and a list of requirement lines, return a
#    sorted list of problems as strings:
#      "missing: <name>"                     if not installed
#      "conflict: <name> <version> (needs <specifier>)"   if the version doesn't satisfy
#    check_environment({"numpy": "2.1.0"}, ["numpy<2", "torch"])
#    -> ["conflict: numpy 2.1.0 (needs <2)", "missing: torch"]
def check_environment(installed, requirement_lines):
    raise NotImplementedError
