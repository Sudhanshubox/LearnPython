"""m09 exercises: files and JSON. Replace each `raise NotImplementedError` with your solution.

Every `path` argument may be a str or a pathlib.Path. Always use encoding="utf-8".
"""

import csv
import json
from pathlib import Path


# 1. Write each string in `lines` to the file on its own line (each followed by "\n").
#    Create any missing parent folders first.
def write_lines(path, lines):
    raise NotImplementedError


# 2. Count the lines in a file without reading it all into memory (loop over the file).
def count_lines(path):
    raise NotImplementedError


# 3. Return the longest line in the file, without its trailing newline.
#    If several lines tie, return the first. Return "" for an empty file.
def longest_line(path):
    raise NotImplementedError


# 4. Save `data` as pretty JSON (indent=2), keeping non-ASCII characters readable
#    (ensure_ascii=False), then write load_json to read it back.
def save_json(path, data):
    raise NotImplementedError


def load_json(path):
    raise NotImplementedError


# 5. Read a CSV file with a header row and return the average of the numeric
#    values in `column`. Skip rows where that value is empty or not a number.
#    Return None if there are no valid values.
#    For a file with header "name,score" and rows "a,10", "b,", "c,x", "d,20" -> 15.0
def column_average(path, column):
    raise NotImplementedError


# 6. JSONL: write a list of dicts, one JSON object per line, and read them back.
#    read_jsonl must skip blank lines.
def write_jsonl(path, records):
    raise NotImplementedError


def read_jsonl(path):
    raise NotImplementedError


# 7. Find all files with the given extension (like ".py") in `folder` and all its
#    subfolders. Return their paths RELATIVE to folder, as strings with forward
#    slashes, sorted. find_files(root, ".txt") -> ["a.txt", "sub/b.txt"]
def find_files(folder, extension):
    raise NotImplementedError
