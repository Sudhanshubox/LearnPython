# m09 · Files, paths and JSON

**By the end you can:** read and write text files safely, work with paths on any operating system, process files too big for memory, and load and save CSV, JSON and JSONL data.

**Why it matters for AI:** every model starts with data on disk. Most LLM training and fine-tuning datasets are **JSONL** files (one JSON object per line), experiment configs are JSON or YAML, and tabular data arrives as CSV. Being fluent with files is not optional.

---

## 1. Paths with pathlib

```python
from pathlib import Path

data_dir = Path("data")
file = data_dir / "train.jsonl"     # join with / : works on Windows, macOS and Linux
file.name          # 'train.jsonl'
file.stem          # 'train'
file.suffix        # '.jsonl'
file.parent        # Path('data')
file.exists()
file.is_file(), data_dir.is_dir()
data_dir.mkdir(parents=True, exist_ok=True)   # create folders, no error if they exist
Path.cwd()         # the current working directory
```

**Tip:** a relative path like `"data/train.jsonl"` is relative to where you *ran* Python from, not where the script lives. Use `Path(__file__).parent / "data"` to be relative to the script.

## 2. Reading and writing text

```python
with open("notes.txt", "w", encoding="utf-8") as f:   # "w" overwrites, "a" appends
    f.write("first line\n")
    f.write("second line\n")

with open("notes.txt", encoding="utf-8") as f:        # default mode is "r" (read)
    content = f.read()                                 # the whole file as one string

with open("notes.txt", encoding="utf-8") as f:
    for line in f:                                     # one line at a time
        print(line.rstrip("\n"))
```

`with` guarantees the file is closed, even if an exception happens inside the block.

Shortcuts for small files: `Path("notes.txt").read_text(encoding="utf-8")` and `.write_text(...)`.

**Always pass `encoding="utf-8"`.** Without it, Windows may use a different default encoding and your Hindi or emoji text breaks.

## 3. Files bigger than memory

`f.read()` loads everything at once. A 50 GB dataset won't fit in RAM, but iterating line by line uses almost no memory:

```python
longest = 0
with open("huge.txt", encoding="utf-8") as f:
    for line in f:              # streams: only one line in memory at a time
        longest = max(longest, len(line))
```

## 4. CSV

```python
import csv

with open("people.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):          # each row is a dict keyed by the header
        print(row["name"], row["age"])     # values are always strings!

with open("out.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "age"])
    writer.writerow(["Asha", 21])
```

Use the `csv` module instead of `line.split(",")`: real CSV has quoted fields with commas inside them (`"Delhi, India"`).

## 5. JSON

JSON is a text format for nested data: objects (dicts), arrays (lists), strings, numbers, `true`/`false`/`null`.

```python
import json

config = {"model": "tiny-gpt", "layers": 4, "dropout": 0.1, "tags": ["test"]}

text = json.dumps(config, indent=2)        # dict -> JSON string
data = json.loads(text)                    # JSON string -> dict

with open("config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)   # keep Unicode readable
with open("config.json", encoding="utf-8") as f:
    config = json.load(f)
```

Watch out: tuples become lists, dict keys become strings (`{1: "a"}` → `{"1": "a"}`), and sets can't be saved at all.

## 6. JSONL: one record per line

```
{"prompt": "What is 2+2?", "answer": "4"}
{"prompt": "Capital of France?", "answer": "Paris"}
```

JSONL can be streamed line by line, appended to without rewriting the file, and split across machines easily. That's why large datasets use it.

```python
with open("data.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

## 7. Finding files

```python
for path in Path("data").glob("*.csv"):     # this folder only
    ...
for path in Path("data").rglob("*.json"):   # this folder and every subfolder
    ...
```

---

## Problem-solving habit #9: test with tiny fixtures

When code reads files, don't test it on the real 2 GB dataset. Create a 3-line file that includes one tricky case (an empty line, a missing value, Unicode) and run against that. The tests in this module do exactly this with pytest's `tmp_path`.

## Go deeper (optional, research-level)

1. What actually happens when you call `f.write()`? Read about buffering and `f.flush()` / `os.fsync()`. Why can a program crash and lose data you "already wrote"?
2. Writing a file in place risks leaving it half-written if the program crashes. Research the "write to a temporary file, then rename" pattern (`os.replace`) and why it is atomic.
3. Compare JSONL, CSV and Parquet for a 10-million-row dataset: file size, reading speed, and reading only some columns. Why does the ML world increasingly use Parquet and Arrow?

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**.
