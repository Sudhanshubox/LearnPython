import json

import pytest

from exercises import (
    column_average,
    count_lines,
    find_files,
    load_json,
    longest_line,
    read_jsonl,
    save_json,
    write_jsonl,
    write_lines,
)


def test_write_lines_creates_folders(tmp_path):
    target = tmp_path / "deep" / "folder" / "notes.txt"
    write_lines(target, ["one", "two", "नमस्ते"])
    assert target.read_text(encoding="utf-8") == "one\ntwo\nनमस्ते\n"


def test_write_lines_accepts_str_paths(tmp_path):
    target = tmp_path / "s.txt"
    write_lines(str(target), ["x"])
    assert target.read_text(encoding="utf-8") == "x\n"


def test_count_lines(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("a\nb\n\nc\n", encoding="utf-8")
    assert count_lines(f) == 4
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    assert count_lines(empty) == 0


def test_longest_line(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("short\nthe longest one\nmedium line\nsame length 123\n", encoding="utf-8")
    assert longest_line(f) == "the longest one"
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    assert longest_line(empty) == ""


def test_json_round_trip(tmp_path):
    data = {"model": "tiny", "layers": [64, 64], "lang": "हिन्दी", "ok": True, "drop": None}
    path = tmp_path / "cfg.json"
    save_json(path, data)
    text = path.read_text(encoding="utf-8")
    assert "हिन्दी" in text, "use ensure_ascii=False"
    assert '\n  "model"' in text, "use indent=2"
    assert load_json(path) == data


def write_csv(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_column_average(tmp_path):
    f = write_csv(tmp_path / "s.csv", "name,score\na,10\nb,\nc,x\nd,20\n")
    assert column_average(f, "score") == 15.0


def test_column_average_handles_quotes_and_floats(tmp_path):
    f = write_csv(tmp_path / "s.csv", 'city,temp\n"Delhi, India",30.5\n"Pune, India",25.5\n')
    assert column_average(f, "temp") == 28.0


def test_column_average_no_values(tmp_path):
    f = write_csv(tmp_path / "s.csv", "name,score\na,\nb,n/a\n")
    assert column_average(f, "score") is None


def test_jsonl_round_trip(tmp_path):
    records = [{"prompt": "2+2?", "answer": "4"}, {"prompt": "राजधानी?", "answer": "दिल्ली"}]
    path = tmp_path / "data.jsonl"
    write_jsonl(path, records)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1]) == records[1]
    assert read_jsonl(path) == records


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_find_files(tmp_path):
    for rel in ["a.txt", "b.py", "sub/c.txt", "sub/deeper/d.txt", "sub/e.md"]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
    assert find_files(tmp_path, ".txt") == ["a.txt", "sub/c.txt", "sub/deeper/d.txt"]
    assert find_files(str(tmp_path), ".md") == ["sub/e.md"]
    assert find_files(tmp_path, ".csv") == []
