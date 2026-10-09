# m02 · Strings in depth

**By the end you can:** slice and search text, clean messy input, build strings efficiently, and explain how text becomes bytes.

**Why it matters for AI:** language models never see text directly. Text is turned into bytes, then into **tokens**. Cleaning, splitting and encoding text is the first step of every NLP and LLM pipeline.

---

## 1. Indexing and slicing

A string is a sequence of characters. Positions start at 0, and negative indices count from the end.

```python
s = "PYTHON"
#    012345
s[0]      # 'P'
s[-1]     # 'N'
s[1:4]    # 'YTH'   start included, stop excluded
s[:2]     # 'PY'
s[3:]     # 'HON'
s[::2]    # 'PTO'   every 2nd character
s[::-1]   # 'NOHTYP' reversed
```

**Tip:** slicing never raises an error for out-of-range bounds: `s[2:100]` gives `'THON'`. Indexing does: `s[100]` is an `IndexError`.

## 2. Strings are immutable

```python
s = "cat"
s[0] = "b"        # TypeError!
s = "b" + s[1:]   # make a new string instead -> "bat"
```

Every "change" to a string creates a new string object.

## 3. The methods you'll use every day

```python
"  hello  ".strip()            # 'hello'          (also lstrip / rstrip)
"Hello".lower(), "Hello".upper()
"a,b,,c".split(",")            # ['a', 'b', '', 'c']
"one  two\tthree".split()      # ['one', 'two', 'three']  no argument = split on any whitespace
"-".join(["2026", "10", "09"]) # '2026-10-09'
"banana".replace("an", "AN")   # 'bANANa'
"banana".count("a")            # 3
"banana".find("n")             # 2   (-1 if not found)
"banana".index("z")            # ValueError if not found
"report.pdf".endswith(".pdf")  # True
"42".isdigit(), "abc".isalpha(), "a1".isalnum()
"python" in "I love python"    # True
```

## 4. Looping over strings

```python
for ch in "abc":
    print(ch)

for i, ch in enumerate("abc"):
    print(i, ch)       # 0 a, 1 b, 2 c
```

## 5. Building strings efficiently

```python
# Slow for large inputs: each + creates a brand new string (O(n²) overall)
result = ""
for word in words:
    result += word + " "

# Fast: collect pieces, join once (O(n))
result = " ".join(words)
```

**Tip:** whenever you build a string in a loop, collect parts in a list and `"".join(parts)` at the end.

## 6. Characters are numbers: Unicode

Every character has a number called its **code point**.

```python
ord("A")    # 65
chr(97)     # 'a'
ord("अ")    # 2309
len("नमस्ते")   # 6   (characters, not "letters" as you'd perceive them)
len("👍🏽")    # 2   (thumbs up + a skin-tone modifier)
```

To store or send text, it's **encoded** into bytes. UTF-8 is the standard:

```python
"A".encode("utf-8")    # b'A'            1 byte
"अ".encode("utf-8")    # b'\xe0\xa4\x85'  3 bytes
"👍".encode("utf-8")   # 4 bytes
b"caf\xc3\xa9".decode("utf-8")   # 'café'
```

GPT-style tokenizers start from these UTF-8 bytes, which is why Hindi text often costs more tokens than English: each character is 3 bytes.

## 7. Escape sequences and raw strings

```python
print("line1\nline2")    # \n newline, \t tab, \\ backslash, \" quote
print(r"C:\new\table")    # raw string: backslashes stay literal (great for regex and Windows paths)
print("""Multi-line
strings use triple quotes""")
```

---

## Problem-solving habit #2: normalize first

Messy real-world text (extra spaces, mixed case, punctuation) makes simple problems look hard. Clean the input into a standard form first, then solve the easy version. Exercise 2 is a classic example.

## Go deeper (optional, research-level)

1. Why is `s += x` in a loop sometimes fast in CPython anyway? Search for "CPython string concatenation in-place optimization", and explain why you still shouldn't rely on it.
2. Read how UTF-8 works (the bit patterns for 1–4 byte characters). Why can you always tell where a character starts, even in the middle of a byte stream?
3. Look at how Byte Pair Encoding (BPE) builds a tokenizer from byte pairs (Sennrich et al., 2016, *Neural Machine Translation of Rare Words with Subword Units*). You'll implement it in Phase 7.

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**.
