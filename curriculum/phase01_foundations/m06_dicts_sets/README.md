# m06 · Dictionaries and sets

**By the end you can:** look things up instantly by key, count and group data, use set algebra, and explain why hashing makes all of this fast.

**Why it matters for AI:** a tokenizer's vocabulary is a dict from token to ID. Counting words is how classic NLP works. And in exercise 9 you'll build a **bigram language model**: a dict that predicts the next word from the previous one. It's the great-great-grandparent of GPT.

---

## 1. Dictionaries map keys to values

```python
ages = {"Asha": 21, "Ben": 19}
ages["Chen"] = 22          # add or update
ages["Asha"]               # 21
ages["Zoe"]                # KeyError!
ages.get("Zoe")            # None, no error
ages.get("Zoe", 0)         # 0, a default
"Ben" in ages              # True   (checks keys)
del ages["Ben"]
len(ages)                  # 2
```

Since Python 3.7, dicts keep keys in the order they were inserted.

## 2. Looping over a dict

```python
for name in ages:                  # keys
for age in ages.values():          # values
for name, age in ages.items():     # both: the most common form
```

## 3. Dict comprehensions

```python
squares = {n: n * n for n in range(5)}           # {0: 0, 1: 1, 2: 4, ...}
passed = {name: s for name, s in scores.items() if s >= 40}
flipped = {v: k for k, v in d.items()}
```

## 4. Two patterns you'll write a hundred times

**Counting:**

```python
counts = {}
for word in words:
    counts[word] = counts.get(word, 0) + 1
```

**Grouping:**

```python
groups = {}
for word in words:
    groups.setdefault(word[0], []).append(word)   # group by first letter
```

The standard library has shortcuts for both:

```python
from collections import Counter, defaultdict

Counter(words).most_common(3)      # the 3 most frequent words with counts

groups = defaultdict(list)         # missing keys start as []
for word in words:
    groups[word[0]].append(word)
```

**Tip:** learn the manual versions first (the exercises ask for them), then use `Counter` and `defaultdict` in real code.

## 5. Sets: unique items, fast membership

```python
seen = {1, 2, 3}
empty = set()            # NOT {} (that's an empty dict)
seen.add(4)
seen.discard(10)         # no error if missing (remove() would raise)
3 in seen                # True, and fast

a = {1, 2, 3}
b = {2, 3, 4}
a | b    # {1, 2, 3, 4}   union
a & b    # {2, 3}         intersection
a - b    # {1}            difference
a ^ b    # {1, 4}         symmetric difference (in exactly one)
set("hello")             # {'h', 'e', 'l', 'o'}: duplicates removed
```

## 6. Why lookups are fast: hashing

A dict doesn't search its keys one by one. It computes `hash(key)`, a number, and uses it to jump straight to the right slot in an internal array. That's why `key in d` and `x in s` are **O(1) on average**, while `x in some_list` is O(n).

```python
hash("cat")          # some big integer, stable within one run
hash((1, 2))         # tuples are hashable
hash([1, 2])         # TypeError: unhashable type: 'list'
```

Keys (and set items) must be **hashable**, which in practice means immutable: numbers, strings, tuples of hashables. A list can't be a key, because if it changed after insertion, its hash would no longer match its slot.

**Tip:** whenever you see `if x in big_list` inside a loop, ask whether a set would turn O(n²) into O(n). It's one of the most common speed-ups in real code.

---

## Problem-solving habit #6: trade memory for speed

Many problems that look like they need two nested loops can be solved in one pass by *remembering what you've seen* in a set or dict. Exercise 6 (pair with a given sum) is the classic: O(n²) by brute force, O(n) with a set.

## Go deeper (optional, research-level)

1. Python randomizes string hashes each run (`PYTHONHASHSEED`). Read about "hash flooding" denial-of-service attacks to see why.
2. How does a dict handle two keys landing in the same slot (a collision)? Read about open addressing, then skim the comments at the top of CPython's `dictobject.c`.
3. Your bigram model assigns probability 0 to any word pair it never saw. Read about *Laplace (add-one) smoothing* and why language models needed it before neural networks. Then think about why a neural model doesn't have this problem in the same way.

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**.
