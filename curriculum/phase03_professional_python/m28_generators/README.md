# m28 · Iterators and generators

**By the end you can:** explain the iterator protocol behind every `for` loop, write generators that produce values lazily, build memory-efficient data pipelines, and use `itertools` fluently.

**Why it matters for AI:** training data rarely fits in memory. A data loader reads, decodes, augments and batches examples **lazily**, one batch at a time, while the GPU trains on the previous one. Hugging Face `datasets` streaming mode, PyTorch `IterableDataset`, and LLM APIs that **stream** tokens back to you as they're generated are all built on these ideas.

---

## 1. Iterables and iterators

```python
nums = [10, 20]
it = iter(nums)      # calls nums.__iter__(): get an iterator
next(it)             # 10   (calls it.__next__())
next(it)             # 20
next(it)             # raises StopIteration: no more items
```

- An **iterable** can give you an iterator (`list`, `str`, `dict`, `file`, `range`...).
- An **iterator** produces values one at a time with `next()` and remembers where it is. Once exhausted, it stays exhausted.

A `for` loop is just:

```python
it = iter(nums)
while True:
    try:
        x = next(it)
    except StopIteration:
        break
    ...  # loop body
```

## 2. Writing an iterator class

```python
class Countdown:
    def __init__(self, start):
        self.current = start
    def __iter__(self):
        return self              # an iterator is its own iterator
    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1
```

It works, but it's a lot of ceremony. Generators do the same in a few lines.

## 3. Generators

A function containing `yield` is a **generator function**. Calling it doesn't run the body; it returns a generator, which runs the body up to each `yield` when you ask for the next value, and pauses there:

```python
def countdown(start):
    while start > 0:
        yield start          # hand out a value, pause here
        start -= 1

for n in countdown(3):
    print(n)                 # 3, 2, 1
```

The function's local variables survive between `yield`s. When the function ends, the generator raises `StopIteration` for you.

**Generator expressions** are the lazy version of list comprehensions:

```python
squares = [x * x for x in range(10**9)]   # tries to build a billion-item list: out of memory
squares = (x * x for x in range(10**9))   # instant: nothing computed yet
sum(x * x for x in range(10**6))           # fine: values flow straight into sum
```

## 4. Laziness and infinite sequences

Because generators compute on demand, they can be **infinite**:

```python
def naturals():
    n = 0
    while True:
        yield n
        n += 1
```

Only take what you need: `itertools.islice(naturals(), 5)` → 0, 1, 2, 3, 4.

## 5. Pipelines

Chain generators so each item flows through every stage before the next item is even read:

```python
def read_lines(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")

def parse(lines):
    for line in lines:
        if line:
            yield json.loads(line)

def long_texts(records, min_len):
    for r in records:
        if len(r["text"]) >= min_len:
            yield r

pipeline = long_texts(parse(read_lines("data.jsonl")), 100)
for record in pipeline:        # memory use stays tiny, however big the file is
    ...
```

`yield from other_generator` hands out every value of another iterable, handy for recursion (flattening nested structures) and for splitting a generator into parts.

## 6. itertools: the toolbox

```python
from itertools import islice, chain, count, cycle, groupby, accumulate, pairwise, product, combinations, batched

list(islice(count(10), 3))          # [10, 11, 12]
list(chain([1, 2], [3]))            # [1, 2, 3]
list(accumulate([1, 2, 3]))         # [1, 3, 6]   running totals
list(pairwise("abc"))               # [('a','b'), ('b','c')]
list(batched(range(5), 2))          # [(0,1), (2,3), (4,)]   (Python 3.12+)
```

Read the itertools docs once; you'll recognize problems they solve for years.

## 7. Gotchas

- A generator can only be consumed **once**. `list(gen)` twice gives `[]` the second time.
- Nothing runs until something asks for values, so errors inside a generator appear late, where the values are *consumed*.
- `len()` doesn't work on generators: they don't know their length in advance.

---

## Problem-solving habit #26: stream, don't load

When processing data, ask: "do I need everything in memory at once?" Usually you only need one item (or one batch) at a time. Writing the pipeline as generators makes the program work for 1 MB and for 100 GB with the same code.

## Go deeper (optional, research-level)

1. Generators can also *receive* values: `value = yield x` together with `gen.send(...)`. Read PEP 342. This is how Python's `async`/`await` (m33) was originally built.
2. Look at how PyTorch's `DataLoader` uses worker processes to prefetch batches. Why is loading in parallel with training so important for GPU utilization?
3. When an LLM API streams its answer, the client library gives you a generator of text chunks (the mentor in this repo uses one: see `stream.text_stream` in `mentor/client.py`). What are the user-experience and memory benefits of streaming?

## Your turn

Open the **Exercises** tab, solve each part, then press **Run tests**. Several tests feed your functions infinite generators, so make sure everything is lazy.
