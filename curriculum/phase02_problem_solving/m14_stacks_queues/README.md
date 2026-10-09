# m14 · Stacks and queues

**By the end you can:** use stacks and queues to solve matching, parsing, "next greater element" and scheduling problems, and pick the right Python tool (`list` or `collections.deque`) for each.

**Why it matters for AI:** the call stack runs every recursive function you write. Expression parsers, compilers and tokenizers are built on stacks. Queues schedule jobs on GPU clusters and serve requests to model APIs. Breadth-first search (graphs, Phase 2) is a queue, and beam search for text generation keeps a queue-like set of candidates.

---

## 1. Stack: last in, first out (LIFO)

Think of a stack of plates: you add and remove at the **top**.

```python
stack = []
stack.append("a")     # push
stack.append("b")
stack[-1]             # peek: 'b'
stack.pop()           # pop: 'b'
not stack             # is it empty?
```

A Python list is a perfect stack: `append` and `pop()` at the end are O(1).

## 2. Classic stack problem: matching brackets

```python
PAIRS = {")": "(", "]": "[", "}": "{"}

def balanced(text):
    stack = []
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in PAIRS:
            if not stack or stack.pop() != PAIRS[ch]:
                return False
    return not stack       # anything left open is unbalanced
```

Whenever a problem has **nesting** (brackets, HTML tags, function calls, undo history), think stack.

## 3. Evaluating expressions

Humans write `3 + 4 * 2` (infix). Computers prefer **Reverse Polish Notation** (postfix): `3 4 2 * +`. It needs no parentheses and evaluates with one stack: push numbers; on an operator, pop two, compute, push the result.

```
3 4 2 * +      stack: [3] → [3, 4] → [3, 4, 2] → [3, 8] → [11]
```

The **shunting-yard algorithm** (Dijkstra, 1961) converts infix to postfix, with a stack for operators. You'll implement it in exercise 4.

## 4. Monotonic stack: "next greater element"

For each day's temperature, how many days until a warmer one? Brute force is O(n²). Keep a stack of indices whose answer is still unknown, with temperatures decreasing from bottom to top. Each new temperature resolves everything smaller on top of the stack:

```python
def days_until_warmer(temps):
    answer = [0] * len(temps)
    stack = []                           # indices, temperatures decreasing
    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            j = stack.pop()
            answer[j] = i - j
        stack.append(i)
    return answer
```

Each index is pushed and popped at most once: **O(n)**.

## 5. Queue: first in, first out (FIFO)

Like a line at a shop: join at the back, leave from the front.

```python
from collections import deque

q = deque()
q.append("a")         # enqueue at the back
q.append("b")
q.popleft()           # dequeue from the front: 'a'
q.appendleft("z")     # deques work at both ends
```

**Never use `list.pop(0)` as a queue.** It shifts every remaining item: O(n) per pop, O(n²) overall. `deque.popleft()` is O(1).

A `deque(maxlen=k)` automatically drops the oldest item when full. That's a ready-made sliding window, for example the last 100 losses for a smoothed training curve.

## 6. Designing your own data structure

Interviews (and real systems) often ask you to build a structure with specific costs. For example, a **min-stack** supports push, pop and `get_min` all in O(1). The trick: alongside each item, store the minimum *at the time it was pushed*.

To build one, you need a **class**: a way to bundle data with the functions that use it. You'll study classes properly in Phase 3; for now this pattern is all you need:

```python
class Counter:
    def __init__(self):          # runs when you create one: Counter()
        self.count = 0           # self.count is stored on this object

    def increment(self):
        self.count += 1
        return self.count

c = Counter()
c.increment()                    # 1
c.increment()                    # 2
```

---

## Problem-solving habit #13: what am I waiting for?

Stacks fit problems where items **wait** for something later to resolve them (a closing bracket, a warmer day, an operator's operands), and the most recent waiting item is resolved first. If the *oldest* waiting item is resolved first, it's a queue.

## Go deeper (optional, research-level)

1. Python's own interpreter is a **stack machine**. Run `import dis; dis.dis(lambda a, b: a + b * 2)` and trace the stack by hand for each bytecode instruction.
2. Beam search (used to generate text from language models) keeps the k best partial sequences at each step. How is it different from greedy decoding, and why can it still miss the overall most likely sequence?
3. Look up "amortized analysis". Prove that a queue built from **two stacks** has O(1) amortized cost per operation, even though a single dequeue can take O(n). (You'll implement it in exercise 7.)

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**.
