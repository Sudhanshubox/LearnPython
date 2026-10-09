# m10 · Debugging like an engineer

**By the end you can:** find bugs systematically instead of by guessing, use `print`, `breakpoint()` and `logging` well, and recognize the bug patterns that cause most real-world failures.

**Why it matters for AI:** in normal software, bugs usually crash. In machine learning, they usually *don't*: the code runs, the loss goes down a bit, and the model is quietly worse than it should be. A shuffled label column, a wrong normalization, an off-by-one in a sequence model. Debugging discipline is what separates people who ship working models from people who don't.

---

## 1. The debugging loop

1. **Reproduce** the bug reliably, with the smallest input you can find.
2. **Read** the error (bottom of the traceback first, m08).
3. **Hypothesize**: "I think `total` is wrong after the first iteration."
4. **Test** the hypothesis by looking at real values: print them, or stop in the debugger.
5. **Fix** the cause, not the symptom, then **add a test** so it never comes back.

The most common mistake is skipping step 4: changing code at random until the error disappears. That usually creates a second bug.

## 2. Print debugging, done well

```python
print(f"{i=} {total=} {item=}")       # prints: i=3 total=17 item='x'
print(f"{type(value)=} {len(batch)=}")
```

The `=` inside an f-string prints the expression *and* its value. Print **types and lengths**, not only values: many bugs are a `str` where you expected an `int`, or a list one item too short.

## 3. The debugger: breakpoint()

```python
def train_step(batch):
    loss = compute_loss(batch)
    breakpoint()          # execution pauses here, with an interactive prompt
    ...
```

At the `(Pdb)` prompt:

| Command | Does |
|---|---|
| `p expr` | print any expression, e.g. `p batch[0]` |
| `n` | run the next line |
| `s` | step *into* a function call |
| `c` | continue until the next breakpoint |
| `l` | list the code around you |
| `w` | show the call stack (where am I?) |
| `q` | quit |

VS Code and PyCharm have the same features with a visual interface: click left of a line number to set a breakpoint.

## 4. logging instead of print

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

log.debug("batch shape %s", shape)        # hidden unless level=DEBUG
log.info("epoch %d loss %.4f", epoch, loss)
log.warning("skipped %d bad rows", bad)
log.error("could not load %s", path)
```

Logs have levels, timestamps, and can go to a file. Leave them in your code; delete debugging `print`s.

## 5. Bisection: halve the search space

If a pipeline of 10 steps gives a wrong result, check the output of step 5. Wrong? The bug is in steps 1–5. Right? It's in 6–10. Repeat. You find the bug in about 4 checks instead of 10. The same trick works on data (which of 100,000 rows breaks the parser?) and on history (`git bisect` finds which commit introduced a bug).

## 6. The usual suspects

Most bugs you'll ever write fall into a few families. Learn to recognize them on sight; every one of them is planted in this module's exercises.

- **Off-by-one:** `range(len(x) - 1)` vs `range(len(x))`, slices one item too long or short, `<` vs `<=`.
- **Integer vs float:** `//` where you meant `/`.
- **Operator precedence:** `a - b / c - d` is not `(a - b) / (c - d)`.
- **Changing a list while looping over it:** items get skipped.
- **Aliasing:** `result = original` and then modifying `result` also modifies `original` (m01, m05).
- **Wrong initial value:** starting a count at 1, a minimum at 0.
- **Late-binding closures:** functions created in a loop all see the loop variable's *final* value.
- **Shadowing:** naming a variable `list`, `sum` or `max` breaks the built-in for the rest of the scope.

## 7. Sanity checks for data and ML code

- Look at a few raw examples before training. Then look at them *after* preprocessing.
- Check shapes, value ranges and counts (`min`, `max`, `mean`, number of NaNs).
- **Overfit a tiny batch:** a correct model should reach near-zero loss on 10 examples. If it can't, there's a bug.
- Compare with a dumb baseline (always predict the most common class).

---

## Problem-solving habit #10: explain it out loud

"Rubber duck debugging": explain your code line by line to someone (a rubber duck works, and so does your mentor). Saying *why* each line is correct often reveals the line that isn't. Try it in the mentor panel: paste a function and explain it before asking for help.

## Go deeper (optional, research-level)

1. Read Andrej Karpathy's blog post *A Recipe for Training Neural Networks* (2019). Which of its advice is really about debugging?
2. How does `breakpoint()` work? Read PEP 553 and find out how the `PYTHONBREAKPOINT` environment variable lets you switch debuggers or disable all breakpoints.
3. Learn `git bisect` on a toy repo: make 16 commits, plant a bug in one of them, and find it with `git bisect run pytest`.

## Your turn

Every function in `exercises.py` has **exactly one bug**. Run the tests, read the failures, find and fix each bug, and add a short comment saying what was wrong. Don't rewrite the functions from scratch: practising *finding* the bug is the point.
