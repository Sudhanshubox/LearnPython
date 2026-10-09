import asyncio
import inspect
import time

import pytest

import exercises
from code_checks import uses_any
from exercises import (
    collect,
    evaluate_prompts,
    fetch_all,
    fetch_limited,
    fetch_with_retry,
    fetch_with_timeout,
    first_result,
    process_with_workers,
    stream_words,
    thread_map,
)


def run(coro):
    return asyncio.run(coro)


class FakeAPI:
    """A fake async API that tracks how many calls run at once."""

    def __init__(self, delay=0.1, fail_times=None, delays=None):
        self.delay = delay
        self.delays = delays or {}
        self.fail_times = dict(fail_times or {})   # item -> how many times to fail first
        self.running = 0
        self.max_running = 0
        self.calls = []

    async def __call__(self, item):
        self.calls.append(item)
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            await asyncio.sleep(self.delays.get(item, self.delay))
            if self.fail_times.get(item, 0) > 0:
                self.fail_times[item] -= 1
                raise ConnectionError(f"temporary failure for {item}")
            return f"answer:{item}"
        finally:
            self.running -= 1


def timed(coro):
    start = time.perf_counter()
    result = run(coro)
    return result, time.perf_counter() - start


def test_no_blocking_sleep():
    for name in ["fetch_all", "fetch_limited", "fetch_with_retry", "stream_words", "evaluate_prompts"]:
        source = inspect.getsource(getattr(exercises, name))
        assert "time.sleep" not in source, f"{name}: use await asyncio.sleep, not time.sleep"


def test_fetch_all_is_concurrent_and_ordered():
    api = FakeAPI(delay=0.1)
    results, elapsed = timed(fetch_all(range(20), api))
    assert results == [f"answer:{i}" for i in range(20)]
    assert elapsed < 0.5, "the calls should overlap, not run one after another"
    assert api.max_running == 20


def test_fetch_limited_respects_limit():
    api = FakeAPI(delay=0.05)
    results, elapsed = timed(fetch_limited(range(20), api, limit=4))
    assert results == [f"answer:{i}" for i in range(20)]
    assert api.max_running == 4
    assert 0.2 < elapsed < 1.0


def test_fetch_with_timeout():
    api = FakeAPI(delays={"slow": 1.0, "fast": 0.01})
    assert run(fetch_with_timeout(api, "fast", 0.5)) == "answer:fast"
    result, elapsed = timed(fetch_with_timeout(api, "slow", 0.1, default="gave up"))
    assert result == "gave up"
    assert elapsed < 0.5


def test_fetch_with_retry_recovers():
    api = FakeAPI(delay=0, fail_times={"x": 2})
    assert run(fetch_with_retry(api, "x", attempts=3, base_delay=0.01)) == "answer:x"
    assert api.calls == ["x", "x", "x"]


def test_fetch_with_retry_backoff_and_give_up():
    api = FakeAPI(delay=0, fail_times={"x": 10})
    start = time.perf_counter()
    with pytest.raises(ConnectionError):
        run(fetch_with_retry(api, "x", attempts=4, base_delay=0.02))
    elapsed = time.perf_counter() - start
    assert len(api.calls) == 4
    assert elapsed >= 0.02 + 0.04 + 0.08 - 0.01, "wait base, 2*base, 4*base between the 4 tries"


def test_first_result_cancels_the_rest():
    api = FakeAPI(delays={"a": 0.5, "b": 0.05, "c": 0.5})
    result, elapsed = timed(first_result(["a", "b", "c"], api))
    assert result == "answer:b"
    assert elapsed < 0.3, "return as soon as the first one finishes and cancel the others"


def test_process_with_workers():
    api = FakeAPI(delay=0.05)
    results, elapsed = timed(process_with_workers(list(range(12)), api, n_workers=3))
    assert results == {i: f"answer:{i}" for i in range(12)}
    assert api.max_running == 3
    assert elapsed < 0.6


def test_streaming():
    agen = stream_words("the quick brown fox", delay=0.01)
    assert inspect.isasyncgen(agen)
    assert run(collect(agen)) == "the quick brown fox"


def test_evaluate_prompts():
    api = FakeAPI(
        delay=0.02,
        fail_times={"flaky": 1, "broken": 99},
        delays={"hangs": 5.0},
    )
    prompts = ["p1", "flaky", "p2", "broken", "hangs", "p3"]
    results, elapsed = timed(evaluate_prompts(prompts, api, limit=3, timeout=0.1, attempts=3, base_delay=0.01))
    assert results == ["answer:p1", "answer:flaky", "answer:p2", None, None, "answer:p3"]
    assert api.max_running <= 3
    assert elapsed < 2.0
    assert api.calls.count("hangs") == 3, "a timed-out attempt should be retried"


def test_thread_map():
    def slow_square(x):
        time.sleep(0.1)          # a BLOCKING call, like requests.get
        return x * x

    start = time.perf_counter()
    assert thread_map(slow_square, range(16), max_workers=16) == [x * x for x in range(16)]
    assert time.perf_counter() - start < 0.6
    assert uses_any(exercises.thread_map, "ThreadPoolExecutor")
