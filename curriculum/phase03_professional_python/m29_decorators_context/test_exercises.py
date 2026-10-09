import math
import os
import random
import time

import pytest

import exercises
from code_checks import uses_any
from exercises import (
    ACTIVATIONS,
    Timer,
    check_types,
    collect_errors,
    count_calls,
    get_activation,
    memoize,
    register,
    retry,
    temporary_seed,
    timed,
    working_directory,
)


def test_count_calls():
    @count_calls
    def greet(name):
        """Say hi."""
        return f"hi {name}"

    assert greet.calls == 0
    assert greet("a") == "hi a"
    greet(name="b")
    assert greet.calls == 2
    assert greet.__name__ == "greet" and greet.__doc__ == "Say hi."


def test_timed():
    @timed
    def nap(seconds):
        time.sleep(seconds)
        return "done"

    assert nap(0.05) == "done"
    assert 0.04 < nap.last_elapsed < 1
    assert nap.__name__ == "nap"


def test_memoize():
    calls = []

    @memoize
    def slow_square(n):
        calls.append(n)
        return n * n

    assert [slow_square(3), slow_square(3), slow_square(4)] == [9, 9, 16]
    assert calls == [3, 4]
    assert slow_square.cache_info() == {"hits": 1, "misses": 2, "size": 2}
    assert slow_square.__name__ == "slow_square"


def test_memoize_makes_recursion_fast():
    @memoize
    def fib(n):
        return n if n < 2 else fib(n - 1) + fib(n - 2)

    assert fib(200) == 280571172992510140037611932413038677189525
    assert not uses_any(exercises.memoize, "cache", "lru_cache")


def test_retry_succeeds():
    attempts = []

    @retry(times=3, exceptions=(TimeoutError,))
    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise TimeoutError
        return "ok"

    assert flaky() == "ok"
    assert len(attempts) == 3
    assert flaky.__name__ == "flaky"


def test_retry_gives_up_and_ignores_other_errors():
    attempts = []

    @retry(times=2)
    def always():
        attempts.append(1)
        raise ConnectionError(len(attempts))

    with pytest.raises(ConnectionError, match="2"):
        always()

    other = []

    @retry(times=5, exceptions=(TimeoutError,))
    def wrong_kind():
        other.append(1)
        raise ValueError

    with pytest.raises(ValueError):
        wrong_kind()
    assert len(other) == 1


def test_retry_delay():
    @retry(times=3, delay=0.05)
    def fails():
        raise RuntimeError

    start = time.perf_counter()
    with pytest.raises(RuntimeError):
        fails()
    assert time.perf_counter() - start >= 0.09


def test_registry():
    assert ACTIVATIONS["relu"](-2) == 0 and ACTIVATIONS["relu"](3) == 3
    assert get_activation("identity")(7) == 7
    assert get_activation("sigmoid")(0) == pytest.approx(0.5)
    with pytest.raises(KeyError, match="relu"):
        get_activation("swish")


def test_register_returns_function_and_rejects_duplicates():
    @register("square_test")
    def square(x):
        return x * x

    assert square(3) == 9
    assert ACTIVATIONS["square_test"] is square
    with pytest.raises(ValueError):
        @register("square_test")
        def other(x):
            return x


def test_check_types():
    @check_types
    def scale(values: list, factor: float, label="x"):
        return [v * factor for v in values]

    assert scale([1, 2], 2.0) == [2.0, 4.0]
    assert scale([1], factor=1.5, label=3) == [1.5]
    with pytest.raises(TypeError, match="factor"):
        scale([1], "2")
    with pytest.raises(TypeError, match="values"):
        scale(values=(1, 2), factor=1.0)
    assert scale.__name__ == "scale"


def test_timer_context():
    with Timer() as t:
        time.sleep(0.05)
    assert 0.04 < t.elapsed < 1
    with pytest.raises(ZeroDivisionError):
        with Timer() as t2:
            1 / 0
    assert t2.elapsed >= 0


def test_working_directory(tmp_path):
    start = os.getcwd()
    with working_directory(tmp_path):
        assert os.path.samefile(os.getcwd(), tmp_path)
    assert os.getcwd() == start
    with pytest.raises(KeyError):
        with working_directory(tmp_path):
            raise KeyError
    assert os.getcwd() == start


def test_temporary_seed_is_reproducible_and_restores_state():
    random.seed(123)
    before = random.getstate()
    with temporary_seed(42):
        first = [random.random() for _ in range(3)]
    assert random.getstate() == before
    with temporary_seed(42):
        second = [random.random() for _ in range(3)]
    assert first == second
    random.seed(42)
    assert first == [random.random() for _ in range(3)]


def test_temporary_seed_restores_after_error():
    random.seed(5)
    before = random.getstate()
    with pytest.raises(RuntimeError):
        with temporary_seed(1):
            raise RuntimeError
    assert random.getstate() == before


def test_collect_errors():
    with collect_errors(ValueError, KeyError) as errors:
        int("not a number")
    assert len(errors) == 1 and isinstance(errors[0], ValueError)
    with collect_errors(ValueError) as errors:
        pass
    assert errors == []
    with pytest.raises(ZeroDivisionError):
        with collect_errors(ValueError):
            1 / 0
