import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from exercises import (
    count_syllables,
    figure_problems,
    flesch_reading_ease,
    format_mean_std,
    learning_curve_figure,
    missing_sections,
    results_table,
    unsupported_claims,
)


def test_format_mean_std():
    assert format_mean_std([0.961, 0.969, 0.959], 1, 100) == "96.3 ± 0.5"
    assert format_mean_std([2.0], 2) == "2.00 ± 0.00"


def test_results_table():
    results = {"Baseline": {"Acc": [0.90, 0.91, 0.92], "Loss": [0.40, 0.42, 0.41]},
               "Ours": {"Acc": [0.93, 0.94, 0.95], "Loss": [0.45, 0.44, 0.46]}}
    table = results_table(results, ["Acc", "Loss"], {"Acc": True, "Loss": False}, decimals=1, scale=100)
    assert table.splitlines() == [
        "| Method | Acc | Loss |",
        "|---|---|---|",
        "| Baseline | 91.0 ± 1.0 | **41.0 ± 1.0** |",
        "| Ours | **94.0 ± 1.0** | 45.0 ± 1.0 |",
    ]


DRAFT = ("We study arithmetic in small transformers. Our method outperforms the baseline. "
         "Reversed answers improve accuracy from 76% to 99% over 4 seeds. "
         "This is significantly better than prior work [3]. It is also faster (Table 2). "
         "Training is clearly superior with our trick! We release code.")


def test_unsupported_claims():
    assert unsupported_claims(DRAFT) == ["Our method outperforms the baseline.",
                                         "Training is clearly superior with our trick!"]
    assert unsupported_claims("Figure 3 shows the best run is better.") == []
    assert unsupported_claims("Our approach is State-of-the-art.") == ["Our approach is State-of-the-art."]


def test_syllables_and_readability():
    assert [count_syllables(w) for w in ["cat", "make", "table", "free", "beautiful", "rhythm", "a"]] == [1, 1, 2, 1, 3, 1, 1]
    easy = "The cat sat on the mat. It was a good cat. We fed it fish."
    hard = ("Notwithstanding considerable computational expenditure, contemporary transformer-based "
            "architectures demonstrate substantially heterogeneous generalization characteristics.")
    assert flesch_reading_ease(easy) > 90
    assert flesch_reading_ease(hard) < 0
    words, sentences = 15, 3
    syll = 15
    assert flesch_reading_ease("The cat sat on the mat. It was a good cat. We fed it fish.") == \
        pytest.approx(206.835 - 1.015 * words / sentences - 84.6 * syll / words)
    assert flesch_reading_ease("") == 0.0


def test_figure_problems():
    fig, (a, b) = plt.subplots(1, 2)
    a.plot([1, 2], [3, 4])
    a.set_xlabel("step")
    b.plot([1, 2], [3, 4], label="x")
    b.plot([1, 2], [4, 5], label="y")
    b.set_xlabel("step")
    b.set_ylabel("loss")
    assert figure_problems(fig) == ["axes 0: missing y label", "axes 1: several lines but no legend"]
    b.legend()
    a.set_ylabel("acc")
    assert figure_problems(fig) == []
    plt.close(fig)


def test_learning_curve_figure():
    curves = {"plain": [[3.0, 2.0, 1.5], [3.2, 2.2, 1.7]], "reversed": [[3.0, 1.5, 1.0]]}
    fig = learning_curve_figure(curves)
    ax = fig.axes[0]
    assert figure_problems(fig) == []
    assert ax.get_xlabel() == "Training step" and ax.get_ylabel() == "Validation loss"
    lines = ax.get_lines()
    assert [l.get_label() for l in lines] == ["plain (n=2)", "reversed (n=1)"]
    assert np.allclose(lines[0].get_ydata(), [3.1, 2.1, 1.6]) and list(lines[0].get_xdata()) == [1, 2, 3]
    assert len(ax.collections) >= 1, "shade the spread across seeds"
    plt.close(fig)


def test_missing_sections():
    draft = "# Title\n\n## Results\ntext\n## Introduction\n\n### Method\n#not a heading\n"
    missing, out_of_order = missing_sections(draft, ["Introduction", "Method", "Results", "Limitations"])
    assert missing == ["Limitations"] and out_of_order is True
    ok = "## Introduction\n## Method\n## results\n## Limitations"
    assert missing_sections(ok, ["Introduction", "Method", "Results", "Limitations"]) == ([], False)
