import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure

from exercises import (
    confusion_heatmap,
    histogram,
    image_grid,
    line_plot,
    loss_curves,
    ranked_bars,
    save_figure,
    scatter_by_class,
)

rng = np.random.default_rng(48)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def legend_texts(ax):
    legend = ax.get_legend()
    assert legend is not None, "add a legend"
    return [t.get_text() for t in legend.get_texts()]


def test_line_plot():
    x = [0, 1, 2]
    fig = line_plot(x, {"a": [1, 2, 3], "b": [3, 2, 1]}, "Title", "x axis", "y axis")
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == ("Title", "x axis", "y axis")
    assert len(ax.get_lines()) == 2
    assert list(ax.get_lines()[1].get_ydata()) == [3, 2, 1]
    assert legend_texts(ax) == ["a", "b"]


def test_loss_curves():
    train = [2.0, 1.0, 0.5, 0.3, 0.2]
    val = [2.1, 1.2, 0.8, 0.9, 1.1]
    fig = loss_curves(train, val)
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == ("Loss", "epoch", "loss")
    lines = {l.get_label(): l for l in ax.get_lines()}
    assert list(lines["train"].get_xdata()) == [1, 2, 3, 4, 5]
    assert list(lines["validation"].get_ydata()) == val
    points = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(points) == 1
    np.testing.assert_allclose(points[0].get_offsets()[0], [3, 0.8])


def test_histogram():
    values = rng.normal(5, 1, 1000)
    fig = histogram(values, 20, "Feature")
    ax = fig.axes[0]
    assert ax.get_title() == "Feature"
    assert len(ax.patches) == 20
    dashed = [l for l in ax.get_lines() if l.get_linestyle() == "--"]
    assert len(dashed) == 1
    assert dashed[0].get_xdata()[0] == pytest.approx(values.mean())


def test_scatter_by_class():
    X = rng.normal(size=(30, 2))
    labels = np.array([0] * 10 + [1] * 12 + [2] * 8)
    fig = scatter_by_class(X, labels, ["cat", "dog", "bird"])
    ax = fig.axes[0]
    groups = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert [len(g.get_offsets()) for g in groups] == [10, 12, 8]
    assert legend_texts(ax) == ["cat", "dog", "bird"]
    np.testing.assert_allclose(groups[1].get_offsets(), X[labels == 1])


def test_ranked_bars():
    s = pd.Series({"pen": 30.0, "lamp": 120.0, "notebook": 75.0})
    fig = ranked_bars(s, "revenue")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "revenue"
    bars = sorted(ax.patches, key=lambda p: p.get_y())          # bottom to top
    assert [round(b.get_width()) for b in bars] == [30, 75, 120], "largest bar on top"
    fig.canvas.draw()
    labels_bottom_to_top = [t.get_text() for t in ax.get_yticklabels()]
    assert labels_bottom_to_top == ["pen", "notebook", "lamp"]


def test_image_grid():
    images = rng.random((6, 8, 8))
    titles = [f"img {i}" for i in range(6)]
    fig = image_grid(images, titles, ncols=4)
    assert len(fig.axes) == 8                                   # 2 rows x 4 columns
    shown = [ax for ax in fig.axes if ax.images]
    assert len(shown) == 6
    assert [ax.get_title() for ax in shown] == titles
    assert all(not ax.axison for ax in fig.axes)


def test_confusion_heatmap():
    cm = np.array([[5, 1, 0], [2, 7, 1], [0, 0, 9]])
    fig = confusion_heatmap(cm, ["a", "b", "c"])
    ax = fig.axes[0]
    assert len(fig.axes) == 2, "add a colorbar"
    assert ax.images and np.array_equal(ax.images[0].get_array(), cm)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("predicted", "true")
    assert sorted(t.get_text() for t in ax.texts) == sorted(str(v) for v in cm.flatten())
    fig.canvas.draw()
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]


def test_save_figure(tmp_path):
    fig = line_plot([0, 1], {"a": [0, 1]}, "t", "x", "y")
    path = tmp_path / "plot.png"
    save_figure(fig, path, dpi=50)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not plt.fignum_exists(fig.number), "close the figure after saving"
