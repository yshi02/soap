import matplotlib.pyplot as plt
import numpy as np

num_sms_required = [
    16,
    8,
    8,
    8,
    8,
    5,
    5,
    5,
    5,
    5,
    4,
    4,
    4,
    4,
    4,
    4,
    3,
    3,
    3,
    3,
    3,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
]
times = [
    13.60692308,
    7.469230769,
    4.484615385,
    10.42846154,
    4.468461538,
    4.410769231,
    3.180769231,
    4.670769231,
    3.182307692,
    3.156923077,
    3.156153846,
    4.233846154,
    2.766923077,
    2.774615385,
    2.763846154,
    4.221538462,
    2.609230769,
    2.276923077,
    2.293076923,
    3.422307692,
    2.298461538,
    2.263076923,
    2.106153846,
    2.685384615,
    2.116923077,
    2.116923077,
    2.119230769,
    2.097692308,
    3.366153846,
    2.66,
    2.116153846,
    2.113076923,
]

_COLORS = ["#808080", "#DA1E28", "#0072C3", "#007D79", "#8A3FFC"]


def scale_tail(A, B, scale_A, scale_B):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    A[-4:] *= scale_A
    B[-4:] *= scale_B
    return A, B


def plot_variable_width_bars(series, title="", xlabel="", ylabel=""):
    """Plot one or more series as variable-width bars.

    Args:
        series: list of (A, B) or (A, B, label) tuples.
                A controls bar height; B controls bar width.
        title, xlabel, ylabel: axis labels.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    for idx, s in enumerate(series):
        A = np.array(s[0], dtype=float)
        B = np.array(s[1], dtype=float)
        label = s[2] if len(s) > 2 else None
        assert len(A) == len(B), "A and B must have equal length"

        lefts = np.concatenate([[0], np.cumsum(B[:-1])])
        color = _COLORS[idx % len(_COLORS)]
        ax.bar(
            lefts,
            A,
            width=B,
            align="edge",
            color=color,
            alpha=0.6,
            label=label,
            linewidth=0,
        )

    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)

    if any(len(s) > 2 for s in series):
        ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_variable_width_bars(
        [
            (
                *scale_tail(num_sms_required, times, 20, 1),
                "conv + 20x angles in parallel",
            ),
            (
                *scale_tail(num_sms_required, times, 10, 2),
                "conv + 10x (2x angles) in parallel",
            ),
            (
                *scale_tail(num_sms_required, times, 7, 3),
                "conv + 7x (3x angles) in parallel",
            ),
            (
                *scale_tail(num_sms_required, times, 5, 4),
                "conv + 5x (4x angles) in parallel",
            ),
            (
                *scale_tail(num_sms_required, times, 4, 5),
                "conv + 4x (5x angles) in parallel",
            ),
        ],
        title="Number of SMs required vs. runtime of the layer for conv+angles layers",
        xlabel="Cumulative time (μs)",
        ylabel="SMs required",
    )
