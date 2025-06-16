from collections import Counter
import numpy as np


def create_meshgrid(x1_range, x2_range=None, num_points=1000):
    x2_range = x2_range if x2_range is not None else x1_range

    x1 = np.linspace(x1_range[0], x1_range[1], num_points)
    x2 = np.linspace(x2_range[0], x2_range[1], num_points)
    x = np.array(np.meshgrid(x1, x2)).reshape(-1, 2).astype(np.float64)
    return x


def mode(x):
    counter = Counter(x)
    most_common = counter.most_common(1)
    if most_common:
        return most_common[0][0]
    else:
        return None


def save_figs(idx, solution_figs, hist_figs):
    method_titles = ["lrs", "grs", "hc"]
    for fig, title in zip(solution_figs, method_titles):
        fig.savefig(f"images/solution_{idx}_{title}.png")
    for fig, title in zip(hist_figs, method_titles):
        fig.savefig(f"images/hist_{idx}_{title}.png")
