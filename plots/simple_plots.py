from collections import Counter
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_counter(
    counter: Counter, orient: str = "h", color: str = "lightblue", figsize: tuple = (20, 13)
):
    plt.figure(figsize=figsize)
    keys = [k[0] for k in counter]
    vals = [int(k[1]) for k in counter]
    ax = sns.barplot(x=vals, y=keys, orient=orient, color=color)
    ax.bar_label(ax.containers[0])
    return ax


def plot_counter_lemmas(
    df: pd.DataFrame,
    filter_dict: Dict[str, str],
    number: int = 30,
    orient: str = "h",
    color: str = "lightblue",
    figsize: tuple = (20, 13),
):
    filtered_lemmas = df[(df[key] == value for key, value in filter_dict.items())]["lemmas"]
    counter = Counter(filtered_lemmas.sum()).most_common(number)
    plot_counter(counter, orient, color, figsize)
    plt.show()
