from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt


def _matrix_split(
    topic_probs: pd.DataFrame, covariates_set: pd.DataFrame, num_topics: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged_df = topic_probs.merge(covariates_set, on=topic_probs.index.name)
    matrix_numeric = merged_df.select_dtypes("number")
    matrix_nonnumeric = merged_df.select_dtypes("object")
    matrix_numeric = (matrix_numeric - matrix_numeric.mean()) / matrix_numeric.std()
    X = matrix_numeric.iloc[:, :num_topics]
    Y = matrix_numeric.iloc[:, num_topics:]
    return X, Y, matrix_nonnumeric


def _calculate_cca_repr(
    X: pd.DataFrame, Y: pd.DataFrame, n_components: int = 2
) -> Tuple[np.array, np.array]:
    ca = CCA(n_components=n_components)
    ca.fit(X, Y)
    X_c, Y_c = ca.transform(X, Y)
    return X_c, Y_c


def cca(
    topic_probs: pd.DataFrame, covariates_set: pd.DataFrame, num_topics: int, n_components: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, Y, matrix_nonnumeric = _matrix_split(topic_probs, covariates_set, num_topics)
    X_c, Y_c = _calculate_cca_repr(X, Y, n_components)
    cc_df = pd.DataFrame(np.hstack([X_c, Y_c]))
    cc_df.columns = ["CCX_" + str(i) for i in range(1, n_components + 1)] + [
        "CCY_" + str(i) for i in range(n_components)
    ]
    cc_df.set_index(X.index, inplace=True)
    return cc_df, X, Y, matrix_nonnumeric


def cca_biplot(
    cca_result: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    d1: int = 1,
    d2: int = 2,
    canonical_variables_from: str = "X",
    type: str = "variables",
    figsize: tuple = (12, 12),
    colors: Tuple[str, str] = ("red", "blue"),
    save_path: Optional[str] = None,
):
    cc_df, X, Y, _ = cca_result
    d = [f"CC{canonical_variables_from}_{d1}", f"CC{canonical_variables_from}_{d2}"]
    corr_X_xscores = cc_df[d].apply(X.corrwith)
    corr_Y_xscores = cc_df[d].apply(Y.corrwith)

    plt.figure(figsize=figsize)
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))

    if type == "variables":
        for var_i in range(corr_X_xscores.shape[0]):
            x = corr_X_xscores.iloc[var_i, 0]
            y = corr_X_xscores.iloc[var_i, 1]

            plt.arrow(0, 0, x, y)
            plt.text(x, y, X.columns[var_i], color=colors[0])

        for var_i in range(corr_Y_xscores.shape[0]):
            x = corr_Y_xscores.iloc[var_i, 0]
            y = corr_Y_xscores.iloc[var_i, 1]

            plt.arrow(0, 0, x, y)
            plt.text(x, y, Y.columns[var_i], color=colors[1])
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
