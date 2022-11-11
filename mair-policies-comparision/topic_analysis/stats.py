import pandas as pd
from skbio.stats.distance import mantel
from statsmodels.multivariate.manova import MANOVA


def mantel_test_pval(
    distance_matrix_A: pd.DataFrame,
    distance_matrix_B: pd.DataFrame,
    n_permutations: int = 1000,
    corr_method: str = "pearson",
) -> float:
    return mantel(
        distance_matrix_A, distance_matrix_B, permutations=n_permutations, method=corr_method
    )[:2]


def manova_significant_difference_pval(topic_probs: pd.DataFrame, cluster: pd.Series) -> float:
    manova = MANOVA(endog=topic_probs, exog=cluster)
    test_res = manova.mv_test()
    pval = test_res.results["x0"]["stat"]["Pr > F"]["Pillai's trace"]
    return pval
