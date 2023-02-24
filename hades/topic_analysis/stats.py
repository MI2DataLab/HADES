import pandas as pd
from statsmodels.multivariate.manova import MANOVA


def manova_significant_difference_pval(topic_probs: pd.DataFrame, cluster: pd.Series) -> float:
    """Performs a MANOVA test to check if there is a significant difference between the topic probabilities of two clusters."""
    manova = MANOVA(endog=topic_probs, exog=cluster)
    test_res = manova.mv_test()
    pval = test_res.results["x0"]["stat"]["Pr > F"]["Pillai's trace"]
    return pval
