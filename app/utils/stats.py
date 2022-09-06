from statsmodels.multivariate.manova import MANOVA
import pandas as pd
import numpy as np


def manova_significant_difference_pval(topic_probs: pd.DataFrame, cluster: pd.Series) -> float:
    manova = MANOVA(endog=topic_probs.values, exog=cluster.astype(int))
    test_res = manova.mv_test()
    pval = test_res.results["x0"]["stat"]["Pr > F"]["Pillai's trace"]
    return pval
