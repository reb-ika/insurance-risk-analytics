
"""
hypothesis_tests.py
-------------------
Reusable statistical hypothesis testing functions for insurance risk analysis.
Tests: chi-squared, t-test, z-test for A/B hypothesis testing.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple


def compute_claim_frequency(group: pd.Series) -> float:
    """Proportion of policies with at least one claim."""
    return (group > 0).mean()


def compute_claim_severity(group: pd.Series) -> float:
    """Average claim amount given a claim occurred."""
    claims = group[group > 0]
    return claims.mean() if len(claims) > 0 else 0.0


def ttest_two_groups(
    group_a: pd.Series,
    group_b: pd.Series,
    alpha: float = 0.05
) -> dict:
    """
    Run independent two-sample t-test.
    
    Args:
        group_a: Control group values
        group_b: Test group values
        alpha: Significance level (default 0.05)
    
    Returns:
        Dictionary with t_stat, p_value, decision, mean_a, mean_b
    """
    t_stat, p_value = stats.ttest_ind(
        group_a.dropna(), group_b.dropna(), equal_var=False)
    return {
        "test": "Welch t-test",
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "mean_a": round(group_a.mean(), 4),
        "mean_b": round(group_b.mean(), 4),
        "difference": round(group_b.mean() - group_a.mean(), 4),
        "decision": "Reject H0" if p_value < alpha else "Fail to Reject H0",
        "significant": p_value < alpha
    }


def ztest_two_proportions(
    count_a: int, nobs_a: int,
    count_b: int, nobs_b: int,
    alpha: float = 0.05
) -> dict:
    """
    Run z-test for two proportions (claim frequency).
    
    Args:
        count_a: Number of claims in group A
        nobs_a: Total policies in group A
        count_b: Number of claims in group B
        nobs_b: Total policies in group B
        alpha: Significance level
    
    Returns:
        Dictionary with z_stat, p_value, decision, freq_a, freq_b
    """
    from statsmodels.stats.proportion import proportions_ztest
    counts = np.array([count_a, count_b])
    nobs = np.array([nobs_a, nobs_b])
    z_stat, p_value = proportions_ztest(counts, nobs)
    return {
        "test": "Z-test (proportions)",
        "z_statistic": round(z_stat, 4),
        "p_value": round(p_value, 6),
        "freq_a": round(count_a / nobs_a, 6),
        "freq_b": round(count_b / nobs_b, 6),
        "decision": "Reject H0" if p_value < alpha else "Fail to Reject H0",
        "significant": p_value < alpha
    }


def chisq_test(
    group_a: pd.Series,
    group_b: pd.Series,
    alpha: float = 0.05
) -> dict:
    """
    Run chi-squared test on two categorical distributions.
    
    Args:
        group_a: Control group categorical series
        group_b: Test group categorical series
        alpha: Significance level
    
    Returns:
        Dictionary with chi2, p_value, decision
    """
    all_cats = pd.concat([group_a, group_b]).unique()
    obs_a = group_a.value_counts().reindex(all_cats, fill_value=0)
    obs_b = group_b.value_counts().reindex(all_cats, fill_value=0)
    contingency = pd.DataFrame({'A': obs_a, 'B': obs_b})
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency.T)
    return {
        "test": "Chi-squared",
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 6),
        "degrees_of_freedom": dof,
        "decision": "Reject H0" if p_value < alpha else "Fail to Reject H0",
        "significant": p_value < alpha
    }


def print_result(hypothesis: str, result: dict) -> None:
    """Print formatted hypothesis test result."""
    print(f"\n{'='*60}")
    print(f"Hypothesis: {hypothesis}")
    print(f"{'='*60}")
    for k, v in result.items():
        print(f"  {k}: {v}")
    if result['significant']:
        print(f"  ✅ SIGNIFICANT — Reject H0 (p < 0.05)")
    else:
        print(f"  ❌ NOT SIGNIFICANT — Fail to Reject H0 (p >= 0.05)")
