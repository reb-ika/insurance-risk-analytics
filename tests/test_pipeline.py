
"""
test_pipeline.py
----------------
Unit tests for insurance risk analytics pipeline.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import clean_data
from src.hypothesis_tests import (
    ttest_two_groups, ztest_two_proportions,
    label_sentiment, compute_claim_frequency,
    compute_claim_severity
)
from src.modeling import risk_based_premium


# ── FIXTURES ──

@pytest.fixture
def sample_df():
    """Sample insurance DataFrame for testing."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        'TotalPremium': np.random.uniform(0, 500, n),
        'TotalClaims': np.concatenate([
            np.zeros(990),
            np.random.uniform(1000, 50000, 10)
        ]),
        'Province': np.random.choice(
            ['Gauteng', 'Western Cape', 'KwaZulu-Natal'], n),
        'VehicleType': np.random.choice(
            ['Passenger Vehicle', 'Heavy Commercial'], n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'SumInsured': np.random.uniform(5000, 500000, n),
        'RegistrationYear': np.random.randint(2000, 2015, n),
        'TransactionMonth': pd.date_range(
            '2014-01-01', periods=n, freq='h'),
        'AlarmImmobiliser': np.random.choice(['Yes', 'No'], n),
        'TrackingDevice': np.random.choice(['Yes', 'No'], n),
        'CrossBorder': np.nan,
        'NumberOfVehiclesInFleet': np.nan
    })


# ── DATA LOADER TESTS ──

class TestDataLoader:

    def test_clean_drops_high_missing_columns(self, sample_df):
        """Columns >90% missing should be dropped."""
        result = clean_data(sample_df.copy())
        assert 'CrossBorder' not in result.columns
        assert 'NumberOfVehiclesInFleet' not in result.columns

    def test_clean_adds_loss_ratio(self, sample_df):
        """clean_data should add LossRatio column."""
        result = clean_data(sample_df.copy())
        assert 'LossRatio' in result.columns

    def test_clean_adds_margin(self, sample_df):
        """clean_data should add Margin column."""
        result = clean_data(sample_df.copy())
        assert 'Margin' in result.columns

    def test_clean_adds_has_claim(self, sample_df):
        """clean_data should add HasClaim binary column."""
        result = clean_data(sample_df.copy())
        assert 'HasClaim' in result.columns

    def test_loss_ratio_calculation(self, sample_df):
        """LossRatio should equal TotalClaims/TotalPremium."""
        result = clean_data(sample_df.copy())
        mask = result['TotalPremium'] > 0
        expected = (result.loc[mask, 'TotalClaims'] /
                    result.loc[mask, 'TotalPremium'])
        pd.testing.assert_series_equal(
            result.loc[mask, 'LossRatio'].round(6),
            expected.round(6),
            check_names=False)

    def test_margin_calculation(self, sample_df):
        """Margin should equal TotalPremium - TotalClaims."""
        result = clean_data(sample_df.copy())
        expected = result['TotalPremium'] - result['TotalClaims']
        pd.testing.assert_series_equal(
            result['Margin'], expected, check_names=False)

    def test_preserves_row_count(self, sample_df):
        """clean_data should not drop rows."""
        result = clean_data(sample_df.copy())
        assert len(result) == len(sample_df)


# ── HYPOTHESIS TEST TESTS ──

class TestHypothesisTests:

    def test_ttest_rejects_different_means(self):
        """t-test should reject H0 for clearly different groups."""
        np.random.seed(42)
        group_a = pd.Series(np.random.normal(1000, 100, 500))
        group_b = pd.Series(np.random.normal(5000, 100, 500))
        result = ttest_two_groups(group_a, group_b)
        assert result['significant'] == True
        assert result['p_value'] < 0.05

    def test_ttest_fails_similar_means(self):
        """t-test should fail to reject H0 for similar groups."""
        np.random.seed(42)
        group_a = pd.Series(np.random.normal(1000, 100, 500))
        group_b = pd.Series(np.random.normal(1001, 100, 500))
        result = ttest_two_groups(group_a, group_b)
        assert result['significant'] == False

    def test_ttest_returns_required_keys(self):
        """t-test result must have required keys."""
        group_a = pd.Series([1, 2, 3, 4, 5])
        group_b = pd.Series([10, 20, 30, 40, 50])
        result = ttest_two_groups(group_a, group_b)
        required = ['test', 't_statistic', 'p_value',
                    'mean_a', 'mean_b', 'decision', 'significant']
        for key in required:
            assert key in result

    def test_ztest_returns_required_keys(self):
        """z-test result must have required keys."""
        result = ztest_two_proportions(10, 1000, 50, 1000)
        required = ['test', 'z_statistic', 'p_value',
                    'freq_a', 'freq_b', 'decision', 'significant']
        for key in required:
            assert key in result

    def test_ztest_rejects_different_proportions(self):
        """z-test should reject H0 for clearly different proportions."""
        result = ztest_two_proportions(5, 1000, 200, 1000)
        assert result['significant'] == True

    def test_claim_frequency_zero_claims(self):
        """Claim frequency should be 0 when no claims."""
        series = pd.Series([0.0, 0.0, 0.0, 0.0])
        assert compute_claim_frequency(series) == 0.0

    def test_claim_frequency_all_claims(self):
        """Claim frequency should be 1 when all have claims."""
        series = pd.Series([100.0, 200.0, 300.0])
        assert compute_claim_frequency(series) == 1.0

    def test_claim_severity_excludes_zeros(self):
        """Claim severity should only average non-zero claims."""
        series = pd.Series([0.0, 0.0, 1000.0, 2000.0])
        assert compute_claim_severity(series) == 1500.0

    def test_claim_severity_all_zeros(self):
        """Claim severity should return 0 when no claims."""
        series = pd.Series([0.0, 0.0, 0.0])
        assert compute_claim_severity(series) == 0.0


# ── MODELING TESTS ──

class TestModeling:

    def test_risk_premium_positive(self):
        """Risk premium should always be positive."""
        premium = risk_based_premium(0.1, 10000)
        assert premium > 0

    def test_risk_premium_formula(self):
        """Risk premium should follow the actuarial formula."""
        p = 0.1
        severity = 10000
        expected = p * severity * (1 + 0.15 + 0.10)
        result = risk_based_premium(p, severity)
        assert abs(result - expected) < 0.01

    def test_risk_premium_zero_probability(self):
        """Zero claim probability should give zero premium."""
        assert risk_based_premium(0.0, 10000) == 0.0

    def test_risk_premium_custom_loadings(self):
        """Custom expense and profit loadings should work."""
        result = risk_based_premium(0.1, 10000, 0.20, 0.05)
        expected = 0.1 * 10000 * (1 + 0.20 + 0.05)
        assert abs(result - expected) < 0.01

    def test_risk_premium_scales_with_severity(self):
        """Higher severity should produce higher premium."""
        p1 = risk_based_premium(0.1, 5000)
        p2 = risk_based_premium(0.1, 10000)
        assert p2 > p1

    def test_risk_premium_scales_with_probability(self):
        """Higher claim probability should produce higher premium."""
        p1 = risk_based_premium(0.1, 10000)
        p2 = risk_based_premium(0.5, 10000)
        assert p2 > p1
