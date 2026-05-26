
"""
modeling.py
-----------
Modular modeling functions for insurance risk analytics.
Supports Linear Regression, Random Forest, and XGBoost.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score,
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score)
from xgboost import XGBRegressor, XGBClassifier


def train_severity_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate three severity regression models.
    Returns dict of results with RMSE and R2.
    """
    results = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results['Linear Regression'] = {
        'model': lr,
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=100, random_state=42,
        n_jobs=-1, max_depth=10)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results['Random Forest'] = {
        'model': rf,
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=100, random_state=42,
        max_depth=6, learning_rate=0.1, verbosity=0)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    results['XGBoost'] = {
        'model': xgb,
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }

    return results


def train_classifier_models(X_train, X_test, y_train, y_test,
                             pos_weight=1.0):
    """
    Train and evaluate three classification models.
    Returns dict of results with F1 and AUC.
    """
    results = {}

    # Logistic Regression
    lr = LogisticRegression(
        class_weight='balanced', random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]
    results['Logistic Regression'] = {
        'model': lr,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42,
        class_weight='balanced', n_jobs=-1, max_depth=10)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    results['Random Forest'] = {
        'model': rf,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=100, random_state=42,
        scale_pos_weight=pos_weight,
        max_depth=6, learning_rate=0.1,
        verbosity=0, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    results['XGBoost'] = {
        'model': xgb,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }

    return results


def risk_based_premium(p_claim, predicted_severity,
                       expense_loading=0.15, profit_margin=0.10):
    """
    Compute risk-based premium using actuarial formula.
    Premium = P(claim) * Severity * (1 + expense + profit)
    """
    return p_claim * predicted_severity * (1 + expense_loading + profit_margin)
