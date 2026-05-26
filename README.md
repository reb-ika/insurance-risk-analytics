# insurance-risk-analyticsgit checkout main

# Insurance Risk Analytics & Predictive Modeling
**Kifiya AI Training Program — Week 3**
*AlphaCare Insurance Solutions (ACIS) | South Africa Auto Insurance*

---

## Business Objective
ACIS needs evidence-driven strategies to optimize marketing investments
and refine pricing models for the South African auto-insurance market.

This project delivers:
1. **EDA** — Understanding 1M+ insurance policies across 18 months
2. **DVC** — Reproducible, versioned data pipeline
3. **Hypothesis Testing** — Statistical validation of risk drivers
4. **Predictive Modeling** — Claim severity and probability models
5. **Risk-Based Pricing** — Dynamic premium formula using ML

---

## Project Structure
insurance-risk-analytics/
├── .github/workflows/ci.yml     # CI/CD pipeline
├── data/                        # Tracked by DVC, not Git
│   ├── MachineLearningRating_v3.txt
│   └── MachineLearningRating_v3.txt.dvc
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_hypothesis_testing.ipynb  # A/B Hypothesis Testing
│   └── 03_modeling.ipynb        # ML Models & SHAP
├── src/
│   ├── data_loader.py           # Data loading & cleaning
│   ├── hypothesis_tests.py      # Statistical test functions
│   └── modeling.py              # Model training functions
├── tests/                       # Unit tests
├── reports/final_report.md      # Final report
├── dvc.yaml                     # DVC pipeline config
└── requirements.txt
---

## Setup Instructions

### 1. Clone repository
```bash
git clone https://github.com/reb-ika/insurance-risk-analytics.git
cd insurance-risk-analytics
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

### 3. Pull data with DVC
```bash
dvc pull
```

### 4. Run notebooks in order
notebooks/01_eda.ipynb
notebooks/02_hypothesis_testing.ipynb
notebooks/03_modeling.ipynb
---

## Data Version Control (DVC)

```bash
# Pull data
dvc pull

# Push new data version
dvc add data/MachineLearningRating_v3.txt
dvc push
```

### Data Versions
- **Version 1**: Raw dataset — 1,000,098 rows, 52 columns
- **Version 2**: Cleaned — dropped CrossBorder (99.9% missing) and NumberOfVehiclesInFleet (100% missing)

---

## Dataset Overview

| Field Group | Key Fields |
|------------|-----------|
| Policy | PolicyID, UnderwrittenCoverID |
| Transaction | TransactionMonth (Feb 2014 – Aug 2015) |
| Location | Province, PostalCode |
| Vehicle | VehicleType, Make, RegistrationYear |
| Financial | TotalPremium, TotalClaims, SumInsured |

**Derived Metrics:**
- Loss Ratio = TotalClaims / TotalPremium
- Margin = TotalPremium − TotalClaims

---

## Key Findings

### EDA
- Portfolio Loss Ratio: **0.79** (losing money overall)
- Total Margin: **-R2.96M** (claims exceed premiums)
- Claim Frequency: **0.28%** (very low but high severity)
- Highest risk province: **Gauteng** (LR = 1.22)
- Highest risk vehicle: **Heavy Commercial** (LR = 1.63)

### Hypothesis Testing
| Hypothesis | Test | P-Value | Decision |
|-----------|------|---------|---------|
| H01: Province risk differences (severity) | t-test | 0.031 | ✅ Reject H0 |
| H01: Province risk differences (frequency) | z-test | 0.000 | ✅ Reject H0 |
| H02: Zip code risk differences | t-test | 0.700 | ❌ Fail to Reject |
| H03: Zip code margin differences | t-test | 0.244 | ❌ Fail to Reject |
| H04: Gender risk differences | t-test | 0.568 | ❌ Fail to Reject |

### Modeling Results
**Claim Severity (Regression):**
| Model | RMSE | R² |
|-------|------|-----|
| Linear Regression | — | — |
| Random Forest | — | — |
| XGBoost | — | — |

*(Update with actual results from notebook)*

---

## Tools & Libraries

| Tool | Purpose |
|------|---------|
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| SciPy / Statsmodels | Statistical testing |
| Scikit-learn | ML models |
| XGBoost | Gradient boosting |
| SHAP | Model interpretability |
| DVC | Data version control |
| pytest | Unit testing |

---

## Author
**Rebika Woldeyesus**
Kifiya AI Training Program — Week 3
GitHub: [@reb-ika](https://github.com/reb-ika)
