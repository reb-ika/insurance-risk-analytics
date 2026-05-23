# insurance-risk-analytics
## Data Version Control (DVC)

This project uses DVC to version and track datasets.

### Reproduce the data pipeline
pip install dvc
dvc pull

### Data Versions
- Version 1: Raw dataset (MachineLearningRating_v3.txt) — 1,000,098 rows, 52 columns
- Version 2: Cleaning notes documenting dropped columns (100% and 99.9% missing)
