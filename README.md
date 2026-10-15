# Hospital Readmission Equity Audit

A research-only readmission baseline with subgroup error-rate reporting and explicit clinical-use safeguards.

> **Portfolio prototype.** Code is provided; no benchmark results are claimed until you run it on the cited data and record the output.

## Use case

A research-only readmission baseline with subgroup error-rate reporting and explicit clinical-use safeguards. The command below reads a local data export, applies a transparent baseline, writes holdout or decision-support outputs, and saves the fitted model where applicable.

## Data and provenance

[UCI Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us-hospitals+for+years+1999-2008). A patient-encounter CSV and binary target (for example a derived 30-day readmission indicator). Choose a subgroup column such as `race` only for audit, not as an automatic care rule. Dataset files are not bundled. Respect the source terms and cite the dataset when publishing results.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Download/prepare the source data locally, then run:
python main.py --input data/diabetic_data.csv --target readmit_30d --positive-label 1 --group race --out outputs
```

## Outputs

The `outputs/` directory contains the model-specific results (metrics and/or ranked CSV) and a serialized estimator where applicable. Outputs are reproducible with the same input, package versions, and random seed. The repository intentionally excludes downloaded data and generated artifacts.

## Evaluation and operating notes

Historical observational data with sensitive health information. This is not a clinical decision system; do not use for individual care, resource denial, or care prioritization.

For a production review, preserve a time-based, untouched holdout; compare against a simple business baseline; inspect calibration and subgroup error; document data lineage; and monitor drift. The included baseline is not a production service.

## License

No code license file is included yet. Third-party datasets remain subject to their original terms.
