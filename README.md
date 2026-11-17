# Hospital Readmission Equity Audit

A research-only benchmark for the historical UCI Diabetes 130-US Hospitals dataset. It predicts the dataset's **encounter-level `<30` readmission label** from encounter fields, and reports race/gender subgroup errors for audit. This is not a current clinical model, care-priority tool, or individual risk estimate.

## Data and run

The source contains 101,766 inpatient encounters across 130 U.S. hospitals, 1999–2008. UCI marks race, gender, and age as potentially sensitive. The source archive and row-level records are downloaded locally and are not committed.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python fetch_data.py
python benchmark.py
```

The model predicts `readmitted == '<30'` versus `'>30'` or `'NO'`. A seeded 20% patient-group holdout keeps all encounters for one patient in a single partition. The split is not chronological because this source file does not provide encounter dates.

## Observed result

On 20,488 held-out encounters from 14,304 distinct patients, the positive rate was 11.6%. Class-weighted logistic regression produced average precision 0.211 (prevalence baseline 0.116), ROC-AUC 0.640, and Brier score 0.228.

At a fixed 0.5 threshold, precision was 16.8%, recall 53.8%, and balanced accuracy 0.595, with 6,303 false positives and 1,273 true positives. Subgroup rates in `reports/benchmark.json` are descriptive: for example, race-group recall ranged from 33.3% (Asian, n=110) to 59.0% (African American, n=3,942), while false-positive rates ranged from 21.8% to 36.3%. These gaps need investigation; they do not establish a cause or a fairness guarantee.

## Modeling choices and limits

- Patient number and encounter ID are used only to split/check overlap, never as predictors.
- Race, gender, age, weight, discharge disposition, and the target are excluded from predictors. Race and gender appear only in subgroup reporting.
- Features include encounter-level diagnoses, utilization counts, labs, medications, and admission details. All preprocessing is fitted on training data only.
- This historical cohort has selection criteria and missingness that limit generalization. Patient-group holdout prevents same-person leakage but does not simulate a future deployment.
- The 0.5 threshold is not calibrated to clinical utility or resource capacity. Errors can affect patient care; do not use these outputs to make individual decisions, restrict services, or allocate care.

## Source and attribution

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014), [Diabetes 130-US Hospitals for Years 1999-2008, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us-hospitals+for-years-1999-2008), DOI [10.24432/C5230J](https://doi.org/10.24432/C5230J). UCI lists the dataset under CC BY 4.0. This repository does not include the dataset itself.

