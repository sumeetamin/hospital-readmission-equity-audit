"""Patient-group holdout baseline for UCI's 30-day readmission label."""
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DROP = {"encounter_id", "patient_nbr", "readmitted", "race", "gender", "age", "weight",
        "discharge_disposition_id"}


def main():
    source = Path("data/diabetic_data.csv")
    df = pd.read_csv(source, low_memory=False)
    required = {"patient_nbr", "readmitted", "race", "gender"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing expected UCI columns: {sorted(missing)}")

    df["readmit_30d"] = (df["readmitted"].astype(str) == "<30").astype(int)
    groups = df["patient_nbr"].astype(str)
    x_cols = [c for c in df.columns if c not in DROP and c != "readmit_30d"]
    X = df[x_cols].replace("?", pd.NA)
    y = df["readmit_30d"]
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_i, test_i = next(split.split(X, y, groups=groups))
    train_x, test_x = X.iloc[train_i], X.iloc[test_i]
    y_train, y_test = y.iloc[train_i], y.iloc[test_i]
    train_groups, test_groups = set(groups.iloc[train_i]), set(groups.iloc[test_i])
    if train_groups & test_groups:
        raise RuntimeError("Patient leakage across partitions.")

    numeric = train_x.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in train_x.columns if c not in numeric]
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
        ]), categorical),
    ])
    model = Pipeline([
        ("features", prep),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear")),
    ])
    model.fit(train_x, y_train)
    proba = model.predict_proba(test_x)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    audit = {}
    test_meta = df.iloc[test_i][["race", "gender"]].reset_index(drop=True)
    for col in ["race", "gender"]:
        rows = {}
        for label, positions in test_meta.groupby(col, dropna=False).indices.items():
            positions = list(positions)
            if len(positions) < 100:
                continue
            yy = y_test.iloc[positions].to_numpy()
            pp = pred[positions]
            if len(set(yy)) < 2:
                tpr = fpr = None
            else:
                tpr = float(((pp == 1) & (yy == 1)).sum() / max(1, (yy == 1).sum()))
                fpr = float(((pp == 1) & (yy == 0)).sum() / max(1, (yy == 0).sum()))
            rows[str(label)] = {"n": len(positions), "positive_rate": float(yy.mean()),
                                "recall_at_0_5": tpr, "false_positive_rate_at_0_5": fpr}
        audit[col] = rows

    report = {
        "dataset": "UCI Diabetes 130-US Hospitals for Years 1999-2008",
        "target": "readmitted == '<30' versus '>30' or 'NO'; encounter-level label",
        "rows": {"all_encounters": int(len(df)), "train_encounters": int(len(train_i)),
                 "test_encounters": int(len(test_i)), "train_unique_patients": len(train_groups),
                 "test_unique_patients": len(test_groups), "patient_overlap": 0},
        "positive_prevalence": {"train": float(y_train.mean()), "test": float(y_test.mean())},
        "model": "Class-weighted logistic regression with train-only imputation/scaling/one-hot encoding",
        "features": x_cols,
        "excluded_predictors": sorted(DROP),
        "test_metrics_at_threshold_0_5": {
            "average_precision": float(average_precision_score(y_test, proba)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "brier_score": float(brier_score_loss(y_test, proba)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
        },
        "subgroup_audit": audit,
        "source_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "limitations": [
            "Retrospective 1999-2008 data from 130 hospitals; the benchmark is historical and may not transport to current care.",
            "GroupShuffleSplit holds out patients but is not chronological because dates are not supplied in this file.",
            "A fixed 0.5 threshold is not calibrated to clinical benefit, capacity, or harms.",
            "Sensitive fields and discharge disposition are excluded as predictors; race and gender are shown only for descriptive audits.",
            "This research baseline must not be used for individual care decisions, service eligibility, or resource denial.",
        ],
    }
    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

