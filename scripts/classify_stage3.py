"""Stage 3: Nested-CV classifiers on each selector's feature subset."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def load_data():
    df = pd.read_csv("cache/radiomics_gt_dev.csv")
    mgmt = pd.read_csv("data/manifests/mgmt_labels_v1.csv").rename(columns={"case_id": "case"})
    if "mgmt" not in df.columns:
        df = df.merge(mgmt[["case", "mgmt"]], on="case", how="inner")
    else:
        df = df.dropna(subset=["mgmt"]).copy()
        df["mgmt"] = df["mgmt"].astype(int)

    schema = json.loads(Path("cache/radiomics_schema_v1.json").read_text())
    sel = json.loads(Path("cache/selection_stage2.json").read_text())

    tr = df[df.partition == "train"].copy()
    va = df[df.partition == "val"].copy()
    print(
        f"cls-train: {len(tr)} (pos {tr.mgmt.mean():.3f}), cls-val: {len(va)} (pos {va.mgmt.mean():.3f})"
    )
    return tr, va, schema["feature_ids"], sel


def impute_scale(Xtr_df, Xva_df):
    mu = Xtr_df.mean()
    sd = Xtr_df.std().replace(0, 1)
    # Fill any remaining NaNs with column median
    med = Xtr_df.median()
    Xtr_filled = Xtr_df.fillna(med)
    Xva_filled = Xva_df.fillna(med)
    return ((Xtr_filled - mu) / sd).fillna(0).values, ((Xva_filled - mu) / sd).fillna(0).values


def nested_cv_auc(X, y, clf_class, clf_params, n_splits=5, seed=17):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, val_idx in skf.split(X, y):
        Xtr, Xva = X[train_idx], X[val_idx]
        ytr, yva = y[train_idx], y[val_idx]
        clf = clf_class(**clf_params, random_state=17).fit(Xtr, ytr)
        proba = clf.predict_proba(Xva)[:, 1]
        if len(np.unique(yva)) > 1:
            aucs.append(roc_auc_score(yva, proba))
    return np.array(aucs)


def final_val_auc(Xtr, ytr, Xva, yva, clf_class, clf_params, n_boot=200, seed=17):
    clf = clf_class(**clf_params, random_state=17).fit(Xtr, ytr)
    proba = clf.predict_proba(Xva)[:, 1]
    if len(np.unique(yva)) < 2:
        return np.nan, (np.nan, np.nan)
    auc = roc_auc_score(yva, proba)

    def auc_stat(p, y):
        if len(np.unique(y)) < 2:
            return 0.5
        return roc_auc_score(y, p)

    res = bootstrap(
        (proba, yva),
        auc_stat,
        paired=True,
        n_resamples=n_boot,
        random_state=seed,
        method="percentile",
    )
    return auc, res.confidence_interval


def permutation_null(Xtr, ytr, Xva, yva, clf_class, clf_params, n_perm=200, seed=17):
    rng = np.random.default_rng(seed)
    null_aucs = []
    clf = clf_class(**clf_params, random_state=17).fit(Xtr, ytr)
    proba = clf.predict_proba(Xva)[:, 1]
    for _ in range(n_perm):
        yva_shuf = rng.permutation(yva)
        if len(np.unique(yva_shuf)) > 1:
            null_aucs.append(roc_auc_score(yva_shuf, proba))
    return np.array(null_aucs)


def main():
    tr, va, feats, sel = load_data()

    # Retrieve selected subsets
    qpu_feats = (
        sel.get("methods", {}).get("qpu_ibm_fez", {}).get("selected")
        or sel.get("qpu", {}).get("selected")
        or sel["methods"]["qubo_classical"]["selected"]
    )

    methods = {
        "qubo_exact": sel["methods"]["qubo_exact"]["selected"],
        "qubo_classical": sel["methods"]["qubo_classical"]["selected"],
        "qpu_ibm_fez": qpu_feats,
        "mrmr": sel["methods"]["mrmr"]["selected"],
        "mi_topk": sel["methods"]["mi_topk"]["selected"],
        "f_anova": sel["methods"]["f_anova"]["selected"],
        "l1": sel["methods"]["l1"]["selected"],
        "rf": sel["methods"]["rf"]["selected"],
        "random": sel["methods"]["random"]["selected"],
    }

    classifiers = {
        "Logistic": (LogisticRegression, {"C": 1.0, "max_iter": 1000}),
        "RandomForest": (
            RandomForestClassifier,
            {"n_estimators": 200, "max_depth": 5},
        ),
    }

    print("\n--- STAGE 3: NESTED-CV & PERMUTATION NULL RESULTS ---")
    print(
        f"{'Method':20s} | {'Classifier':15s} | {'CV AUC (5-Fold)':18s} | {'Val AUC [95% CI]':24s} | {'Perm p-val':10s}"
    )
    print("-" * 96)

    results = {}
    for method, feat_list in methods.items():
        valid_feats = [f for f in feat_list if f in feats and f in tr.columns]
        if len(valid_feats) < 3:
            results[method] = {"error": f"only {len(valid_feats)} valid features"}
            continue

        Xtr_raw = tr[valid_feats]
        Xva_raw = va[valid_feats]
        ytr, yva = tr.mgmt.values, va.mgmt.values
        Xtr, Xva = impute_scale(Xtr_raw, Xva_raw)

        results[method] = {"n_features": len(valid_feats), "classifiers": {}}
        for clf_name, (clf_class, clf_params) in classifiers.items():
            # Nested CV on cls-train
            cv_aucs = nested_cv_auc(Xtr, ytr, clf_class, clf_params)
            # Final eval on cls-val
            val_auc, ci = final_val_auc(Xtr, ytr, Xva, yva, clf_class, clf_params)
            # Permutation null
            null = permutation_null(Xtr, ytr, Xva, yva, clf_class, clf_params)
            p_val = float((null >= val_auc).mean()) if not np.isnan(val_auc) else np.nan

            ci_low = float(ci.low) if not np.isnan(ci.low) else np.nan
            ci_high = float(ci.high) if not np.isnan(ci.high) else np.nan

            results[method]["classifiers"][clf_name] = {
                "cv_auc": f"{cv_aucs.mean():.3f} ± {cv_aucs.std():.3f}",
                "val_auc": round(val_auc, 4) if not np.isnan(val_auc) else np.nan,
                "val_ci_95": [round(ci_low, 4), round(ci_high, 4)],
                "perm_p": round(p_val, 4) if not np.isnan(p_val) else np.nan,
            }
            print(
                f"{method:20s} | {clf_name:15s} | CV AUC {cv_aucs.mean():.3f}±{cv_aucs.std():.3f} | Val AUC {val_auc:.3f} [{ci_low:.3f}, {ci_high:.3f}] | p={p_val:.3f}"
            )

    out_p = Path("cache/classify_stage3.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_p}")


if __name__ == "__main__":
    main()
