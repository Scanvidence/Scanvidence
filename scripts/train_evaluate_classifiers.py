"""Stage-3 & Stage-4: Downstream Classification, Bootstrap CIs, and SHAP Explainability.

Evaluates classifiers (Logistic Regression, Random Forest, XGBoost) on feature panels from Stage 2.
Computes 5-fold CV out-of-fold predictions, 2,000 bootstrap 95% CIs, and SHAP explanations (Hypotheses H1a & H2b).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean and 95% bootstrap confidence intervals for ROC-AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_aucs = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        # Ensure both classes are present in sample
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc = roc_auc_score(y_true[idx], y_prob[idx])
        boot_aucs.append(auc)

    mean_auc = float(np.mean(boot_aucs))
    ci_lower = float(np.percentile(boot_aucs, 2.5))
    ci_upper = float(np.percentile(boot_aucs, 97.5))
    return mean_auc, ci_lower, ci_upper


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3 & 4 Classifiers and SHAP Explainability.")
    ap.add_argument(
        "--stage2-json",
        default="cache/selection_stage2.json",
        help="Stage 2 feature selection output JSON",
    )
    ap.add_argument(
        "--data-csv",
        default="cache/radiomics_gt_dev.csv",
        help="Radiomics features CSV",
    )
    ap.add_argument(
        "--out",
        default="cache/classification_stage3.json",
        help="Stage 3 output summary JSON",
    )
    ap.add_argument(
        "--n-bootstraps",
        type=int,
        default=2000,
        help="Number of bootstrap iterations for 95% CIs",
    )
    args = ap.parse_args()

    print("--- Stage 3: Downstream Classification & Bootstrap Evaluation ---")

    # 1. Load Data & Stage 2 Selections
    st2 = json.loads(Path(args.stage2_json).read_text())
    df = pd.read_csv(args.data_csv)

    df_labeled = df.dropna(subset=["mgmt"]).copy()
    df_labeled["mgmt"] = df_labeled["mgmt"].astype(int)

    tr = df_labeled[df_labeled["partition"] == "train"].copy()
    va = df_labeled[df_labeled["partition"] == "val"].copy()
    cal = df_labeled[df_labeled["partition"] == "calibration"].copy()

    # Define feature panels to benchmark
    panels: dict[str, list[str]] = {}
    for m, v in st2["methods"].items():
        if "selected" in v and len(v["selected"]) > 0:
            panels[m] = v["selected"]

    # Add candidate pool and full 253 as baselines
    panels["candidate_pool_20"] = st2["candidate_pool"]
    with open("cache/radiomics_schema_v1.json") as f:
        schema = json.load(f)
    panels["all_253_features"] = schema["feature_ids"]

    # Impute missing values with training medians
    all_needed_feats = sorted(list(set([f for sub in panels.values() for f in sub])))
    medians = tr[all_needed_feats].median()
    tr[all_needed_feats] = tr[all_needed_feats].fillna(medians)
    va[all_needed_feats] = va[all_needed_feats].fillna(medians)
    cal[all_needed_feats] = cal[all_needed_feats].fillna(medians)

    y_train = tr["mgmt"].values
    y_val = va["mgmt"].values

    print(f"Cohorts: Train={len(tr)} cases, Val={len(va)} cases, Calibration={len(cal)} cases")
    print(f"Evaluating {len(panels)} feature panels across 3 model families...\n")

    results: dict[str, Any] = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for panel_name, feat_list in panels.items():
        results[panel_name] = {
            "n_features": len(feat_list),
            "features": feat_list,
            "models": {},
        }

        X_tr_raw = tr[feat_list].values
        X_va_raw = va[feat_list].values

        # Test model families
        model_factories = {
            "LogisticRegression": lambda: LogisticRegression(
                penalty="l2", C=1.0, max_iter=500, random_state=42
            ),
            "RandomForest": lambda: RandomForestClassifier(
                n_estimators=100, max_depth=4, random_state=42
            ),
            "XGBoost": lambda: XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                random_state=42,
                eval_metric="logloss",
            ),
        }

        for mname, mfactory in model_factories.items():
            oof_preds = np.zeros(len(tr))

            # 5-fold CV on Train
            for train_idx, test_idx in skf.split(X_tr_raw, y_train):
                scaler = StandardScaler()
                X_t = scaler.fit_transform(X_tr_raw[train_idx])
                X_te = scaler.transform(X_tr_raw[test_idx])

                clf = mfactory()
                clf.fit(X_t, y_train[train_idx])
                oof_preds[test_idx] = clf.predict_proba(X_te)[:, 1]

            # 5-fold CV bootstrap metrics
            mean_auc, ci_low, ci_high = bootstrap_ci(
                y_train,
                oof_preds,
                n_bootstraps=args.n_bootstraps,
                seed=42,
            )
            brier = float(brier_score_loss(y_train, oof_preds))

            # Train on full train set & evaluate on independent validation set
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_raw)
            X_va_scaled = scaler.transform(X_va_raw)

            full_clf = mfactory()
            full_clf.fit(X_tr_scaled, y_train)
            val_preds = full_clf.predict_proba(X_va_scaled)[:, 1]
            val_auc = float(roc_auc_score(y_val, val_preds))
            val_brier = float(brier_score_loss(y_val, val_preds))

            results[panel_name]["models"][mname] = {
                "cv_auc_mean": round(mean_auc, 4),
                "cv_auc_95ci": [round(ci_low, 4), round(ci_high, 4)],
                "cv_brier": round(brier, 4),
                "val_auc": round(val_auc, 4),
                "val_brier": round(val_brier, 4),
            }

    # Summary table for presentation
    summary_rows = []
    for pname, pdata in results.items():
        for mname, mdata in pdata["models"].items():
            ci_str = f"[{mdata['cv_auc_95ci'][0]:.3f}, {mdata['cv_auc_95ci'][1]:.3f}]"
            summary_rows.append(
                {
                    "Feature Panel": pname,
                    "k": pdata["n_features"],
                    "Classifier": mname,
                    "5-Fold CV AUC (95% CI)": f"{mdata['cv_auc_mean']:.3f} {ci_str}",
                    "Val AUC": f"{mdata['val_auc']:.3f}",
                    "Val Brier": f"{mdata['val_brier']:.3f}",
                }
            )

    sum_df = pd.DataFrame(summary_rows)
    print("--- STAGE 3 BENCHMARK: 5-FOLD CV & VALIDATION RESULTS ---")
    print(sum_df.to_string(index=False))

    # --- Stage 4: SHAP Explainability on the winning QUBO panel ---
    print("\n--- Stage 4: SHAP Explainability on QUBO Optimal Biomarker Panel ---")
    qubo_features = results["qubo_exact"]["features"]
    scaler = StandardScaler()
    X_tr_qubo = scaler.fit_transform(tr[qubo_features].values)
    X_va_qubo = scaler.transform(va[qubo_features].values)

    rf_qubo = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf_qubo.fit(X_tr_qubo, y_train)

    explainer = shap.TreeExplainer(rf_qubo)
    shap_values = explainer.shap_values(X_va_qubo)

    # In binary classification, take class 1 SHAP values
    shap_vals_class1 = (
        shap_values[:, :, 1]
        if len(np.shape(shap_values)) == 3
        else shap_values[1]
        if isinstance(shap_values, list)
        else shap_values
    )

    mean_abs_shap = np.mean(np.abs(shap_vals_class1), axis=0)
    shap_rank = pd.Series(mean_abs_shap, index=qubo_features).sort_values(ascending=False)

    print("SHAP Feature Importance (Mean |SHAP| on Validation Cohort):")
    for i, (fname, imp) in enumerate(shap_rank.items()):
        print(f"  {i + 1:2d}. {fname:55s} | Mean |SHAP| = {imp:.4f}")

    # Top-k ablation faithfulness test (Hypothesis H2b)
    print("\nValidating Hypothesis H2b (SHAP Attribution Faithfulness)...")
    ablation_scores = []
    for k_drop in range(1, len(qubo_features)):
        top_k_drop = shap_rank.head(k_drop).index.tolist()
        keep_feats = [f for f in qubo_features if f not in top_k_drop]

        sc = StandardScaler()
        X_t_abl = sc.fit_transform(tr[keep_feats].values)
        X_v_abl = sc.transform(va[keep_feats].values)

        m = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        m.fit(X_t_abl, y_train)
        score = roc_auc_score(y_val, m.predict_proba(X_v_abl)[:, 1])
        ablation_scores.append(
            {
                "k_dropped": k_drop,
                "dropped_features": top_k_drop,
                "auc_after_ablation": round(score, 4),
            }
        )

    results["shap_explainability"] = {
        "panel": "qubo_exact",
        "feature_ranking": shap_rank.to_dict(),
        "faithfulness_ablation": ablation_scores,
    }

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2))
    print(f"\nSaved complete Stage 3 & 4 results to {out_p}")


if __name__ == "__main__":
    main()
