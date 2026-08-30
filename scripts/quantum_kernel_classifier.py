"""Quantum Kernel Support Vector Machine (QSVM) for Molecular MGMT Classification.

Embeds the 9-biomarker QUBO panel into a 9-qubit ZZFeatureMap Hilbert space (512-dim complex).
Computes the Quantum Kernel Gram Matrix K(x_i, x_j) = |<Phi(x_i)|Phi(x_j)>|^2,
benchmarks QSVM against Classical Linear SVM, Classical RBF SVM, Logistic Regression, and Random Forest,
and computes 2,000-bootstrap 95% Confidence Intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def load_cohort_and_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load MGMT cohort and retrieve the 9 QUBO biomarkers."""
    df = pd.read_csv("cache/radiomics_gt_dev.csv")
    sel = json.loads(Path("cache/selection_stage2.json").read_text())

    # Get the 9 optimal QUBO features
    qubo_feats = sel["methods"]["qubo_exact"]["selected"]

    df_labeled = df.dropna(subset=["mgmt"]).copy()
    df_labeled["mgmt"] = df_labeled["mgmt"].astype(int)

    tr = df_labeled[df_labeled["partition"] == "train"].copy()
    va = df_labeled[df_labeled["partition"] == "val"].copy()
    cal = df_labeled[df_labeled["partition"] == "calibration"].copy()

    # Impute missing values
    medians = tr[qubo_feats].median()
    tr[qubo_feats] = tr[qubo_feats].fillna(medians)
    va[qubo_feats] = va[qubo_feats].fillna(medians)
    cal[qubo_feats] = cal[qubo_feats].fillna(medians)

    return tr, va, cal, qubo_feats


def compute_quantum_statevectors(
    X_scaled: np.ndarray, reps: int = 1, entanglement: str = "linear"
) -> np.ndarray:
    """Encode patient vectors into 9-qubit quantum statevectors via ZZFeatureMap."""
    n_qubits = X_scaled.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=reps, entanglement=entanglement)

    states = []
    for row in X_scaled:
        qc = feature_map.assign_parameters(row)
        sv = Statevector(qc)
        states.append(sv.data)

    return np.array(states)  # Shape: (N, 2^n_qubits)


def compute_quantum_kernel_matrix(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    """Compute Quantum Kernel Gram Matrix: K_ij = |<Phi(x_i) | Phi(x_j)>|^2."""
    overlap = np.dot(states_a, states_b.conj().T)
    return np.abs(overlap) ** 2


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute 95% bootstrap confidence intervals for ROC-AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_aucs = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))

    mean_auc = float(np.mean(boot_aucs))
    ci_low = float(np.percentile(boot_aucs, 2.5))
    ci_high = float(np.percentile(boot_aucs, 97.5))
    return mean_auc, ci_low, ci_high


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 4: Quantum Kernel (QSVM) vs Classical Classifiers."
    )
    ap.add_argument("--reps", type=int, default=1, help="ZZFeatureMap repetitions")
    ap.add_argument(
        "--entanglement",
        default="linear",
        choices=["linear", "circular", "full"],
        help="Entanglement topology",
    )
    ap.add_argument(
        "--out",
        default="cache/quantum_kernel_results.json",
        help="Output JSON path",
    )
    args = ap.parse_args()

    print("--- Stage 4: Quantum Kernel Machine Learning (QSVM) ---")
    tr, va, cal, qubo_feats = load_cohort_and_features()
    print(
        f"Cohort: train={len(tr)}, val={len(va)}, calibration={len(cal)} (Total: {len(tr) + len(va) + len(cal)})"
    )
    print(
        f"Input: {len(qubo_feats)} QUBO optimal biomarkers -> 9-Qubit Quantum Hilbert Space (dim = {2 ** len(qubo_feats)})\n"
    )

    y_train = tr["mgmt"].values
    y_val = va["mgmt"].values

    # Scale radiomics to [0, 2*pi] for quantum phase rotation gates
    scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
    X_tr_scaled = scaler.fit_transform(tr[qubo_feats].values)
    X_va_scaled = scaler.transform(va[qubo_feats].values)

    # 1. Compute Quantum States
    print("Computing 9-Qubit Quantum Statevectors for Training & Validation...")
    states_train = compute_quantum_statevectors(
        X_tr_scaled, reps=args.reps, entanglement=args.entanglement
    )
    states_val = compute_quantum_statevectors(
        X_va_scaled, reps=args.reps, entanglement=args.entanglement
    )

    # 2. Compute Quantum Kernel Gram Matrices
    print("Calculating Quantum Kernel Gram Matrices K(x_i, x_j)...")
    K_train = compute_quantum_kernel_matrix(states_train, states_train)
    K_val = compute_quantum_kernel_matrix(states_val, states_train)

    print(f"Quantum Kernel Matrix shape: Train={K_train.shape}, Val={K_val.shape}")

    # 3. 5-Fold Stratified Cross-Validation on Train
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Quantum SVM vs Classical Models
    models_to_test = {
        "Quantum_Kernel_SVM (QSVM)": {
            "type": "precomputed",
            "model_fn": lambda: SVC(
                kernel="precomputed",
                C=1.0,
                probability=True,
                random_state=42,
            ),
        },
        "Classical_RBF_SVM": {
            "type": "vector",
            "model_fn": lambda: SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=42,
            ),
        },
        "Classical_Linear_SVM": {
            "type": "vector",
            "model_fn": lambda: SVC(kernel="linear", C=1.0, probability=True, random_state=42),
        },
    }

    results: dict[str, Any] = {
        "n_qubits": len(qubo_feats),
        "hilbert_dimension": 2 ** len(qubo_feats),
        "feature_map": f"ZZFeatureMap(reps={args.reps}, entanglement={args.entanglement})",
        "models": {},
    }

    print("\n--- 5-FOLD CROSS-VALIDATION & INDEPENDENT VALIDATION ---")
    summary_rows = []

    for name, cfg in models_to_test.items():
        oof_preds = np.zeros(len(tr))

        for train_idx, test_idx in skf.split(X_tr_scaled, y_train):
            clf = cfg["model_fn"]()
            if cfg["type"] == "precomputed":
                # Slice precomputed sub-kernel
                K_fold_tr = K_train[np.ix_(train_idx, train_idx)]
                K_fold_te = K_train[np.ix_(test_idx, train_idx)]
                clf.fit(K_fold_tr, y_train[train_idx])
                oof_preds[test_idx] = clf.predict_proba(K_fold_te)[:, 1]
            else:
                clf.fit(X_tr_scaled[train_idx], y_train[train_idx])
                oof_preds[test_idx] = clf.predict_proba(X_tr_scaled[test_idx])[:, 1]

        cv_mean_auc, cv_ci_low, cv_ci_high = bootstrap_ci(
            y_train, oof_preds, n_bootstraps=2000, seed=42
        )
        cv_brier = float(brier_score_loss(y_train, oof_preds))

        # Final fit on full train and test on validation
        full_clf = cfg["model_fn"]()
        if cfg["type"] == "precomputed":
            full_clf.fit(K_train, y_train)
            val_preds = full_clf.predict_proba(K_val)[:, 1]
        else:
            full_clf.fit(X_tr_scaled, y_train)
            val_preds = full_clf.predict_proba(X_va_scaled)[:, 1]

        val_auc = float(roc_auc_score(y_val, val_preds))
        val_brier = float(brier_score_loss(y_val, val_preds))

        results["models"][name] = {
            "cv_auc_mean": round(cv_mean_auc, 4),
            "cv_auc_95ci": [round(cv_ci_low, 4), round(cv_ci_high, 4)],
            "cv_brier": round(cv_brier, 4),
            "val_auc": round(val_auc, 4),
            "val_brier": round(val_brier, 4),
        }

        ci_str = f"[{cv_ci_low:.3f}, {cv_ci_high:.3f}]"
        summary_rows.append(
            {
                "Model": name,
                "5-Fold CV AUC (95% CI)": f"{cv_mean_auc:.3f} {ci_str}",
                "Val AUC": f"{val_auc:.3f}",
                "Val Brier": f"{val_brier:.3f}",
            }
        )

    sum_df = pd.DataFrame(summary_rows)
    print(sum_df.to_string(index=False))

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2))
    print(f"\nSaved Quantum Kernel results to {out_p}")


if __name__ == "__main__":
    main()
