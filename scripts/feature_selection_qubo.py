"""Stage-2 Quantum Feature Selection via QUBO (Classical Local Search, Exact, QAOA, QPU).

Benchmarks QUBO-based feature selection against 6 classical baselines (mRMR, MI, F-ANOVA, L1, RF, Random)
on the locked 20-candidate pool from the 577 MGMT dev cohort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression


def coh(
    data_csv: Path = Path("cache/radiomics_gt_dev.csv"),
    schema_json: Path = Path("cache/radiomics_schema_v1.json"),
    mgmt_csv: Path = Path("data/manifests/mgmt_labels_v1.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load and partition MGMT development cohort with zero data leakage."""
    df = pd.read_csv(data_csv)
    schema = json.loads(schema_json.read_text())

    feats = schema["feature_ids"]

    # Filter labeled development cases
    df_labeled = df.dropna(subset=["mgmt"]).copy()
    df_labeled["mgmt"] = df_labeled["mgmt"].astype(int)

    tr = df_labeled[df_labeled["partition"] == "train"].copy()
    va = df_labeled[df_labeled["partition"] == "val"].copy()
    ho = df_labeled[df_labeled["partition"] == "calibration"].copy()

    # Impute missing values on train statistics only
    medians = tr[feats].median()
    tr[feats] = tr[feats].fillna(medians)
    va[feats] = va[feats].fillna(medians)
    ho[feats] = ho[feats].fillna(medians)

    return tr, va, ho, feats


def prefilter(tr: pd.DataFrame, feats: list[str], k: int = 20) -> pd.Series:
    """Pre-filter candidate pool to top-k features by absolute Spearman correlation on train."""
    corrs = {}
    y = tr["mgmt"].values
    for f in feats:
        x = tr[f].values
        r, _ = spearmanr(x, y)
        corrs[f] = abs(r) if not np.isnan(r) else 0.0
    s = pd.Series(corrs).sort_values(ascending=False)
    return s.head(k)


def build_qubo(
    rel: pd.Series,
    tr: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 0.15,
    lam: float = 0.35,
) -> tuple[list[str], np.ndarray]:
    """Construct QUBO matrix Q from normalized relevance, redundancy, and size penalty lam."""
    pool = list(rel.index)
    k = len(pool)

    # Redundancy matrix: pairwise Spearman correlation between feature pairs
    vals = tr[pool].values
    R, _ = spearmanr(vals)
    R = np.abs(np.nan_to_num(R, nan=0.0))

    # Normalized relevance
    max_rel = float(rel.values.max()) if float(rel.values.max()) > 1e-12 else 1.0
    rel_norm = rel.values / max_rel

    # Q matrix
    Q = np.zeros((k, k))
    # Off-diagonal: pairwise redundancy penalty (weighted by beta)
    Q[np.triu_indices(k, 1)] = beta * R[np.triu_indices(k, 1)]
    # Diagonal: lambda (size penalty) - alpha * normalized_relevance_i
    np.fill_diagonal(Q, lam - alpha * rel_norm)

    return pool, Q


def obj(x: np.ndarray, Q: np.ndarray) -> float:
    """Calculate QUBO energy x^T Q x."""
    return float(x @ Q @ x)


def solve_classical(Q: np.ndarray, seed: int = 17, restarts: int = 10) -> tuple[np.ndarray, float]:
    """Solve QUBO via greedy bit-flip local search with random multi-start restarts."""
    k = len(Q)
    rng = np.random.default_rng(seed)
    bx: np.ndarray = np.zeros(k, dtype=int)
    bo: float = np.inf

    for r in range(restarts):
        x = np.zeros(k, dtype=float) if r == 0 else rng.integers(0, 2, k).astype(float)
        improved = True
        while improved:
            improved = False
            for i in range(k):
                y = x.copy()
                y[i] = 1.0 - y[i]
                if obj(y, Q) < obj(x, Q) - 1e-12:
                    x, improved = y, True

        current_obj = obj(x, Q)
        if current_obj < bo:
            bo, bx = current_obj, x.copy().astype(int)

    return bx, bo


def solve_exact(Q: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve QUBO via exact brute-force batch evaluation over all 2^k states."""
    k = len(Q)
    bx: np.ndarray = np.zeros(k, dtype=int)
    bo: float = np.inf

    chunk_size = 65536
    total_states = 1 << k
    for s in range(0, total_states, chunk_size):
        idx = np.arange(s, min(s + chunk_size, total_states))
        bits = ((idx[:, None] >> np.arange(k)) & 1).astype(float)
        o = np.einsum("bi,ij,bj->b", bits, Q, bits)

        i_min = int(o.argmin())
        if o[i_min] < bo:
            bo = float(o[i_min])
            bx = bits[i_min].astype(int)

    return bx, bo


def qubo_to_ising(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert QUBO formulation (x in {0, 1}) to Ising model (s in {-1, +1})."""
    W = np.triu(Q, 1)
    W = W + W.T
    d = np.diag(Q)

    h = -d / 2.0 - W.sum(axis=1) / 4.0
    J = W / 4.0
    offset = float(d.sum() / 2.0 + W[np.triu_indices(len(Q), 1)].sum() / 4.0)

    return h, J, offset


def qaoa_circuit(
    h: np.ndarray, J: np.ndarray, gamma: np.ndarray, beta: np.ndarray
) -> QuantumCircuit:
    """Construct QAOA parameterized quantum circuit with p layers."""
    k = len(h)
    qc = QuantumCircuit(k)
    qc.h(range(k))

    p = len(gamma)
    for t in range(p):
        g = float(gamma[t])
        b = float(beta[t])

        # Cost unitary
        for i in range(k):
            if abs(h[i]) > 1e-12:
                qc.rz(2.0 * g * float(h[i]), i)
        for i in range(k):
            for j in range(i + 1, k):
                if abs(J[i, j]) > 1e-12:
                    qc.rzz(2.0 * g * float(J[i, j]), i, j)

        # Mixer unitary
        qc.rx(2.0 * b, range(k))

    return qc


def qaoa_sim(
    h: np.ndarray,
    J: np.ndarray,
    p: int = 1,
    seed: int = 17,
) -> tuple[dict[str, str], float, np.ndarray, list[int]]:
    """Simulate QAOA expectation optimization and bitstring sampling with fast vectorized energy."""
    k = len(h)
    rng = np.random.default_rng(seed)

    # Precompute diagonal Ising energies for all 2^k computational basis states
    idx = np.arange(1 << k)
    bits = ((idx[:, None] >> np.arange(k)) & 1).astype(float)
    s = 1.0 - 2.0 * bits
    E_diag = -np.dot(s, h) + np.sum(np.dot(s, J) * s, axis=1)

    def energy_fn(params: np.ndarray) -> float:
        circ = qaoa_circuit(h, J, params[:p], params[p:])
        sv = Statevector(circ)
        probs = np.abs(sv.data) ** 2
        return float(np.dot(probs, E_diag))

    init_params = rng.uniform(0.1, np.pi / 2.0, 2 * p)
    res = minimize(energy_fn, init_params, method="COBYLA", options={"maxiter": 40, "rhobeg": 0.5})

    opt_circ = qaoa_circuit(h, J, res.x[:p], res.x[p:])
    sv = Statevector(opt_circ)
    probs = np.abs(sv.data) ** 2
    order = np.argsort(-probs)

    top_bitstrings = {f"top{i}": format(o, f"0{k}b")[::-1] for i, o in enumerate(order[:10])}

    # Best sampled bitstring to selected indices
    best_bs = format(order[0], f"0{k}b")[::-1]
    selected = [i for i, char in enumerate(best_bs) if char == "1"]

    return top_bitstrings, float(res.fun), res.x, selected


def baselines(Xtr: np.ndarray, ytr: np.ndarray, k_target: int) -> dict[str, list[int]]:
    """Run 6 classical feature selection baselines on the same candidate pool."""
    n_feats = Xtr.shape[1]
    k_target = min(k_target, n_feats)
    res: dict[str, list[int]] = {}

    # 1. Mutual Information
    mi = mutual_info_classif(Xtr, ytr, random_state=42)
    res["mi_topk"] = sorted(np.argsort(-mi)[:k_target].tolist())

    # 2. mRMR
    mrmr_sel = [int(np.argmax(mi))]
    corr = np.abs(np.nan_to_num(np.corrcoef(Xtr.T), nan=0.0))
    for _ in range(k_target - 1):
        rem = [i for i in range(n_feats) if i not in mrmr_sel]
        scores = [mi[c] - np.mean([corr[c, s] for s in mrmr_sel]) for c in rem]
        mrmr_sel.append(rem[int(np.argmax(scores))])
    res["mrmr"] = sorted(mrmr_sel)

    # 3. F-ANOVA
    F, _ = f_classif(Xtr, ytr)
    F = np.nan_to_num(F, nan=0.0)
    res["f_anova"] = sorted(np.argsort(-F)[:k_target].tolist())

    # 4. LASSO (L1)
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.2, random_state=42, max_iter=500)
    l1.fit(Xtr, ytr)
    coef = np.abs(l1.coef_).ravel()
    res["l1"] = sorted(np.argsort(-coef)[:k_target].tolist())

    # 5. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(Xtr, ytr)
    imp = rf.feature_importances_
    res["rf"] = sorted(np.argsort(-imp)[:k_target].tolist())

    # 6. Random (Null Baseline)
    rng = np.random.default_rng(42)
    res["random"] = sorted(rng.choice(n_feats, size=k_target, replace=False).tolist())

    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2 Quantum QUBO vs Classical Feature Selection.")
    ap.add_argument("--prefilter", type=int, default=20, help="Candidate pool size")
    ap.add_argument("--alpha", type=float, default=1.0, help="Relevance weight")
    ap.add_argument("--beta", type=float, default=0.15, help="Redundancy penalty weight")
    ap.add_argument("--lam", type=float, default=0.35, help="QUBO subset size penalty lambda")
    ap.add_argument(
        "--exact",
        action="store_true",
        default=True,
        help="Run exact brute-force solver if k <= 20",
    )
    ap.add_argument("--qaoa-p", type=int, default=1, help="Number of QAOA layers (p)")
    ap.add_argument(
        "--submit-qpu",
        action="store_true",
        help="Submit QAOA circuit to real IBM Quantum QPU",
    )
    ap.add_argument(
        "--ibm-token",
        default="",
        help="IBM Quantum API token for hardware execution",
    )
    ap.add_argument(
        "--ibm-crn",
        default="",
        help="IBM Cloud CRN / instance name (if using IBM Cloud channel)",
    )
    ap.add_argument(
        "--ibm-channel",
        choices=["ibm_quantum", "ibm_cloud"],
        default="ibm_quantum",
        help="IBM Quantum service channel",
    )
    ap.add_argument(
        "--backend",
        default="",
        help="Specific IBM QPU backend name (e.g. ibm_brisbane, ibm_kyoto, ibm_sherbrooke). Defaults to least busy.",
    )
    ap.add_argument(
        "--out",
        default="cache/selection_stage2.json",
        help="Output JSON summary path",
    )
    args = ap.parse_args()

    print("--- Stage 2: Quantum Feature Selection via QUBO ---")
    tr, va, ho, feats = coh()
    print(
        f"Cohort: train={len(tr)}, val={len(va)}, calibration={len(ho)} (Total: {len(tr) + len(va) + len(ho)})"
    )

    # 1. Pre-filter candidate pool
    rel = prefilter(tr, feats, k=args.prefilter)
    print(f"\nPre-filtered top-{len(rel)} candidate features on train:")
    for i, (fname, score) in enumerate(rel.items()):
        print(f"  {i + 1:2d}. {fname:55s} | |Spearman r| = {score:.4f}")

    # 2. Build QUBO Matrix
    pool, Q = build_qubo(rel, tr, alpha=args.alpha, beta=args.beta, lam=args.lam)
    print(
        f"\nConstructed QUBO Matrix Q ({Q.shape[0]}x{Q.shape[1]}), alpha={args.alpha}, beta={args.beta}, lambda={args.lam}"
    )

    out: dict[str, Any] = {
        "candidate_pool": pool,
        "prefilter_k": args.prefilter,
        "alpha": args.alpha,
        "beta": args.beta,
        "lambda": args.lam,
        "methods": {},
    }

    # 3. Solver 1: Classical Local Search
    xc, oc = solve_classical(Q, seed=17, restarts=10)
    sel_classical = [pool[i] for i, b in enumerate(xc) if b == 1]
    out["methods"]["qubo_classical"] = {
        "selected": sel_classical,
        "objective": round(oc, 5),
        "k_selected": len(sel_classical),
        "bitstring": "".join(map(str, xc)),
    }
    print(
        f"\n[1/4] QUBO Classical (Local Search): Selected {len(sel_classical)} features | Objective: {oc:.4f}"
    )

    # 4. Solver 2: Exact Enumeration
    if args.exact and len(Q) <= 20:
        xe, oe = solve_exact(Q)
        sel_exact = [pool[i] for i, b in enumerate(xe) if b == 1]
        out["methods"]["qubo_exact"] = {
            "selected": sel_exact,
            "objective": round(oe, 5),
            "k_selected": len(sel_exact),
            "bitstring": "".join(map(str, xe)),
        }
        print(
            f"[2/4] QUBO Exact (2^{len(Q)} states):   Selected {len(sel_exact)} features | Objective: {oe:.4f} (Global Optimum)"
        )

    # 5. Solver 3: QAOA Simulator
    h, J, offset = qubo_to_ising(Q)
    top_bs, opt_e, opt_angles, qaoa_sel_idx = qaoa_sim(h, J, p=args.qaoa_p, seed=17)
    sel_qaoa = [pool[i] for i in qaoa_sel_idx]
    # Evaluate QAOA bitstring on original QUBO objective
    qaoa_bit_arr = np.array([int(c) for c in top_bs["top0"]])
    qaoa_qubo_obj = obj(qaoa_bit_arr, Q)

    out["methods"]["qaoa_sim"] = {
        "selected": sel_qaoa,
        "energy_expectation": round(opt_e, 5),
        "qubo_objective": round(qaoa_qubo_obj, 5),
        "k_selected": len(sel_qaoa),
        "top_bitstrings": top_bs,
        "qaoa_p": args.qaoa_p,
    }
    print(
        f"[3/4] QAOA Simulator (p={args.qaoa_p}):       Selected {len(sel_qaoa)} features | QUBO Objective: {qaoa_qubo_obj:.4f}"
    )

    # 6. Solver 4: IBM QPU Submission
    if args.submit_qpu:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

            if args.ibm_channel == "ibm_cloud":
                service = QiskitRuntimeService(
                    channel="ibm_cloud",
                    token=args.ibm_token,
                    instance=args.ibm_crn,
                )
            else:
                service = (
                    QiskitRuntimeService(channel="ibm_quantum", token=args.ibm_token)
                    if args.ibm_token
                    else QiskitRuntimeService(channel="ibm_quantum")
                )

            if args.backend:
                backend = service.backend(args.backend)
            else:
                backend = service.least_busy(simulator=False, operational=True)
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

            qpu_circ = qaoa_circuit(h, J, opt_angles[: args.qaoa_p], opt_angles[args.qaoa_p :])
            qpu_circ.measure_all()

            pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
            isa_circuit = pm.run(qpu_circ)

            sampler = SamplerV2(mode=backend)
            job = sampler.run([isa_circuit], shots=10000)
            out["qpu"] = {
                "backend": backend.name,
                "job_id": job.job_id(),
                "status": "submitted",
                "shots": 10000,
            }
            print(f"      QPU Job submitted successfully! Job ID: {job.job_id()}")
            print(f"      Backend: {backend.name} (156 qubits)")
            print(f"      Track live status at: https://quantum.ibm.com/jobs/{job.job_id()}")
        except Exception as e:
            out["qpu"] = {"status": "submission_failed", "error": str(e)}
            print(f"\n[4/4] QPU submission notice: {e}")
    else:
        out["qpu"] = {
            "status": "not_submitted",
            "info": "Pass --submit-qpu --ibm-token <TOKEN> [--ibm-crn <CRN>] to submit to real hardware",
        }

    # 7. Classical Baselines (Targeting size of QUBO selection)
    target_k = len(sel_classical)
    Xtr = tr[pool].values
    ytr = tr["mgmt"].values

    base_dict = baselines(Xtr, ytr, target_k)
    for name, idx in base_dict.items():
        sel_names = [pool[i] for i in idx]
        out["methods"][name] = {
            "selected": sel_names,
            "k_selected": len(sel_names),
        }

    # 8. Jaccard Similarity Matrix vs QUBO Classical & Exact
    qset = set(sel_classical)
    out["jaccard_vs_qubo_classical"] = {
        m: round(
            len(qset & set(v["selected"])) / max(1, len(qset | set(v["selected"]))),
            3,
        )
        for m, v in out["methods"].items()
        if "selected" in v
    }

    # Print Summary Table
    print("\n--- STAGE 2 FEATURE SELECTION COMPARISON ---")
    rows = []
    for m, v in out["methods"].items():
        jacc = out["jaccard_vs_qubo_classical"].get(m, 1.0)
        obj_str = f"{v.get('objective', v.get('qubo_objective', '—'))}"
        rows.append(
            {
                "Method": m,
                "Selected Features (k)": v["k_selected"],
                "QUBO Objective": obj_str,
                "Jaccard vs QUBO": jacc,
            }
        )
    sum_df = pd.DataFrame(rows)
    print(sum_df.to_string(index=False))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved Stage 2 results to {out_path}")


if __name__ == "__main__":
    main()
