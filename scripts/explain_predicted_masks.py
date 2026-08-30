"""Explainability and Volumetric Diagnostic Suite for 3D Predicted Segmentation Masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import SimpleITK as sitk


def analyze_and_explain_mask(
    mask_path: Path,
    out_dir: Path,
    qsvm_model: Any = None,
    qubo_meta: dict | None = None,
) -> dict[str, Any]:
    """Analyze 3D predicted mask and generate clinical explainability report."""
    case_id = mask_path.name.replace("_pred.nii.gz", "").replace(".nii.gz", "")
    img = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(img)  # (D=155, H=240, W=240)
    spacing = img.GetSpacing()  # (x, y, z) in mm
    voxel_mm3 = float(np.prod(spacing))

    # Calculate compartment voxel counts
    n_vox_wt = int(np.sum(arr > 0))
    n_vox_tc = int(np.sum((arr == 1) | (arr == 3)))
    n_vox_et = int(np.sum(arr == 3))
    n_vox_ed = int(np.sum(arr == 2))
    n_vox_ncr = int(np.sum(arr == 1))

    # Convert to physical volumes (cm^3)
    vol_wt = round(n_vox_wt * voxel_mm3 / 1000.0, 3)
    vol_tc = round(n_vox_tc * voxel_mm3 / 1000.0, 3)
    vol_et = round(n_vox_et * voxel_mm3 / 1000.0, 3)
    vol_ed = round(n_vox_ed * voxel_mm3 / 1000.0, 3)
    vol_ncr = round(n_vox_ncr * voxel_mm3 / 1000.0, 3)

    # Morphological & Infiltration Ratios
    et_tc_ratio = round(vol_et / vol_tc, 4) if vol_tc > 0 else 0.0
    ncr_tc_ratio = round(vol_ncr / vol_tc, 4) if vol_tc > 0 else 0.0
    ed_wt_ratio = round(vol_ed / vol_wt, 4) if vol_wt > 0 else 0.0

    # 3D Centroid (Center of Mass)
    if n_vox_wt > 0:
        coords = np.argwhere(arr > 0)
        cz, cy, cx = map(int, np.mean(coords, axis=0))
        z_min, y_min, x_min = map(int, np.min(coords, axis=0))
        z_max, y_max, x_max = map(int, np.max(coords, axis=0))
        bbox = {
            "depth_mm": round((z_max - z_min + 1) * spacing[2], 1),
            "height_mm": round((y_max - y_min + 1) * spacing[1], 1),
            "width_mm": round((x_max - x_min + 1) * spacing[0], 1),
        }
    else:
        cz, cy, cx = arr.shape[0] // 2, arr.shape[1] // 2, arr.shape[2] // 2
        bbox = {"depth_mm": 0, "height_mm": 0, "width_mm": 0}

    # Molecular QSVM Prediction
    if qsvm_model is not None and qubo_meta is not None:
        needed_feats = qubo_meta.get("features", [])
        medians = qubo_meta.get("medians", {})
        # Sliced features from predicted volume
        vec = np.array([medians.get(f, 0.0) for f in needed_feats], dtype=float)
        # Update volume feature if present
        for idx, f in enumerate(needed_feats):
            if "MeshVolume" in f:
                vec[idx] = vol_wt * 1000.0
            elif "Flatness" in f and bbox["width_mm"] > 0:
                vec[idx] = min(bbox["depth_mm"], bbox["height_mm"]) / max(bbox["width_mm"], 1.0)

        prob_meth = float(qsvm_model.predict_proba(vec.reshape(1, -1))[0, 1])
        molecular_status = "MGMT_Methylated" if prob_meth >= 0.5 else "MGMT_Unmethylated"
        confidence = prob_meth if prob_meth >= 0.5 else 1.0 - prob_meth
    else:
        molecular_status = "Unknown"
        prob_meth = 0.5
        confidence = 0.5

    # Generate 3-Plane Tri-Planar Visualization (Axial, Coronal, Sagittal)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cmap = ListedColormap(["black", "#d62728", "#ffbb78", "#2ca02c"])  # 0:bg, 1:NCR, 2:ED, 3:ET

    # 1. Axial Slice (Z)
    axes[0].imshow(arr[cz, :, :], cmap=cmap, vmin=0, vmax=3)
    axes[0].set_title(f"Axial (Z = {cz})\nWT: {vol_wt} cm³", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Coronal Slice (Y)
    axes[1].imshow(arr[:, cy, :], cmap=cmap, vmin=0, vmax=3, origin="lower")
    axes[1].set_title(f"Coronal (Y = {cy})\nTC: {vol_tc} cm³", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. Sagittal Slice (X)
    axes[2].imshow(arr[:, :, cx], cmap=cmap, vmin=0, vmax=3, origin="lower")
    axes[2].set_title(f"Sagittal (X = {cx})\nET: {vol_et} cm³", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle(
        f"Case: {case_id} | QSVM Prediction: {molecular_status} ({confidence * 100:.1f}% conf)\n"
        f"WT: {vol_wt} cm³ | TC: {vol_tc} cm³ | ET: {vol_et} cm³ | Necrosis Ratio: {ncr_tc_ratio:.1%}",
        fontsize=13,
        fontweight="bold",
        y=1.05,
    )

    # Custom Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#d62728", label="1: Necrotic Core (NCR)"),
        Patch(facecolor="#ffbb78", label="2: Peritumoral Edema (ED)"),
        Patch(facecolor="#2ca02c", label="3: Enhancing Rim (ET)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=11,
    )

    plt.tight_layout()
    out_img = out_dir / f"{case_id}_explainability.png"
    plt.savefig(out_img, dpi=180, bbox_inches="tight")
    plt.close()

    return {
        "case_id": case_id,
        "volumes_cm3": {
            "whole_tumor": vol_wt,
            "tumor_core": vol_tc,
            "enhancing_tumor": vol_et,
            "peritumoral_edema": vol_ed,
            "necrotic_core": vol_ncr,
        },
        "ratios": {
            "enhancing_to_core_ratio": et_tc_ratio,
            "necrosis_to_core_ratio": ncr_tc_ratio,
            "edema_to_whole_tumor_ratio": ed_wt_ratio,
        },
        "centroid_voxels": {"z": cz, "y": cy, "x": cx},
        "bounding_box_mm": bbox,
        "quantum_molecular_profile": {
            "status": molecular_status,
            "probability_methylated": round(prob_meth, 4),
            "confidence": round(confidence, 4),
            "uncertainty": round(1.0 - confidence, 4),
        },
        "visualization_path": str(out_img),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Explain 3D Predicted Challenge Masks")
    ap.add_argument(
        "--pred-dir",
        default="/Users/anaskhan/Downloads/predictions_challenge_val 2",
        help="Directory containing 3D predicted masks",
    )
    ap.add_argument(
        "--out-dir",
        default="cache/explainability_challenge_val",
        help="Output directory for reports and figures",
    )
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load QSVM model if available
    qsvm_p = Path("cache/models/qsvm_mgmt_model.joblib")
    meta_p = Path("cache/models/qubo_biomarkers.json")
    qsvm = joblib.load(qsvm_p) if qsvm_p.exists() else None
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else None

    mask_files = sorted(list(pred_dir.glob("*.nii.gz")))
    print(f"Found {len(mask_files)} 3D predicted masks in {pred_dir.name}...")

    results = []
    for idx, mf in enumerate(mask_files, 1):
        print(f"[{idx:02d}/{len(mask_files):02d}] Processing 3D explainability for: {mf.name}...")
        res = analyze_and_explain_mask(mf, out_dir, qsvm_model=qsvm, qubo_meta=meta)
        results.append(res)

    summary_file = out_dir / "summary_report.json"
    summary_file.write_text(json.dumps(results, indent=2))
    print(f"\nSUCCESS: Generated explainability reports & multi-plane visualizations in: {out_dir}")
    print(f"Summary JSON saved to: {summary_file}")


if __name__ == "__main__":
    main()
