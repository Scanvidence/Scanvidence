"""Generate 3D Spatial Grad-CAM heatmaps for SegResNetB0 on patient MRI volumes.

Runs on CPU, Apple Silicon MPS, or NVIDIA CUDA GPUs (e.g. NVIDIA T1000).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch

from scanvidence.models.backbone import SegResNetB0
from scanvidence.xai import GradCAM


def load_patient_mri(
    pdir: Path, patch_size: int = 96
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Load and normalize patient 4-channel MRI scan, extracting central tumor volume."""
    t1n_p = list(pdir.glob("*t1n.nii.gz"))[0]
    t1c_p = list(pdir.glob("*t1c.nii.gz"))[0]
    t2w_p = list(pdir.glob("*t2w.nii.gz"))[0]
    t2f_p = list(pdir.glob("*t2f.nii.gz"))[0]

    t1n = sitk.GetArrayFromImage(sitk.ReadImage(str(t1n_p)))
    t1c = sitk.GetArrayFromImage(sitk.ReadImage(str(t1c_p)))
    t2w = sitk.GetArrayFromImage(sitk.ReadImage(str(t2w_p)))
    flair = sitk.GetArrayFromImage(sitk.ReadImage(str(t2f_p)))

    # Crop centered 3D patch
    d, h, w = t1c.shape
    cd, ch, cw = d // 2, h // 2, w // 2
    r = patch_size // 2

    patch = np.stack(
        [
            t1n[cd - r : cd + r, ch - r : ch + r, cw - r : cw + r],
            t1c[cd - r : cd + r, ch - r : ch + r, cw - r : cw + r],
            t2w[cd - r : cd + r, ch - r : ch + r, cw - r : cw + r],
            flair[cd - r : cd + r, ch - r : ch + r, cw - r : cw + r],
        ],
        axis=0,
    ).astype(np.float32)

    # Intensity z-score normalization
    for c in range(4):
        std = patch[c].std()
        patch[c] = (patch[c] - patch[c].mean()) / (std if std > 0 else 1.0)

    raw_t1c = t1c[cd - r : cd + r, ch - r : ch + r, cw - r : cw + r]
    raw_flair = flair[cd - r : cd + r, ch - r : ch + r, cw - r : cw + r]
    return torch.from_numpy(patch).unsqueeze(0), raw_t1c, raw_flair


def main() -> None:
    ap = argparse.ArgumentParser(description="3D Grad-CAM for SegResNetB0")
    ap.add_argument(
        "--patient-dir",
        default="data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00002-000",
        help="Path to patient MRI directory",
    )
    ap.add_argument(
        "--checkpoint",
        default="checkpoints/best.pt",
        help="Path to SegResNetB0 checkpoint",
    )
    ap.add_argument(
        "--target-class",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Target class (1: Necrosis, 2: Edema, 3: Enhancing Tumor)",
    )
    ap.add_argument(
        "--out",
        default="cache/gradcam_3d_slice.png",
        help="Path to save output visualization",
    )
    args = ap.parse_args()

    # Device selection (CUDA for NVIDIA GPU / MPS for Mac / CPU)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using Compute Device: {device}")

    # 1. Load SegResNetB0 model
    model = SegResNetB0()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(
            f"Checkpoint '{ckpt_path}' not found locally. Auto-downloading from Hugging Face Hub..."
        )
        try:
            from huggingface_hub import hf_hub_download

            ckpt_path = Path(
                hf_hub_download(
                    repo_id="Falcon7211/Scanvidence-SegResNetB0",
                    filename="best.pt",
                    local_dir="checkpoints",
                )
            )
            print(f"Downloaded checkpoint to: {ckpt_path}")
        except Exception as e:
            raise FileNotFoundError(
                f"Could not find '{args.checkpoint}' and auto-download failed: {e}"
            )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 2. Attach 3D Grad-CAM
    gradcam = GradCAM(model)

    # 3. Load patient scan
    pdir = Path(args.patient_dir)
    print(f"Loading MRI scan for: {pdir.name}...")
    x, raw_t1c, raw_flair = load_patient_mri(pdir)
    x = x.to(device)

    # 4. Generate 3D Spatial Heatmap
    class_names = {1: "Necrotic Core (NCR)", 2: "Peritumoral Edema (ED)", 3: "Enhancing Tumor (ET)"}
    print(f"Computing 3D Grad-CAM backpropagation for target: {class_names[args.target_class]}...")
    heatmap_3d = gradcam.generate_heatmap(x, target_class=args.target_class)

    # 5. Extract peak activation axial slice
    peak_z = int(np.argmax(np.mean(heatmap_3d, axis=(1, 2))))
    print(f"Peak 3D Activation located at Axial Slice: z = {peak_z}")

    # 6. Plot & Save Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # T1c Contrast MRI Slice
    axes[0].imshow(raw_t1c[peak_z], cmap="gray")
    axes[0].set_title(f"T1c MRI (Slice z={peak_z})", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # FLAIR MRI Slice
    axes[1].imshow(raw_flair[peak_z], cmap="gray")
    axes[1].set_title(f"FLAIR MRI (Slice z={peak_z})", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # 3D Grad-CAM Spatial Heatmap Overlay
    axes[2].imshow(raw_t1c[peak_z], cmap="gray")
    im = axes[2].imshow(heatmap_3d[peak_z], cmap="jet", alpha=0.55, vmin=0, vmax=1)
    axes[2].set_title(
        f"3D Grad-CAM Overlay: {class_names[args.target_class]}", fontsize=12, fontweight="bold"
    )
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Attribution Intensity")

    plt.tight_layout()
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=200, bbox_inches="tight")
    print(f"Successfully saved 3D Grad-CAM visualization to: {out_p}")


if __name__ == "__main__":
    main()
