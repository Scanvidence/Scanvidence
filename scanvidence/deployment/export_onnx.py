import argparse
from pathlib import Path

import torch

from scanvidence.models.backbone import SegResNetB0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    model = SegResNetB0(in_channels=4, num_classes=4)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Fallback if someone saves just the raw state_dict without metadata
        model.load_state_dict(checkpoint)

    model.eval()

    dummy_input = torch.randn(1, 4, 96, 96, 96)
    save_path = out_path / "B0_onnx.onnx"

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_volume"],
        output_names=["output_logits"],
        dynamic_axes=None,
    )
    print(f"ONNX export successful! Saved to {save_path}")


if __name__ == "__main__":
    main()
