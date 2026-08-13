import argparse
import torch
from pathlib import Path

from scanvidence.models.backbone.SegResNetB0 import SegResNetB0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    
    model = SegResNetB0(in_channels=4, num_classes=4)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
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
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_volume"],
        output_names=["output_logits"],
        dynamic_axes=None
    )
    print(f"ONNX export successful! Saved to {save_path}")

if __name__ == "__main__":
    main()