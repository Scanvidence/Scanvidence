import argparse
import torch
import coremltools as ct
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

    print("Tracing model...")
    traced_model = torch.jit.trace(model, dummy_input)

    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=dummy_input.shape, name="input_volume")],
    )

    save_path = out_path / "B0_coreml.mlpackage"
    mlmodel.save(str(save_path))
    print(f"Core ML export successful! Saved to {save_path}")

if __name__ == "__main__":
    main()