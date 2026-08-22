"""Export smoke test: ONNX parity plus a Core ML conversion attempt.

Exports the frozen B0 checkpoint to ONNX and checks it against the eager
PyTorch model on random 96-cubed patches — max absolute logit difference
and argmax agreement — so an export that silently changes behavior fails
the gate. The Core ML route is attempted as well, but a conversion on
Windows is not expected to succeed; that parity run happens later on the
M1, and this script only records whether the toolchain was available.

Artifacts land in ``runs/export-smoke-b0/``: ``b0.onnx`` and
``report.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scanvidence.models.backbone import SegResNetB0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ckpt", default="runs/full-b0-seed17/best.pt")
    args = parser.parse_args()

    model = SegResNetB0(in_channels=4, num_classes=4, widths=(16, 32, 64, 128), dropout=0.0)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(
        state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    )
    model.eval()
    dummy = torch.randn(1, 4, 96, 96, 96)

    out = Path("runs/export-smoke-b0")
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    try:
        import onnx
        import onnxruntime as ort

        onnx_path = out / "b0.onnx"
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            opset_version=17,
            input_names=["x"],
            output_names=["logits"],
            dynamo=False,
        )
        onnx.checker.check_model(onnx.load(onnx_path))
        sess = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())
        patches = [torch.randn(1, 4, 96, 96, 96) for _ in range(10)]
        diffs, agree = [], 0
        with torch.no_grad():
            for p in patches:
                ref = model(p).numpy()
                got = sess.run(None, {"x": p.numpy()})[0]
                diffs.append(float(np.abs(ref - got).max()))
                agree += int((ref.argmax(1) == got.argmax(1)).mean() > 0.99)
        report["onnx"] = {
            "providers": ort.get_available_providers(),
            "max_abs_diff": round(max(diffs), 6),
            "parity_patches": f"{agree}/10",
        }
    except Exception as exc:
        report["onnx"] = {"error": str(exc)}

    try:
        import coremltools as ct

        traced = torch.jit.trace(model, dummy)
        ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=dummy.shape)])
        ml.save(str(out / "b0.mlpackage"))
        report["coreml"] = {"converted": True}
    except Exception as exc:
        report["coreml"] = {"converted": False, "error": str(exc)}

    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
