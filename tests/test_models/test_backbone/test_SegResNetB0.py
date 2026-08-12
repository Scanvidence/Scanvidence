"""CPU-only tests for the B0 SegResNet-style reference CNN."""

import pytest
import torch

from scanvidence.models import SegResNetB0
from scanvidence.models.backbone import SegResNetB0 as BackboneSegResNetB0


def _patch(spatial: int, in_channels: int = 4, batch: int = 1) -> torch.Tensor:
    return torch.randn(batch, in_channels, spatial, spatial, spatial)


def test_public_symbol_is_backbone_class():
    assert SegResNetB0 is BackboneSegResNetB0


def test_forward_shape_small_patch():
    model = SegResNetB0()
    out = model(_patch(32))
    assert out.shape == (1, 4, 32, 32, 32)


def test_forward_matches_batch_size():
    model = SegResNetB0()
    out = model(_patch(32, batch=2))
    assert out.shape[0] == 2


def test_forward_single_modality_input():
    model = SegResNetB0(in_channels=1)
    out = model(_patch(32, in_channels=1))
    assert out.shape == (1, 4, 32, 32, 32)


def test_custom_widths_and_classes():
    model = SegResNetB0(in_channels=4, num_classes=3, widths=(8, 16, 32))
    out = model(_patch(32))
    assert out.shape == (1, 3, 32, 32, 32)


def test_forward_logits_finite():
    model = SegResNetB0()
    out = model(_patch(32))
    assert torch.isfinite(out).all()


def test_default_parameter_count_is_compact():
    model = SegResNetB0()
    count = model.parameter_count
    assert 1_000_000 < count < 5_000_000


def test_groupnorm_only_no_batchnorm():
    model = SegResNetB0()
    named = dict(model.named_modules())
    assert any(isinstance(m, torch.nn.GroupNorm) for m in named.values())
    assert not any(isinstance(m, torch.nn.BatchNorm3d) for m in named.values())


def test_deterministic_under_fixed_seed():
    torch.manual_seed(17)
    a = SegResNetB0()(_patch(32))
    torch.manual_seed(17)
    b = SegResNetB0()(_patch(32))
    assert torch.equal(a, b)


def test_gradients_flow_to_all_parameters():
    model = SegResNetB0()
    loss = model(_patch(32)).sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"


def test_head_outputs_one_logit_per_class():
    model = SegResNetB0(num_classes=3)
    out = model(_patch(32))
    assert out.shape[1] == 3


def test_from_config_matches_constructor():
    config = {
        "in_channels": 1,
        "num_classes": 3,
        "widths": [8, 16, 32],
        "num_groups": 4,
        "dropout": 0.1,
    }
    model = SegResNetB0.from_config(config)
    assert model.parameter_count == SegResNetB0(**config).parameter_count
    with torch.no_grad():
        out = model(_patch(32, in_channels=1))
    assert out.shape == (1, 3, 32, 32, 32)


def test_dropout_matches_config():
    assert SegResNetB0.from_config({"dropout": 0.5}).dropout == 0.5
    assert SegResNetB0().dropout == 0.0


def test_from_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown SegResNetB0 config keys"):
        SegResNetB0.from_config({"transformer_depth": 4})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_channels": 0},
        {"num_classes": 0},
        {"widths": [16]},
        {"widths": [16, 32, 64, 128, 256, 512]},
        {"widths": [16, 0, 64]},
        {"num_groups": 0},
        {"dropout": 1.0},
        {"dropout": -0.1},
    ],
)
def test_invalid_configs_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        SegResNetB0(**kwargs)


def test_forward_rejects_non_divisible_spatial_dims():
    model = SegResNetB0()
    with pytest.raises(ValueError, match="divisible by"):
        model(_patch(100))


def test_forward_rejects_wrong_rank():
    model = SegResNetB0()
    with pytest.raises(ValueError, match="5-D"):
        model(torch.randn(4, 32, 32, 32))


@pytest.mark.slow
def test_forward_shape_design_point_96_cubed():
    """The production config: 96-cubed patch, batch size 1, 4 modalities."""
    model = SegResNetB0()
    out = model(_patch(96))
    assert out.shape == (1, 4, 96, 96, 96)
    assert torch.isfinite(out).all()
