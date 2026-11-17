import pytest
import torch

from sanet.data.dataset import LaneDataset, collate
from sanet.models.attention import AMPM, ERA, active_descriptor
from sanet.models.backbone import CameraFPN
from sanet.models.losses import LaneLoss, discriminative_embedding
from sanet.models.network import SANet


@pytest.mark.parametrize("mode", ["image", "bev"])
def test_integrated_forward_loss_backward(config, records, mode):
    config.model.mode = mode
    dataset = LaneDataset(config, "train")
    batch = collate([dataset[0], dataset[1]])
    model = SANet(config.model)
    result = model(batch, warmup=True)
    target_shape = (32, 64) if mode == "image" else (16, 16)
    assert result["confidence"].shape == (2, 1, *target_shape)
    torch.testing.assert_close(result["routing"].sum(1), torch.ones(2))
    losses = LaneLoss(config)(result, batch["targets"])
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.camera.lateral[0].weight.grad.abs().sum() > 0
    assert model.ampm.branches[0].kernels.grad.abs().sum() > 0
    if mode == "bev":
        assert losses["depth"] > 0
        assert model.fusion.conv.weight.grad.abs().sum() > 0
        assert model.fusion.lidar.net[0].weight.grad.abs().sum() > 0
    else:
        assert losses["depth"] == 0 and losses["height"] == 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def test_multiview_bev_camera_only_and_empty_scene(config, records):
    config.model.mode = "bev"
    dataset = LaneDataset(config, "train")
    batch = collate([dataset[0]])
    for key in ["images", "intrinsics", "transforms"]:
        batch[key] = batch[key].repeat(1, 2, *([1] * (batch[key].ndim - 2)))
    batch["points"] = [torch.empty(0, 4)]
    model = SANet(config.model).eval()
    result = model(batch)
    assert result["confidence"].shape == (1, 1, 16, 16)
    assert result["sparsity"]["lidar_cells"] == 0
    assert torch.isfinite(result["confidence"]).all()


def test_all_empty_labels_finite(config, records):
    dataset = LaneDataset(config, "train")
    batch = collate([dataset[0]])
    for k in ["instance", "image_instance", "height_valid"]:
        batch["targets"][k].zero_()
    model = SANet(config.model)
    losses = LaneLoss(config)(model(batch), batch["targets"])
    assert losses["embedding"] == 0
    losses["total"].backward()
    assert torch.isfinite(losses["total"])


def test_embedding_margin_separates_instances():
    labels = torch.tensor([[[1, 1, 2, 2]]])
    bad = torch.zeros(1, 2, 1, 4, requires_grad=True)
    good = torch.tensor([[[[0.0, 0, 4, 4]], [[0.0, 0, 0, 0]]]])
    assert discriminative_embedding(bad, labels) > discriminative_embedding(good, labels)


def test_attention_empty_support_and_conditional_kernels():
    x = torch.arange(2 * 8 * 4 * 4, dtype=torch.float32).reshape(2, 8, 4, 4) / 100
    active = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
    descriptor = active_descriptor(x, active, torch.tensor(0.5))
    assert not descriptor.any()
    ampm = AMPM(8, [1, 2], 3)
    output, alpha = ampm(x, torch.ones_like(active))
    assert not torch.allclose(alpha[0], alpha[1])
    assert ERA(8)(output).shape == x.shape


def test_resnet34_fpn_no_download():
    net = CameraFPN("resnet34", 8, pretrained=False).eval()
    with torch.no_grad():
        output = net(torch.zeros(1, 3, 64, 128))
    assert output.shape == (1, 8, 16, 32)
    assert torch.isfinite(output).all()


def test_bev_empty_camera_and_lidar_support(config, records):
    config.model.mode = "bev"
    sample = LaneDataset(config, "train")[0]
    batch = collate([sample])
    batch["points"] = [torch.empty(0, 4)]
    batch["transforms"][:, :, 0, 3] = 1000.0
    result = SANet(config.model)(batch)
    assert result["sparsity"]["union_cells"] == 0
    assert torch.isfinite(result["confidence"]).all()
    result["confidence"].sum().backward()
