import numpy as np
import pytest
import torch
from torch.nn import functional as F

from sanet.geometry import backproject, project, resize_matrix, validate_calibration
from sanet.models.fusion import SparseLift, VoxelPointEncoder
from sanet.models.sparse import SubmanifoldConv, aggregate, union_concat


def test_projection_and_resize_roundtrip():
    k, t = torch.eye(3), torch.eye(4)
    uv, z = torch.tensor([[2.0, 3.0], [4.0, 6.0]]), torch.tensor([2.0, 5.0])
    xyz = backproject(uv, z, k, t)
    recovered, depth = project(xyz, k, t)
    torch.testing.assert_close(recovered, uv)
    torch.testing.assert_close(depth, z)
    a = resize_matrix(720, 1280, 320, 800, 80)
    new_uv = torch.cat([uv, torch.ones(2, 1)], 1) @ a.T
    xyz_new = backproject(new_uv[:, :2], z, a @ k, t)
    torch.testing.assert_close(xyz, xyz_new, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("kind", ["singular", "reflection", "nonrigid", "nonfinite"])
def test_bad_calibration(kind):
    k, t = torch.eye(3), torch.eye(4)
    if kind == "singular":
        k[0] = 0
    elif kind == "reflection":
        t[0, 0] = -1
    elif kind == "nonrigid":
        t[1, 1] = 2
    else:
        k[1, 1] = float("nan")
    with pytest.raises(ValueError):
        validate_calibration(k, t)


def test_weighted_aggregation_and_union_gradient():
    coords = torch.tensor([[0, 1, 1], [0, 1, 1], [0, 2, 2]])
    values = torch.tensor([[2.0], [6.0], [10.0]], requires_grad=True)
    a = aggregate(coords, values, 1, (4, 4), torch.tensor([1.0, 3.0, 1.0]))
    assert a.values[:, 0].tolist() == [5.0, 10.0]
    b = aggregate(torch.tensor([[0, 3, 3]]), torch.tensor([[7.0]], requires_grad=True), 1, (4, 4))
    union = union_concat(a, b)
    torch.testing.assert_close(union.values, torch.tensor([[5.0, 0.0], [10.0, 0.0], [0.0, 7.0]]))
    union.values.sum().backward()
    torch.testing.assert_close(values.grad[:, 0], torch.tensor([0.25, 0.75, 1.0]))


def test_sparse_conv_matches_dense_and_gradients():
    coords = torch.tensor([[0, 0, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0], [1, 3, 3]])
    values = torch.arange(10, dtype=torch.float32).reshape(5, 2).requires_grad_()
    sparse = aggregate(coords, values, 2, (4, 4))
    conv = SubmanifoldConv(2, 3)
    actual = conv(sparse).values
    dense, _ = sparse.dense()
    expected = F.conv2d(dense, conv.weight.permute(3, 2, 0, 1), conv.bias, padding=1)
    selected = expected[sparse.coords[:, 0], :, sparse.coords[:, 1], sparse.coords[:, 2]]
    torch.testing.assert_close(actual, selected)
    ga = torch.autograd.grad(actual.square().sum(), (values, conv.weight), retain_graph=True)
    gb = torch.autograd.grad(selected.square().sum(), (values, conv.weight))
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b)


def test_empty_sparse_conv_backward():
    values = torch.zeros(0, 3, requires_grad=True)
    sparse = aggregate(torch.empty(0, 3, dtype=torch.long), values, 2, (4, 4))
    output = SubmanifoldConv(3, 2)(sparse)
    dense, active = output.dense()
    assert dense.shape == (2, 2, 4, 4) and not active.any()
    dense.sum().backward()
    assert values.grad is not None


def test_known_camera_lift_and_gradients(config):
    config.model.mode = "bev"
    lift = SparseLift(config.model)
    f = torch.ones(1, 1, 8, 1, 2, requires_grad=True)
    s = torch.zeros(1, 1, 1, 1, 2, requires_grad=True)
    depth = torch.arange(10, dtype=torch.float32).reshape(1, 1, 5, 1, 2).requires_grad_()
    k = torch.tensor([[[[2.0, 0, 0.5], [0, 2, 0], [0, 0, 1]]]])
    t = torch.tensor([[[[0.0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]]])
    sparse, stats = lift(f, s, depth, k, t, (1, 2))
    assert len(sparse.values) > 0 and stats["selected_pixels"] == 2
    assert sparse.coords[:, 1].max() < 16
    sparse.values.sum().backward()
    assert f.grad is not None and torch.isfinite(f.grad).all()
    assert depth.grad is not None and torch.isfinite(depth.grad).all()


def test_voxel_encoder_filters_bounds_and_handles_absence(config):
    encoder = VoxelPointEncoder(config.model)
    p = torch.tensor(
        [[2.0, 0, 0, 0.5], [2.1, 0, 0, 0.5], [16.0, 0, 0, 1.0], [float("nan"), 0, 0, 0]]
    )
    result = encoder([p, torch.empty(0, 4)], 2)
    assert len(result.coords) == 1
    result.values.sum().backward()
    assert any(p.grad is not None for p in encoder.parameters())
    empty = encoder([torch.empty(0, 4)], 1)
    assert empty.values.shape == (0, config.model.channels)


def test_floor_boundaries_do_not_wrap():
    # A neighbor across the final column must not wrap to the next row/batch.
    values = torch.ones(2, 1)
    a = aggregate(torch.tensor([[0, 0, 3], [0, 1, 0]]), values, 1, (3, 4))
    layer = SubmanifoldConv(1, 1)
    layer.weight.data.fill_(1)
    layer.bias.data.zero_()
    np.testing.assert_allclose(layer(a).values.detach(), [[1.0], [1.0]])
