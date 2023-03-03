import numpy as np
import torch


def rect_iou(rects1, rects2, bound=None):
    r"""Intersection over union.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    assert rects1.shape == rects2.shape

    if bound is not None:
        # bounded rects1
        rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
        rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
        rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
        rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
        # bounded rects2
        rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
        rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
        rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
        rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

    rects_inter = _intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)

    return ious


def rect_diou(rects1, rects2, bound=None):
    assert rects1.shape == rects2.shape
    if len(rects1.shape) == 1:
        rects1 = rects1.reshape(-1, rects1.shape[0])
        rects2 = rects2.reshape(-1, rects2.shape[0])

    if bound is not None:
        # bounded rects1
        rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
        rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
        rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
        rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
        # bounded rects2
        rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
        rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
        rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
        rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

    # Get IoU
    rects_inter = _intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)
    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter
    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)

    # Get Minimum Covering BBOX
    x1 = np.minimum(rects1[:, 0], rects2[:, 0])
    y1 = np.minimum(rects1[:, 1], rects2[:, 1])
    x2 = np.maximum(rects1[:, 0] + rects1[:, 2], rects2[:, 0] + rects2[:, 2])
    y2 = np.maximum(rects1[:, 1] + rects1[:, 3], rects2[:, 1] + rects2[:, 3])
    covbbox_diag_square = np.square(x1 - x2) + np.square(y1 - y2)

    # Get Center
    rects1_cx, rects1_cy = rects1[:, 0] + rects1[:, 2] / 2.0, rects1[:, 1] + rects1[:, 3] / 2.0
    rects2_cx, rects2_cy = rects2[:, 0] + rects2[:, 2] / 2.0, rects2[:, 1] + rects2[:, 3] / 2.0
    bbox_center_square = np.square(rects1_cx - rects2_cx) + np.square(rects1_cy - rects2_cy)

    # Get DIoU
    dious = ious - np.divide(bbox_center_square, covbbox_diag_square)

    return dious


def rect_iou_torch(rects1, rects2, bound=None):
    assert rects1.shape == rects2.shape

    if bound is not None:
        # Bounded Rects1
        rects1[:, 0] = torch.clip(rects1[:, 0], 0, bound[0])
        rects1[:, 1] = torch.clip(rects1[:, 1], 0, bound[1])
        rects1[:, 2] = torch.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
        rects1[:, 3] = torch.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
        # Bounded Rects2
        rects2[:, 0] = torch.clip(rects2[:, 0], 0, bound[0])
        rects2[:, 1] = torch.clip(rects2[:, 1], 0, bound[1])
        rects2[:, 2] = torch.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
        rects2[:, 3] = torch.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

    rects_inter = _intersection_torch(rects1, rects2)
    areas_inter = torch.prod(rects_inter[..., 2:], dim=-1)

    areas1 = torch.prod(rects1[..., 2:], dim=-1)
    areas2 = torch.prod(rects2[..., 2:], dim=-1)
    areas_union = areas1 + areas2 - areas_inter

    eps = torch.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = torch.clip(ious, 0.0, 1.0)

    return ious


def rect_diou_torch(rects1, rects2, bound=None):
    assert rects1.shape == rects2.shape

    if bound is not None:
        # Bounded Rects1
        rects1[:, 0] = torch.clip(rects1[:, 0], 0, bound[0])
        rects1[:, 1] = torch.clip(rects1[:, 1], 0, bound[1])
        rects1[:, 2] = torch.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
        rects1[:, 3] = torch.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
        # Bounded Rects2
        rects2[:, 0] = torch.clip(rects2[:, 0], 0, bound[0])
        rects2[:, 1] = torch.clip(rects2[:, 1], 0, bound[1])
        rects2[:, 2] = torch.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
        rects2[:, 3] = torch.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

    # Get IoU
    rects_inter = _intersection_torch(rects1, rects2)
    areas_inter = torch.prod(rects_inter[..., 2:], dim=-1)
    areas1 = torch.prod(rects1[..., 2:], dim=-1)
    areas2 = torch.prod(rects2[..., 2:], dim=-1)
    areas_union = areas1 + areas2 - areas_inter
    eps = torch.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = torch.clip(ious, 0.0, 1.0)

    # Get Minimum Covering BBOX
    x1 = torch.minimum(rects1[:, 0], rects2[:, 0])
    y1 = torch.minimum(rects1[:, 1], rects2[:, 1])
    x2 = torch.maximum(rects1[:, 0] + rects1[:, 2], rects2[:, 0] + rects2[:, 2])
    y2 = torch.maximum(rects1[:, 1] + rects1[:, 3], rects2[:, 1] + rects2[:, 3])
    covbbox_diag_square = torch.square(x1 - x2) + torch.square(y1 - y2)

    # Get Center
    rects1_cx, rects1_cy = rects1[:, 0] + rects1[:, 2] / 2.0, rects1[:, 1] + rects1[:, 3] / 2.0
    rects2_cx, rects2_cy = rects2[:, 0] + rects2[:, 2] / 2.0, rects2[:, 1] + rects2[:, 3] / 2.0
    bbox_center_square = torch.square(rects1_cx - rects2_cx) + torch.square(rects1_cy - rects2_cy)

    # Get DIoU
    dious = ious - torch.divide(bbox_center_square, covbbox_diag_square)

    return dious


def _intersection(rects1, rects2):
    r"""Rectangle intersection.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    assert rects1.shape == rects2.shape
    x1 = np.maximum(rects1[..., 0], rects2[..., 0])
    y1 = np.maximum(rects1[..., 1], rects2[..., 1])
    x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                    rects2[..., 0] + rects2[..., 2])
    y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                    rects2[..., 1] + rects2[..., 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    return np.stack([x1, y1, w, h]).T


def _intersection_torch(rects1, rects2):
    assert rects1.shape == rects2.shape
    x1 = torch.maximum(rects1[..., 0], rects2[..., 0])
    y1 = torch.maximum(rects1[..., 1], rects2[..., 1])
    x2 = torch.minimum(rects1[..., 0] + rects1[..., 2],
                       rects2[..., 0] + rects2[..., 2])
    y2 = torch.minimum(rects1[..., 1] + rects1[..., 3],
                       rects2[..., 1] + rects2[..., 3])
    w = torch.maximum(x2 - x1, torch.zeros_like(x2))
    h = torch.maximum(y2 - y1, torch.zeros_like(y2))

    return torch.stack([x1, y1, w, h]).T

if __name__ == "__main__":
    bboxes1 = [
        [100, 100, 20, 20],
        [110, 90, 20, 30],
    ]
    bboxes2 = [
        [110, 110, 20, 20],
        [150, 150, 40, 40]
    ]
    bboxes1_gpu = torch.Tensor(bboxes1)
    bboxes2_gpu = torch.Tensor(bboxes2)

    # IoU Torch
    # tt = rect_iou_torch(bboxes1_gpu, bboxes2_gpu)
    tt = rect_diou_torch(bboxes1_gpu, bboxes2_gpu)