import cv2
import numpy as np
import torch
from wingman.utils import cpuize


def _count_contours(torch_image: torch.Tensor) -> np.ndarray:
    """Finds the number of parent-only contours given a [C, H, W] torch Tensor."""
    assert len(torch_image.shape) == 3

    image = cpuize(torch_image)
    image = image.astype(np.uint8)
    image *= 255

    num_contours = []
    for i in range(image.shape[0]):
        contours, _ = cv2.findContours(
            image[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        num_contours.append(len(contours))

    return np.array(num_contours)


def compute_precision_recall_contours(
    prediction: torch.Tensor, label: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the contourwise channel-level precision and recall of a prediction against a label.

    Given two equal shaped boolean tensors of shape (C, H, W),
    computes the precision and recall of the prediction against the label along the channel dimension.

    Args:
        prediction (torch.Tensor): prediction
        label (torch.Tensor): label

    Returns:
        tuple[np.ndarray, np.ndarray]: two (C) long vectors for precision and recall.
    """
    assert prediction.shape == label.shape
    assert prediction.dtype == torch.bool
    assert label.dtype == torch.bool
    assert len(prediction.shape) == 3
    assert len(label.shape) == 3

    # get true pos, true neg, false pos, false neg maps
    guess = prediction
    truth = label
    joint = prediction & label

    # count metrics for each map
    TP = np.clip(_count_contours(joint), a_min=0, a_max=None)
    FP = np.clip(_count_contours(guess) - TP, a_min=0, a_max=None)
    FN = np.clip(_count_contours(truth) - TP, a_min=0, a_max=None)

    # compute precision recall
    precision = TP / (TP + FP + 1e-6)
    precision[TP + FP == 0] = 1.0
    recall = TP / (TP + FN + 1e-6)
    recall[TP + FN == 0] = 1.0

    return precision, recall


def compute_precision_recall_pixel(
    prediction: torch.Tensor, label: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the pixelwise channel-level precision and recall of a prediction against a label.

    Given two equal shaped boolean tensors of shape (C, H, W),
    computes the precision and recall of the prediction against the label along the channel dimension.

    Args:
        prediction (torch.Tensor): input of shape (C, H, W).
        label (torch.Tensor): input of shape (C, H, W).

    Returns:
        tuple[np.ndarray, np.ndarray]: two (C) long vectors for precision and recall.
    """
    assert prediction.shape == label.shape
    assert prediction.dtype == torch.bool
    assert label.dtype == torch.bool
    assert len(prediction.shape) == 3
    assert len(label.shape) == 3

    # compute TP, TN, FP, FN
    TP = (prediction & label).sum(dim=[-1, -2])
    TN = (~prediction & ~label).sum(dim=[-1, -2])  # noqa: F841
    FP = (prediction & ~label).sum(dim=[-1, -2])
    FN = (~prediction & label).sum(dim=[-1, -2])

    # compute precision recall
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    return cpuize(precision), cpuize(recall)
