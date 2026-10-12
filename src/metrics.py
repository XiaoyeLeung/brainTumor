from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_scores(
    targets: Sequence[int],
    predictions: Sequence[int],
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(targets, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(targets, predictions, average="macro", zero_division=0)
        ),
    }


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(1, targets.ndim))
    intersection = (probabilities * targets).sum(dim=dimensions)
    total = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    dice = (2.0 * intersection + smooth) / (total + smooth)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
) -> torch.Tensor:
    """Combine pixel-wise BCE and soft Dice losses."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return bce_weight * bce + (1.0 - bce_weight) * soft_dice_loss(logits, targets)


def _binary_overlap(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    metric: str,
    smooth: float = 1e-6,
) -> torch.Tensor:
    predictions = (torch.sigmoid(logits.detach()) >= threshold).float()
    targets = (targets.detach() >= 0.5).float()
    dimensions = tuple(range(1, targets.ndim))
    intersection = (predictions * targets).sum(dim=dimensions)

    if metric == "dice":
        denominator = predictions.sum(dim=dimensions) + targets.sum(dim=dimensions)
        scores = (2.0 * intersection + smooth) / (denominator + smooth)
    else:
        union = (
            predictions.sum(dim=dimensions) + targets.sum(dim=dimensions) - intersection
        )
        scores = (intersection + smooth) / (union + smooth)
    return scores.mean()


def dice_coefficient(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Mean hard Dice score for a batch of binary masks."""
    return _binary_overlap(logits, targets, threshold, metric="dice")


def iou_coefficient(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Mean intersection-over-union score for binary masks."""
    return _binary_overlap(logits, targets, threshold, metric="iou")
