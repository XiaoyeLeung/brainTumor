from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch


@dataclass(frozen=True)
class SegmentationPostprocessConfig:
    """Inference-time cleanup applied to thresholded segmentation masks."""

    min_component_pixels: int = 0
    no_tumor_gate_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.min_component_pixels < 0:
            raise ValueError("min_component_pixels must be non-negative")
        if self.no_tumor_gate_threshold is not None and not (
            0.0 <= self.no_tumor_gate_threshold <= 1.0
        ):
            raise ValueError("no_tumor_gate_threshold must be in [0, 1]")

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def load_postprocess_config(path: Path) -> SegmentationPostprocessConfig:
    """Load the post-processing block stored by threshold calibration."""
    with path.open() as file:
        payload = json.load(file)
    values = payload.get("segmentation_postprocessing", payload)
    if not isinstance(values, dict):
        raise ValueError(f"{path} does not contain a post-processing object")
    return SegmentationPostprocessConfig(
        min_component_pixels=int(values.get("min_component_pixels", 0)),
        no_tumor_gate_threshold=values.get("no_tumor_gate_threshold"),
    )


def align_no_tumor_probabilities(
    segmentation_frame: pd.DataFrame,
    classification_predictions: pd.DataFrame,
    *,
    id_column: str = "filename",
) -> list[float]:
    """Align classifier no-tumor probabilities to segmentation row order."""
    required = {id_column, "no_tumor_probability"}
    if missing := required - set(classification_predictions.columns):
        raise ValueError(
            f"Classification predictions are missing columns: {sorted(missing)}"
        )
    if segmentation_frame[id_column].duplicated().any():
        raise ValueError(f"Segmentation rows contain duplicate {id_column}s")
    if classification_predictions[id_column].duplicated().any():
        raise ValueError(f"Classification predictions contain duplicate {id_column}s")

    segmentation_ids = set(segmentation_frame[id_column])
    classification_ids = set(classification_predictions[id_column])
    if segmentation_ids != classification_ids:
        raise ValueError(
            "Classification and segmentation predictions do not cover the same samples"
        )

    probabilities = classification_predictions.set_index(id_column)[
        "no_tumor_probability"
    ].reindex(segmentation_frame[id_column])
    if probabilities.isna().any() or not probabilities.between(0.0, 1.0).all():
        raise ValueError("no_tumor_probability must contain values in [0, 1]")
    return probabilities.astype(float).tolist()


def filter_small_components(
    mask: torch.Tensor,
    min_component_pixels: int,
) -> tuple[torch.Tensor, int, int]:
    """Remove 8-connected foreground components smaller than the given size."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, received shape {tuple(mask.shape)}")
    if min_component_pixels < 0:
        raise ValueError("min_component_pixels must be non-negative")
    if min_component_pixels <= 1 or not mask.any():
        return mask.bool().clone(), 0, 0

    import numpy as np
    from scipy import ndimage

    array = mask.detach().to(device="cpu", dtype=torch.uint8).numpy()
    labels, component_count = ndimage.label(
        array,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    areas = np.bincount(labels.ravel(), minlength=component_count + 1)
    small_labels = np.flatnonzero(areas[1:] < min_component_pixels) + 1
    keep = array.copy()
    keep[np.isin(labels, small_labels)] = 0
    removed_components = len(small_labels)
    removed_pixels = int(areas[small_labels].sum())

    filtered = torch.from_numpy(keep).to(device=mask.device, dtype=torch.bool)
    return filtered, removed_components, removed_pixels


def postprocess_masks(
    masks: torch.Tensor,
    config: SegmentationPostprocessConfig,
    no_tumor_probabilities: Sequence[float] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Filter components and optionally suppress classifier-confirmed normal masks."""
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(
            "Expected masks with shape (batch, 1, height, width), "
            f"received {tuple(masks.shape)}"
        )
    output_device = masks.device
    cpu_masks = masks.detach().to(device="cpu", dtype=torch.bool)
    if config.no_tumor_gate_threshold is not None:
        if no_tumor_probabilities is None:
            raise ValueError(
                "Classifier probabilities are required when classifier gating is enabled"
            )
        probabilities = torch.as_tensor(
            no_tumor_probabilities,
            device="cpu",
            dtype=torch.float32,
        )
        if probabilities.shape != (len(masks),):
            raise ValueError("Expected one no-tumor probability per mask")
        if not torch.isfinite(probabilities).all() or not (
            (probabilities >= 0.0) & (probabilities <= 1.0)
        ).all():
            raise ValueError("no_tumor_probabilities must be finite values in [0, 1]")
    else:
        probabilities = None

    processed: list[torch.Tensor] = []
    metadata: list[dict[str, object]] = []
    pixel_count = cpu_masks.shape[-2] * cpu_masks.shape[-1]
    for index, batch_mask in enumerate(cpu_masks):
        raw_mask = batch_mask[0]
        filtered, removed_components, removed_pixels = filter_small_components(
            raw_mask,
            config.min_component_pixels,
        )
        filtered_area = int(filtered.sum().item())
        classifier_gated = bool(
            probabilities is not None
            and (
                probabilities[index] >= config.no_tumor_gate_threshold
            ).item()
        )
        final_mask = torch.zeros_like(filtered) if classifier_gated else filtered
        processed.append(final_mask.unsqueeze(0))
        metadata.append(
            {
                "raw_predicted_mask_area_fraction": (
                    raw_mask.sum().item() / pixel_count
                ),
                "filtered_mask_area_fraction": filtered_area / pixel_count,
                "removed_component_count": removed_components,
                "removed_pixel_count": removed_pixels,
                "classifier_gated": classifier_gated,
            }
        )

    return torch.stack(processed).to(output_device), metadata
