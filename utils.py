"""Utility helpers for drawing detections and loading class labels."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def load_class_names(names_file: str | Path) -> list[str]:
    """Return a list of class names loaded from a plain-text file (one per line)."""
    path = Path(names_file)
    if not path.exists():
        raise FileNotFoundError(f"Class names file not found: {path.resolve()}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def draw_detections(
    image: np.ndarray,
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    class_ids: Sequence[int],
    class_names: Sequence[str] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.6,
    text_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Draw bounding boxes and optional labels onto *image* (in-place) and return it.

    Args:
        image: BGR image as a NumPy array.
        boxes: Sequence of ``[x1, y1, x2, y2]`` bounding-box coordinates.
        scores: Confidence score for each detection.
        class_ids: Integer class index for each detection.
        class_names: Optional sequence mapping class index to name string.
        color: BGR colour for the box rectangle.
        thickness: Line thickness in pixels.
        font_scale: Font scale used for label text.
        text_color: BGR colour for the label text (default: black).

    Returns:
        The annotated image (same object as *image*).
    """
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        label_parts = []
        if class_names and 0 <= cls_id < len(class_names):
            label_parts.append(class_names[cls_id])
        label_parts.append(f"{score:.2f}")
        label = " ".join(label_parts)

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y = max(y1, text_h + baseline)
        cv2.rectangle(
            image,
            (x1, label_y - text_h - baseline),
            (x1 + text_w, label_y),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            image,
            label,
            (x1, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
        )

    return image
