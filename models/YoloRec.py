from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict


@dataclass
class DetectionResult:
    """Stores a single YOLOv5 detection result.

    Fields:
    - x_center, y_center: center coordinates of the box
    - x1, y1, x2, y2: coordinates of two diagonal vertices (any order)
    - confidence: detection confidence score (optional)
    - class_id: numeric class id (optional)
    - label: human-readable class label (optional)
    """
    x_center: float
    y_center: float
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: Optional[float] = None
    class_id: Optional[int] = None
    label: Optional[str] = None

    def width(self) -> float:
        return abs(self.x2 - self.x1)

    def height(self) -> float:
        return abs(self.y2 - self.y1)

    def area(self) -> float:
        return self.width() * self.height()

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Return box as (x_min, y_min, x_max, y_max)."""
        x_min = min(self.x1, self.x2)
        y_min = min(self.y1, self.y2)
        x_max = max(self.x1, self.x2)
        y_max = max(self.y1, self.y2)
        return x_min, y_min, x_max, y_max

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Return box as (x_center, y_center, width, height)."""
        x_min, y_min, x_max, y_max = self.to_xyxy()
        w = x_max - x_min
        h = y_max - y_min
        cx = x_min + w / 2.0
        cy = y_min + h / 2.0
        return cx, cy, w, h

    def as_dict(self) -> Dict:
        return {
            "x_center": self.x_center,
            "y_center": self.y_center,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "label": self.label,
        }

    def __repr__(self) -> str:
        return (
            f"DetectionResult(label={self.label}, class_id={self.class_id}, "
            f"conf={self.confidence:.3f} if self.confidence is not None else None, "
            f"cx={self.x_center:.1f}, cy={self.y_center:.1f}, "
            f"x1={self.x1:.1f}, y1={self.y1:.1f}, x2={self.x2:.1f}, y2={self.y2:.1f})"
        )


class YoloRec:
    """Container for multiple `DetectionResult` entries."""

    def __init__(self, detections: Optional[List[DetectionResult]] = None):
        self.detections: List[DetectionResult] = detections or []

    def add(self, det: DetectionResult) -> None:
        self.detections.append(det)

    def extend(self, others: List[DetectionResult]) -> None:
        self.detections.extend(others)

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self):
        return iter(self.detections)

    def to_dicts(self) -> List[Dict]:
        return [d.as_dict() for d in self.detections]
