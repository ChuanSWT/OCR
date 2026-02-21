from math import hypot
from typing import List, Dict, Optional, Tuple


def find_best_historical_label(det_box: Dict, history_frames: List[List[Dict]]) -> Optional[Tuple[str, Optional[float]]]:
    """
    Search recent history for the detection with a non-empty OCR label whose center
    is closest (Euclidean) to the current detection center.

    det_box: dict with keys 'x_center','y_center','x1','y1','x2','y2'
    history_frames: list of frames, each frame is list of detection dicts

    Returns (label, confidence) if a suitable historical entry is found and its
    center lies inside the current box; otherwise returns None.
    """
    cx = det_box.get('x_center')
    cy = det_box.get('y_center')
    if cx is None or cy is None:
        return None

    best = None
    best_dist = float('inf')
    best_conf = None

    # Flatten history and search
    for frame in history_frames:
        for rec in frame:
            label = rec.get('label')
            if not label:
                continue
            # treat 'unknown' (case-insensitive) as empty
            if isinstance(label, str) and label.strip().lower() == 'unknown':
                continue

            hx = rec.get('x_center')
            hy = rec.get('y_center')
            if hx is None or hy is None:
                continue
            d = hypot(cx - hx, cy - hy)
            if d < best_dist:
                best_dist = d
                best = rec
                best_conf = rec.get('confidence')

    if best is None:
        return None

    # Check whether the historical center still lies within current detection box
    x1 = det_box.get('x1')
    y1 = det_box.get('y1')
    x2 = det_box.get('x2')
    y2 = det_box.get('y2')
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None

    # normalize
    bx1, bx2 = min(x1, x2), max(x1, x2)
    by1, by2 = min(y1, y2), max(y1, y2)

    if bx1 <= best.get('x_center') <= bx2 and by1 <= best.get('y_center') <= by2:
        return best.get('label'), best_conf

    return None
