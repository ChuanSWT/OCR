import cv2
from typing import Optional, Tuple


def draw_type_labels(
    img,
    green_label: str,
    blue_label: Optional[str],
    type2_box: Tuple[float, float, float, float],
    type0_box: Optional[Tuple[float, float, float, float]] = None,
    green_color=(0, 255, 0),
    blue_color=(255, 0, 0),
    font_scale: float = 0.6,
    thickness: int = 2,
):
    """
    Draw the green (type2) label and optional blue (type0) label on `img`.

    Parameters:
    - img: OpenCV image to draw onto (modified in place).
    - green_label: OCR text for the type-2 box (drawn in green).
    - blue_label: OCR text for the type-0 box (drawn in blue), or None to skip.
    - type2_box: (x1,y1,x2,y2) coordinates of the type-2 detection box.
    - type0_box: (x1,y1,x2,y2) coords of the type-0 detection box (unused for placement,
                 but included for API completeness).
    - green_color, blue_color: BGR tuples for text colors.
    - font_scale, thickness: text rendering params.

    Behavior mirrors previous inline logic: place green text just outside the
    right-top of the type-2 box, and blue text 20px below the green text.
    Text placement is adjusted if it would go off the top of the image.
    """
    det_x1, det_y1, det_x2, det_y2 = map(int, type2_box)

    # Preferred position: right of type2 box, near its top
    text_x = det_x2 + 5
    text_y = det_y1 + 15

    # If text would be too close to top, place inside box slightly below top
    if text_y < 10:
        text_y = det_y1 + 15

    if green_label:
        cv2.putText(
            img,
            green_label,
            (int(text_x), int(text_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            green_color,
            thickness,
        )

    if blue_label:
        text_x_blue = text_x
        text_y_blue = text_y + 20
        cv2.putText(
            img,
            blue_label,
            (int(text_x_blue), int(text_y_blue)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            blue_color,
            thickness,
        )
