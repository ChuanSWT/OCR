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

def draw_type1_with_leader_line(
    img,
    text_color: Tuple[int, int, int],
    line_color: Tuple[int, int, int],
    position: str,
    type1_box: Tuple[float, float, float, float],
    ocr_text: str,
    font_scale: float = 0.6,
    thickness: int = 2,
    line_thickness: int = 2,
):
    """
    绘制类型1识别框的OCR文本，并用引出线连接到识别框。

    参数:
    - img: 目标图片（原地修改）
    - text_color: 文本颜色 (BGR)
    - line_color: 线条颜色 (BGR)
    - position: "left" 或 "right"，表示文本显示的位置
    - type1_box: (x1, y1, x2, y2) 识别框坐标
    - ocr_text: 要显示的OCR文本
    - font_scale: 文字大小缩放
    - thickness: 文字粗细
    - line_thickness: 引出线的粗细

    逻辑:
    - 根据position参数，在左边或右边显示文本
    - 从文本位置引出一条水平线连接到识别框的相应侧面中点
    """
    height, width = img.shape[:2]
    bx1, by1, bx2, by2 = map(int, type1_box)
    
    # 计算识别框的中心y坐标
    box_center_y = int((by1 + by2) / 2)
    
    # 获取文本大小
    text_size = cv2.getTextSize(ocr_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_width = text_size[0]
    text_height = text_size[1]
    
    if position == "left":
        # 文本显示在左边
        text_margin = 10  # 文本距离左边的距离
        text_x = text_margin
        text_y = box_center_y + text_height // 2
        
        # 文本引出点（文本右侧）
        text_right = text_x + text_width + 5
        
        # 识别框的左侧中点
        box_left_center_x = bx1
        box_left_center_y = box_center_y
        
        # 绘制文本
        cv2.putText(
            img,
            ocr_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
        )
        
        # 绘制引出线（水平线从文本右侧到识别框左侧）
        cv2.line(
            img,
            (text_right, box_center_y),
            (box_left_center_x, box_left_center_y),
            line_color,
            line_thickness,
        )
        
    elif position == "right":
        # 文本显示在右边
        text_margin = 10  # 文本距离右边的距离
        text_x = width - text_width - text_margin
        text_y = box_center_y + text_height // 2
        
        # 文本引出点（文本左侧）
        text_left = text_x - 5
        
        # 识别框的右侧中点
        box_right_center_x = bx2
        box_right_center_y = box_center_y
        
        # 绘制文本
        cv2.putText(
            img,
            ocr_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
        )
        
        # 绘制引出线（水平线从识别框右侧到文本左侧）
        cv2.line(
            img,
            (box_right_center_x, box_right_center_y),
            (text_left, box_center_y),
            line_color,
            line_thickness,
        )