import os
import pandas as pd
import cv2
from paddleocr import PaddleOCR
import numpy as np

#====准备工作-------------------------------------
#创建侵蚀核
kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (3, 1)
)

#霍夫直线变换
def HoughLinesP(edges):
    lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=60,
    minLineLength=50,
    maxLineGap=10
    )
    return lines

#==逻辑实现------------------------------
def EdgeDetect(frame):
    img=frame.copy()
    #检测边缘(canny)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edge = cv2.Canny(gray, threshold1=100, threshold2=200)

    #横向侵蚀 以提取横边
    _, img_edge = cv2.threshold(img_edge, 128, 255, cv2.THRESH_BINARY)
    img_edge_eroded = cv2.erode(img_edge, kernel, iterations=1)
    
    #线段检测/绘制
    img_lines=HoughLinesP(img_edge_eroded)
    return img_lines

#判断是上边还是下边
def EdgeClassify(gray, line, offset: int = 5, thickness: int = 6):
    """Classify a line as 'up' or 'down' by comparing median brightness of
    two narrow regions formed by shifting the original line up and down.

    Parameters:
    - gray: grayscale image (2D numpy array)
    - line: single line returned by HoughLinesP (expects line[0] = (x1,y1,x2,y2))
    - offset: pixels to shift the line perpendicular to its direction
    - thickness: width (in pixels) of the sampling band around the shifted line

    Returns:
    - 'down' if the upper region's median brightness > lower region's median
      brightness, otherwise 'up' (as requested).
    """
    x1, y1, x2, y2 = line[0]

    # line vector and its length
    dx = x2 - x1
    dy = y2 - y1
    length = max(1.0, np.hypot(dx, dy))

    # unit perpendicular (normal) vector: (-dy, dx) / length
    nx = -dy / length
    ny = dx / length

    # shifted line endpoints for upper and lower regions
    up_shift = (nx * offset, ny * offset)
    down_shift = (-nx * offset, -ny * offset)

    def polygon_for_shift(shift):
        sx, sy = shift
        # shifted endpoints (float)
        p1 = (x1 + sx, y1 + sy)
        p2 = (x2 + sx, y2 + sy)
        # half thickness vector along normal
        half_t = thickness / 2.0
        tx = nx * half_t
        ty = ny * half_t
        # polygon corners: p1+T, p1-T, p2-T, p2+T
        poly = np.array([
            [p1[0] + tx, p1[1] + ty],
            [p1[0] - tx, p1[1] - ty],
            [p2[0] - tx, p2[1] - ty],
            [p2[0] + tx, p2[1] + ty],
        ], dtype=np.int32)
        return poly

    h, w = gray.shape[:2]

    up_poly = polygon_for_shift(up_shift)
    down_poly = polygon_for_shift(down_shift)

    # create masks and compute median inside polygons
    mask_up = np.zeros((h, w), dtype=np.uint8)
    mask_down = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_up, [up_poly], 255)
    cv2.fillPoly(mask_down, [down_poly], 255)

    up_vals = gray[mask_up == 255]
    down_vals = gray[mask_down == 255]

    # handle empty regions
    if up_vals.size == 0 and down_vals.size == 0:
        return "up"
    if up_vals.size == 0:
        return "up"
    if down_vals.size == 0:
        return "down"

    up_med = float(np.median(up_vals))
    down_med = float(np.median(down_vals))

    if up_med > down_med:
        return "down"
    return "up"

    