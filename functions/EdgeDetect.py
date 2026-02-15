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
    threshold=80,
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
def EdgeClassify(gray,line):
    x1, y1, x2, y2 = line[0]

    h = 2
    y = int((y1 + y2) / 2)
    x_start = min(x1, x2)
    x_end = max(x1, x2)

    upper_y1 = max(0, y - h)
    upper_y2 = y
    lower_y1 = y
    lower_y2 = min(gray.shape[0], y + h)

    upper_roi = gray[upper_y1:upper_y2, x_start:x_end]
    lower_roi = gray[lower_y1:lower_y2, x_start:x_end]

    upper_val = np.median(upper_roi)
    lower_val = np.median(lower_roi)

    if upper_val > lower_val:
        return "down"
    return "up"

    