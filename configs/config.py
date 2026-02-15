import os
import pandas as pd
import cv2
from paddleocr import PaddleOCR
#临时展示窗口的属性设置
#cv2.namedWindow("window", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("window", 800, 600)

#创建ocr识别模型
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

#====定义常量-------------------------------------
yolo_class=["wire","slot","head"]
colors=[(255,0,0),(0,255,0),[0,0,255]]