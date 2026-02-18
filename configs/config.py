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

#====视频处理配置-------------------------------------
# 视频帧范围设置
# start_frame: 开始帧号（从0开始计数），设为None表示从第一帧开始
# end_frame: 结束帧号（包括该帧），设为None表示处理到视频结尾
VIDEO_CONFIG = {
    'start_frame': None,  # 设置为None或0表示从开始，或指定具体帧号如100
    'end_frame': None     # 设置为None表示到结尾，或指定具体帧号如500
}

#====边缘和直线检测配置-------------------------------------
# 是否使用自适应参数进行边缘和直线检测
# True: 根据图像特性自动调整Canny和HoughLinesP参数
# False: 使用固定参数
EDGE_DETECT_CONFIG = {
    'adaptive': False,  # 启用自适应检测
}
