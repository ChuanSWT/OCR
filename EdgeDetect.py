import os
import pandas as pd
import cv2
from paddleocr import PaddleOCR
import numpy as np
#====准备工作-------------------------------------
#临时展示窗口的属性设置
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("window", 800, 600)

#创建边缘检测核
kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (2, 1)
)

# 3. 霍夫直线变换（概率）
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


#导入视频和元数据
cap = cv2.VideoCapture("data/wires/video.mp4")
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

#获取编码器和写出视频流
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("output_edge.mp4", fourcc,fps,(width,height))

#====定义常量-------------------------------------
colors=[(255,0,0),(0,255,0),[0,0,255]]


#====正式逻辑-------------------------------------

index=0

while True:
    ret, img = cap.read()
    index+=1
    if not ret:
        print("视频读取结束")
        break

    #检测边缘(canny)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edge = cv2.Canny(gray, threshold1=100, threshold2=200)

    #横向侵蚀 以提取横边
    _, img_edge = cv2.threshold(img_edge, 128, 255, cv2.THRESH_BINARY)
    img_edge_eroded = cv2.erode(img_edge, kernel, iterations=1)
    
    #线段检测/绘制
    img_lines=HoughLinesP(img_edge_eroded)
    if img_lines is not None:
        print("got lines")
        for x1, y1, x2, y2 in img_lines[:, 0]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            print({x1,y1},{x2,y2})

    #====写出-------------------------------------
    output=img #取出写出帧
    #op1:写彩色
    cv2.imwrite(f"data/wires/outputs/{index}.png",output)
    writer.write(output)
    #op2:写灰白
    #cv2.imwrite(f"data/wires/outputs/{index}.png",cv2.cvtColor(output, cv2.COLOR_GRAY2BGR))
    #writer.write(cv2.cvtColor(output, cv2.COLOR_GRAY2BGR))
    cv2.imshow('window',output)
    cv2.waitKey(1)
    print(f"index-{index} wrote")

    
writer.release()
cap.release()
cv2.destroyAllWindows()
