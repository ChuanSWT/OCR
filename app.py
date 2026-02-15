import os
import pandas as pd
import cv2
from paddleocr import PaddleOCR
import configs.config as config
import base
#====准备工作-------------------------------------
#==导入函数---------------------
Detect=base.Detect
#==导入模型---------------------
ocr=config.ocr
colors=config.colors
#导入视频和元数据
cap = cv2.VideoCapture("data/wires/video.mp4")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

#获取编码器和写出视频流
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("output.mp4", fourcc,fps,(width,height))

#====正式逻辑-------------------------------------

index=0

while True:
    ret, img = cap.read()
    index+=1
    if not ret:
        print("视频读取结束")
        break
    output=Detect(img,index)
    cv2.imwrite(f"data/wires/outputs/{index}.png",output)
    writer.write(output)
    #cv2.imshow('window',output)
    #cv2.waitKey(1)
    print(f"index-{index} wrote")

    
writer.release()
cap.release()
cv2.destroyAllWindows()
