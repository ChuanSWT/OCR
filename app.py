import os
import pandas as pd
import cv2
from paddleocr import PaddleOCR
#====准备工作-------------------------------------
#临时展示窗口的属性设置
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("window", 800, 600)

#创建ocr识别模型
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

#导入视频和元数据
cap = cv2.VideoCapture("data/wires/video.mp4")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

#获取编码器和写出视频流
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("output.mp4", fourcc,fps,(width,height))

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

    output=img.copy()
    #读取标签文件
    label_path=f"data/wires/labels/video_{str(index)}.txt"

    if not os.path.exists(label_path):
        cv2.imwrite(f"data/wires/outputs/{index}.png",output)
        writer.write(output)
        print(f"index-{index} wrote")
        continue

    df = pd.read_csv(f"data/wires/labels/video_{str(index)}.txt", sep=" ",header=None)
    
    for x in range(0,df.shape[0]):
        #获取标记信息
        yolo_type=int(df.loc[x,0])

        x_center=int(width*df.loc[x,1])
        y_center=int(height*df.loc[x,2])

        x_width=int(width*df.loc[x,3])
        y_width=int(height*df.loc[x,4])
    
        x1=max(0,x_center-x_width//2)
        y1=max(0,y_center-y_width//2)
        x2=min(width,x_center+x_width//2)
        y2=min(height,y_center+y_width//2)#标记是一个从(x1,y1)到(x2,y2)所确定的矩形

        #截取标记所覆盖的图片
        crop=img[y1:y2,x1:x2]
        #识别
        result = ocr.predict(input=crop)

        
        if len(result[0]['rec_texts']) == 0:
            continue
        #for res in result:
        #    print(res['rec_texts'],res["rec_scores"])
        # 画矩形
        cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        color=colors[yolo_type],   # BGR
        thickness=2
        )
        # 写文字
        cv2.putText(
            output,
            " ".join([result[0]['rec_texts'][0],"conf:",format(result[0]["rec_scores"][0], ".2f")]),
            (x1, y1 - 10),      # 文字左下角
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,                # 字体大小
            colors[yolo_type],
            2
        )
    cv2.imwrite(f"data/wires/outputs/{index}.png",output)
    writer.write(output)
    cv2.imshow('window',output)
    cv2.waitKey(1)
    print(f"index-{index} wrote")

    
writer.release()
cap.release()
cv2.destroyAllWindows()
