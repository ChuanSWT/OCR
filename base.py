#识别单张图片
import os
import pandas as pd
import cv2
from paddleocr import PaddleOCR
import configs.config as config
import functions.EdgeDetect
from models.YoloRec import DetectionResult, YoloRec
#====准备工作-------------------------------------
#==导入函数---------------------
EdgeDetect=functions.EdgeDetect.EdgeDetect
EdgeClassify=functions.EdgeDetect.EdgeClassify
#==导入模型---------------------
ocr=config.ocr
colors=config.colors

#====正式逻辑-------------------------------------
recs=[]
def Detect(img,index, return_rec: bool = False):
    height, width = img.shape[:2]

    rec = YoloRec()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = img.copy()
    #读取标签文件
    label_path=f"data/wires/labels/video_{str(index)}.txt"

    if not os.path.exists(label_path):
        cv2.imwrite(f"data/wires/outputs/{index}.png",output)
        return output

    df = pd.read_csv(f"data/wires/labels/video_{str(index)}.txt", sep=" ",header=None)
    #进行ocr识别
    for x in range(0,df.shape[0]):
        #获取信息
        #标记是一个从(x1,y1)到(x2,y2)所确定的矩形
        yolo_type=int(df.loc[x,0])

        x_center=int(width*df.loc[x,1])
        y_center=int(height*df.loc[x,2])

        x_width=int(width*df.loc[x,3])
        y_width=int(height*df.loc[x,4])
    
        x1=max(0,x_center-x_width//2)
        y1=max(0,y_center-y_width//2)
        x2=min(width,x_center+x_width//2)
        y2=min(height,y_center+y_width//2)

        #截取标记所覆盖的图片
        crop=img[y1:y2,x1:x2]
        #识别
        result = ocr.predict(input=crop)

        #没有识别到就退出
        if len(result[0]['rec_texts']) == 0:
            continue

        # 保存检测结果到 YoloRec
        det = DetectionResult(
            x_center=x_center,
            y_center=y_center,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=float(result[0]["rec_scores"][0]) if result[0].get("rec_scores") else None,
            class_id=yolo_type,
            label=result[0]['rec_texts'][0] if result[0].get('rec_texts') else None,
        )
        rec.add(det)

        #进行图像标记
        cv2.rectangle(output,(x1, y1),(x2, y2),color=colors[yolo_type],thickness=2)
        cv2.putText(
            output,
            " ".join([result[0]['rec_texts'][0],"conf:",format(result[0]["rec_scores"][0], ".2f")]),
            (x1, y1 - 10),      # 文字左下角
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,                # 字体大小
            colors[yolo_type],
            2
        )
    recs.append(rec)
    #
    lines=EdgeDetect(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if(abs(x1-x2)<abs(y1-y2)):
                continue
            cv2.line(output,(x1,y1),(x2,y2),colors[2],2)

    if return_rec:
        return output, recs
    return output

