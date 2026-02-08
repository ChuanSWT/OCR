#识别单张图片
import pandas as pd
import cv2
from paddleocr import PaddleOCR
#====准备工作-------------------------------------
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("window", 800, 600)
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

#====正式逻辑-------------------------------------
df = pd.read_csv("data/test/labels/test.txt", sep=" ",header=None)
#print(df)

img=cv2.imread("data/test/images/test.png")
height,width,deepth=img.shape
for x in range(0,df.shape[0]):

    x_center=int(width*df.loc[x,1])
    y_center=int(height*df.loc[x,2])

    x_width=int(width*df.loc[x,3])
    y_width=int(height*df.loc[x,4])
    
    x1=max(0,x_center-x_width//2)
    y1=max(0,y_center-y_width//2)
    x2=min(width,x_center+x_width//2)
    y2=min(height,y_center+y_width//2)
    crop=img[y1:y2,x1:x2]
    result = ocr.predict(input=crop)
    #ok
    for res in result:
        print(res['rec_texts'],res["rec_scores"][0])

    
    # 画矩形
    cv2.rectangle(
    img,
    (x1, y1),
    (x2, y2),
    color=(0, 255, 0),   # BGR
    thickness=2
    )
    # 写文字
    cv2.putText(
        img,
        " ".join([res['rec_texts'][0],"conf:",str(res["rec_scores"][0])]),
        (x1, y1 - 10),      # 文字左下角
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,                # 字体大小
        (0, 255, 0),
        2
    )

#cv2.imwrite("data/test/output/output.jpg",img)
cv2.imshow("window",img)
cv2.waitKey(0)