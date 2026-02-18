import os
import pandas as pd
import cv2
from paddleocr import PaddleOCR
import configs.config as config
import base
#====准备工作-------------------------------------
#==导入函数---------------------
Detect=base.Detect
from functions.EdgeDetect import EdgeClassify
from functions.Lines import should_skip_line
#==导入模型和配置---------------------
ocr=config.ocr
colors=config.colors
video_config=config.VIDEO_CONFIG

#导入视频和元数据
cap = cv2.VideoCapture("data/wires/video.mp4")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

#获取编码器和写出视频流
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("output.mp4", fourcc,fps,(width,height))

#====正式逻辑-------------------------------------
# 获取视频帧范围配置
start_frame = video_config.get('start_frame')
end_frame = video_config.get('end_frame')

# 如果指定了起始帧，跳过前面的帧
if start_frame is not None and start_frame > 0:
    for _ in range(start_frame):
        cap.read()

index = 0 if start_frame is None else start_frame

while True:
    ret, img = cap.read()
    if not ret:
        print("视频读取结束")
        break
    
    # 检查是否超过结束帧
    if end_frame is not None and index > end_frame:
        print(f"已处理到指定的结束帧 {end_frame}")
        break
    
    index+=1
    rst=Detect(img,index)
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # draw horizontal midline in purple to show top/bottom halves
    mid = height / 2.0
    purple = (255, 0, 255)
    cv2.line(output, (0, int(mid)), (width, int(mid)), purple, 2)
    '''
    for det in rst.recs:
        x1, y1, x2, y2 = det.to_xyxy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(output,(x1, y1),(x2, y2),color=colors[det.class_id],thickness=2)
        cv2.putText(
            output,
            " ".join([det.label,"conf:",format(det.confidence, ".2f")]),
            (x1, y1 - 10),      # 文字左下角
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,                # 字体大小
            colors[det.class_id],
            2
        )
    '''
    # 收集所有类型0的识别中心点和标签
    type0_detections = []
    blue = (255, 0, 0)
    for det in rst.recs:
        if det.class_id == 0:
            type0_detections.append({
                'x': det.x_center,
                'y': det.y_center,
                'label': det.label or "Unknown"
            })
            # 绘制类型0识别框的中心点（较小的蓝色点）
            center_x, center_y = int(det.x_center), int(det.y_center)
            cv2.circle(output, (center_x, center_y), 3, blue, -1)  # 填充圆形，半径为3
    
    for line in (rst.lines or []):
        # EdgeClassify expects a line in the format where line[0] = (x1,y1,x2,y2)
        cls = EdgeClassify(gray, [line])
        x1, y1, x2, y2 = line
        
        # 检查线段的斜率，如果斜率大于1则跳过
        if should_skip_line(x1, y1, x2, y2, slope_threshold=1):
            continue
        
        # normalize line bbox
        lx_min, lx_max = min(x1, x2), max(x1, x2)
        ly_min, ly_max = min(y1, y2), max(y1, y2)

        # Check if the line crosses the horizontal middle boundary
        crosses_mid = (ly_min < mid and ly_max > mid)

        drawn = False

        # If the line crosses mid, it should NOT be considered for contained+annotated drawing;
        # it will be drawn as red fallback below.
        if not crosses_mid:
            # Determine which half the line belongs to (both endpoints are on same side)
            in_top = ly_max <= mid
            in_bottom = ly_min >= mid

            # required classification per half (per request: top -> 'down', bottom -> 'up')
            required_cls = None
            if in_top:
                required_cls = "up"
            elif in_bottom:
                required_cls = "down"
            #to chat:请检查一下opencv的坐标系

            # Only draw colored+annotated line if it's fully contained in a detection box
            # and classification matches the half-specific requirement
            if required_cls is not None and cls == required_cls:
                for det in rst.recs:
                    if det.class_id == 0:
                        continue
                    bx1, by1, bx2, by2 = det.to_xyxy()
                    if lx_min >= bx1 and ly_min >= by1 and lx_max <= bx2 and ly_max <= by2:
                        color = colors[det.class_id] if det.class_id is not None else (0, 255, 0)
                        
                        # 计算线段向右延长到画面右边的端点
                        # 如果线段是垂直线
                        if abs(x2 - x1) < 1e-6:
                            # 垂直线，保持 x 坐标，y 坐标不变
                            x1_ext, y1_ext = x1, y1
                            x2_ext, y2_ext = x2, y2
                        else:
                            # 计算直线斜率
                            slope = (y2 - y1) / (x2 - x1)
                            # 向右延长到 x = width
                            x_right = width
                            if x1 <= x_right:
                                y_right = y1 + slope * (x_right - x1)
                                x1_ext, y1_ext = int(x1), int(y1)
                                x2_ext, y2_ext = int(x_right), int(y_right)
                            else:
                                x1_ext, y1_ext = int(x1), int(y1)
                                x2_ext, y2_ext = int(x2), int(y2)
                        
                        cv2.line(output, (x1_ext, y1_ext), (x2_ext, y2_ext), color, 2)
                        
                        # 寻找最近的类型0识别中心点
                        # 对于上方线段，找直线上方的点；对于下方线段，找直线下方的点
                        best_type0 = None
                        best_distance = float('inf')
                        
                        for type0_det in type0_detections:
                            t0_x, t0_y = type0_det['x'], type0_det['y']
                            
                            # 计算点到直线的距离
                            # 直线方程：(y2-y1)*x - (x2-x1)*y + (x2-x1)*y1 - (y2-y1)*x1 = 0
                            if abs(x2 - x1) < 1e-6:
                                # 垂直线
                                dist = abs(t0_x - x1)
                                point_y = t0_y
                                line_y = y1
                            else:
                                # 一般直线
                                A = y2 - y1
                                B = -(x2 - x1)
                                C = (x2 - x1) * y1 - (y2 - y1) * x1
                                dist = abs(A * t0_x + B * t0_y + C) / (A*A + B*B)**0.5
                                # 计算点在直线上的投影
                                point_y = t0_y
                                line_y_at_t0_x = y1 + (y2 - y1) / (x2 - x1) * (t0_x - x1)
                            
                            # 检查点是否在正确的一侧
                            if in_top:
                                # 上方线段，需要点在直线上方（y值更小）
                                if point_y < line_y_at_t0_x and dist < best_distance:
                                    best_distance = dist
                                    best_type0 = type0_det
                            elif in_bottom:
                                # 下方线段，需要点在直线下方（y值更大）
                                if point_y > line_y_at_t0_x and dist < best_distance:
                                    best_distance = dist
                                    best_type0 = type0_det
                        
                        # 并排绘制绿色文字（所在识别框的OCR结果）和蓝色文字（类型0的OCR结果）
                        mx = int((x1 + x2) / 2)
                        my = int((y1 + y2) / 2)
                        green = (0, 255, 0)
                        blue = (255, 0, 0)
                        
                        # 绿色文字：线段所处识别框的OCR结果
                        line_label = det.label or ""
                        cv2.putText(
                            output,
                            line_label,
                            (mx, my - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            green,
                            2,
                        )
                        
                        # 蓝色文字：类型0的OCR识别结果，在绿色文字右侧
                        if best_type0:
                            type0_label = best_type0['label']
                            # 计算绿色文字的宽度（粗略估计）
                            green_text_width = len(line_label) * 10
                            # 蓝色文字位置：在绿色文字右侧
                            text_x = mx + green_text_width + 10
                            text_y = my - 10
                            cv2.putText(
                                output,
                                type0_label,
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                blue,
                                2,
                            )
                        
                        drawn = True
                        break

        # If not drawn by the above rules, draw as red fallback (no OCR label, containment not required)
        if not drawn:
            if should_skip_line(x1, y1, x2, y2, slope_threshold=1):
                continue
            red = (0, 0, 255)
            cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)), red, 2)
    cv2.imwrite(f"data/wires/outputs/{index}.png",output)
    writer.write(output)
    #cv2.imshow('window',output)
    #cv2.waitKey(1)
    print(f"index-{index} wrote")

    
writer.release()
cap.release()
cv2.destroyAllWindows()
