import os
import pandas as pd
import cv2
from paddleocr import PaddleOCR
import configs.config as config
import base
import time
from collections import deque
from functions.ocr_history import find_best_historical_label
#====准备工作-------------------------------------
# 临时展示窗口的属性设置（如需打开可取消注释）
#cv2.namedWindow("window", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("window", 800, 600)
#==导入函数---------------------
Detect=base.Detect
from functions.EdgeDetect import EdgeClassify
from functions.Lines import should_skip_line, extend_line_to_right
#==导入模型和配置---------------------
ocr=config.ocr
colors=config.colors
video_config=config.VIDEO_CONFIG

#导入视频和元数据
cap = cv2.VideoCapture(str(config.VIDEO_PATH))

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

#获取编码器和写出视频流
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(config.OUTPUT_VIDEO), fourcc, fps, (width, height))

# 初始化数据收集列表
data_records = []

# 最近 N 帧的历史识别记录（存放原始识别结果，不包含后续的替换）
RECENT_FRAMES_HISTORY = deque(maxlen=5)

#====正式逻辑-------------------------------------
# 获取视频帧范围配置
start_frame = video_config.get('start_frame')
end_frame = video_config.get('end_frame')

# 如果指定了起始帧，跳过前面的帧
if start_frame is not None and start_frame > 0:
    for _ in range(start_frame):
        cap.read()

index = 0 if start_frame is None else start_frame
frame_count = 0

#==开始读取视频---------------------------
while True:
    ret, img = cap.read()

    if not ret:
        print("视频读取结束")
        break

    # 检查是否超过结束帧
    if end_frame is not None and index > end_frame:
        print(f"已处理到指定的结束帧 {end_frame}")
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = img.copy()
    index+=1
    frame_count += 1
    #获取图的所有文本结果:
    #1.ocr结果
    #2.边缘检测结果
    rst=Detect(img,index)
    # 保存本帧的原始识别结果（在可能的替换之前）到局部变量，以便写入历史记录
    orig_recs_info = [det.as_dict().copy() for det in rst.recs]

    # 对识别结果中为 Unknown / None 的条目，尝试用历史记录中的最近有 OCR 的条目替换
    if getattr(config, 'USE_OCR_HISTORY', False) and RECENT_FRAMES_HISTORY:
        for det in rst.recs:
            lab = det.label
            if lab is None or (isinstance(lab, str) and lab.strip().lower() == 'unknown'):
                candidate = find_best_historical_label(det.as_dict(), list(RECENT_FRAMES_HISTORY))
                if candidate:
                    new_label, new_conf = candidate
                    det.label = new_label
                    # 置信度使用历史记录中的置信度（如果存在）
                    det.confidence = new_conf
    
    # draw horizontal midline in purple to show top/bottom halves
    mid = height / 2.0
    purple = (255, 0, 255)
    cv2.line(output, (0, int(mid)), (width, int(mid)), purple, 2)
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
        elif det.class_id == 1:
            # 类型1：使用紫色绘制识别框，并在框上侧用紫色标注OCR识别结果
            bx1, by1, bx2, by2 = det.to_xyxy()
            cv2.rectangle(output, (int(bx1), int(by1)), (int(bx2), int(by2)), purple, 2)
            label = det.label or ""
            text_x = int(bx1)
            text_y = int(by1) - 5  # 在框上方5个像素
            # 如果文本会超出图片顶端，则改为放在框内上方
            if text_y < 10:
                text_y = int(by1) + 15
            cv2.putText(
                output,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                purple,
                2,
            )
    
    #拿到边缘检测结果
    edge_lines = rst.lines or []
    
    for line in edge_lines:
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
                    # Only consider Type-2 recognition boxes (class_id == 2)
                    # Previously this considered non-zero classes (Type1). Change
                    # to require class_id == 2 per request.
                    if det.class_id != 2:
                        continue
                    bx1, by1, bx2, by2 = det.to_xyxy()
                    if lx_min >= bx1 and ly_min >= by1 and lx_max <= bx2 and ly_max <= by2:
                        color = (0, 255, 0)
                        
                        # 使用函数将线段向右延长到画面右边
                        x1_ext, y1_ext, x2_ext, y2_ext = extend_line_to_right(x1, y1, x2, y2, width)
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
                        
                        # 基于识别框的相对位置来绘制文字
                        # 获取识别框的坐标
                        det_x1, det_y1, det_x2, det_y2 = bx1, by1, bx2, by2
                        det_width = det_x2 - det_x1
                        det_height = det_y2 - det_y1
                        
                        green = (0, 255, 0)
                        # 使用统一绘制函数绘制绿色(类型2)文字和蓝色(类型0)文字
                        from functions.labels.draw_labels import draw_type_labels

                        line_label = det.label or ""
                        type0_label = best_type0['label'] if best_type0 else None
                        draw_type_labels(
                            output,
                            green_label=line_label,
                            blue_label=type0_label,
                            type2_box=(det_x1, det_y1, det_x2, det_y2),
                            type0_box=(best_type0['x'] if best_type0 else None,
                                       best_type0['y'] if best_type0 else None,
                                       best_type0['x'] if best_type0 else None,
                                       best_type0['y'] if best_type0 else None),
                        )
                        #==开始准备表格数据---------------------
                        # 计算百分比制坐标
                        x1_pct = (x1 / width) * 100
                        y1_pct = (y1 / height) * 100
                        x2_pct = (x2 / width) * 100
                        y2_pct = (y2 / height) * 100
                        
                        det_cx_pct = (det.x_center / width) * 100
                        det_cy_pct = (det.y_center / height) * 100
                        
                        type0_cx_pct = (best_type0['x'] / width) * 100 if best_type0 else None
                        type0_cy_pct = (best_type0['y'] / height) * 100 if best_type0 else None
                        
                        record = {
                            'Frame': index,
                            'Line_X1_Pct': round(x1_pct, 2),
                            'Line_Y1_Pct': round(y1_pct, 2),
                            'Line_X2_Pct': round(x2_pct, 2),
                            'Line_Y2_Pct': round(y2_pct, 2),
                            'Type2_Center_X_Pct': round(det_cx_pct, 2),
                            'Type2_Center_Y_Pct': round(det_cy_pct, 2),
                            'Type2_OCR_Text': line_label,
                            'Type2_Confidence': round(det.confidence, 4) if det.confidence else None,
                            'Type0_Center_X_Pct': round(type0_cx_pct, 2) if type0_cx_pct is not None else None,
                            'Type0_Center_Y_Pct': round(type0_cy_pct, 2) if type0_cy_pct is not None else None,
                            'Type0_OCR_Text': best_type0['label'] if best_type0 else None,
                            'Type0_Confidence': None  # Type0的置信度在识别框中未记录
                        }
                        data_records.append(record)
                        
                        drawn = True
                        break

        # If not drawn by the above rules, draw as red fallback (no OCR label, containment not required)
        if not drawn:
            if should_skip_line(x1, y1, x2, y2, slope_threshold=1):
                continue
            red = (0, 0, 255)
            cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)), red, 2)
    
    
    out_file = config.WIRES_OUTPUTS_DIR / f"{index}.png"
    cv2.imwrite(str(out_file), output)
    writer.write(output)
    # 将本帧的原始识别结果加入历史（用于后续帧的回溯替换）
    if getattr(config, 'USE_OCR_HISTORY', False):
        RECENT_FRAMES_HISTORY.append(orig_recs_info)
    
    
    #cv2.imshow('window',output)
    #cv2.waitKey(1)
    print(f"index-{index} wrote")

    
writer.release()
cap.release()
cv2.destroyAllWindows()

# 导出数据到CSV
if data_records:
    df = pd.DataFrame(data_records)
    csv_filename = config.CSV_PATH
    df.to_csv(str(csv_filename), index=False, encoding='utf-8-sig')
    print(f"\n数据已导出到 {csv_filename}")
    print(f"总共记录了 {len(data_records)} 条线段-识别框对应关系")
else:
    print("没有有效的数据记录")
