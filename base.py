#识别单张图片
import os
import pandas as pd
import cv2
from paddleocr import PaddleOCR
import configs.config as config
import functions.EdgeDetect
from models.YoloRec import DetectionResult, YoloRec, DetectRunResult
#====准备工作-------------------------------------
#==导入函数---------------------
EdgeDetect=functions.EdgeDetect.EdgeDetect
EdgeClassify=functions.EdgeDetect.EdgeClassify
#==导入模型---------------------
ocr=config.ocr
colors=config.colors
edge_detect_config=config.EDGE_DETECT_CONFIG

#====正式逻辑-------------------------------------

def Detect(img,index):
    height, width = img.shape[:2]

    rec = YoloRec()
    run_result = DetectRunResult(recs=rec, lines=[], meta={"index": index})

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = img.copy()
    # 读取标签文件（使用相对路径配置），标签文件名基于配置中的视频名
    video_stem = config.VIDEO_PATH.stem
    label_path = config.WIRES_LABELS_DIR / f"{video_stem}_{index}.txt"

    if not label_path.exists():
        out_path = config.WIRES_OUTPUTS_DIR / f"{index}.png"
        cv2.imwrite(str(out_path), output)
        run_result.meta["output_path"] = str(out_path)
        return run_result

    df = pd.read_csv(str(label_path), sep=" ", header=None)
    
    # 第一步：收集所有需要识别的区域及其信息
    crops_data = []
    for x in range(0, df.shape[0]):
        yolo_type = int(df.loc[x, 0])
        x_center = int(width * df.loc[x, 1])
        y_center = int(height * df.loc[x, 2])
        x_width = int(width * df.loc[x, 3])
        y_width = int(height * df.loc[x, 4])
        
        x1 = max(0, x_center - x_width // 2)
        y1 = max(0, y_center - y_width // 2)
        x2 = min(width, x_center + x_width // 2)
        y2 = min(height, y_center + y_width // 2)
        
        crop = img[y1:y2, x1:x2]
        crops_data.append({
            'crop': crop,
            'yolo_type': yolo_type,
            'x_center': x_center,
            'y_center': y_center,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        })
    
    # 第二步：一次性进行批量OCR识别
    if crops_data:
        # 将所有裁剪的图像提供给OCR进行批量识别
        crops_list = [data['crop'] for data in crops_data]
        results = ocr.predict(input=crops_list)
        
        # 第三步：处理识别结果
        for idx, data in enumerate(crops_data):
            result = results[idx] if idx < len(results) else None
            
            # 没有识别到就跳过
            #if not result or len(result.get('rec_texts', [])) == 0:
            #    continue
            
            # 保存检测结果到 YoloRec
            # 从 OCR 结果中选取置信度最高的识别文本及其置信度（若有）
            best_label = None
            best_confidence = None
            if result:
                rec_scores = result.get('rec_scores')
                rec_texts = result.get('rec_texts', [])
                if rec_scores:
                    try:
                        # 选择置信度最高的索引（支持列表或可转为 float 的元素）
                        max_i = max(range(len(rec_scores)), key=lambda i: float(rec_scores[i]))
                        best_confidence = float(rec_scores[max_i])
                        best_label = rec_texts[max_i] if max_i < len(rec_texts) else None
                    except Exception:
                        # 退化到第一个识别结果
                        best_label = rec_texts[0] if rec_texts else None
                        try:
                            best_confidence = float(rec_scores[0]) if rec_scores else None
                        except Exception:
                            best_confidence = None
                elif rec_texts:
                    best_label = rec_texts[0]
                    best_confidence = None

            det = DetectionResult(
                x_center=data['x_center'],
                y_center=data['y_center'],
                x1=data['x1'],
                y1=data['y1'],
                x2=data['x2'],
                y2=data['y2'],
                confidence=best_confidence,
                class_id=data['yolo_type'],
                label=best_label,
            )
            rec.add(det)
            run_result.add_detection(det)
    
    lines = EdgeDetect(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            run_result.add_line((x1, y1, x2, y2))
    # 保存 output image path in meta (optional)
    run_result.meta.setdefault("output_saved", False)
    return run_result

