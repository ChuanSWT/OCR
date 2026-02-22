import os
import pandas as pd
import cv2
import paddle
from paddleocr import PaddleOCR
import os
from pathlib import Path
import pandas as pd
import cv2
from paddleocr import PaddleOCR

# 创建ocr识别模型（在此处初始化一次以便复用）
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

# ====定义常量-------------------------------------
yolo_class = ["slot", "head", "wire"]
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# ==== Path configuration (all paths are relative to project root) ----
# Project root is the parent directory of this configs folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
WIRES_DIR = DATA_DIR / "3class_right"
WIRES_LABELS_DIR = WIRES_DIR / "labels"
WIRES_OUTPUTS_DIR = WIRES_DIR / "outputs"

# Runs directory for numbered output folders
RUNS_DIR = PROJECT_ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Default input video path (relative)
VIDEO_PATH = WIRES_DIR / "video2.mp4"

# Default output files (these will be overridden by the auto-numbered run folder)
OUTPUT_VIDEO = PROJECT_ROOT / "output.mp4"
CSV_PATH = PROJECT_ROOT / "video_ocr_results.csv"

# Ensure output directories exist
WIRES_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ====输出文件夹管理函数-------------------------------------
def get_next_run_number():
    """
    获取下一个运行编号，根据runs文件夹下现有的文件夹情况
    返回格式：run_1, run_2, run_3...
    """
    if not RUNS_DIR.exists():
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 获取所有run_*文件夹
    existing_runs = [d.name for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")]
    
    if not existing_runs:
        return 1
    
    # 提取数字并找到最大值
    try:
        run_numbers = [int(name.split("_")[1]) for name in existing_runs]
        return max(run_numbers) + 1
    except (IndexError, ValueError):
        return 1


def setup_run_directory():
    """
    创建新的运行输出文件夹，并返回该文件夹的路径和相关输出文件路径
    返回字典包含：
      - run_dir: 运行文件夹路径
      - frames_dir: 帧输出文件夹路径
      - output_video: 输出视频路径
      - csv_path: 输出CSV路径
    """
    run_number = get_next_run_number()
    run_dir = RUNS_DIR / f"run_{run_number}"
    frames_dir = run_dir / "frames"
    
    # 创建文件夹结构
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    output_video = run_dir / "output.mp4"
    csv_path = run_dir / "results.csv"
    
    return {
        'run_number': run_number,
        'run_dir': run_dir,
        'frames_dir': frames_dir,
        'output_video': output_video,
        'csv_path': csv_path
    }

# ====视频处理配置-------------------------------------
# 视频帧范围设置
# start_frame: 开始帧号（从0开始计数），设为None表示从第一帧开始
# end_frame: 结束帧号（包括该帧），设为None表示处理到视频结尾
VIDEO_CONFIG = {
    'start_frame': None,  # 设置为None或0表示从开始，或指定具体帧号如100
    'end_frame': None     # 设置为None表示到结尾，或指定具体帧号如500
}

# ====边缘和直线检测配置-------------------------------------
# 是否使用自适应参数进行边缘和直线检测
# True: 根据图像特性自动调整Canny和HoughLinesP参数
# False: 使用固定参数
EDGE_DETECT_CONFIG = {
    'adaptive': False,  # 启用自适应检测
}

# Whether to enable history-based OCR label replacement for unknown results.
# If True, the pipeline will search recent frames for a nearby historical
# OCR result to replace an 'unknown' label (the history still stores the
# original OCR outputs).
USE_OCR_HISTORY = True
