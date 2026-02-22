"""
线段处理函数模块
"""


def calculate_slope(x1, y1, x2, y2):
    """
    计算两点之间的斜率的绝对值。
    
    参数:
        x1, y1: 第一个点的坐标
        x2, y2: 第二个点的坐标
    
    返回:
        float: 斜率的绝对值。如果是垂直线，返回float('inf')
    """
    if abs(x2 - x1) < 1e-6:
        # 垂直线
        return float('inf')
    else:
        slope = abs((y2 - y1) / (x2 - x1))
        return slope


def should_skip_line(x1, y1, x2, y2, slope_threshold=1):
    """
    判断线段是否应该被跳过（基于斜率阈值）。
    
    参数:
        x1, y1: 第一个点的坐标
        x2, y2: 第二个点的坐标
        slope_threshold: 斜率阈值，默认为1
    
    返回:
        bool: 如果斜率大于阈值则返回True（应该跳过），否则返回False
    """
    slope = calculate_slope(x1, y1, x2, y2)
    return slope > slope_threshold


def extend_line_to_right(x1, y1, x2, y2, frame_width):
    """
    将线段向右延长到画面右边界。
    
    参数:
        x1, y1: 线段起点坐标
        x2, y2: 线段终点坐标
        frame_width: 画面宽度（右边界）
    
    返回:
        tuple: (x1_ext, y1_ext, x2_ext, y2_ext) 延长后的线段端点坐标，均为整数
    """
    # 如果线段是垂直线
    if abs(x2 - x1) < 1e-6:
        # 垂直线，保持 x 坐标，y 坐标不变
        x1_ext, y1_ext = x1, y1
        x2_ext, y2_ext = x2, y2
    else:
        # 计算直线斜率
        slope = (y2 - y1) / (x2 - x1)
        # 向右延长到 x = frame_width
        x_right = frame_width
        if x1 <= x_right:
            y_right = y1 + slope * (x_right - x1)
            x1_ext, y1_ext = int(x1), int(y1)
            x2_ext, y2_ext = int(x_right), int(y_right)
        else:
            x1_ext, y1_ext = int(x1), int(y1)
            x2_ext, y2_ext = int(x2), int(y2)
    
    return x1_ext, y1_ext, x2_ext, y2_ext


def extend_line_to_left(x1, y1, x2, y2, frame_width):
    """
    将线段向左延长到画面左边界 (x=0)。

    参数:
        x1, y1: 线段起点坐标
        x2, y2: 线段终点坐标
        frame_width: 画面宽度（仅用于签名一致性）

    返回:
        tuple: (x1_ext, y1_ext, x2_ext, y2_ext) 延长后的线段端点坐标，均为整数
    """
    # 如果线段是垂直线
    if abs(x2 - x1) < 1e-6:
        x1_ext, y1_ext = x1, y1
        x2_ext, y2_ext = x2, y2
    else:
        slope = (y2 - y1) / (x2 - x1)
        # 向左延长到 x = 0
        x_left = 0
        if x2 >= x_left:
            # 以 x2 为基准向左延伸到 x=0，计算对应的 y
            y_left = y2 + slope * (x_left - x2)
            x1_ext, y1_ext = int(x1), int(y1)
            x2_ext, y2_ext = int(x_left), int(y_left)
        else:
            x1_ext, y1_ext = int(x1), int(y1)
            x2_ext, y2_ext = int(x2), int(y2)

    return x1_ext, y1_ext, x2_ext, y2_ext
