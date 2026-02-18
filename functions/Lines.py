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
