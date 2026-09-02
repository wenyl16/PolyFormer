import os
import pickle

from Simulator import PROJECT_ROOT

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib.pyplot as plt
casename = 'simple_case'
# casename = 'mg_case'
data_path = f'{PROJECT_ROOT}\\results\\safe_region\\{casename}\\training_history.pkl'
with open(data_path, "rb") as f:
    training_history = pickle.load(f)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# IEEE风格配置
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 8,
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})


def sliding_average(data, window_size=5):
    """滑动平均平滑处理（窗口大小可调整）"""
    if window_size < 2:
        return data
    # 确保窗口不超过数据长度
    window = min(window_size, len(data) // 5)  # 窗口最大为数据长度的1/5
    return np.convolve(data, np.ones(window) / window, mode='same')


def plot_smoothed_curve(training_history, window_size=5, break_points = [1100,1300]):
    # 检查数据
    if len(training_history['feas']) != len(training_history['opt']):
        raise ValueError("feas和opt长度必须一致")

    feas = training_history['feas']
    opt = training_history['opt']
    epochs = np.arange(1, len(feas) + 1)

    # 平滑处理
    feas_smoothed = sliding_average(feas, window_size)
    opt_smoothed = sliding_average(opt, window_size)

    # 创建画布
    fig, ax = plt.subplots(figsize=(3.346, 2.51))  # IEEE单栏尺寸
    # 第一段：x从0到1100（浅蓝透明）
    ax.axvspan(0, break_points[0], color='#F0F0F0', alpha=0.7, zorder=0)  # 浅灰
    # 第二段（1100-1300）：浅绿色（与曲线颜色对比明显）
    ax.axvspan(break_points[0], break_points[1], color='#E6F4EA', alpha=0.7, zorder=0)  # 浅绿
    ax.axvline(x=break_points[0], color='gray', linestyle='--', linewidth=0.8,
               zorder=1.5)  # 线宽较细，位于曲线下方、网格上方
    # 绘制原始平滑线（浅色）+ 平滑线（深色）
    # ax.plot(epochs, feas, color='#0073CF', alpha=0.3, linewidth=0.8)  # 原始波动线
    ax.plot(epochs, feas_smoothed,
            color='#0073CF')

    # ax.plot(epochs, opt, color='#D95319', alpha=0.3, linewidth=0.8)  # 原始波动线
    ax.plot(epochs, opt_smoothed,
            color='#D95319')

    # 对数坐标（适应大波动）
    ax.set_yscale('log')
    ax.set_xlim([0,break_points[1]])
    # 标签与网格
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Error')
    # ax.grid(True, linestyle='--', linewidth=0.3, which='both')
    # ax.legend()
    plt.tight_layout(pad=0.5)
    plt.show()
    # 保存：fig.savefig('smoothed_curve.pdf', dpi=600, bbox_inches='tight')

print(len(training_history['feas']))
# 调用（window_size可根据波动程度调整，越大越平滑）
plot_smoothed_curve(training_history, window_size=10)

