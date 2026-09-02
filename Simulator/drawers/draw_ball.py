import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
from Simulator.cases.basic_cases import case_ball
model_type = 'pretrainnet'
from Simulator import PROJECT_ROOT
import re

dim_list = np.hstack([2,4,6,8,10,15,20,30,40,50,60,70,80,90,100,120,140,160,180,200])

# files = [f'{PROJECT_ROOT}\\results\\ball\\results_dim{dim}.txt' for dim in dim_list]
#
# # 存储结果的列表
# data = []
#
# # 遍历每个文件
# for filename in files:
#     # 从文件名中提取维度数字
#     dim = int(re.search(r'dim(\d+)', filename).group(1))
#
#     # 读取文件内容
#     with open(filename, 'r') as f:
#         content = f.read()
#
#         # 提取各个数据
#         initial_distance = re.search(r'Initial: distance=([\d.e+-]+)', content)
#         final_distance = re.search(r'Final: distance=([\d.e+-]+)', content)
#         ideal_distance = re.search(r'ideal_distance=([\d.e+-]+)', content)
#         approx_rate = re.search(r'approximating_rate=([\d.e+-]+)', content)
#
#         # 将数据添加到列表
#         data.append({
#             'dim': dim,
#             'initial_distance': float(initial_distance.group(1)) if initial_distance else None,
#             'final_distance': float(final_distance.group(1)) if final_distance else None,
#             'ideal_distance': float(ideal_distance.group(1)) if ideal_distance else None,
#             'approximating_rate': float(approx_rate.group(1)) if approx_rate else None
#         })
#
# # 创建DataFrame
# df = pd.DataFrame(data)
#
# # 按dim排序
# df = df.sort_values('dim').reset_index(drop=True)
# csv_save_path = f'{PROJECT_ROOT}\\results\\ball\\res.csv'
# # 保存为CSV（使用pandas的to_csv方法）
# df.to_csv(csv_save_path, index=False)

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
from Simulator import PROJECT_ROOT

# Nature风格参数设置
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 14
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5
rcParams['xtick.major.size'] = 3
rcParams['ytick.major.size'] = 3
rcParams['grid.alpha'] = 0.3
rcParams['grid.linewidth'] = 0.3

# 读取CSV文件
df = pd.read_csv(f'{PROJECT_ROOT}\\results\\ball\\res.csv', sep=r',')

# 提取数据
dim = df['dim']
initial_distance = df['initial_distance']
final_distance = df['final_distance']
ideal_distance = df['ideal_distance']
approximating_rate = df['approximating_rate']

# 创建图形和轴
fig, ax1 = plt.subplots(figsize=(11, 3))

# Nature风格配色方案 - 使用箱线图的配色
color_blue = '#4DBBD5'      # 蓝色（Initial error）
color_green = '#00A087'     # 绿色（Ideal error）
color_red = '#E64B35'       # 红色（Converged error）
color_gray = '#7E6148'      # 灰褐色（柱状图）

markersize = 10
# 绘制三条曲线（主Y轴）
line1 = ax1.plot(dim, initial_distance,
                 color=color_blue,
                 linewidth=1,
                 marker='s',
                 markersize=markersize,
                 markerfacecolor=color_blue,
                 markeredgecolor='black',
                 markeredgewidth=0.5,
                 label='Initial error',
                 alpha=0.8)

line2 = ax1.plot(dim, ideal_distance,
                 color=color_green,
                 linewidth=1,
                 marker='s',
                 markersize=markersize,
                 markerfacecolor=color_green,
                 markeredgecolor='black',
                 markeredgewidth=0.5,
                 label='Ideal error',
                 alpha=0.8)

line3 = ax1.plot(dim, final_distance,
                 color=color_red,
                 linewidth=1,
                 marker='s',
                 markersize=markersize,
                 markerfacecolor=color_red,
                 markeredgecolor='black',
                 markeredgewidth=0.5,
                 label='Converged error',
                 alpha=0.8)

# 设置主Y轴（对数坐标）
ax1.set_xlabel('Dimension', fontweight='normal')
ax1.set_yscale('log')
ax1.set_xscale('log')

# 设置X轴刻度
ax1.set_xticks([2, 20, 200])
ax1.xaxis.set_minor_locator(mticker.NullLocator())


# 设置主Y轴刻度
left_ticks = np.logspace(-2, 4, 4)
ax1.set_yticks(left_ticks)
ax1.tick_params(axis='y', labelcolor='black', width=0.5, length=3)
ax1.minorticks_off()

# 网格设置 - Nature风格
ax1.yaxis.grid(True, linestyle='--', linewidth=0.3, alpha=0.5, zorder=0)
ax1.grid(False, axis='x')
ax1.set_axisbelow(True)

# 创建右侧纵轴（副Y轴）用于柱状图
ax2 = ax1.twinx()

# 绘制柱状图（Approximating Rate）
ax2.bar(dim, approximating_rate,
        width=dim/12,
        alpha=0.3,
        color=color_gray,
        edgecolor='black',
        linewidth=0.5,
        label='Approximating Rate')

# 设置副Y轴
ax2.tick_params(axis='y', labelcolor=color_gray, width=0.5, length=3)
ax2.set_ylim(0.991, 1.0)
ax2.set_yticks([0.991, 0.994, 0.997, 1.0])

# 副Y轴不显示网格（避免与主Y轴网格重叠）
ax2.grid(False)

# 移除顶部边框（Nature风格）
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# 调整所有边框线宽
for spine in ax1.spines.values():
    spine.set_linewidth(0.5)
for spine in ax2.spines.values():
    spine.set_linewidth(0.5)

# 添加图例（可选）
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2,
#            loc='upper center', frameon=False, fontsize=12, ncol=2)

plt.tight_layout()

# 保存图片
# plt.savefig(f'{PROJECT_ROOT}\\results\\ball\\res_fig.svg',
#             dpi=300, bbox_inches='tight', format='svg')

plt.show()