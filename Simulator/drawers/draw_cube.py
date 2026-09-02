import os

import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet, BiasNet, FullNet
import numpy as np
from Simulator.Plotter import ShapeDrawer_2D
from Simulator.Counter import PolyBallHausdorffCalculator
import os
import re

import torch
from Simulator.cases.basic_cases import case_cube
model_type = 'pretrainnet'
from Simulator import PROJECT_ROOT
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

results = []


def extract_data_from_txt(file_path):
    """
    从格式固定的txt结果文件中提取Initial FeasErr和Converged Epoch
    已知文件格式：
    - 第1行：Initial: FeasErr=xxx, OptErr=xxx （含FeasErr和OptErr，逗号分隔）
    - 第2行：Converged at epoch: xxx （含收敛迭代次数）
    :param file_path: txt文件路径
    :return: feas_err (float): 初始可行性误差, epoch (int): 收敛迭代次数
    """
    # 读取文件所有行，过滤空行并去除每行首尾空白（避免换行符/空格干扰）
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # 第一步：校验文件格式是否符合预期（必须是2行有效内容）
    if len(lines) != 2:
        raise ValueError(
            f"文件 {file_path} 格式异常！\n"
            f"预期2行有效内容（第1行Initial信息，第2行Converged信息），\n"
            f"实际检测到{len(lines)}行有效内容，请检查文件完整性。"
        )

    # ---------------------- 提取Initial FeasErr（第1行数据） ----------------------
    # 第1行结构："Initial: FeasErr=数值1, OptErr=数值2"
    # 先按"Initial: "拆分，去掉前缀；再按", "拆分，分离FeasErr和OptErr字段
    initial_part = lines[0].split("Initial: ")[-1]  # 得到 "FeasErr=xxx, OptErr=xxx"
    feaserr_field = initial_part.split(", ")[0]  # 得到 "FeasErr=xxx"（取第一个字段）

    # 提取FeasErr的数值部分并转浮点数（支持科学计数法）
    try:
        feas_err = float(feaserr_field.split("FeasErr=")[-1])
    except ValueError:
        raise ValueError(
            f"文件 {file_path} 第1行FeasErr格式错误！\n"
            f"当前FeasErr字段内容：{feaserr_field}\n"
            f"预期格式为'FeasErr=数字'（如FeasErr=3.9883e-01），请检查数值格式。"
        )

    # ---------------------- 提取Converged Epoch（第2行数据） ----------------------
    # 第2行结构："Converged at epoch: 数值"
    # 按": "拆分，取后半部分即为迭代次数
    try:
        epoch = int(lines[1].split(": ")[-1])
    except ValueError:
        raise ValueError(
            f"文件 {file_path} 第2行Epoch格式错误！\n"
            f"当前第2行内容：{lines[1]}\n"
            f"预期格式为'Converged at epoch: 整数'（如Converged at epoch: 40），请检查数值格式。"
        )

    return feas_err, epoch

# dim_list = np.hstack([2,4,6,8,10,15,20,30,40,50,60,70,80,90,100,120,140,160,180,200])
# # dim_list = np.hstack([200])
#
# # 存储提取的数据
# initial_feaserr_list = []
# converged_epoch_list = []
# for dim in dim_list:
#     print(dim)
#     file_path = f'{PROJECT_ROOT}\\results\\cube\\results_dim{dim}.txt'
#     feas_err, epoch = extract_data_from_txt(file_path)
#     initial_feaserr_list.append(feas_err)
#     converged_epoch_list.append(epoch+int(50*np.sqrt(dim/2)))
#
# data_df = pd.DataFrame({
#     "Dimension (dim)": dim_list,  # 维度列（列名标注含义，便于后续分析）
#     "Initial_FeasErr": initial_feaserr_list,  # 初始可行性误差列
#     "Converged_Epoch": converged_epoch_list  # 收敛迭代次数列
# })
#
# csv_save_path = f'{PROJECT_ROOT}\\results\\cube\\res.csv'
#
# # 保存CSV：index=False 去除默认的行索引（避免冗余），encoding='utf-8' 确保中文列名正常显示
# data_df.to_csv(
#     path_or_buf=csv_save_path,
#     index=False,  # 关键参数：不保存行号，使CSV数据更简洁
#     encoding='utf-8',
#     float_format='%.4e'  # 可选：将浮点数（FeasErr）按科学计数法保留4位有效数字（与文档格式一致）
# )
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
from matplotlib import rcParams

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
df = pd.read_csv(f'{PROJECT_ROOT}\\results\\cube\\res.csv', sep=r',')

# 提取数据
dimensions = df['Dimension']
initial_feaserr = df['Initial_FeasErr']
converged_epoch = df['Converged_Epoch']

# 创建图形和轴
fig, ax1 = plt.subplots(figsize=(10, 3))

# Nature风格配色方案 - 使用类似的颜色
color_blue = '#5589bd'  # 蓝色系
color_orange = '#d55c60'  # 橙色系

# 绘制曲线图 (Initial_FeasErr)
line1 = ax1.plot(dimensions, initial_feaserr,
                 color=color_blue,
                 linewidth=1,
                 marker='s',
                 markersize=10,
                 markerfacecolor=color_blue,
                 markeredgecolor='black',
                 markeredgewidth=0.5,
                 label='Initial feasibility error',
                 alpha=0.8)

# 设置横坐标为对数坐标
ax1.set_xscale('log')

# 设置横坐标刻度
ax1.set_xticks([2, 20, 200])
ax1.xaxis.set_minor_locator(mticker.NullLocator())

# 设置左侧纵轴
ax1.set_xlabel('Dimension', fontweight='normal')
ax1.tick_params(axis='y', labelcolor=color_blue, width=0.5, length=3)
left_ticks = np.logspace(-1, 5, 4)
ax1.set_yscale('log')
ax1.set_yticks(left_ticks)

# 网格设置 - Nature风格
ax1.yaxis.grid(True, linestyle='--', linewidth=0.3, alpha=0.5, zorder=0)
ax1.grid(False, axis='x')
ax1.set_axisbelow(True)

# 创建右侧纵轴
ax2 = ax1.twinx()

# 绘制曲线图 (Converged_Epoch)
line2 = ax2.plot(dimensions, converged_epoch,
                 color=color_orange,
                 linewidth=1,
                 marker='s',
                 markersize=10,
                 markerfacecolor=color_orange,
                 markeredgecolor='black',
                 markeredgewidth=0.5,
                 label='Steps to convergence',
                 alpha=0.8)

# 设置右侧纵轴
ax2.tick_params(axis='y', labelcolor=color_orange, width=0.5, length=3)
right_ticks = np.logspace(1, 4, 4)
ax2.set_yscale('log')
ax2.set_yticks(right_ticks)
ax2.set_ylim(ax2.get_ylim())
ax2.minorticks_off()

# 网格设置
ax2.yaxis.grid(True, linestyle='--', linewidth=0.3, alpha=0.5, zorder=0)
ax2.grid(False, axis='x')
ax2.set_axisbelow(True)

# 移除顶部和右侧边框（Nature风格）
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# 调整边框线宽
for spine in ax1.spines.values():
    spine.set_linewidth(0.5)
for spine in ax2.spines.values():
    spine.set_linewidth(0.5)

# 添加图例（可选）
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2,
#            loc='upper left', frameon=False, fontsize=12)

# 调整布局
plt.tight_layout()

# 保存图片
# plt.savefig(f'{PROJECT_ROOT}\\results\\cube\\res_fig.svg',
#             dpi=300, bbox_inches='tight', format='svg')

plt.show()


