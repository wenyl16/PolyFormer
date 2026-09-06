import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Simulator import PROJECT_ROOT
import seaborn as sns
from matplotlib.colors import LogNorm

model_type = 'fullnet'
file_path = f'{PROJECT_ROOT}\\results\\ds_proj\\td_results\\td_experiment_results.xlsx'
df = pd.read_excel(file_path, sheet_name=model_type)


# Data from the provided table

tscases = [
           'case4gs_ts',
           'case118_ts',
           'case300_ts'
           ]

dscases = [
           'case10ba_ds',
           'case17me_ds',
           'case33bw_ds',
           'case51ga_ds',
           'case74_ds',
           'case118zh_ds',
           'case136ma_ds',
           'case533mt_hi_ds',
           'case36real_3phase_ds',
           ]
def draw_feas_error(df):
    # Prepare data for feas_error
    df['dscasename'] = df['dscasename'].str.replace('_ds', '')
    df['tscasename'] = df['tscasename'].str.replace('_ts', '')
    pivot_table = df.pivot(index='tscasename', columns='dscasename', values='max_error')
    pivot_table = pivot_table.reindex(index=[t.replace('_ts', '') for t in tscases],
                                     columns=[d.replace('_ds', '') for d in dscases])

    # Create figure and axis
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')  # Changed from 'seaborn' to 'ggplot'

    # Create heatmap with logarithmic scale
    sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', norm=plt.cm.colors.LogNorm(vmin=pivot_table.min().min(), vmax=pivot_table.max().max()),
                cbar_kws={'label': 'max feas error'}, linewidths=0.5, linecolor='white')

    # Customize the plot
    plt.title('Heatmap of max_feas_error', fontsize=12, pad=20)
    plt.xlabel('dscase', fontsize=10)
    plt.ylabel('tscase', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)

    # Adjust layout
    plt.tight_layout()

    # Save and show the plot
    plt.savefig('feas_error_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def draw_obj_error(df):
    # Prepare data for obj_error
    df['dscasename'] = df['dscasename'].str.replace('_ds', '')
    df['tscasename'] = df['tscasename'].str.replace('_ts', '')
    df['obj_error'] = (df['apx_obj']-df['ipopt_obj'])/df['ipopt_obj']

    pivot_table = df.pivot(index='tscasename', columns='dscasename', values='obj_error')
    pivot_table = pivot_table.reindex(index=[t.replace('_ts', '') for t in tscases],
                                     columns=[d.replace('_ds', '') for d in dscases])

    abs_pivot_table = pivot_table.abs()

    # Create annotation array with original values formatted as strings
    annot_array = pivot_table.map(lambda x: f'{x:.1e}' if pd.notnull(x) else '')

    # Create figure and axis
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')

    # Create heatmap with logarithmic scale on absolute values, original values in annotations
    sns.heatmap(abs_pivot_table, annot=annot_array, fmt='s', cmap='YlOrRd',
                norm=LogNorm(vmin=abs_pivot_table.min().min(), vmax=abs_pivot_table.max().max()),
                cbar_kws={'label': 'obj error'}, linewidths=0.5, linecolor='white')

    # Customize the plot
    plt.title('Heatmap of obj error', fontsize=12, pad=20)
    plt.xlabel('dscasename', fontsize=10)
    plt.ylabel('tscasename', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)

    # Adjust layout
    plt.tight_layout()

    # Save and show the plot
    plt.savefig('obj_error_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def draw_memory(df):
    # Create a figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=False)
    plt.style.use('ggplot')  # Changed from 'seaborn' to 'ggplot'

    for idx, tscase in enumerate(tscases):
        # Filter data for the current tscasename
        tscase_data = df[df['tscasename'] == tscase]
        dscasenames = tscase_data['dscasename'].str.replace('_ds', '')
        apx_peak_memory_MB = tscase_data['apx_peak_memory_MB']
        ipopt_peak_memory_MB = tscase_data['ipopt_peak_memory_MB']

        # Set up bar positions
        x = np.arange(len(dscasenames))
        width = 0.35  # Width of the bars

        # Plot bars
        ax = axes[idx]
        ax.bar(x - width / 2, apx_peak_memory_MB, width, label='APX', color='#1f77b4', edgecolor='black')
        ax.bar(x + width / 2, ipopt_peak_memory_MB, width, label='Full', color='#ff7f0e', edgecolor='black')

        # Customize the plot
        ax.set_title(f'Memory Comparison for {tscase}', fontsize=10, pad=8)
        ax.set_xlabel('dscasename', fontsize=10)
        ax.set_ylabel('memory(MB)', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(dscasenames, rotation=45, ha='right', fontsize=6)
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.7)
        ax.legend()

        # Adjust y-axis limits to ensure all bars are visible
        max_ncons = max(max(apx_peak_memory_MB), max(ipopt_peak_memory_MB))
        min_ncons = min(min(apx_peak_memory_MB), min(ipopt_peak_memory_MB))
        ax.set_ylim(min_ncons / 2, max_ncons * 2)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save and show the plot
    plt.savefig('memory_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def draw_comp_time(df):
    # Create a figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=False)
    plt.style.use('ggplot')  # Changed from 'seaborn' to 'ggplot'

    for idx, tscase in enumerate(tscases):
        # Filter data for the current tscasename
        tscase_data = df[df['tscasename'] == tscase]
        dscasenames = tscase_data['dscasename'].str.replace('_ds', '')
        apx_time = tscase_data['apx_time']
        ipopt_time = tscase_data['ipopt_time']

        # Set up bar positions
        x = np.arange(len(dscasenames))
        width = 0.35  # Width of the bars

        # Plot bars
        ax = axes[idx]
        ax.bar(x - width / 2, apx_time, width, label='APX', color='#1f77b4', edgecolor='black')
        ax.bar(x + width / 2, ipopt_time, width, label='Full', color='#ff7f0e', edgecolor='black')

        # Customize the plot
        ax.set_title(f'Comp Time Comparison for {tscase}', fontsize=10, pad=8)
        ax.set_xlabel('dscasename', fontsize=10)
        ax.set_ylabel('comp time (s)', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(dscasenames, rotation=45, ha='right', fontsize=6)
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.7)
        ax.legend()

        # Adjust y-axis limits to ensure all bars are visible
        max_ncons = max(max(apx_time), max(ipopt_time))
        min_ncons = min(min(apx_time), min(ipopt_time))
        ax.set_ylim(min_ncons / 2, max_ncons * 2)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save and show the plot
    plt.savefig('comp_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def draw_time_and_memory(df):
    # 设置基础绘图风格
    plt.rcParams['font.family'] = 'Calibri'
    plt.rcParams['font.size'] = 14
    text_font_size = 10
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5

    apx_peak_memory_MB = []
    ipopt_peak_memory_MB = []
    apx_time = []
    ipopt_time = []
    for idx, tscase in enumerate(tscases):
        # Filter data for the current tscasename
        tscase_data = df[df['tscasename'] == tscase]
        apx_peak_memory_MB.append(tscase_data['apx_peak_memory_MB'])
        ipopt_peak_memory_MB.append(tscase_data['ipopt_peak_memory_MB'])
        apx_time.append(tscase_data['apx_time'])
        ipopt_time.append(tscase_data['ipopt_time'])
    trans_list = tscases
    dis_list = dscases
    method1_time = np.vstack(ipopt_time)
    method2_time = np.vstack(apx_time)
    method1_memory = np.vstack(ipopt_peak_memory_MB)
    method2_memory = np.vstack(apx_peak_memory_MB)
    # 创建两个子图：上方为计算时间，下方为内存占用
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.01})

    # 准备数据
    cases = [f'{t}-{d}' for t in trans_list for d in dis_list]
    x_pos = np.arange(len(cases))
    method1_time_flat = method1_time.flatten()
    method2_time_flat = method2_time.flatten()
    method1_memory_flat = method1_memory.flatten()
    method2_memory_flat = method2_memory.flatten()

    # 计算削减率
    time_reduction = ((method1_time_flat - method2_time_flat) / method1_time_flat) * 100
    memory_reduction = ((method1_memory_flat - method2_memory_flat) / method1_memory_flat) * 100

    # ========== 上方子图：计算时间 lollipop 图 ==========
    for i, (m1, m2) in enumerate(zip(method1_time_flat, method2_time_flat)):
        # 连接线
        ax1.plot([i, i], [m1, 0], color='#e74c3c', linewidth=2.5, alpha=0.6, zorder=1)
        ax1.plot([i, i], [m2, 0], color='#3498db', linewidth=2.5, alpha=0.6, zorder=1)

        # 端点圆圈
        ax1.scatter(i, m1, s=180, c='#e74c3c', alpha=0.8, edgecolors='white',
                    linewidth=2, zorder=3, label='Method 1 (Traditional)' if i == 0 else '')
        ax1.scatter(i, m2, s=180, c='#3498db', alpha=0.8, edgecolors='white',
                    linewidth=2, zorder=3, label='Method 2 (Improved)' if i == 0 else '')

        # 时间标签（保持原始标注）
        # ax1.text(i, m1 + 0.3 * m1, f'{m1:.2f}s', ha='center', va='bottom',
        #          fontsize=text_font_size, color='#e74c3c', fontweight='bold')
        # ax1.text(i, m2 + 1.2 * m2, f'{m2:.2f}s', ha='center', va='bottom',
        #          fontsize=text_font_size, color='#3498db', fontweight='bold',
        #          bbox=dict(facecolor='white', edgecolor='none', alpha=0.8)
        #          )
        # ax1.text(i-0.45, m1*1.15, f'{m1:.2f}s', ha='center', va='bottom',
        #          fontsize=text_font_size, color='#e74c3c', fontweight='bold')
        # ax1.text(i-0.45, m2*1.15, f'{m2:.2f}s', ha='center', va='bottom',
        #          fontsize=text_font_size, color='#3498db', fontweight='bold',
        #          # bbox=dict(facecolor='white', edgecolor='none', alpha=0.8)
        #          )
    # 在顶部标注时间削减率（无边框）
    # y_max_time = max(method1_time_flat.max(), method2_time_flat.max())
    # for i, reduction in enumerate(time_reduction):
    #     ax1.text(i, method2_time_flat[i]*1.5, f'(-{reduction:.1f}%)', ha='center', va='bottom',
    #              fontsize=text_font_size - 1, color='#3498db', fontweight='bold',
    #              bbox=dict(facecolor='white', edgecolor='none', alpha=0.8)
    #              )

    # 设置计算时间子图标签
    # ax1.set_ylabel('Computation Time (seconds)', fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.25, axis='y', linewidth=0.5, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(0.8)
    ax1.spines['bottom'].set_linewidth(0.8)
    # ax1.legend(loc='upper left', framealpha=0.9, fontsize=11)
    ax1.tick_params(axis='x', bottom=False)  # 去除上方子图底部刻度线

    # 调整y轴范围
    min_time = min(method1_time_flat.min(), method2_time_flat.min())
    max_time = max(method1_time_flat.max(), method2_time_flat.max())
    ax1.set_ylim(6e-3, max_time * 2)

    # ========== 下方子图：内存占用堆叠柱状图 ==========
    bar_width = 0.4

    # 绘制堆叠柱状图（先画方法2在底部，再画方法1-方法2的差值部分）
    # 蓝色部分（方法2）在底部
    ax2.bar(x_pos, method2_memory_flat, width=bar_width,
            color='#3498db', alpha=0.7, label='Method 2 (Improved)', edgecolor='white', linewidth=1)
    # 红色部分（方法1减去方法2的差值）叠加在上面
    ax2.bar(x_pos, method1_memory_flat - method2_memory_flat, width=bar_width,
            color='#e74c3c', alpha=0.7, label='Method 1 (Traditional)', edgecolor='white', linewidth=1,
            bottom=method2_memory_flat)
    ax2.set_yscale('log')
    # 标注内存数据
    # for i, (m1, m2) in enumerate(zip(method1_memory_flat, method2_memory_flat)):
    #     # 方法1的标注（在柱子顶部）
    #     ax2.text(i, m1*1.1, f'{m1:.1f}Mb', ha='center', va='bottom',
    #              fontsize=text_font_size, color='#e74c3c', fontweight='bold')
    #     # 方法2的标注（在蓝色部分中间或底部）
    #     ax2.text(i, m2*1.3, f'{m2:.1f}Mb', ha='center', va='center',
    #              fontsize=text_font_size, color='#3498db', fontweight='bold',
    #              # bbox=dict(facecolor='white', edgecolor='none', alpha=0.8)
    #              )

    # 在顶部标注内存削减率（无边框）
    y_max_memory = method1_memory_flat.max()
    # for i, reduction in enumerate(memory_reduction):
    #     ax2.text(i, method2_memory_flat[i]*1.08, f'(-{reduction:.1f}%)', ha='center', va='bottom',
    #              fontsize=text_font_size - 1, color='#3498db', fontweight='bold',
    #              bbox=dict(facecolor='white', edgecolor='none', alpha=0.8)
    #              )

    # 设置内存占用子图标签
    # ax2.set_ylabel('Memory Usage (MB)', fontweight='bold')
    # ax2.set_xlabel('Test Cases (Transmission-Distribution)', fontweight='bold')
    ax2.grid(True, alpha=0.25, axis='y', linewidth=0.5, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(0.8)
    ax2.spines['bottom'].set_linewidth(0.8)
    # ax2.legend(loc='upper left', framealpha=0.9, fontsize=11)
    ax2.set_ylim(7e-2, y_max_memory * 3)

    # 设置x轴标签
    ax2.set_xticks(x_pos)
    # ax2.set_xticklabels(cases, rotation=45, ha='right', va='top')
    ax2.set_xticklabels([])

    # 输电系统分组分隔线
    for i in range(1, len(trans_list)):
        ax1.axvline(x=i * len(dis_list) - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.4, zorder = 4)
        ax2.axvline(x=i * len(dis_list) - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.4, zorder = 4)

    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()

def draw_errors(df):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import LogNorm
    plt.rcParams['font.family'] = 'Calibri'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    text_font_size = 13

    feas_error = []
    opt_error = []
    for idx, tscase in enumerate(tscases):
        # Filter data for the current tscasename
        tscase_data = df[df['tscasename'] == tscase]
        feas_error.append(tscase_data['max_error'])
        opt_error.append((tscase_data['apx_obj']-tscase_data['ipopt_obj'])/tscase_data['ipopt_obj'])
    feas_error = np.vstack(feas_error)
    opt_error = np.vstack(opt_error)
    trans_list = tscases
    dis_list = dscases
    # 创建1行2列子图（横向对比两个误差）
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4))  # 适当调整高度以适应纵向标签

    # 数据源（仅保留feas和opt误差）
    data_sets = [
        (feas_error, 'Feasibility Error', 'Blues'),
        (opt_error, 'Optimality Error', 'Reds')
    ]

    # 循环绘制两个气泡图
    for idx, (ax, (data, title, cmap)) in enumerate(zip(axes.flat, data_sets)):
        # 转置逻辑：x轴为配电系统索引（j），y轴为输电系统索引（i）
        for i in range(len(trans_list)):  # 输电系统（纵向）
            for j in range(len(dis_list)):  # 配电系统（横向）
                value = data[i, j]  # 数据索引保持不变（i对应输电，j对应配电）

                # 气泡大小（对数缩放）
                scalar = 2500
                size = scalar * (np.log10(np.abs(value)) - np.log10(np.abs(data).min())) / (
                        np.log10(data.max()) - np.log10(np.abs(data).min()))
                size = max(scalar / 10, min(size, scalar * 2))

                # 气泡颜色
                color_val = (np.log10(np.abs(value)) - np.log10(np.abs(data).min())) / (
                        np.log10(data.max()) - np.log10(np.abs(data).min()))
                color = plt.colormaps[cmap](color_val)

                # 绘制气泡：x=配电索引，y=输电索引（实现转置）
                ax.scatter(j, i, s=size, c=[color], alpha=0.7,
                           edgecolors='white', linewidth=1.5)

                # 添加数值标签
                label = f'{value:.1e}'
                ax.text(j, i, label, ha='center', va='center',
                        # fontsize=text_font_size,
                        # fontweight='bold'
                        )

        # 坐标轴设置（关键转置调整）
        ax.set_xticks(range(len(dis_list)))  # x轴为配电系统
        ax.set_yticks(range(len(trans_list)))  # y轴为输电系统
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_xticklabels(dis_list, rotation=45, ha='right')  # x轴标签：配电系统
        ax.set_yticklabels(trans_list)  # y轴标签：输电系统
        ax.set_xlim(-0.5, len(dis_list) - 0.5)  # x轴范围适应配电系统数量
        ax.set_ylim(-0.5, len(trans_list) - 0.5)  # y轴范围适应输电系统数量
        ax.grid(True, alpha=0.2, linewidth=0.5)
        # ax.set_title(f'{chr(97 + idx)}  {title}', loc='left',
        #              fontweight='bold', fontsize=9)

        # 轴标签互换（x=配电，y=输电）
        # ax.set_xlabel('Distribution System', fontweight='bold')
        # ax.set_ylabel('Transmission System', fontweight='bold')

        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=LogNorm(vmin=np.abs(data).min(), vmax=data.max()))
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=text_font_size)

    # 调整布局并保存
    plt.tight_layout()
    plt.show()

import re


def extract_node_number(casename):
    """从case名称中提取节点数"""
    match = re.search(r'case(\d+)', casename)
    if match:
        return int(match.group(1))
    return 0


# NATURE = {
#     "method1": "#E64B35",  # vermillion
#     "method2": "#3C5488",  # indigo
#     "better":  "#00A087",  # teal
#     "worse":   "#E64B35",  # vermillion (same as method1 is OK)
#     "mix1":    "#4DBBD5",  # cyan
#     "mix2":    "#F39B7F",  # peach
# }
NATURE = {
    # 点：统一用中性深灰（两种方法用形状区分）
    "points": "#2F2F2F",   # near-black, print-friendly

    # 箭头：四象限高区分度（比原来 teal/cyan/peach 更拉开）
    "better": "#009E73",   # 绿：Both Better（左下）
    "worse":  "#D62728",   # 红：Both Worse（右上）  ← 你要的
    "mix1":   "#CC79A7",   # 蓝：Feas↑ Obj↓（右下）
    "mix2":   "#0072B2",   # 紫：Feas↓ Obj↑（左上）
}

def draw_feas_obj_errors(df1, use_log_x=True, use_symlog_y=True,
                           y_linear_threshold=1e-8, save_path=None):
    """
    绘制两种方法的误差对比散点图（Nature风格）
    """
    from matplotlib import rcParams

    # Nature风格参数
    rcParams['font.family'] = 'Arial'
    rcParams['font.size'] = 14
    rcParams['axes.linewidth'] = 0.6
    rcParams['xtick.major.width'] = 0.6
    rcParams['ytick.major.width'] = 0.6
    rcParams['xtick.major.size'] = 3
    rcParams['ytick.major.size'] = 3

    m1_marker = 'o'  # circle

    # 统一散点颜色
    point_color = NATURE["points"]


    # 输电系统和配电系统列表
    tscases = ['case4gs_ts', 'case118_ts', 'case300_ts']
    dscases = ['case10ba_ds', 'case17me_ds', 'case33bw_ds', 'case51ga_ds',
               'case74_ds', 'case118zh_ds', 'case136ma_ds', 'case533mt_hi_ds',
               'case36real_3phase_ds']

    ts_nodes = [extract_node_number(ts) for ts in tscases]
    ds_nodes = [extract_node_number(ds) for ds in dscases]

    fig, ax = plt.subplots(figsize=(8, 6))

    method1_data = {}

    # 收集数据
    marker_size = 150
    for ts, ts_node in zip(tscases, ts_nodes):
        for ds, ds_node in zip(dscases, ds_nodes):
            case_key = (ts, ds)

            case_data1 = df1[(df1['tscasename'] == ts) & (df1['dscasename'] == ds)]
            if not case_data1.empty:
                row = case_data1.iloc[0]
                feas_error = max(row['max_error'],1e-8)
                obj_error = (row['apx_obj'] - row['ipopt_obj']) / row['ipopt_obj']
                complexity = ts_node * ds_node
                size = np.log10(complexity) * marker_size
                method1_data[case_key] = (feas_error, obj_error, size)

    # 绘制散点（同色，不同形状）
    for case_key, (x, y, size) in method1_data.items():
        pc = point_color
        if np.isfinite(x) and np.isfinite(y):
            ax.scatter(
                x, y, s=size,
                c='White', marker=m1_marker,
                alpha=0.70,
                edgecolors=pc, linewidth=1.0,
                zorder=2
            )

    # 坐标轴设置
    if use_log_x:
        ax.set_xscale('log')

    if use_symlog_y:
        ax.set_yscale('symlog', linthresh=y_linear_threshold)
        ax.axhline(y=0, color='#333333', linewidth=0.8, alpha=0.2)

    ax.set_xlabel('Feasibility Error', fontsize=14)
    ax.set_ylabel('Objective Error', fontsize=14)
    ax.set_ylim([-1e-3, 1e-3])
    ax.set_xlim([1e-9, 1e-4])
    # 边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_visible(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg')

    plt.show()

# draw_feas_obj_errors(df, use_log_x=True,  use_symlog_y=True, y_linear_threshold=1e-5)

def get_direction_color(x1, y1, x2, y2):
    """
    dx = feas_error change (x2-x1)
    dy = obj_error change  (y2-y1)

    dx<0 means feas improves (error smaller)
    dy<0 means obj improves  (error smaller)
    """
    dx = x2 - x1
    dy = y2 - y1

    if dx < 0 and dy < 0:      # 左下：两者都更好
        return NATURE["better"], (dx, dy)
    elif dx > 0 and dy > 0:    # 右上：两者都更差
        return NATURE["worse"],  (dx, dy)
    elif dx > 0 and dy < 0:    # 右下：feas变差、obj变好
        return NATURE["mix1"],   (dx, dy)
    else:                      # 左上：feas变好、obj变差
        return NATURE["mix2"],   (dx, dy)


def draw_errors_comparison(df1, df2, method1_name='Method 1', method2_name='Method 2',
                           link_style='arrow', use_log_x=True, use_symlog_y=True,
                           y_linear_threshold=1e-8, save_path=None):
    """
    绘制两种方法的误差对比散点图（Nature风格）
    """
    from matplotlib import rcParams

    # Nature风格参数
    rcParams['font.family'] = 'Arial'
    rcParams['font.size'] = 14
    rcParams['axes.linewidth'] = 0.6
    rcParams['xtick.major.width'] = 1.2
    rcParams['ytick.major.width'] = 1.2
    rcParams['xtick.major.size'] = 3
    rcParams['ytick.major.size'] = 3

    m1_marker = 'o'  # circle
    m2_marker = 'D'  # triangle

    # 统一散点颜色
    point_color = NATURE["points"]


    # 输电系统和配电系统列表
    tscases = ['case4gs_ts', 'case118_ts', 'case300_ts']
    dscases = ['case10ba_ds', 'case17me_ds', 'case33bw_ds', 'case51ga_ds',
               'case74_ds', 'case118zh_ds', 'case136ma_ds', 'case533mt_hi_ds',
               'case36real_3phase_ds']

    ts_nodes = [extract_node_number(ts) for ts in tscases]
    ds_nodes = [extract_node_number(ds) for ds in dscases]

    fig, ax = plt.subplots(figsize=(6, 6))

    method1_data = {}
    method2_data = {}

    # 收集数据
    marker_size = 50
    avg_feas_data1 = 0
    avg_opt_data1 = 0
    avg_feas_data2 = 0
    avg_opt_data2 = 0
    for ts, ts_node in zip(tscases, ts_nodes):
        for ds, ds_node in zip(dscases, ds_nodes):
            case_key = (ts, ds)

            case_data1 = df1[(df1['tscasename'] == ts) & (df1['dscasename'] == ds)]
            if not case_data1.empty:
                row = case_data1.iloc[0]
                feas_error = max(row['max_error'],1e-12)
                avg_feas_data1+=feas_error/(len(tscases)*len(dscases))
                # feas_error = row['max_error']
                obj_error = (row['apx_obj'] - row['ipopt_obj']) / row['ipopt_obj']
                avg_opt_data1 += obj_error / (len(tscases) * len(dscases))
                complexity = ts_node * ds_node
                size = np.log10(complexity) * marker_size
                method1_data[case_key] = (feas_error, obj_error, size)

            case_data2 = df2[(df2['tscasename'] == ts) & (df2['dscasename'] == ds)]
            if not case_data2.empty:
                row = case_data2.iloc[0]
                feas_error = max(row['max_error'], 1e-12)
                avg_feas_data2 += feas_error / (len(tscases) * len(dscases))
                # feas_error = row['max_error']
                obj_error = (row['apx_obj'] - row['ipopt_obj']) / row['ipopt_obj']
                avg_opt_data2 += obj_error / (len(tscases) * len(dscases))

                complexity = ts_node * ds_node
                size = np.log10(complexity) * marker_size
                method2_data[case_key] = (feas_error, obj_error, size)
    print(avg_feas_data1, avg_opt_data1, avg_feas_data2, avg_opt_data2)
    # 突出显示的特殊案例
    special_case_key = ('case4gs_ts', 'case36real_3phase_ds')
    special_case_key = None
    # 绘制连线（根据方向着色）
    if link_style != 'none':
        for case_key in method1_data:
            if case_key in method2_data:
                x1, y1, _ = method1_data[case_key]
                x2, y2, _ = method2_data[case_key]

                # 获取Nature风格的方向颜色
                arrow_color, direction = get_direction_color(x1, y1, x2, y2)
                # print(direction, case_key)

                if all(np.isfinite([x1, y1, x2, y2])):
                    if link_style == 'arrow':
                        lw = 1.5 if not case_key==special_case_key else 3
                        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                    arrowprops=dict(arrowstyle='->',
                                                    color=arrow_color,
                                                    alpha=0.4,
                                                    lw=lw,
                                                    mutation_scale=10,
                                                    shrinkA=3, shrinkB=3))
                    elif link_style == 'line':
                        ax.plot([x1, x2], [y1, y2], color=arrow_color,
                                alpha=0.5, lw=1.5, zorder=1)
                    elif link_style == 'dashed':
                        ax.plot([x1, x2], [y1, y2], color=arrow_color,
                                alpha=0.5, lw=1.5, linestyle='--', zorder=1)


    # 为特殊案例指定标记
    special_case_color = '#FF5733'  # 使用一个醒目的颜色，例如亮橙色
    # 绘制散点（同色，不同形状）
    for case_key, (x, y, size) in method1_data.items():
        pc = point_color if not case_key==special_case_key else special_case_color
        if np.isfinite(x) and np.isfinite(y):
            ax.scatter(
                x, y, s=size,
                c=NATURE['better'], marker=m2_marker,
                alpha=0.70,
                edgecolors='white', linewidth=1.0,
                zorder=2
            )
    for case_key, (x, y, size) in method2_data.items():
        pc = point_color if not case_key==special_case_key else special_case_color

        if np.isfinite(x) and np.isfinite(y):
            ax.scatter(
                x, y, s=size,
                c=NATURE['mix2'], marker=m1_marker,
                alpha=0.70,
                edgecolors='white', linewidth=1.0,
                zorder=2
            )

    # 坐标轴设置
    if use_log_x:
        ax.set_xscale('log')

    if use_symlog_y:
        ax.set_yscale('symlog', linthresh=y_linear_threshold)
        ax.axhline(y=0, color='#333333', linewidth=0.8, alpha=0.2)

    ax.set_xlabel('Feasibility Error', fontsize=14)
    ax.set_ylabel('Objective Error', fontsize=14)
    ax.set_ylim([-1e-3, 1e-2])
    ax.set_xlim([1e-12, 1e-4])
    # ax.set_xticklabels(['0', '1e-11', 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5])
    # Nature风格网格
    # ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.4, zorder=0)
    # ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='minor', length=0)  # 将小刻度长度设为0

    # 边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_visible(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg')

    plt.show()


    # 标题


draw_feas_obj_errors(df,
                      use_log_x=True,         # x轴取绝对值
                      use_symlog_y=True,      # y轴使用对称对数
                      y_linear_threshold=1e-5 # y轴线性区间阈值
                       )
def draw_time_and_memory_2d(df):
    # 设置基础绘图风格
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    text_font_size = 10
    plt.rcParams['axes.linewidth'] = 0.6
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.size'] = 3
    apx_peak_memory_MB = []
    ipopt_peak_memory_MB = []
    apx_time = []
    ipopt_time = []

    for idx, tscase in enumerate(tscases):
        # Filter data for the current tscasename
        tscase_data = df[df['tscasename'] == tscase]
        apx_peak_memory_MB.append(tscase_data['apx_peak_memory_MB'])
        ipopt_peak_memory_MB.append(tscase_data['ipopt_peak_memory_MB'])
        apx_time.append(tscase_data['apx_time'])
        ipopt_time.append(tscase_data['ipopt_time'])

    trans_list = tscases
    dis_list = dscases
    method1_time = np.vstack(ipopt_time).flatten()  # IPOPT
    method2_time = np.vstack(apx_time).flatten()  # APX
    method1_memory = np.vstack(ipopt_peak_memory_MB).flatten()
    method2_memory = np.vstack(apx_peak_memory_MB).flatten()

    # 提取节点数并计算散点大小
    ts_nodes = [extract_node_number(ts) for ts in trans_list]
    ds_nodes = [extract_node_number(ds) for ds in dis_list]

    # 计算每个测试用例的大小（基于复杂度的对数）
    marker_size = 80
    sizes = []
    for ts_node in ts_nodes:
        for ds_node in ds_nodes:
            complexity = ts_node * ds_node
            size = np.log10(complexity) * marker_size
            # size = complexity/100
            sizes.append(size)

    sizes = np.array(sizes)

    # 创建单个图形
    fig, ax = plt.subplots(figsize=(6, 6))

    # 准备测试用例标签
    cases = [f'{t}-{d}' for t in trans_list for d in dis_list]

    # 绘制连接线（同一测试用例的两个方法之间）
    for i in range(len(method1_time)):
        ax.plot([method1_time[i], method2_time[i]],
                [method1_memory[i], method2_memory[i]], linestyle = '-',
                color='gray', linewidth=1.5, alpha=0.4, zorder=1)

    # 绘制IPOPT方法的点（圆形，使用计算的大小）
    scatter1 = ax.scatter(method1_time, method1_memory,
                          s=sizes, c='#e74c3c', alpha=0.8,
                          edgecolors='white', linewidth=1,
                          marker='o', zorder=3,
                          label='IPOPT (Traditional)')

    # 绘制APX方法的点（方形，使用计算的大小）
    scatter2 = ax.scatter(method2_time, method2_memory,
                          s=sizes, c='#3498db', alpha=0.8,
                          edgecolors='white', linewidth=1,
                          marker='o', zorder=3,
                          label='Proposed Method')

    # 添加测试用例标注（可选，避免过于拥挤）
    # for i, case in enumerate(cases):
    #     # 在APX点附近标注
    #     ax.annotate(case, (method2_time[i], method2_memory[i]),
    #                xytext=(5, 5), textcoords='offset points',
    #                fontsize=8, alpha=0.7)

    # 设置对数坐标轴
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 设置坐标轴标签
    ax.set_xlabel('Computation Time (seconds)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Memory Usage (MB)', fontweight='bold', fontsize=14)
    ax.set_ylim([1e-2, 1e4])
    ax.set_xlim([1e-2, 1e4])
    # 添加网格
    ax.grid(False)

    # 添加图例
    # ax.legend(loc='upper left', framealpha=0.9, fontsize=12,
    #           edgecolor='gray', fancybox=True)

    # 添加辅助对角线（可选）- 显示性能相等的参考线
    # 这条线表示如果两种方法性能相等，点应该在的位置
    min_val = min(method1_time.min(), method2_time.min(),
                  method1_memory.min(), method2_memory.min())
    max_val = max(method1_time.max(), method2_time.max(),
                  method1_memory.max(), method2_memory.max())
    # ax.plot([min_val, max_val], [min_val, max_val],
    #        'k--', alpha=0.3, linewidth=1, label='Equal Performance')

    # 计算并显示平均改进
    avg_time_reduction = np.mean((method1_time - method2_time) / method1_time * 100)
    avg_memory_reduction = np.mean((method1_memory - method2_memory) / method1_memory * 100)
    ax.tick_params(axis='both', which='minor', length=0)  # 将小刻度长度设为0
    # 在图上添加文字说明改进情况
    text_str = f'Average Improvements:\nTime: {avg_time_reduction:.1f}%\nMemory: {avg_memory_reduction:.1f}%'
    # ax.text(0.98, 0.02, text_str, transform=ax.transAxes,
    #         fontsize=11, verticalalignment='bottom', horizontalalignment='right',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_visible(True)
    plt.tight_layout()
    plt.show()
draw_time_and_memory_2d(df)
