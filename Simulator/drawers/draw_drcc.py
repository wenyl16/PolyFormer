import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
from matplotlib import cm, rcParams
from Simulator import PROJECT_ROOT



def _pareto_front_maxX_minY(X: np.ndarray, Y: np.ndarray):
    """
    右下角偏好：X 越大越好，Y 越小越好。
    返回帕累托前沿点（按 X 从小到大排序）。
    判定：点 i 被支配 当且仅当 存在点 j 使得 Xj>=Xi 且 Yj<=Yi 且至少一项严格。
    """
    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()

    # 按 X 降序排序；同 X 情况下 Y 升序更利于筛选
    idx = np.lexsort((Y, -X))
    Xs, Ys = X[idx], Y[idx]

    front_x = []
    front_y = []
    best_y = np.inf
    for x, y in zip(Xs, Ys):
        # 在 X 从大到小扫描时，只要 y 比当前最小 y 还小，就一定是不被支配点
        if y < best_y:
            front_x.append(x)
            front_y.append(y)
            best_y = y

    # 为了画线好看：按 X 从小到大输出
    front_x = np.array(front_x[::-1])
    front_y = np.array(front_y[::-1])
    return front_x, front_y


import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

def plot_kde_scatter_with_pareto(
    res_path: str,
    label_org: str = 'Classical linearization of DRCC',
    label_apx: str = 'Proposed method',
    color_org: str = '#ec7063',
    color_apa: str = '#5dade2',
    pareto_org: str = '#b03a2e',
    pareto_apa: str = '#1f4e79',
    pareto_alpha: float = 0.75,
    figsize=(8, 5),
    scalar: float = 0.2,
    major_contour: int = 5,
    mini_contour: int = 50,
    show_pareto: bool = True,

    # ===== 新增：坐标范围/刻度自定义 =====
    xlim: tuple | None = None,          # (xmin, xmax)
    ylim: tuple | None = None,          # (ymin, ymax)
    xticks: list | np.ndarray | None = None,
    yticks: list | np.ndarray | None = None,
    xtick_step: float | None = None,    # 与 xlim 搭配：按步长生成刻度
    ytick_step: float | None = None,    # 与 ylim 搭配：按步长生成刻度
    x_nbins: int = 4,                   # 默认 MaxNLocator 的 nbins
    y_nbins: int = 4,
    save_path: str | None = None,   # 例如 "figures/kde.svg"
):
    # =========================
    # Nature 风格参数设置
    # =========================
    rcParams['font.family'] = 'Arial'
    rcParams['font.size'] = 14
    rcParams['axes.linewidth'] = 0.5
    rcParams['xtick.major.width'] = 0.5
    rcParams['ytick.major.width'] = 0.5
    rcParams['xtick.major.size'] = 3
    rcParams['ytick.major.size'] = 3
    rcParams['axes.unicode_minus'] = False

    # =========================
    # 读取数据
    # =========================
    with open(res_path, 'rb') as f:
        res_list = pickle.load(f)

    obj_org_list = sum([item['obj_org'] for item in res_list], [])
    obj_apx_list = sum([item['obj_apx'] for item in res_list], [])
    vv_org_list  = sum([item['vv_org']  for item in res_list], [])
    vv_apx_list  = sum([item['vv_apx']  for item in res_list], [])

    X1 = np.array(obj_org_list, dtype=float)
    Y1 = np.array(vv_org_list,  dtype=float)
    X2 = np.array(obj_apx_list, dtype=float)
    Y2 = np.array(vv_apx_list,  dtype=float)

    # =========================
    # 打印两个方法的均值（不画图上）
    # =========================
    x1_mean, y1_mean = float(np.mean(X1)), float(np.mean(Y1))
    x2_mean, y2_mean = float(np.mean(X2)), float(np.mean(Y2))
    print(f'[{label_org}] mean X={x1_mean:.6f}, mean Y={y1_mean:.6f}')
    print(f'[{label_apx}] mean X={x2_mean:.6f}, mean Y={y2_mean:.6f}')

    # =========================
    # 创建图形
    # =========================
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # KDE
    kde1 = gaussian_kde(np.vstack([X1, Y1]))
    kde2 = gaussian_kde(np.vstack([X2, Y2]))

    X_all = np.hstack([X1, X2])
    Y_all = np.hstack([Y1, Y2])

    # ===== 范围：如果用户传了 xlim/ylim 就用用户的，否则用自动范围 =====
    if xlim is None:
        x_min = np.min(X_all) - scalar * (np.max(X_all) - np.min(X_all))
        x_max = np.max(X_all) + scalar * (np.max(X_all) - np.min(X_all))
    else:
        x_min, x_max = float(xlim[0]), float(xlim[1])

    if ylim is None:
        y_min = np.min(Y_all) - scalar * (np.max(Y_all) - np.min(Y_all))
        y_max = np.max(Y_all) + scalar * (np.max(Y_all) - np.min(Y_all))
    else:
        y_min, y_max = float(ylim[0]), float(ylim[1])

    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    Z1 = kde1(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    Z2 = kde2(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    vmin1, vmax1 = np.min(Z1), np.max(Z1)
    vmin2, vmax2 = np.min(Z2), np.max(Z2)

    levels1_fill = np.linspace(vmin1, vmax1, mini_contour)
    levels2_fill = np.linspace(vmin2, vmax2, mini_contour)

    ax.contourf(xx, yy, Z1,
                levels=levels1_fill[int(mini_contour/major_contour):],
                cmap=cm.Reds, alpha=0.35)
    ax.contourf(xx, yy, Z2,
                levels=levels2_fill[int(mini_contour/major_contour):],
                cmap=cm.Blues, alpha=0.35)

    levels1 = np.linspace(Z1.min(), Z1.max(), major_contour)
    levels2 = np.linspace(Z2.min(), Z2.max(), major_contour)
    ax.contour(xx, yy, Z1, levels=levels1[1:], colors=color_org, linewidths=0.7, alpha=0.8)
    ax.contour(xx, yy, Z2, levels=levels2[1:], colors=color_apa, linewidths=0.7, alpha=0.8)

    # 散点
    ax.scatter(X1, Y1, color=color_org, s=8, label=label_org,
               alpha=0.6, edgecolors='black', linewidth=0)
    ax.scatter(X2, Y2, color=color_apa, s=8, label=label_apx,
               alpha=0.6, edgecolors='black', linewidth=0)

    # =========================
    # 帕累托前沿（右下角偏好）
    # =========================
    if show_pareto:
        fx1, fy1 = _pareto_front_maxX_minY(X1, Y1)
        fx2, fy2 = _pareto_front_maxX_minY(X2, Y2)

        ax.plot(fx1, fy1,
                color=pareto_org, linewidth=2, alpha=pareto_alpha,
                linestyle='-', label=f'{label_org} Pareto')
        ax.plot(fx2, fy2,
                color=pareto_apa, linewidth=2, alpha=pareto_alpha,
                linestyle='-', label=f'{label_apx} Pareto')

        ax.scatter(fx1, fy1, s=15, color=pareto_org, alpha=pareto_alpha,
                   edgecolors='white', linewidth=0.6, zorder=5)
        ax.scatter(fx2, fy2, s=15, color=pareto_apa, alpha=pareto_alpha,
                   edgecolors='white', linewidth=0.6, zorder=5)

    # 坐标轴
    ax.set_xlabel('Objective', fontweight='normal')
    ax.set_ylabel('Violation', fontweight='normal')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # ===== 刻度：优先级 xticks/yticks > step > MaxNLocator =====
    if xticks is not None:
        ax.set_xticks(list(xticks))
    elif xtick_step is not None:
        ax.set_xticks(np.arange(x_min, x_max + 0.5 * xtick_step, xtick_step))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=x_nbins))

    if yticks is not None:
        ax.set_yticks(list(yticks))
    elif ytick_step is not None:
        ax.set_yticks(np.arange(y_min, y_max + 0.5 * ytick_step, ytick_step))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_nbins))

    for spine in ax.spines.values():
        spine.set_linewidth(1)

    ax.set_facecolor('white')
    ax.grid(False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.show()

    return (X1, Y1, X2, Y2)





def plot_case_counts_bar(
    case_name: str,
    methods: list,
    cons_counts: dict,
    vars_counts: dict,
    colors: dict = None,
    figsize=(5, 2.5),
    y_log=True,
    y_lim=(10, 1e6),
    save_path: str | None = None,  # 例如 "figures/kde.svg"

):
    """
    一次只画一个 case：不同方法的 Constraints & Variables 数量对比柱状图。

    参数
    - case_name: 例如 'Case 1'
    - methods: 方法名列表，顺序决定绘图顺序，如 ['APA', 'Original']
    - cons_counts: dict(method -> value) 该 case 的约束数量
    - vars_counts: dict(method -> value) 该 case 的变量数量
    - colors: dict(method -> hex color) 颜色映射；不传则给默认 Nature-ish 配色
    """
    # ===== Nature 风格参数 =====
    rcParams['font.family'] = 'Arial'
    rcParams['font.size'] = 10
    rcParams['axes.linewidth'] = 0.6
    rcParams['xtick.major.width'] = 0.6
    rcParams['ytick.major.width'] = 0.6
    rcParams['xtick.major.size'] = 3
    rcParams['ytick.major.size'] = 3

    # 默认配色（和谐、低饱和，接近你之前的）
    if colors is None:
        default_palette = ['#5dade1', '#ed8075', '#58d68d', '#af7ac5', '#f5b041', '#48c9b0']
        colors = {m: default_palette[i % len(default_palette)] for i, m in enumerate(methods)}

    # ===== 准备数据 =====
    categories = ['Constraints', 'Variables']
    x = np.arange(len(categories))  # [0, 1]
    n_methods = len(methods)
    group_width = 0.78
    bar_w = group_width / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=figsize)

    for i, m in enumerate(methods):
        vals = [cons_counts[m], vars_counts[m]]
        # 让多方法在每个类别下并排
        offset = (i - (n_methods - 1) / 2.0) * bar_w
        ax.bar(
            x + offset, vals, width=bar_w * 0.95,
            label=m,
            color=colors[m], alpha=0.75,
            edgecolor='None', linewidth=0.6,
            zorder=3
        )

    # ===== 坐标轴 & 样式 =====
    ax.set_xticks(x)
    ax.set_xticklabels([f'{case_name}\nConstraints', f'{case_name}\nVariables'])

    ax.set_ylabel('Count', fontsize=11)

    if y_log:
        ax.set_yscale('log')
    if y_lim is not None:
        ax.set_ylim(*y_lim)

    # 去除上右边框（Nature 常见）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 柔和网格线
    ax.yaxis.grid(False)
    # ax.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks([])
    ax.tick_params(axis='y', which='minor', length=0)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.show()

    return fig, ax
N_var_list = [50,120,150,300,400]
N_levels_list = [2, 2, 3, 5,8]
N_samples_list = [150, 240, 300, 900,1280]

case_idx = 4 # 0->Case1, 1->Case2, 2->Case3, 3->Case4(如果你有)

dims = N_var_list
apa_cons = np.array([255, 605, 757, 1511,2017]) - np.array(dims )
org_cons = np.array([15655, 58685, 91957, 549311, 1044897]) - np.array(dims )
apa_vars = np.array([100, 240, 300, 600, 800]) - np.array(dims )
org_vars = np.array([15404, 58324, 91206, 545110,1035056]) - np.array(dims )
methods = ['APA', 'Original']
cons_counts = {'APA': int(apa_cons[case_idx]), 'Original': int(org_cons[case_idx])}
vars_counts = {'APA': int(apa_vars[case_idx]), 'Original': int(org_vars[case_idx])}
colors = {'APA': '#5dade1', 'Original': '#ed8075'}  # 和你原来一致
save_path = f'{PROJECT_ROOT}\\results\\DRCC\\bars.svg'
plot_case_counts_bar(
    case_name=f'Case {case_idx+1}',
    methods=methods,
    cons_counts=cons_counts,
    vars_counts=vars_counts,
    colors=colors,
    figsize=(4, 3),
    y_log=True,
    y_lim=(10, 1e7),
    save_path=save_path
)

# # 读取数据
# N_var = N_var_list[case_idx]
# N_levels = N_levels_list[case_idx]
# N_samples = N_samples_list[case_idx]
# res_path = f'{PROJECT_ROOT}\\results\\DRCC\\x{N_var}g{N_levels}s{N_samples}\\test_result.pkl'
# # ===== 用法示例 =====
#
# case_figure_params = [dict(xlim = (0.0371, 0.0377),ylim = (-0.1e-4, 2e-4),xtick_step=0.0002,yticks=[0, 0.5e-4, 1e-4, 1.5e-4, 2e-4]),
#                     dict(xlim = (0.036, 0.038),ylim = (-0.5e-4, 5e-4),xtick_step=0.0005,yticks=[0, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4]),
#                     dict(xlim = (0.0345, 0.037),ylim = (1.5e-4, 3.5e-4),xtick_step=0.0005,ytick_step=0.5e-4),
#                     dict(xlim = (0.035, 0.038),ylim = (0, 1.6e-4),xtick_step=0.001,ytick_step=0.4e-4),
#                       dict(xlim=(0.0380, 0.0392), ylim=(0.3e-4, 1.5e-4), xtick_step=0.0004, ytick_step=0.3e-4),
#                       ]
# save_path = f'{PROJECT_ROOT}\\results\\DRCC\\distributions.svg'
# stats = plot_kde_scatter_with_pareto(res_path,**case_figure_params[case_idx], save_path=save_path)
# print(stats)

