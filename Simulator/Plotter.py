import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon, Ellipse, Rectangle, Circle
import os
from PIL import Image
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

from typing import List, Tuple, Optional
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class ShapeDrawer_2D:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.shapes = []  # 统一存储所有图形对象
        self.ax.grid(True)

        self.error_history = {
            'iterations': [],
            'error_feas': [],
            'error_opt': []
        }

    def _add_shape(self, patch, shape_type, ** kwargs):
        """统一添加图形到存储列表"""
        shape_id = len(self.shapes)
        shape_info = {
            'patch': patch,
            'shape': shape_type,
            'id': shape_id,
            ** kwargs  # 存储其他自定义属性
        }
        self.shapes.append(shape_info)
        return shape_id

    def plot_epigraph(self, x_range=(0, 1), f_min_func=lambda x: x ** 2,
        num_points = 500, alpha = 0.3,
        facecolor = 'skyblue',
        f_color = 'b-', label = 'epigraph', title = None,
        xlim = None, ylim = None):
        """绘制x² ≤ f ≤1类型的可行域
        参数：
            x_range: x轴数据生成范围元组，默认(0,1)
            f_min_func: 下限函数，默认x²
            f_max_val: 上限值，默认1
            num_points: 采样点数量，默认500
            alpha: 填充透明度，默认0.3
            facecolor: 填充颜色，默认天蓝
            f_min_color: 下限曲线样式，默认红色虚线
            f_max_color: 上限曲线样式，默认蓝色实线
            xlim: 显示范围x轴界限，默认None（自动适配）
            ylim: 显示范围y轴界限，默认None（自动适配）
        """
        # 生成数据点
        x = np.linspace(x_range[0], x_range[1], num_points)
        f_min = f_min_func(x)
        f_max_val = f_min_func(x_range[1])
        f_max = np.full_like(x, f_max_val)

        # 填充可行域（生成PolyCollection对象）
        fill = self.ax.fill_between(x, f_min, f_max,
                                    alpha=alpha,
                                    facecolor=facecolor,
                                    label=label)

        # 绘制边界曲线
        line_min, = self.ax.plot(x, f_min, f_color)
        line_max, = self.ax.plot(x, f_max, f_color)
        # 新增x=0边界线
        x0 = 0
        y_min_x0 = f_min_func(x0)
        line_x0, = self.ax.plot([x0, x0], [y_min_x0, f_max_val], f_color)
        # 坐标范围设置（优先级：手动设置 > 自动计算）
        display_xlim = xlim if xlim is not None else x_range
        self.ax.set_xlim(display_xlim)

        if ylim is not None:
            self.ax.set_ylim(ylim)
        else:
            ymin = f_min.min() * 0.9 if len(f_min) > 0 else 0
            ymax = f_max_val * 1.1
            self.ax.set_ylim(ymin, ymax)

        # 统一注册所有图形元素（关键修改点）
        shape_id = self._add_shape(
            patch=fill,  # 将PolyCollection作为patch存储
            shape_type='epigraph',
            lines=[line_min, line_max],  # 存储边界线对象
            params={
                'x_range': x_range,
                'f_min_func': f_min_func,
                'f_max_val': f_max_val,
                'xlim': xlim,
                'ylim': ylim,
                'artist_type': 'collection'  # 标记特殊类型
            },
            alpha=alpha,
            facecolor=facecolor,
            label=label
        )

        # 标题设置
        if title:
            self.ax.set_title(title)
        else:
            default_title = f"Epigraph: {x_range[0]} ≤ x ≤ {x_range[1]} | {f_min_func.__name__} ≤ f ≤ {f_max_val}"
            self.ax.set_title(default_title)

        return shape_id
    def plot_polygon(self, A, b, xlim, ylim, alpha=0.2, edgecolor='navy',
                     facecolor='blue', label=None, title=None):
        """绘制多边形"""
        # 计算交点
        vertices = []
        for i in range(len(A)):
            # if np.linalg.norm(A[i])<=1e-4:
            #     continue
            for j in range(i + 1, len(A)):
                try:
                    x, y = np.linalg.solve(np.array([A[i], A[j]]), np.array([b[i], b[j]]))
                    if np.all(A @ np.array([x, y]) <= b + 1e-5):
                        vertices.append((x, y))
                except np.linalg.LinAlgError:
                    pass

        if not vertices:
            return None

        # 顶点排序
        center = np.mean(vertices, axis=0)
        angles = np.arctan2([v[1] - center[1] for v in vertices],
                            [v[0] - center[0] for v in vertices])
        sorted_vertices = np.array([v for _, v in sorted(zip(angles, vertices))])

        # 创建并添加多边形
        polygon = Polygon(sorted_vertices, closed=True, alpha=alpha,
                          edgecolor=edgecolor, facecolor=facecolor, label=label)
        patch = self.ax.add_patch(polygon)

        # 统一存储
        shape_id = self._add_shape(
            patch=patch,
            shape_type='polygon',
            vertices=sorted_vertices,
            alpha=alpha,
            edgecolor=edgecolor,
            facecolor=facecolor,
            label=label
        )

        # 设置坐标范围和标题
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        # 移除所有边框和网格
        # self.ax.set_xticks([])
        # self.ax.set_yticks([])
        # self.ax.spines['top'].set_visible(False)
        # self.ax.spines['right'].set_visible(False)
        # self.ax.spines['bottom'].set_visible(False)
        # self.ax.spines['left'].set_visible(False)
        self.ax.grid(False)


        if title:
            self.ax.set_title(title)

        return shape_id

    def plot_convex_hull(self, points, alpha=0.2, facecolor='none',
                         edgecolor='green', vertices_marker='o',
                         vertices_color='green', label=None, title=None):
        """
        绘制点集的凸包多边形

        参数:
            points: 二维点的集合，形状为(n, 2)的NumPy数组
            alpha: 填充透明度，默认0.3
            facecolor: 填充颜色，默认'lightcoral'
            edgecolor: 边界线颜色，默认'red'
            linewidth: 边界线宽度，默认2
            vertices_marker: 凸包顶点标记样式，默认's'(方块)
            vertices_color: 凸包顶点颜色，默认'red'
            label: 图例标签
            title: 标题
        """


        # 获取凸包顶点并按顺序排列
        vertices = points

        # 创建多边形对象
        polygon = Polygon(vertices, closed=True,
                          alpha=alpha,
                          facecolor=facecolor,
                          edgecolor=edgecolor,
                          label=label)

        # 添加多边形到坐标轴
        patch = self.ax.add_patch(polygon)

        # 绘制凸包顶点标记
        vertices_x = vertices[:, 0]
        vertices_y = vertices[:, 1]
        vertices_scatter = self.ax.scatter(vertices_x, vertices_y,
                                           marker=vertices_marker,
                                           color=vertices_color,
                                           edgecolors=edgecolor,
                                           s=5, zorder=3)

        # 存储多边形和顶点标记作为同一图形
        shape_id = self._add_shape(
            patch=patch,  # 多边形作为主要补丁
            shape_type='convex_hull',
            scatter=vertices_scatter,  # 存储散点标记
            params={
                'vertices': vertices,
                'points': points,
                'vertices_color': vertices_color
            },
            alpha=alpha,
            facecolor=facecolor,
            edgecolor=edgecolor,
            label=label
        )

        return shape_id

    def plot_ellipse(self, Sigma, xlim, ylim, alpha=0.2, edgecolor='green',
                     facecolor='none', label=None, title=None):
        """绘制椭圆（基于协方差矩阵）"""
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(Sigma)
        lambda1, lambda2 = eigenvalues
        v1, v2 = eigenvectors[:, 0], eigenvectors[:, 1]

        # 计算椭圆参数
        width = 2 / np.sqrt(lambda1)
        height = 2 / np.sqrt(lambda2)
        angle = np.degrees(np.arctan2(v1[1], v1[0]))

        # 创建并添加椭圆
        ellipse = Ellipse(xy=(0, 0), width=width, height=height, angle=angle,
                          alpha=alpha, edgecolor=edgecolor,
                          facecolor=facecolor, label=label)
        patch = self.ax.add_patch(ellipse)

        # 统一存储
        shape_id = self._add_shape(
            patch=patch,
            shape_type='ellipse',
            sigma=Sigma,
            alpha=alpha,
            edgecolor=edgecolor,
            facecolor=facecolor,
            label=label
        )

        # 设置坐标范围和标题
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        # 移除所有边框和网格
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.grid(False)
        if title:
            self.ax.set_title(title)

        return shape_id

    def plot_circle_regions(self, theta, xlim, ylim, alpha=0.2, edgecolor='green', facecolor='none', label=None,
                            title=None):
        """绘制单位圆内部和另一圆外部的区域"""
        # 绘制单位圆内部
        circle_inner = Circle((0, 0), 1, alpha=alpha, edgecolor=edgecolor, facecolor=facecolor,label = label)
        patch_inner = self.ax.add_patch(circle_inner)
        shape_id_inner = self._add_shape(
            patch=patch_inner,
            shape_type='circle',
            center=(0, 0),
            radius=1,
            alpha=alpha,
            edgecolor=edgecolor,
            facecolor=facecolor,
            # label=label
        )
        # circle_outer = Circle((theta[0], theta[1]), theta[2], alpha=1, edgecolor=None, facecolor='white',label = label)
        # patch_outer = self.ax.add_patch(circle_outer)
        # shape_id_outer = self._add_shape(
        #     patch=patch_outer,
        #     shape_type='circle',
        #     center=(theta[0], theta[1]),
        #     radius=theta[2],
        #     alpha=1,
        #     edgecolor=None,
        #     facecolor='white',
        #     # label=label
        # )
        # 绘制另一圆的外部区域
        a, b, r = theta[0], theta[1], theta[2]
        # 创建覆盖整个绘图区域的矩形
        rect = Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0],
                         facecolor='white', alpha=1,label = None)
        # 创建圆形路径用于剪裁
        circle_clip = Circle((a, b), r, transform=self.ax.transData,label = None)
        rect.set_clip_path(circle_clip)
        patch_rect = self.ax.add_patch(rect)
        shape_id_outer = self._add_shape(
            patch=patch_rect,
            shape_type='rectangle',
            xlim=xlim,
            ylim=ylim,
            alpha=1,
            facecolor=None,
            edgecolor=edgecolor,
            label = None,
            clip_circle={'center': (a, b), 'radius': r}
        )

        # 设置坐标范围和标题
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        # 移除所有边框和网格
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.grid(False)
        if title:
            self.ax.set_title(title)

        return shape_id_inner, shape_id_outer
    def remove_shape(self, shape_id):
        """删除指定图形"""
        for i, shape in enumerate(self.shapes):
            if shape['id'] == shape_id:
                shape['patch'].remove()
                del self.shapes[i]
                return True
        return False

    def show(self):
        """显示图形"""
        if any(shape.get('label') for shape in self.shapes):
            self.ax.legend(loc='upper right')
        plt.show()

    def save(self, filename, dpi=300, transparent=False, format=None, show_legend=True):
        """Save a string or PathLike filename, inferring its format (default: SVG).

        An explicit format overrides and replaces a conflicting file extension.
        """
        filename = os.fsdecode(os.fspath(filename))
        stem, extension = os.path.splitext(filename)
        format = (format or extension.lstrip('.') or 'svg').lower()
        if extension.lower() != '.' + format:
            filename = (stem if extension else filename) + '.' + format

        if show_legend and any(shape.get('label') for shape in self.shapes):
            self.ax.legend(
                loc='upper right',
                bbox_to_anchor=(1, 1),
                frameon=False  # 无边框图例
            )

        self.fig.savefig(
            filename,
            # dpi=dpi,
            transparent=transparent,
            format=format,
            bbox_inches='tight'
        )

        print(f"图形已保存到 {filename}")




class ShapeDrawer_3D:
    """
    Draw evolving 2D polygons (given as half-space constraints Ax <= b) as 3D prisms
    across an 'iteration' axis, comparing an original polygon vs. a sequence of
    approximation polygons.

    Key features:
    - Reusable helper to convert constraints -> ordered vertices
    - Robust handling of shapes (b can be (n,) or (n,1))
    - Configurable colors/alphas, axis/labels, show/save control
    - Returns (fig, ax) so you can further customize outside the class
    """

    def __init__(
        self,
        *,
        end_face_alpha: float = 0.2,
        tube_alpha: float = 0.05,
        colors: Optional[dict] = None,
        axis_off: bool = True,
        angle = (23,-106),
    ):
        self.end_face_alpha = float(end_face_alpha)
        self.tube_alpha = float(tube_alpha)
        self.axis_off = bool(axis_off)
        self.colors = {
            # original polygon visual style
            "original_tube_face": "lightblue",
            "original_tube_edge": "blue",
            "original_end_face": "blue",
            # "original_end_edge": "None",
            "original_end_edge": "navy",
            # approximations visual style
            "approx_tube_face": "lightcoral",
            "approx_tube_edge": "red",
            "approx_end_face": "red",
            # "approx_end_edge": "None",
            "approx_end_edge": "maroon",
        }
        self.elev, self.azim = angle[0], angle[1]
        if colors:
            self.colors.update(colors)

    # ------------------------
    # Public utility
    # ------------------------
    @staticmethod
    def solve_inequalities_to_vertices(A: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> Optional[np.ndarray]:
        """
        Convert 2D half-space constraints Ax <= b into an ordered list of polygon vertices.
        Returns:
            vertices: (m, 2) array in CCW order, or None if infeasible / empty.
        Notes:
            - Simple pairwise intersection of each line, then inside-test.
            - Works for convex polygons. (This method does not compute a convex hull
              from arbitrary scattered points; it relies on the half-space description.)
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)

        if A.ndim != 2 or A.shape[1] != 2:
            raise ValueError("A must be of shape (n, 2).")
        if b.ndim != 1 or b.shape[0] != A.shape[0]:
            raise ValueError("b must be of shape (n,) matching A.shape[0].")

        n = A.shape[0]
        verts = []

        # Intersect each pair of constraint lines A_i x = b_i, A_j x = b_j
        for i in range(n):
            for j in range(i + 1, n):
                M = np.vstack([A[i], A[j]])  # 2x2
                rhs = np.array([b[i], b[j]])
                try:
                    xy = np.linalg.solve(M, rhs)  # (2,)
                except np.linalg.LinAlgError:
                    continue  # parallel or ill-conditioned pair

                if np.all(A @ xy <= b + 1e-5):  # inside test with small tolerance
                    verts.append(xy)

        if not verts:
            return None

        verts = np.asarray(verts)
        # Order vertices by angle around centroid
        center = verts.mean(axis=0)
        angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
        order = np.argsort(angles)
        ordered = verts[order]

        # Remove near-duplicates (optional stability)
        deduped = [ordered[0]]
        for k in range(1, len(ordered)):
            if np.linalg.norm(ordered[k] - deduped[-1]) > eps:
                deduped.append(ordered[k])
        if len(deduped) > 1 and np.linalg.norm(deduped[0] - deduped[-1]) <= eps:
            deduped.pop()  # drop closing duplicate

        return np.asarray(deduped).reshape(-1, 2) if len(deduped) >= 3 else None


    # ------------------------
    # Main plotting entrypoint
    # ------------------------
    def plot_polygon_evolution(
        self,
        A: np.ndarray,
        b: np.ndarray,
        approximations: List[Tuple[np.ndarray, np.ndarray]],
        *,
        figsize: Tuple[int, int] = (12, 8),
        iteration_axis_label: str = "Iteration",
        y_label: str = "X",
        z_label: str = "Y",
        title: str = "Polygon Evolution: Original (Blue) vs Approximations (Red)",
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the evolution of polygon approximations in 3D.

        Parameters
        ----------
        A, b : define the original polygon as Ax <= b (A: (n,2), b: (n,) or (n,1))
        approximations : list of (C, d) each defining Cx <= d for an iteration
        figsize : figure size
        fig, ax : optionally pass an existing 3D axis to draw on
        show : call plt.show()
        save_path : if provided, saves the figure to this path

        Returns
        -------
        (fig, ax)
        """

        # Convert constraints to vertices
        original_vertices = self.solve_inequalities_to_vertices(A, b)
        if original_vertices is None:
            raise ValueError("Original polygon constraints produce no feasible polygon.")

        # Pre-compute all iteration vertices (some may be None; we skip those)
        approx_vertices_list: List[Optional[np.ndarray]] = []
        for (C, d) in approximations:
            approx_vertices_list.append(self.solve_inequalities_to_vertices(C, d))

        num_iterations = len(approximations)
        x_start = 0
        x_end = num_iterations - 1 if num_iterations > 1 else 1

        # Prepare figure/axes
        if fig is None or ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

        # Clean axis look
        if self.axis_off:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.axis("off")

        # ---- Original polygon tube (constant prism from x=0 to x=end) ----
        side_surfaces = []
        m0 = len(original_vertices)
        for j in range(m0):
            nj = (j + 1) % m0
            side_face = [
                (x_start, original_vertices[j, 0], original_vertices[j, 1]),
                (x_start, original_vertices[nj, 0], original_vertices[nj, 1]),
                (x_end,   original_vertices[nj, 0], original_vertices[nj, 1]),
                (x_end,   original_vertices[j, 0], original_vertices[j, 1]),
            ]
            side_surfaces.append(side_face)

        if side_surfaces:
            side_collection = Poly3DCollection(
                side_surfaces,
                alpha=self.tube_alpha,
                facecolor=self.colors["original_tube_face"],
                edgecolor=self.colors["original_tube_edge"],
            )
            ax.add_collection3d(side_collection)

        # ---- Approximation evolution surfaces + end faces ----
        approx_surfaces = []       # transition tubes
        approx_end_faces = []      # cross-sections at each iteration
        original_end_faces = []    # cross-sections of original at each iteration

        for i, verts in enumerate(approx_vertices_list):
            if verts is None or len(verts) < 3:
                # Skip infeasible or degenerate iterations
                continue

            x_pos = i

            # Connect to previous valid iteration to form transition surfaces
            prev_idx = i - 1
            while prev_idx >= 0 and (approx_vertices_list[prev_idx] is None or len(approx_vertices_list[prev_idx]) < 3):
                prev_idx -= 1

            if prev_idx >= 0:
                prev_verts = approx_vertices_list[prev_idx]
                assert prev_verts is not None
                # pair up by index (fallback: min length)
                min_len = min(len(verts), len(prev_verts))
                for j in range(min_len):
                    nj = (j + 1) % min_len
                    transition_face = [
                        (prev_idx, prev_verts[j, 0], prev_verts[j, 1]),
                        (prev_idx, prev_verts[nj, 0], prev_verts[nj, 1]),
                        (x_pos,    verts[nj, 0],     verts[nj, 1]),
                        (x_pos,    verts[j, 0],      verts[j, 1]),
                    ]
                    approx_surfaces.append(transition_face)

            # Current cross-section faces (approx & original)
            approx_end_faces.append([(x_pos, verts[j, 0], verts[j, 1]) for j in range(len(verts))])
            original_end_faces.append([(x_pos, original_vertices[j, 0], original_vertices[j, 1]) for j in range(m0)])
        if original_end_faces:
            original_faces_collection = Poly3DCollection(
                original_end_faces,
                alpha=self.end_face_alpha+0.1,
                facecolor=self.colors["original_end_face"],
                edgecolor=self.colors["original_end_edge"],
            )
            ax.add_collection3d(original_faces_collection)
        # Add collections
        if approx_surfaces:
            approx_tube_collection = Poly3DCollection(
                approx_surfaces,
                alpha=self.tube_alpha,
                facecolor=self.colors["approx_tube_face"],
                edgecolor=self.colors["approx_tube_edge"],
            )
            ax.add_collection3d(approx_tube_collection)


        if approx_end_faces:
            approx_sections_collection = Poly3DCollection(
                approx_end_faces,
                alpha=self.end_face_alpha,
                facecolor=self.colors["approx_end_face"],
                edgecolor=self.colors["approx_end_edge"],
            )
            ax.add_collection3d(approx_sections_collection)
        # ---- Axis limits (equal-ish aspect) ----
        all_vertices = [original_vertices] + [v for v in approx_vertices_list if v is not None]
        all_points = np.vstack(all_vertices)
        y_min, y_max = all_points[:, 0].min(), all_points[:, 0].max()
        z_min, z_max = all_points[:, 1].min(), all_points[:, 1].max()
        span = max(y_max - y_min, z_max - z_min)
        margin = 0.1 * (span if span > 0 else 1.0)

        ax.set_xlim(0, max(1, num_iterations - 1))
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)

        # Labels / title (only if axis isn't fully off)
        if not self.axis_off:
            ax.set_xlabel(iteration_axis_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_zlabel(z_label, fontsize=12)
            ax.set_title(title, fontsize=14)
        else:
            # Even when axis is off, a title can still be useful:
            ax.set_title(title, fontsize=14)

        plt.tight_layout()
        ax.view_init(elev=self.elev, azim=self.azim)
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight", transparent=True)
        if show:
            plt.show()

        return fig, ax
    @staticmethod
    def generate_ellipse_vertices(Sigma: np.ndarray, num_points: int = 100) -> np.ndarray:
        """
        Generate vertices for an ellipse defined by the equation x' Σ x <= 1.
        Sigma is a 2x2 matrix defining the ellipse.

        Parameters
        ----------
        Sigma : 2x2 matrix defining the ellipse shape.
        num_points : Number of points to generate for the ellipse's perimeter.

        Returns
        -------
        vertices : (num_points, 2) array of ellipse vertices.
        """
        # Generate points on the unit circle (x^2 + y^2 = 1)
        theta = np.linspace(0, 2 * np.pi, num_points)
        circle_points = np.column_stack([np.cos(theta), np.sin(theta)])

        # Transform the circle points by Sigma to get the ellipse
        eigvals, eigvecs = np.linalg.eig(Sigma)
        ellipse_points = eigvecs @ np.diag(np.sqrt(1 / eigvals)) @ circle_points.T
        return ellipse_points.T

    def plot_ellipse_evolution(
            self,
            Sigma: np.ndarray,
            approximations: List[Tuple[np.ndarray, np.ndarray]],
            *,
            figsize: Tuple[int, int] = (12, 8),
            iteration_axis_label: str = "Iteration",
            y_label: str = "X",
            z_label: str = "Y",
            title: str = "Ellipse Evolution: Original (Blue) vs Approximations (Red)",
            fig: Optional[plt.Figure] = None,
            ax: Optional[plt.Axes] = None,
            show: bool = True,
            save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the evolution of ellipse approximations in 3D.

        Parameters
        ----------
        Sigma : 2x2 matrix defining the original ellipse (x' Σ x <= 1).
        approximations : list of (C, d) tuples, each defining an approximation for an iteration.
        figsize : figure size
        fig, ax : optionally pass an existing 3D axis to draw on
        show : call plt.show()
        save_path : if provided, saves the figure to this path

        Returns
        -------
        (fig, ax)
        """
        # Generate the original ellipse vertices
        original_vertices = self.generate_ellipse_vertices(Sigma)

        # Pre-compute all iteration vertices (some may be None; we skip those)
        approx_vertices_list: List[Optional[np.ndarray]] = []
        for (C, d) in approximations:
            approx_vertices_list.append(self.solve_inequalities_to_vertices(C, d))

        num_iterations = len(approximations)
        x_start = 0
        x_end = num_iterations - 1 if num_iterations > 1 else 1

        # Prepare figure/axes
        if fig is None or ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

        # Clean axis look
        if self.axis_off:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.axis("off")

        # ---- Original ellipse tube (constant prism from x=0 to x=end) ----
        ellipse_surfaces = []
        m0 = len(original_vertices)
        for j in range(m0):
            nj = (j + 1) % m0
            ellipse_surfaces.append([
                (x_start, original_vertices[j, 0], original_vertices[j, 1]),
                (x_start, original_vertices[nj, 0], original_vertices[nj, 1]),
                (x_end, original_vertices[nj, 0], original_vertices[nj, 1]),
                (x_end, original_vertices[j, 0], original_vertices[j, 1]),
            ])
        ax.add_collection3d(Poly3DCollection(
            ellipse_surfaces,
            alpha=self.tube_alpha,
            facecolor=self.colors["original_tube_face"],
            edgecolor=None,
        ))

        # ---- Approximation evolution surfaces ----
        approx_surfaces = []
        approx_end_faces = []
        original_end_faces = []
        for i, verts in enumerate(approx_vertices_list):
            if verts is None or len(verts) < 3:
                # Skip infeasible or degenerate iterations
                continue

            x_pos = i

            # Connect to previous valid iteration to form transition surfaces
            prev_idx = i - 1
            while prev_idx >= 0 and (approx_vertices_list[prev_idx] is None or len(approx_vertices_list[prev_idx]) < 3):
                prev_idx -= 1

            if prev_idx >= 0:
                prev_verts = approx_vertices_list[prev_idx]
                assert prev_verts is not None
                # pair up by index (fallback: min length)
                min_len = min(len(verts), len(prev_verts))
                for j in range(min_len):
                    nj = (j + 1) % min_len
                    transition_face = [
                        (prev_idx, prev_verts[j, 0], prev_verts[j, 1]),
                        (prev_idx, prev_verts[nj, 0], prev_verts[nj, 1]),
                        (x_pos,    verts[nj, 0],     verts[nj, 1]),
                        (x_pos,    verts[j, 0],      verts[j, 1]),
                    ]
                    approx_surfaces.append(transition_face)

            # Current cross-section faces (approx & original)
            approx_end_faces.append([(x_pos, verts[j, 0], verts[j, 1]) for j in range(len(verts))])
            original_end_faces.append([(x_pos, original_vertices[j, 0], original_vertices[j, 1]) for j in range(m0)])


        if original_end_faces:
            original_faces_collection = Poly3DCollection(
                original_end_faces,
                alpha=self.end_face_alpha+0.1,
                facecolor=self.colors["original_end_face"],
                edgecolor=self.colors["original_end_edge"],
            )
            ax.add_collection3d(original_faces_collection)
        # Add collections for approximations
        if approx_surfaces:
            ax.add_collection3d(Poly3DCollection(
                approx_surfaces,
                alpha=self.tube_alpha,
                facecolor=self.colors["approx_tube_face"],
                edgecolor=self.colors["approx_tube_edge"],
            ))

        if approx_end_faces:
            ax.add_collection3d(Poly3DCollection(
                approx_end_faces,
                alpha=self.end_face_alpha,
                facecolor=self.colors["approx_end_face"],
                edgecolor=self.colors["approx_end_edge"],
            ))

        # ---- Axis limits ----
        all_vertices = [original_vertices] + [v for v in approx_vertices_list if v is not None]
        all_points = np.vstack(all_vertices)
        y_min, y_max = all_points[:, 0].min(), all_points[:, 0].max()
        z_min, z_max = all_points[:, 1].min(), all_points[:, 1].max()
        span = max(y_max - y_min, z_max - z_min)
        margin = 0.1 * (span if span > 0 else 1.0)

        ax.set_xlim(0, max(1, num_iterations - 1))
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)

        # Labels / title
        if not self.axis_off:
            ax.set_xlabel(iteration_axis_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_zlabel(z_label, fontsize=12)
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(title, fontsize=14)

        plt.tight_layout()
        ax.view_init(elev=self.elev, azim=self.azim)
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight", transparent=True)

        if show:
            plt.show()

        return fig, ax

    @staticmethod
    def generate_circle_region_vertices(center1: Tuple[float, float], radius1: float,
                                        center2: Tuple[float, float], radius2: float,
                                        num_points: int = 200) -> np.ndarray:
        """
        Generate vertices for a non-convex region: circle1 minus the intersection with circle2.
        This creates the region inside circle1 but outside circle2.

        Parameters
        ----------
        center1 : (x, y) center of the first circle (the main region)
        radius1 : radius of the first circle
        center2 : (x, y) center of the second circle (the subtracted region)
        radius2 : radius of the second circle
        num_points : Number of points to generate for the boundary

        Returns
        -------
        vertices : (n, 2) array of boundary vertices, or None if no valid region
        """
        x1, y1 = center1
        x2, y2 = center2

        # Distance between centers
        d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Check if circles intersect
        if d >= radius1 + radius2:
            # No intersection, return full circle1
            theta = np.linspace(0, 2 * np.pi, num_points)
            vertices = np.column_stack([
                x1 + radius1 * np.cos(theta),
                y1 + radius1 * np.sin(theta)
            ])
            return vertices

        if d + radius1 <= radius2:
            # Circle1 completely inside circle2, no region left
            return None

        if d + radius2 <= radius1:
            # Circle2 completely inside circle1, create annulus-like region
            # Outer boundary: circle1, Inner boundary: circle2 (reversed)
            theta1 = np.linspace(0, 2 * np.pi, num_points // 2)
            outer_boundary = np.column_stack([
                x1 + radius1 * np.cos(theta1),
                y1 + radius1 * np.sin(theta1)
            ])

            theta2 = np.linspace(2 * np.pi, 0, num_points // 2)  # Reversed for inner hole
            inner_boundary = np.column_stack([
                x2 + radius2 * np.cos(theta2),
                y2 + radius2 * np.sin(theta2)
            ])

            # Connect outer to inner with a thin bridge to make it simply connected
            bridge_point1 = outer_boundary[0]
            bridge_point2 = inner_boundary[0]

            vertices = np.vstack([outer_boundary, [bridge_point1], inner_boundary, [bridge_point2]])
            return vertices

        # Circles partially intersect - find intersection points
        a = (radius1 ** 2 - radius2 ** 2 + d ** 2) / (2 * d)
        h = np.sqrt(radius1 ** 2 - a ** 2)

        # Intersection point on line between centers
        px = x1 + a * (x2 - x1) / d
        py = y1 + a * (y2 - y1) / d

        # Two intersection points
        intersect1 = np.array([px + h * (y2 - y1) / d, py - h * (x2 - x1) / d])
        intersect2 = np.array([px - h * (y2 - y1) / d, py + h * (x2 - x1) / d])

        # Find angles for intersection points relative to each circle center
        angle1_c1 = np.arctan2(intersect1[1] - y1, intersect1[0] - x1)
        angle2_c1 = np.arctan2(intersect2[1] - y1, intersect2[0] - x1)
        angle1_c2 = np.arctan2(intersect1[1] - y2, intersect1[0] - x2)
        angle2_c2 = np.arctan2(intersect2[1] - y2, intersect2[0] - x2)

        # Normalize angles to [0, 2π]
        angle1_c1 = angle1_c1 % (2 * np.pi)
        angle2_c1 = angle2_c1 % (2 * np.pi)
        angle1_c2 = angle1_c2 % (2 * np.pi)
        angle2_c2 = angle2_c2 % (2 * np.pi)

        # Determine which arc of circle1 to keep (the one outside circle2)
        # Test the midpoint of each possible arc
        if angle2_c1 < angle1_c1:
            angle2_c1 += 2 * np.pi

        # Test arc from angle1 to angle2
        test_angle1 = (angle1_c1 + angle2_c1) / 2
        test_point1 = np.array([x1 + radius1 * np.cos(test_angle1),
                                y1 + radius1 * np.sin(test_angle1)])
        test_dist1 = np.sqrt((test_point1[0] - x2) ** 2 + (test_point1[1] - y2) ** 2)

        # Test complementary arc
        test_angle2 = test_angle1 + np.pi
        test_point2 = np.array([x1 + radius1 * np.cos(test_angle2),
                                y1 + radius1 * np.sin(test_angle2)])
        test_dist2 = np.sqrt((test_point2[0] - x2) ** 2 + (test_point2[1] - y2) ** 2)

        if test_dist1 > radius2:
            # Keep arc from angle1 to angle2
            theta_c1 = np.linspace(angle1_c1, angle2_c1, num_points // 3)
            arc1_points = np.column_stack([
                x1 + radius1 * np.cos(theta_c1),
                y1 + radius1 * np.sin(theta_c1)
            ])
        else:
            # Keep complementary arc
            if angle1_c1 > angle2_c1:
                theta_c1 = np.linspace(angle2_c1, angle1_c1, num_points // 3)
            else:
                theta_c1 = np.concatenate([
                    np.linspace(angle2_c1, 2 * np.pi, num_points // 6),
                    np.linspace(0, angle1_c1, num_points // 6)
                ])
            arc1_points = np.column_stack([
                x1 + radius1 * np.cos(theta_c1),
                y1 + radius1 * np.sin(theta_c1)
            ])

        # Now add the arc of circle2 that forms the boundary (from intersect2 to intersect1)
        # This is the key fix - we need the arc of circle2 that's inside circle1
        if angle1_c2 < angle2_c2:
            # Check which direction gives us the arc inside circle1
            mid_angle = (angle1_c2 + angle2_c2) / 2
            mid_point = np.array([x2 + radius2 * np.cos(mid_angle),
                                  y2 + radius2 * np.sin(mid_angle)])
            mid_dist = np.sqrt((mid_point[0] - x1) ** 2 + (mid_point[1] - y1) ** 2)

            if mid_dist < radius1:
                # Arc from angle2 to angle1 (clockwise)
                theta_c2 = np.linspace(angle2_c2, angle1_c2, num_points // 3)
            else:
                # Arc from angle1 to angle2 (counter-clockwise)
                theta_c2 = np.linspace(angle1_c2, angle2_c2, num_points // 3)
        else:
            # angle2_c2 < angle1_c2, need to check which way around
            mid_angle1 = (angle2_c2 + angle1_c2) / 2
            mid_angle2 = mid_angle1 + np.pi

            mid_point1 = np.array([x2 + radius2 * np.cos(mid_angle1),
                                   y2 + radius2 * np.sin(mid_angle1)])
            mid_dist1 = np.sqrt((mid_point1[0] - x1) ** 2 + (mid_point1[1] - y1) ** 2)

            if mid_dist1 < radius1:
                theta_c2 = np.linspace(angle2_c2, angle1_c2, num_points // 3)
            else:
                # Go the long way around
                theta_c2 = np.concatenate([
                    np.linspace(angle1_c2, 2 * np.pi, num_points // 6),
                    np.linspace(0, angle2_c2, num_points // 6)
                ])

        arc2_points = np.column_stack([
            x2 + radius2 * np.cos(theta_c2),
            y2 + radius2 * np.sin(theta_c2)
        ])

        # Combine the arcs: arc1 + arc2 to form the complete boundary
        vertices = np.vstack([arc1_points, arc2_points])

        return vertices

    def plot_circle_region_evolution(
            self,
            center1: Tuple[float, float],
            radius1: float,
            center2: Tuple[float, float],
            radius2: float,
            approximations: List[Tuple[np.ndarray, np.ndarray]],
            *,
            figsize: Tuple[int, int] = (12, 8),
            iteration_axis_label: str = "Iteration",
            y_label: str = "X",
            z_label: str = "Y",
            title: str = "Circle Region Evolution: Original (Blue) vs Approximations (Red)",
            fig: Optional[plt.Figure] = None,
            ax: Optional[plt.Axes] = None,
            show: bool = True,
            save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the evolution of circle region approximations in 3D.
        The original region is defined as circle1 minus the intersection with circle2.

        Parameters
        ----------
        center1 : (x, y) center of the main circle
        radius1 : radius of the main circle
        center2 : (x, y) center of the subtracted circle
        radius2 : radius of the subtracted circle
        approximations : list of (C, d) tuples, each defining a polyhedral approximation
        figsize : figure size
        fig, ax : optionally pass an existing 3D axis to draw on
        show : call plt.show()
        save_path : if provided, saves the figure to this path

        Returns
        -------
        (fig, ax)
        """
        # Generate the original circle region vertices
        original_vertices = self.generate_circle_region_vertices(center1, radius1, center2, radius2)
        if original_vertices is None:
            raise ValueError("Circle region constraints produce no feasible region.")

        # Pre-compute all iteration vertices (some may be None; we skip those)
        approx_vertices_list: List[Optional[np.ndarray]] = []
        for (C, d) in approximations:
            approx_vertices_list.append(self.solve_inequalities_to_vertices(C, d))

        num_iterations = len(approximations)
        x_start = 0
        x_end = num_iterations - 1 if num_iterations > 1 else 1

        # Prepare figure/axes
        if fig is None or ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

        # Clean axis look
        if self.axis_off:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.axis("off")

        # ---- Original circle region tube (constant prism from x=0 to x=end) ----
        region_surfaces = []
        m0 = len(original_vertices)
        for j in range(m0):
            nj = (j + 1) % m0
            region_surfaces.append([
                (x_start, original_vertices[j, 0], original_vertices[j, 1]),
                (x_start, original_vertices[nj, 0], original_vertices[nj, 1]),
                (x_end, original_vertices[nj, 0], original_vertices[nj, 1]),
                (x_end, original_vertices[j, 0], original_vertices[j, 1]),
            ])

        if region_surfaces:
            ax.add_collection3d(Poly3DCollection(
                region_surfaces,
                alpha=self.tube_alpha,
                facecolor=self.colors["original_tube_face"],
                edgecolor=None,
            ))

        # ---- Approximation evolution surfaces ----
        approx_surfaces = []
        approx_end_faces = []
        original_end_faces = []

        for i, verts in enumerate(approx_vertices_list):
            if verts is None or len(verts) < 3:
                # Skip infeasible or degenerate iterations
                continue

            x_pos = i

            # Connect to previous valid iteration to form transition surfaces
            prev_idx = i - 1
            while prev_idx >= 0 and (approx_vertices_list[prev_idx] is None or len(approx_vertices_list[prev_idx]) < 3):
                prev_idx -= 1

            if prev_idx >= 0:
                prev_verts = approx_vertices_list[prev_idx]
                assert prev_verts is not None
                # pair up by index (fallback: min length)
                min_len = min(len(verts), len(prev_verts))
                for j in range(min_len):
                    nj = (j + 1) % min_len
                    transition_face = [
                        (prev_idx, prev_verts[j, 0], prev_verts[j, 1]),
                        (prev_idx, prev_verts[nj, 0], prev_verts[nj, 1]),
                        (x_pos, verts[nj, 0], verts[nj, 1]),
                        (x_pos, verts[j, 0], verts[j, 1]),
                    ]
                    approx_surfaces.append(transition_face)

            # Current cross-section faces (approx & original)
            approx_end_faces.append([(x_pos, verts[j, 0], verts[j, 1]) for j in range(len(verts))])
            original_end_faces.append([(x_pos, original_vertices[j, 0], original_vertices[j, 1]) for j in range(m0)])

        # Add collections
        if original_end_faces:
            original_faces_collection = Poly3DCollection(
                original_end_faces,
                alpha=self.end_face_alpha+0.1,
                facecolor=self.colors["original_end_face"],
                edgecolor=self.colors["original_end_edge"],
            )
            ax.add_collection3d(original_faces_collection)

        if approx_surfaces:
            ax.add_collection3d(Poly3DCollection(
                approx_surfaces,
                alpha=self.tube_alpha,
                facecolor=self.colors["approx_tube_face"],
                edgecolor=self.colors["approx_tube_edge"],
            ))

        if approx_end_faces:
            ax.add_collection3d(Poly3DCollection(
                approx_end_faces,
                alpha=self.end_face_alpha,
                facecolor=self.colors["approx_end_face"],
                edgecolor=self.colors["approx_end_edge"],
            ))

        # ---- Axis limits ----
        all_vertices = [original_vertices] + [v for v in approx_vertices_list if v is not None]
        all_points = np.vstack(all_vertices)
        y_min, y_max = all_points[:, 0].min(), all_points[:, 0].max()
        z_min, z_max = all_points[:, 1].min(), all_points[:, 1].max()
        span = max(y_max - y_min, z_max - z_min)
        margin = 0.1 * (span if span > 0 else 1.0)

        ax.set_xlim(0, max(1, num_iterations - 1))
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)

        # Labels / title
        if not self.axis_off:
            ax.set_xlabel(iteration_axis_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_zlabel(z_label, fontsize=12)
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(title, fontsize=14)

        plt.tight_layout()
        ax.view_init(elev=self.elev, azim=self.azim)
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight", transparent=True)

        if show:
            plt.show()

        return fig, ax
def plot_errors(error_feas, error_opt, n_train,interporation = True):
    from scipy.interpolate import make_interp_spline

    # Original x values (assuming iterations are evenly spaced up to 1000)
    iterations = np.linspace(0, n_train, len(error_feas))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 3))

    if interporation:
        # Create new x values for smooth curve
        iterations_smooth = np.linspace(0, n_train, 100)  # 300 points for smoothness

        # Create spline interpolations
        feas_spline = make_interp_spline(iterations, np.log10(np.array(error_feas)+1e-10), k=3)
        opt_spline = make_interp_spline(iterations, np.log10(np.array(error_opt)+1e-10), k=3)

        # Evaluate splines at new points
        feas_smooth = feas_spline(iterations_smooth)
        opt_smooth = opt_spline(iterations_smooth)

        # Plot smooth curves without markers
        ax.semilogy(iterations_smooth, np.power(10,feas_smooth) , '-', color='#D95F41', linewidth=2)
        ax.semilogy(iterations_smooth, np.power(10,opt_smooth), '-', color='#3B75AF', linewidth=2)
    else:
        # Plot smooth curves without markers
        ax.semilogy(iterations, error_feas, '-', color='#D95F41', linewidth=2)
        ax.semilogy(iterations, error_opt, '-', color='#3B75AF', linewidth=2)
    # Set axis limits and ticks
    ax.set_xlim(0, n_train)
    ax.set_xticks(np.arange(0, n_train+1, int(n_train/2)))

    fontsize = 30
    ax.set_xticklabels(np.arange(0, n_train+1, int(n_train/2)), fontsize=fontsize )
    ax.set_ylim(1e-4, 1e0)

    # Custom y-axis ticks and labels
    yticks = [1e0, 1e-2, 1e-4]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['1e0', '1e-2', '1e-4'], fontsize=fontsize )
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    # Remove borders and grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.grid(False)

    # Labels and legend
    ax.set_xlabel('Iteration', fontsize=fontsize )
    ax.set_ylabel('Error', fontsize=fontsize )

    plt.tight_layout()
    plt.show()

def create_gif(image_folder, output_gif, duration=300, loop=0):
    """
    使用Pillow生成GIF动画

    参数:
        image_folder: 图片目录路径
        output_gif: 输出GIF文件名
        duration: 每帧持续时间(毫秒)
        loop: 循环次数(0表示无限循环)
    """
    # 获取按数字排序的图片文件
    images = []
    files = sorted([f for f in os.listdir(image_folder)
                    if f.startswith('step') and f.endswith('.png')],
                   key=lambda x: int(x[4:-4]))

    # 加载所有图片帧
    frames = []
    for file in files:
        frame = Image.open(os.path.join(image_folder, file))
        frames.append(frame.copy())
        frame.close()  # 显式关闭文件

    # 保存为GIF（优化处理）
    if len(frames) > 0:
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop,
            optimize=True,  # 启用优化
            # disposal=2  # 每帧恢复背景
        )
        print(f"成功生成GIF: {output_gif} (共{len(frames)}帧)")
    else:
        print("未找到有效图片帧！")

# import seaborn as sns
class ErrorVisualizer:
    def __init__(self):
        """
        独立的误差可视化工具
        跟踪记录两种误差：
        - error_feas: 可行性误差
        - error_opt: 最优性误差
        """
        self.error_history = {
            'iterations': [],
            'error_feas': [],  # 存储每次迭代的可行性误差列表
            'error_opt': []  # 存储每次迭代的最优性误差列表
        }

    def compute_errors(self, model, num_sample=50):
        """
        计算并记录误差（原count_error函数的功能）

        参数:
            model: 需要包含以下方法：
                - optimize_direction()
                - project()
            num_sample: 采样次数
        """
        error_feas = []
        error_opt = []

        for _ in range(num_sample):
            c = np.random.randn(model.dim)
            # print(c)
            # 计算可行性误差
            x_apx = model.optimize_direction(c, in_approx=True)
            x_org = model.project(x_apx) if x_apx is not None else None
            if x_org is not None:
                error_feas.append(np.sum((x_apx - x_org)  ** 2))
            # 计算最优性误差
            x_org = model.optimize_direction(c)
            x_apx = model.project(x_org, to_approx=True) if x_org is not None else None
            if x_apx is not None:
                error_opt.append(np.sum((x_apx - x_org)  ** 2))

        self.error_history['iterations'].append(model._iter)
        self.error_history['error_feas'].append(np.array(error_feas))
        self.error_history['error_opt'].append(np.array(error_opt))

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def plot_dual_violin(self, save_path=None, figsize=(13, 4), is_discrete = False):
        """
        修改后的小提琴图，配色与代码2一致，特别是error_opt的小提琴均值线和极值线为红色，保持error_feas的配色不变
        """
        plt.figure(figsize=figsize)
        label_font_size = 16
        iterations = self.error_history['iterations']
        feas_data = self.error_history['error_feas']
        opt_data = self.error_history['error_opt']

        display_step = max(1, len(iterations) // 10)
        display_indices = range(0, len(iterations), display_step)
        display_iters = [iterations[i] for i in display_indices]

        positions = []
        all_data = []
        for idx in display_indices:
            positions.extend([idx * 2, idx * 2 + 0.8])
            all_data.append(feas_data[idx])
            all_data.append(opt_data[idx])

        data_for_plot = []
        for i in range(len(positions)):
            data_for_plot.append(all_data[i])

        # plt.axhline(y=0, color='black', linestyle='-', linewidth=2)
        parts = plt.violinplot(
            dataset=data_for_plot,
            positions=positions,
            widths=0.8,
            showmeans=True,
            showextrema=True,
        )

        # 配色设置：feasibility error蓝色，optimality error红色
        colors = ['#3498db', '#e74c3c']  # 蓝色用于feasibility，红色用于optimality
        for i, body in enumerate(parts['bodies']):
            body.set_facecolor(colors[i % 2])
            body.set_alpha(0.4)
            body.set_edgecolor('black')
            body.set_linewidth(1)
            if is_discrete:
                body.set_linestyle('--')


        # 设置 error_opt 的均值线、最大值线、最小值线为红色
        color_mid_lines = '#000000'
        parts['cmeans'].set_color(color_mid_lines)  # 均值线颜色
        parts['cmaxes'].set_color(color_mid_lines)  # 最大值线颜色
        parts['cmins'].set_color(color_mid_lines)  # 最小值线颜色
        parts['cbars'].set_color(color_mid_lines)  # 中间竖线颜色

        feas_means_x, feas_means_y = [], []
        opt_means_x, opt_means_y = [], []

        for i, (x, y_line) in enumerate(zip(positions, parts['cmeans'].get_segments())):
            x_coord = (y_line[0][0] + y_line[1][0]) / 2
            y_coord = (y_line[0][1] + y_line[1][1]) / 2
            # plt.plot(x_coord, y_coord, marker='D', color='black', markersize=8, markerfacecolor='white',
            #          linestyle='None')
            if i % 2 == 0:
                feas_means_x.append(x)
                feas_means_y.append(y_coord)
            else:
                opt_means_x.append(x)
                opt_means_y.append(y_coord)

        # 绘制均值曲线
        ls = '--' if is_discrete else '-'
        plt.plot(feas_means_x, feas_means_y, color=colors[0], linestyle=ls, linewidth=2, alpha=0.7,
                 label='Feas Mean Curve')
        plt.plot(opt_means_x, opt_means_y, color=colors[1], linestyle=ls, linewidth=2, alpha=0.7,
                 label='Opt Mean Curve')

        # 设置x轴刻度
        x_ticks = [(feas_means_x[i] + opt_means_x[i]) / 2 for i in range(len(feas_means_x))]
        plt.xticks(x_ticks, [f'{iter}' for iter in display_iters])

        # 设置标签和标题
        # plt.xlabel('Iteration', fontsize=12)
        # plt.ylabel('Error Value', fontsize=12)
        # plt.title('Error Distribution Evolution (Violin)', fontsize=14, pad=20)
        ax = plt.gca()  # 获取当前坐标轴
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        # 修改网格设置，只显示横向网格，取消纵向网格
        plt.grid(True, alpha=0.3, linestyle='--', axis='y')  # 只显示y轴方向的网格
        plt.grid(False, axis='x')  # 取消x轴方向的网格
        plt.ylim(0, 1.4)  # 设置y轴范围为0到1.2，根据你的数据调整

        # 设置x轴刻度标签字体大小
        plt.xticks(fontsize=label_font_size)  # 设置x轴刻度标签字体大小为12
        plt.yticks(fontsize=label_font_size)  # 也可以同时设置y轴刻度标签字体大小

        # 添加图例
        legend_elements = [
            Line2D([0], [0], color=colors[0], lw=4, label='Feasibility error'),
            Line2D([0], [0], color=colors[1], lw=4, label='Optimality error'),
            Line2D([0], [0], color=colors[0], lw=2, linestyle='-', label='Feas mean curve'),
            Line2D([0], [0], color=colors[1], lw=2, linestyle='--', label='Opt mean curve'),
            Line2D([0], [0], marker='D', color='black', label='Mean',
                   markersize=8, linestyle='None', markerfacecolor='white')
        ]
        # plt.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
        plt.subplots_adjust(right=0.75)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()

    def plot_dual_violin_seperated(self, save_path=None, figsize=(15, 5)):
        """
        分为上下子图的小提琴图，上部分展示feasibility error，下部分展示optimality error
        """
        # 创建上下子图，共享x轴
        label_font_size = 13
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'hspace': 0.02})

        iterations = self.error_history['iterations']
        feas_data = self.error_history['error_feas']
        opt_data = self.error_history['error_opt']

        display_step = max(1, len(iterations) // 10)
        display_indices = range(0, len(iterations), display_step)
        display_iters = [iterations[i] for i in display_indices]

        # 准备位置和数据
        positions = [idx * 2 + 0.4 for idx in display_indices]  # 统一位置计算
        feas_data_plot = [feas_data[idx] for idx in display_indices]
        opt_data_plot = [opt_data[idx] for idx in display_indices]

        # 绘制水平线（0基准线）
        # ax1.axhline(y=0, color='black', linestyle='-', linewidth=2)
        # ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)

        # 绘制小提琴图
        colors = ['#3498db', '#e74c3c']

        # 上子图：feasibility error
        parts1 = ax1.violinplot(
            dataset=feas_data_plot,
            positions=positions,
            widths=1,
            showmeans=True,
            showextrema=True
        )
        for body in parts1['bodies']:
            body.set_facecolor(colors[0])
            body.set_alpha(0.6)
            body.set_edgecolor('black')
            body.set_linewidth(1)

        # 下子图：optimality error
        parts2 = ax2.violinplot(
            dataset=opt_data_plot,
            positions=positions,
            widths=1,
            showmeans=True,
            showextrema=True
        )
        for body in parts2['bodies']:
            body.set_facecolor(colors[1])
            body.set_alpha(0.6)
            body.set_edgecolor('black')
            body.set_linewidth(1)
        # 设置均值线和极值线的颜色为红色
        line_color = '#AF535A'
        parts2['cmeans'].set_color(line_color)  # 均值线颜色
        parts2['cmaxes'].set_color(line_color)  # 最大值线颜色
        parts2['cmins'].set_color(line_color)  # 最小值线颜色
        parts2['cbars'].set_color(line_color)  # 中间竖线颜色
        # 处理均值点和连接线
        # 上子图均值
        feas_means_y = []
        for seg in parts1['cmeans'].get_segments():
            y_coord = (seg[0][1] + seg[1][1]) / 2
            x_coord = (seg[0][0] + seg[1][0]) / 2
            # ax1.plot(x_coord, y_coord, marker='D', color='black',
            #          markersize=8, markerfacecolor='white', linestyle='None')
            feas_means_y.append(y_coord)
        ax1.plot(positions, feas_means_y, color=colors[0], linestyle='-', marker = '*',
                 linewidth=3, alpha=0.7, label='Feas Mean Curve')
        ax1.set_ylim(0,None)
        ax1.set_yticks(np.linspace(0, 1.2, 5))  # 从0到0.4均匀分成5个点

        ax1.tick_params(axis='y', labelsize=label_font_size)
        # 下子图均值
        opt_means_y = []
        for seg in parts2['cmeans'].get_segments():
            y_coord = (seg[0][1] + seg[1][1]) / 2
            x_coord = (seg[0][0] + seg[1][0]) / 2

            # ax2.plot(x_coord, y_coord, marker='D', color='black',
            #          markersize=8, markerfacecolor='white', linestyle='None')
            opt_means_y.append(y_coord)
        ax2.plot(positions, opt_means_y, color=colors[1], linestyle='-', marker = '*',
                 linewidth=3, alpha=0.7, label='Opt Mean Curve')

        # 设置x轴刻度
        ax2.set_xticks(positions)
        ax2.set_xticklabels([f'{iter}' for iter in display_iters])
        ax2.set_xlim([-1,19])
        # 设置标签和标题
        # ax1.set_ylabel('Feasibility Error Value', fontsize=label_font_size)
        # ax2.set_ylabel('Optimality Error Value', fontsize=label_font_size)
        # ax2.set_xlabel('Iteration', fontsize=label_font_size)
        # fig.suptitle('Error Distribution Evolution (Violin)', fontsize=14, y=0.95)

        # 添加网格
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(0,0.79)
        ax2.set_yticks(np.linspace(0, 0.6, 4))  # 从0到0.4均匀分成5个点
        ax2.tick_params(axis='y', labelsize=label_font_size)  # 设置x轴刻度标签大小
        ax2.tick_params(axis='x', labelsize=label_font_size)  # 设置x轴刻度标签大小
        # 合并图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=colors[0], lw=4, label='Feasibility error'),
            Line2D([0], [0], color=colors[1], lw=4, label='Optimality error'),
            Line2D([0], [0], color=colors[0], lw=2, linestyle='-', label='Feas mean curve'),
            Line2D([0], [0], color=colors[1], lw=2, linestyle='-', label='Opt mean curve'),
            # Line2D([0], [0], marker='D', color='black', label='Mean',
            #        markersize=8, linestyle='None', markerfacecolor='white')
        ]
        # fig.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
        plt.subplots_adjust(right=0.75)
        # 去除上边框和右边框
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        # 修改网格设置，只显示横向网格，取消纵向网格
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')  # 只显示y轴方向的网格
        ax1.grid(False, axis='x')  # 取消x轴方向的网格
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')  # 只显示y轴方向的网格
        ax2.grid(False, axis='x')  # 取消x轴方向的网格
        if save_path:
            plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        # plt.show()
    def plot_dual_boxplot(self, save_path=None, figsize=(12, 8)):
        """
        精确连接均值点的箱线图版本
        修复内容：
        1. 趋势线准确连接各自箱线图的均值标记
        2. 优化图例和标签显示
        """
        plt.figure(figsize=figsize)

        # 数据准备
        iterations = self.error_history['iterations']
        feas_data = self.error_history['error_feas']
        opt_data = self.error_history['error_opt']

        # 自动调整显示密度
        display_step = max(1, len(iterations) // 10)
        display_indices = range(0, len(iterations), display_step)
        display_iters = [iterations[i] for i in display_indices]

        # 创建绘图位置（每组两个箱体：feas和opt）
        positions = []
        for idx in display_indices:
            positions.extend([idx * 2, idx * 2 + 0.8])  # feas在左，opt在右

        # 合并数据（交替存储feas和opt）
        all_data = []
        for idx in display_indices:
            all_data.append(feas_data[idx])
            all_data.append(opt_data[idx])

        # 配置箱线图属性
        boxprops = dict(linewidth=1.5)
        medianprops = dict(linewidth=2, color='black')
        meanprops = dict(marker='D', markersize=8,
                         markerfacecolor='white',
                         markeredgecolor='black')
        flierprops = dict(marker='o', markersize=5,
                          markerfacecolor='none',
                          markeredgecolor='gray', alpha=0.6)

        # 绘制箱线图
        box = plt.boxplot(
            all_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            showfliers=True,
            boxprops=boxprops,
            medianprops=medianprops,
            meanprops=meanprops,
            flierprops=flierprops,
            whiskerprops=boxprops,
            capprops=boxprops
        )

        # 设置箱体颜色
        colors = ['#3498db', '#e74c3c']
        for i, patch in enumerate(box['boxes']):
            patch.set_facecolor(colors[i % 2])
            patch.set_alpha(0.8)

        # =====================================
        # 关键修复：精确连接各自的均值点
        # =====================================

        # 获取所有均值点的坐标（box['means']返回的Artist对象列表）
        means = box['means']

        # 分离feas和opt的均值点坐标
        feas_means_x = []
        feas_means_y = []
        opt_means_x = []
        opt_means_y = []

        for i, mean in enumerate(means):
            x = mean.get_xdata()[0]  # 均值标记的x坐标
            y = mean.get_ydata()[0]  # 均值标记的y坐标
            if i % 2 == 0:  # 偶数索引是feas
                feas_means_x.append(x)
                feas_means_y.append(y)
            else:  # 奇数索引是opt
                opt_means_x.append(x)
                opt_means_y.append(y)

        # 绘制趋势线（连接各自的实际均值点）
        plt.plot(feas_means_x, feas_means_y,
                 color=colors[0], linestyle='-',
                 linewidth=2, alpha=0.7, label='Feas Mean Curve')

        plt.plot(opt_means_x, opt_means_y,
                 color=colors[1], linestyle='--',
                 linewidth=2, alpha=0.7, label='Opt Mean Curve')

        # =====================================
        # 坐标轴和图例设置
        # =====================================

        # 设置x轴刻度（每组箱线图的中间位置）
        x_ticks = [(feas_means_x[i] + opt_means_x[i]) / 2 for i in range(len(feas_means_x))]
        plt.xticks(
            x_ticks,
            [f' {iter}' for iter in display_iters],
            # rotation=45
        )

        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Error Value', fontsize=12)
        plt.title('Error Distribution Evolution', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')

        # 构建完整图例系统
        from matplotlib.lines import Line2D
        legend_elements = [
            # 误差类型图例
            Line2D([0], [0], color=colors[0], lw=4, label='Feasibility error'),
            Line2D([0], [0], color=colors[1], lw=4, label='Optimality error'),

            # 趋势线图例
            Line2D([0], [0], color=colors[0], lw=2, linestyle='-', label='Feas mean curve'),
            Line2D([0], [0], color=colors[1], lw=2, linestyle='--', label='Opt mean curve'),

            # 统计量图例
            Line2D([0], [0], color='black', lw=2, label='Median'),
            Line2D([0], [0], marker='D', color='black', label='Mean',
                   markersize=8, linestyle='None', markerfacecolor='white'),
            Line2D([0], [0], marker='o', color='gray', label='Outliers',
                   markersize=8, linestyle='None', markerfacecolor='none')
        ]

        plt.legend(handles=legend_elements,
                   loc='upper right',
                   fontsize=12,
                   framealpha=0.9)

        plt.subplots_adjust(right=0.75)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_kde_evolution(self, save_path=None, figsize=(8, 10)):
        """
        修复版KDE可视化方案，解决以下问题：
        1. y轴范围不足导致显示不全
        2. 颜色标尺显示小数迭代次数
        3. 标尺覆盖子图
        4. 为可行性误差添加冲激函数支持
        """
        plt.figure(figsize=figsize)

        # 使用GridSpec精确控制布局
        gs = plt.GridSpec(nrows=3, ncols=1,
                          height_ratios=[1, 1, 0.05],  # 两个子图+颜色条的高度比例
                          hspace=0.5)  # 关键参数：控制子图垂直间距
        # 准备数据
        iterations = np.array(self.error_history['iterations'])
        iterations[0] = 0
        feas_data = self.error_history['error_feas']
        opt_data = self.error_history['error_opt']

        # 颜色映射配置
        cmap = plt.get_cmap('viridis')
        max_iter = len(iterations) - 1
        norm = plt.Normalize(vmin=0, vmax=max_iter)  # 确保归一化到整数迭代

        # 共享的配置参数
        delta_threshold = 1e-2  # 用于判断是否绘制冲激函数的阈值
        spike_marker = '^'  # 冲激函数标记样式
        spike_height_factor = 0.9  # 冲激函数高度比例

        # =====================
        # 子图1：Feasibility Error (添加冲激函数支持)
        # =====================
        ax1 = plt.subplot(gs[0])

        # 动态计算y轴范围（第一遍）
        y_max_values_feas = []
        for errors in feas_data:
            if len(errors) < 2:
                continue
            if np.std(errors) < delta_threshold:
                y_max_values_feas.append(1.0)  # 相对高度基准
            else:
                kde = gaussian_kde(errors)
                x_min, x_max = np.min(errors), np.max(errors)
                x_grid = np.linspace(x_min, x_max, 100)
                y_max_values_feas.append(np.max(kde(x_grid)))

        # 计算全局y轴范围
        global_y_max_feas = max(y_max_values_feas) * 1.2 if y_max_values_feas else 1.0

        # 实际绘制（第二遍）
        for i, errors in enumerate(feas_data):
            if len(errors) < 2:
                continue

            error_std = np.std(errors)
            mean_val = np.mean(errors)

            if error_std < delta_threshold:
                # 绘制冲激标记（高度按比例计算）
                ax1.vlines(mean_val, 0, global_y_max_feas * spike_height_factor,
                           colors=cmap(norm(i)),
                           linewidth=1.5,
                           alpha=0.7)
                ax1.scatter(mean_val, global_y_max_feas * spike_height_factor,
                            color=cmap(norm(i)),
                            marker=spike_marker,
                            s=40,
                            alpha=0.7)
            else:
                kde = gaussian_kde(errors)
                x_min, x_max = np.min(errors), np.max(errors)
                x_grid = np.linspace(x_min, x_max, 100)
                ax1.plot(x_grid, kde(x_grid),
                         color=cmap(norm(i)),
                         alpha=0.7,
                         linewidth=1.5)

        ax1.set_ylim(0, global_y_max_feas)  # 统一设置y轴范围
        ax1.set_xlabel('Feasibility Error Value')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Feasibility Error Distribution Evolution')
        ax1.grid(True, alpha=0.3, linestyle=':')

        # 为第一个子图添加图例说明
        ax1.legend([plt.Line2D([0], [0], color='gray', marker=spike_marker, linestyle='')],
                   ['Concentrated Distribution'],
                   loc='upper right')

        # =====================
        # 子图2：Optimality Error (保持原逻辑)
        # =====================
        ax2 = plt.subplot(gs[1])

        # 动态计算y轴范围
        y_max_values_opt = []
        for errors in opt_data:
            if len(errors) < 2:
                continue
            if np.std(errors) < delta_threshold:
                y_max_values_opt.append(1.0)  # 相对高度基准
            else:
                kde = gaussian_kde(errors)
                x_min, x_max = np.min(errors), np.max(errors)
                x_grid = np.linspace(x_min, x_max, 100)
                y_max_values_opt.append(np.max(kde(x_grid)))

        # 计算全局y轴范围
        global_y_max_opt = max(y_max_values_opt) * 1.2 if y_max_values_opt else 1.0

        # 第二遍：实际绘制
        for i, errors in enumerate(opt_data):
            if len(errors) < 2:
                continue

            error_std = np.std(errors)
            mean_val = np.mean(errors)

            if error_std < delta_threshold:
                # 绘制冲激标记（高度按比例计算）
                ax2.vlines(mean_val, 0, global_y_max_opt * spike_height_factor,
                           colors=cmap(norm(i)),
                           linewidth=1.5,
                           alpha=0.7)
                ax2.scatter(mean_val, global_y_max_opt * spike_height_factor,
                            color=cmap(norm(i)),
                            marker=spike_marker,
                            s=40,
                            alpha=0.7)
            else:
                kde = gaussian_kde(errors)
                x_min, x_max = np.min(errors), np.max(errors)
                x_grid = np.linspace(x_min, x_max, 100)
                ax2.plot(x_grid, kde(x_grid),
                         color=cmap(norm(i)),
                         alpha=0.7,
                         linewidth=1.5)

        ax2.set_ylim(0, global_y_max_opt)  # 统一设置y轴范围
        ax2.set_xlabel('Optimality Error Value')
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Optimality Error Distribution Evolution')
        ax2.grid(True, alpha=0.3, linestyle=':')
        ax2.legend([plt.Line2D([0], [0], color='gray', marker=spike_marker, linestyle='')],
                   ['Concentrated Distribution'],
                   loc='upper right')

        # =====================
        # 颜色标尺（位于底部）
        # =====================
        cax = plt.subplot(gs[2])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        # 设置标尺刻度为整数迭代
        tick_positions = np.linspace(0, max_iter,
                                     # num=min(6, max_iter + 1)
                                     num=max_iter + 1
                                     )
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal',
                            ticks=tick_positions)
        cbar.set_label('Iteration Progress')

        # 将刻度标签转换为实际迭代次数
        tick_labels = [f"{int(iterations[int(pos)])}"
                       for pos in tick_positions]
        cbar.ax.set_xticklabels(tick_labels,
                                # rotation=45,
                                ha='right')

        # 调整整体布局
        plt.subplots_adjust(hspace=0.35)  # 控制子图间距

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_diff_method_violin(self, cube_error, apa_error, save_path=None):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        # Nature风格参数
        rcParams['font.family'] = 'Arial'
        rcParams['font.size'] = 12
        rcParams['axes.linewidth'] = 0.5
        rcParams['xtick.major.width'] = 0.5
        rcParams['ytick.major.width'] = 0.5
        rcParams['xtick.major.size'] = 3
        rcParams['ytick.major.size'] = 3

        org_error = np.zeros(len(apa_error))
        # 三种有数据的方法
        data = [cube_error, org_error, apa_error]
        methods = ['Method 1', 'Method 2', 'Method 3', 'Method 4']  # 共四种方法
        colors = ['#4DBBD5', '#E64B35', '#00A087', '#7E6148']

        fig, ax = plt.subplots(figsize=(3.5, 3))
        positions = np.arange(1, len(methods) + 1)

        # ✅ 只画第1、第2和第4个方法（假设第3个方法无结果）
        valid_positions = [positions[0], positions[2], positions[3]]
        valid_data = [data[0], data[1], data[2]]

        # 绘制小提琴图
        parts = ax.violinplot(valid_data, positions=valid_positions, widths=0.7,
                              showmeans=False, showmedians=False, showextrema=False)

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)

        # 添加散点图（带抖动）
        np.random.seed(42)
        for i, (values, pos) in enumerate(zip(valid_data, valid_positions)):
            jitter = np.random.normal(0, 0.04, size=len(values))
            x = np.full(len(values), pos) + jitter
            ax.scatter(x, values, alpha=0.6, s=8, color=colors[i],
                       edgecolors='black', linewidth=0.15, zorder=3)

        y_min = min([min(d) for d in valid_data])
        y_max = max([max(d) for d in valid_data])
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(-0.05, y_max + margin)

        ax.text(positions[1], (y_min + y_max) / 2, 'N/A',
                ha='center', va='center', fontsize=12,
                color='gray', fontstyle='italic')

        # 坐标轴设置
        ax.set_xticks(positions)
        # 美化
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.3, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg')
        plt.show()
    def plot_diff_method_box(self, cube_error, apa_error, save_path=None):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        # Nature风格参数
        rcParams['font.family'] = 'Arial'
        rcParams['font.size'] = 14
        rcParams['axes.linewidth'] = 0.5
        rcParams['xtick.major.width'] = 0.5
        rcParams['ytick.major.width'] = 0.5
        rcParams['xtick.major.size'] = 3
        rcParams['ytick.major.size'] = 3

        org_error = np.zeros(len(apa_error))
        data = [cube_error, org_error, apa_error]
        methods = ['Method 1', 'Method 2', 'Method 3', 'Method 4']  # 四种方法
        colors = ['#4DBBD5', '#E64B35', '#00A087', '#7E6148']

        fig, ax = plt.subplots(figsize=(5, 3))
        positions = np.arange(1, len(methods) + 1)

        # ✅ 只画第1、第2和第4个方法（假设第3个方法无结果）
        valid_positions = [positions[0], positions[2], positions[3]]
        valid_data = [data[0], data[1], data[2]]
        # 配置箱线图属性
        boxprops = dict(linewidth=1)
        medianprops = dict(linewidth=1, color='black')
        meanprops = dict(marker=None)
        flierprops = dict(marker='o', markersize=5,
                          markerfacecolor='none',
                          markeredgecolor='gray', alpha=0.6)

        # 绘制箱线图
        box = plt.boxplot(
            valid_data,
            positions=valid_positions,
            widths=0.6,
            patch_artist=True,
            showmeans=False,
            showfliers=False,
            boxprops=boxprops,
            medianprops=medianprops,
            meanprops=meanprops,
            flierprops=flierprops,
            whiskerprops=boxprops,
            capprops=boxprops
        )
        # 绘制箱线图
        # box = ax.boxplot(valid_data, positions=valid_positions, widths=0.6,
        #                  patch_artist=True, showcaps=False, boxprops=dict(linewidth=0.6),
        #                  whiskerprops=dict(linewidth=0.6), medianprops=dict(linewidth=0.8, color='black'))

        # 设置颜色
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor('black')

        # 添加散点图（带抖动）
        np.random.seed(42)
        for i, (values, pos) in enumerate(zip(valid_data, valid_positions)):
            jitter = np.random.normal(0, 0.04, size=len(values))
            x = np.full(len(values), pos) + jitter
            ax.scatter(x, values, alpha=0.6, s=8, color=colors[i],
                       edgecolors='black', linewidth=0.15, zorder=3)

        # 添加“N/A”占位
        y_min = min([min(d) for d in valid_data])
        y_max = max([max(d) for d in valid_data])
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(-0.05, y_max + margin)
        # ax.text(positions[1], (y_min + y_max) / 2, '',
        #         ha='center', va='center', fontsize=rcParams['font.size'],
        #         color='gray', fontstyle='italic')

        # 坐标轴设置
        ax.set_xticks(positions)
        # ax.set_xticklabels(methods, rotation=0)

        # 美化
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.3, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg')
        plt.show()
    def plot_constraint_bar(self, save_path=None):
        from matplotlib import rcParams

        # Nature 风格参数
        rcParams['font.family'] = 'Arial'
        rcParams['font.size'] = 14
        rcParams['axes.linewidth'] = 0.5
        rcParams['xtick.major.width'] = 0.5
        rcParams['ytick.major.width'] = 0.5
        rcParams['xtick.major.size'] = 3
        rcParams['ytick.major.size'] = 3

        # 数据
        methods = ['Cube', 'Homothet', 'Original', 'APA']
        constraints = np.array([48, 96, 57696, 96])  # 约束数量
        colors = ['#4DBBD5', '#7E6148', '#E64B35', '#00A087']

        # 绘图
        fig, ax = plt.subplots(figsize=(5, 3))
        x = np.arange(len(methods))

        bars = ax.bar(
            x, constraints, color=colors, edgecolor='black', linewidth=1,
            alpha=0.6, width=0.6, zorder=3
        )

        # 设置 y 轴为对数坐标
        ax.set_yscale('log')
        # ax.set_ylabel('Number of Constraints')

        # 在每个柱顶标注数值
        # for i, (bar, val) in enumerate(zip(bars, constraints)):
        #     if methods[i] == 'Homothet':
        #         # 特殊标注“by design”灰色斜体文字
        #         ax.text(
        #             bar.get_x() + bar.get_width() / 2, val * 1.15,
        #             f'{val}\n(by design)',
        #             ha='center', va='bottom',
        #             fontsize=rcParams['font.size'] - 2,
        #             color='gray', fontstyle='italic'
        #         )
        #     else:
        #         ax.text(
        #             bar.get_x() + bar.get_width() / 2, val * 1.05,
        #             f'{val}',
        #             ha='center', va='bottom',
        #             fontsize=rcParams['font.size'] - 2,
        #             color='black'
        #         )

        # 坐标轴美化
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.3, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg')
        plt.show()

    def plot_error_vs_constraints(self, cube_error, apa_error, save_path=None, background = False):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        from matplotlib.lines import Line2D

        # ========= Nature 风格参数 =========
        rcParams['font.family'] = 'Arial'
        rcParams['font.size'] = 14
        rcParams['axes.linewidth'] = 0.5
        rcParams['xtick.major.width'] = 0.5
        rcParams['ytick.major.width'] = 0.5
        rcParams['xtick.major.size'] = 3
        rcParams['ytick.major.size'] = 3

        # ========= 数据：optimality error & 约束数 =========
        org_error = np.zeros(len(apa_error))

        # 只画有误差数据的三个方法：Cube / Original / APA
        data = [cube_error, org_error, apa_error]
        methods = ['Cube', 'Original', 'APA']
        constraints = np.array([48, 57696, 96])  # 与 methods 对应的 Number of Constraints

        # 颜色：Cube / Original / APA
        colors = ['#4DBBD5',  # Cube
                  '#E64B35',  # Original
                  '#00A087']  # APA

        # 用 log10(约束数) 作为在图上的位置，这样间距类似 log 轴，但本质是线性轴
        # y_pos = np.log10(constraints.astype(float))
        # 用 log10(约束数) 作为在图上的位置（轴本身是线性的，但语义上是 log）
        log_constraints = np.log10(constraints.astype(float))
        y_pos = log_constraints
        fig, ax = plt.subplots(figsize=(5.5, 5))
        # ========= 先确定坐标范围（方便背景和后续一致）=========
        x_min, x_max = -0.1, 2.0
        y_min_plot = y_pos.min() - 0.4
        y_max_plot = y_pos.max() + 0.4

        # ========= 渐变背景：误差↑ + 约束↑ → 颜色变深 =========

        # 在数据坐标系中生成网格
        xx = np.linspace(x_min, x_max, 400)
        yy = np.linspace(y_min_plot, y_max_plot, 400)
        XX, YY = np.meshgrid(xx, yy)  # YY 是“log10(约束数)”的连续值

        # 误差归一化（横轴）
        nx = (XX - x_min) / (x_max - x_min)
        nx = np.clip(nx, 0, 1)

        # 约束归一化（按 log10 归一化，语义上就是对数坐标）
        log_c_min = log_constraints.min()
        log_c_max = log_constraints.max()
        ny = (YY - log_c_min) / (log_c_max - log_c_min)
        ny = np.clip(ny, 0, 1)

        # 圆形/斜向渐变强度：左下亮 → 右上深
        G = np.sqrt(nx ** 2 + ny ** 2)
        # G = np.clip(G, 0, 1)

        if background:
            img = ax.imshow(G,
                      extent=[x_min, x_max, y_min_plot, y_max_plot],
                      origin='lower',
                      cmap='YlGnBu_r',  # 可改成 'Blues' / 'Greens' 等
                      alpha=0.25,  # 渐变透明度
                      zorder=-10,
                      aspect='auto')

            cbar = plt.colorbar(img, ax=ax, orientation='vertical')
            cbar.set_ticks([])  # ← 去掉刻度
            cbar.set_label('Efficiency')

        # ========= 画水平箱线图：横轴为 optimality error，纵轴为 "log10(Number of Constraints)" =========
        boxprops = dict(linewidth=1)
        medianprops = dict(linewidth=1, color='black')
        meanprops = dict(marker=None)
        flierprops = dict(marker='o', markersize=5,
                          markerfacecolor='none',
                          markeredgecolor='gray', alpha=0.6)

        box = ax.boxplot(
            data,
            positions=y_pos,  # 用 y_pos 替代原来的 constraints
            vert=False,  # 水平箱线图
            patch_artist=True,
            showmeans=False,
            showfliers=False,
            widths=0.4,  # 这里是“线性坐标上的高度”，取一个统一的小值即可
            boxprops=boxprops,
            medianprops=medianprops,
            meanprops=meanprops,
            flierprops=flierprops,
            whiskerprops=boxprops,
            capprops=boxprops
        )

        # 设置每个箱线图的颜色
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor('black')

        # ========= 添加散点（带轻微抖动） =========
        np.random.seed(42)
        jitter_scale = 0.03  # y_pos 上的小抖动

        for i, (values, y0) in enumerate(zip(data, y_pos)):
            jitter = np.random.normal(0, jitter_scale, size=len(values))
            y = np.full(len(values), y0) + jitter
            ax.scatter(values, y, alpha=0.6, s=8, color=colors[i],
                       edgecolors='black', linewidth=0.15, zorder=3)

        # ========= 坐标轴设置 =========
        ax.set_xlabel('Optimality error')
        ax.set_ylabel('Number of Constraints')

        # 只保留对应三个点的 ticks：y_pos 对应 48 / 57696 / 96
        ax.set_yticks([])
        # ax.set_yticklabels([str(c) for c in constraints])

        # y 轴范围稍微留点边
        y_min = y_pos.min()
        y_max = y_pos.max()
        ax.set_ylim(y_min - 0.4, y_max + 0.4)
        ax.set_xlim(-0.1, 2)

        # 轴线与网格美化
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.3, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            # ========= 图例（颜色 ↔ 方法） =========
        legend_elements = [
            Line2D([0], [0], color=colors[0], lw=4, label='Cube'),
            Line2D([0], [0], color=colors[1], lw=4, label='Original'),
            Line2D([0], [0], color=colors[2], lw=4, label='APA'),
        ]
        # ax.legend(handles=legend_elements, loc='best', frameon=False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg')
        plt.show()

    def plot_count_discrete(self, save_path=None):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        # Nature风格参数
        rcParams['font.family'] = 'Arial'
        rcParams['font.size'] = 14
        rcParams['axes.linewidth'] = 0.6
        rcParams['xtick.major.width'] = 0.6
        rcParams['ytick.major.width'] = 0.6
        rcParams['xtick.major.size'] = 3
        rcParams['ytick.major.size'] = 3

        # original = [4968, 144, 6000]
        original = [5184, 336, 6291]
        approx = [24, 0, 96]
        colors = ['#E64B35', '#00A087']

        # 设置画布
        fig, ax = plt.subplots(figsize=(5, 5))

        # 坐标与宽度（现在是竖直条形图，因此用 x 作为类别轴）
        x = np.arange(3)
        width = 0.27

        # 竖向条形图（横纵坐标已对调）
        bar1 = ax.bar(x - width / 2, original, width,
                      color=colors[0], alpha=0.3,
                      edgecolor='none', linewidth=1)
        bar2 = ax.bar(x + width / 2, approx, width,
                      color=colors[1], alpha=0.3,
                      edgecolor='none', linewidth=1)

        # 对数坐标（现在用在纵轴）
        ax.set_yscale('log')
        ax.set_ylim(10, 2e4)

        # 类别在 x 轴，把刻度先留空（和你原来 y 轴不显示标签一致）
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg')
        plt.show()


# 使用示例
if __name__ == "__main__":
    # type = 'epigraph'
    # create_gif('figures/'+type, type+'.gif', duration=200, loop=1)
    drawer = ShapeDrawer_2D()
    theta = [1,1,1]  # 圆心 (1,1)，半径1
    drawer.plot_circle_regions(
        theta=theta,
        xlim=(-1.5, 1.5),  # 包含两个圆的可视范围
        ylim=(-1.5, 1.5),
        edgecolor='skyblue',
        facecolor='skyblue',  # 区域填充色
        alpha=0.3,  # 透明度
        title='Circle Regions Demo'
    )

    # 添加辅助元素
    drawer.ax.grid(True)
    drawer.ax.axhline(0, color='black', lw=0.5)
    drawer.ax.axvline(0, color='black', lw=0.5)
    drawer.ax.set_aspect('equal')

    # 显示图形
    plt.show()
