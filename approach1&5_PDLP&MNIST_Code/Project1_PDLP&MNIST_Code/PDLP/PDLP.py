import numpy as np
from ortools.linear_solver import pywraplp
# from ortools.linear_solver import pdlp_pb2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap
from Sampling import *


def solve_barycenter_lp(distributions, A, b, c, meta,
                        solver_name="PDLP",
                        termination_epsilon=1e-6,
                        initial_primal_weight=1.0,
                        verbose=True):
    """求解 Wasserstein barycenter 线性规划问题（支持参数化 PDLP 调参）

    Args:
        distributions: 输入分布列表，每个元素为 (points, weights)
        solver_name: 求解器名称，默认为 "PDLP"
        verbose: 是否打印详细信息

    Returns:
        support: 重心支持点坐标
        weights: 重心权重
        transport_matrices: 运输矩阵列表
    """

    # 创建求解器
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if not solver:
        raise RuntimeError(f"无法创建 {solver_name} 求解器，请确认 OR‑Tools 版本 ≥9.8")

    # === 设置 PDLP 参数字符串（支持 dynamic 实验） ===
    pdlp_params = f"""
    termination_criteria {{
        eps_optimal_relative: {termination_epsilon}
    }}
    initial_primal_weight: {initial_primal_weight}
    verbosity_level: 2
    """
    solver.SetSolverSpecificParametersAsString(pdlp_params)

    # 创建变量
    n_col = A.shape[1]
    x = [solver.NumVar(0.0, solver.infinity(), f"x_{i}") for i in range(n_col)]

    # 添加约束 Ax = b（仅访问非零元素，避免慢速遍历）
    A_csr = A.tocsr()  # 转换为行压缩格式
    for i in range(A_csr.shape[0]):
        constraint = solver.Constraint(b[i], b[i])
        row_start = A_csr.indptr[i]
        row_end = A_csr.indptr[i + 1]
        for idx in range(row_start, row_end):
            j = A_csr.indices[idx]
            val = A_csr.data[idx]
            constraint.SetCoefficient(x[j], val)

    # 设置目标函数
    objective = solver.Objective()
    for j in range(n_col):
        objective.SetCoefficient(x[j], c[j])
    objective.SetMinimization()

    # 求解
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"求解器返回非最优状态: {status}")

    # 提取结果
    solution = np.array([v.solution_value() for v in x])

    # 分离权重和运输矩阵
    w_offset = meta['w_offset']
    weights = solution[w_offset:]

    # 提取运输矩阵
    transport_matrices = []
    for t in range(meta['N']):
        m_t = meta['m_t'][t]
        pi_start = meta['pi_start'][t]
        Pi_t = solution[pi_start:pi_start + meta['m'] * m_t].reshape(meta['m'], m_t)
        transport_matrices.append(Pi_t)
    # 打印结果
    if verbose:
        print("Termination ε:", termination_epsilon)
        print("Initial Primal Weight:", initial_primal_weight)
        print("Optimal objective value :", solver.Objective().Value())
        print("Weight sum:", weights.sum())
    return meta['support'], weights, transport_matrices


def visualize_barycenter(distributions, support, weights,
                         show_density=True, top_k=10):
    """
    Visualize input distributions and the Wasserstein barycenter,
    with optional kernel density background.

    Args:
        distributions: List of input distributions, each as (points, weights)
        support: Coordinates of barycenter support points
        weights: Barycenter weights (array)
        show_density: Whether to add kernel density background
        top_k: Whether to highlight only top-k weighted points
    """
    plt.figure(figsize=(8, 8))

    # Heatmap background (optional)
    if show_density:
        all_points = np.vstack([pts for pts, _ in distributions])
        sns.kdeplot(x=all_points[:, 0], y=all_points[:, 1],
                    cmap="Reds", fill=True, alpha=0.4, levels=100, thresh=0.01)

    # Input distribution points
    for i, (pts, _) in enumerate(distributions):
        plt.scatter(pts[:, 0], pts[:, 1],
                    label=f'Input Distribution {i+1}', s=40, alpha=0.5)

    # All support point locations (black x)
    plt.scatter(support[:, 0], support[:, 1],
                c='black', marker='x', label='Support Points', alpha=0.3)

    # Barycenter weights (blue dots), only show top_k
    if top_k is not None:
        top_indices = np.argsort(weights)[-top_k:]
    else:
        top_indices = np.arange(len(weights))

    plt.scatter(support[top_indices, 0], support[top_indices, 1],
                s=weights[top_indices] * 500, c='blue', alpha=0.6,
                label=f'Barycenter (Top {top_k} Weights)' if top_k else 'Barycenter')

    plt.legend()
    plt.title('Wasserstein Barycenter Visualization' + (' + Density Map' if show_density else ''))
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def plot_barycenter(
    P_locations,
    P_weights,
    X_support,
    w_bary,
    Pis=None,
    flow_thresh=1e-3,
    cmap="Set2",
):
    """
    绘制输入分布 + 重心；若传入 Pis 则同时画运输流(连线).
    参数
    ----
    P_locations, P_weights : list
        输入分布支持点/权重
    X_support              : ndarray (m,2)
        预设重心支持点
    w_bary                 : ndarray (m,)
        求得的重心权重
    Pis                    : list of ndarray, optional
        每个分布的最优运输矩阵，用于画线宽
    flow_thresh            : float
        仅画大于该阈值的 Π_ij
    """
    palette = get_cmap(cmap).colors

    plt.figure(figsize=(7, 7))
    # ① 原始分布
    for t, (locs, wt) in enumerate(zip(P_locations, P_weights)):
        size = wt / wt.max() * 400   # 点大小∝本分布最大权重
        plt.scatter(
            locs[:, 0],
            locs[:, 1],
            s=size,
            color=palette[t % len(palette)],
            alpha=0.7,
            label=f"Input {t+1}",
            edgecolors="k",
        )

    # ② 运输流（细灰线 + 加粗）
    if Pis is not None:
        for t, Pi_t in enumerate(Pis):
            Q_t = P_locations[t]  # 直接使用位置数组
            for i, x_i in enumerate(X_support):
                for j, q_j in enumerate(Q_t):
                    flow = Pi_t[i, j]
                    if flow > flow_thresh:
                        lw = 1 + 8 * flow / Pi_t.max()
                        plt.plot(
                            [x_i[0], q_j[0]],
                            [x_i[1], q_j[1]],
                            color="grey",
                            alpha=0.3,
                            linewidth=lw,
                            zorder=0,
                        )

    # ③ 重心：只画权重大于阈值的支持
    mask = w_bary > (w_bary.max() * 1e-3)
    size = w_bary[mask] / w_bary.max() * 1200
    plt.scatter(
        X_support[mask, 0],
        X_support[mask, 1],
        s=size,
        c="red",
        alpha=0.8,
        marker="o",
        edgecolors="k",
        label="Barycenter",
    )

    plt.title("Wasserstein Barycenter & Transport Plan")
    plt.axis("equal")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dists, A, b, c, meta = load_sample_hdf5("sample_003.h5")
    support_1, weights_1, transport_matrices_1 = solve_barycenter_lp(dists, A, b, c, meta,
                                                                     termination_epsilon=1e-5,
                                                                     initial_primal_weight=0.5)
    support_2, weights_2, transport_matrices_2 = solve_barycenter_lp(dists, A, b, c, meta,
                                                                     termination_epsilon=1e-5,
                                                                     initial_primal_weight=2.0)
    support_3, weights_3, transport_matrices_3 = solve_barycenter_lp(dists, A, b, c, meta,
                                                                     termination_epsilon=1e-3,
                                                                     initial_primal_weight=1.0)
    support_4, weights_4, transport_matrices_4 = solve_barycenter_lp(dists, A, b, c, meta,
                                                                     termination_epsilon=1e-7,
                                                                     initial_primal_weight=1.0)

    # 使用两种不同的可视化方法
    print("\n参数1:initial=0.5, termination_epsilon=1e-5")
    visualize_barycenter(dists, support_1, weights_1, show_density=True, top_k=12)
    print("\n参数2:initial=2.0, termination_epsilon=1e-5")
    visualize_barycenter(dists, support_2, weights_2, show_density=True, top_k=12)
    print("\n参数3:initial=1.0, termination_epsilon=1e-3")
    visualize_barycenter(dists, support_3, weights_3, show_density=True, top_k=12)
    print("\n参数4:initial=1.0, termination_epsilon=1e-7")
    visualize_barycenter(dists, support_4, weights_4, show_density=True, top_k=12)
