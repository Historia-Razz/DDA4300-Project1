import io
import sys
import numpy as np
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap



def make_gaussian_samples(n_dists=3,
                          n_points=30,
                          dim=2,
                          seed=0):
    """生成 N 个高斯分布的(坐标, 权重)."""
    rng = np.random.default_rng(seed)
    dists = []
    for _ in range(n_dists):
        # 随机均值、方差（独立坐标）
        mean = rng.uniform(-5, 5, size=dim)
        std = rng.uniform(0.3, 1.0, size=dim)
        pts = rng.normal(mean, std, size=(n_points, dim))
        # 这里用均匀权重；如需自定义可自行替换
        w = np.full(n_points, 1.0 / n_points)
        dists.append((pts, w))
    return dists


def build_barycenter_lp(distributions, solver_name="PDLP"):
    """给定离散分布列表，构造 Wasserstein barycenter 线性规划."""
    # === 预设重心支持点 ===
    # 这里用所有输入分布的并集；也可先做 k‑means 压缩
    support = np.vstack([pts for pts, _ in distributions])
    m = len(support)  # 重心支持点数量
    N = len(distributions)

    # === 创建求解器 ===
    pdlp_params = r"""
    termination_criteria {
        eps_optimal_relative: 1e-8
    }
    verbosity_level: 3
    """

    solver = pywraplp.Solver.CreateSolver(solver_name)
    solver.SetSolverSpecificParametersAsString(str(pdlp_params))
    if not solver:
        raise RuntimeError(f"无法创建 {solver_name} 求解器，请确认 OR‑Tools 版本 ≥9.8")

    # === 变量 ===
    # 重心权重 w_i
    w = [solver.NumVar(0.0, 1.0, f"w_{i}") for i in range(m)]

    # 运输矩阵 Π^(t)  展平成一维列表保存变量引用
    pi_vars = []
    for t, (pts_t, a_t) in enumerate(distributions):
        m_t = len(pts_t)
        block = []
        for i in range(m):
            row = []
            for j in range(m_t):
                var = solver.NumVar(0.0, solver.infinity(),
                                   f"pi_{t}_{i}_{j}")
                row.append(var)
            block.append(row)
        pi_vars.append(block)

    # === 约束 ===
    # 1) 对每个 t，每个重心支持 i：   Σ_j Π_{ij}^{(t)} = w_i
    for i in range(m):
        for t, (pts_t, _) in enumerate(distributions):
            ct = solver.Constraint(0.0, 0.0)
            for j in range(len(pts_t)):
                ct.SetCoefficient(pi_vars[t][i][j], 1.0)
            ct.SetCoefficient(w[i], -1.0)

    # 2) 对每个 t，每个原分布点 j： Σ_i Π_{ij}^{(t)} = a_j^(t)
    for t, (pts_t, a_t) in enumerate(distributions):
        m_t = len(pts_t)
        for j in range(m_t):
            ct = solver.Constraint(a_t[j], a_t[j])
            for i in range(m):
                ct.SetCoefficient(pi_vars[t][i][j], 1.0)

    # 3) 权重和为 1
    ct = solver.Constraint(1.0, 1.0)
    for i in range(m):
        ct.SetCoefficient(w[i], 1.0)

    # === 目标函数 ===
    objective = solver.Objective()
    for t, (pts_t, _) in enumerate(distributions):
        for i in range(m):
            for j in range(len(pts_t)):
                # 这里用二次欧氏距离 (p=2)
                cost = np.sum((support[i] - pts_t[j]) ** 2)
                objective.SetCoefficient(pi_vars[t][i][j], cost)
    objective.SetMinimization()

    return solver, w, support


def solve_barycenter_lp(distributions, verbose=True):
    solver, w_vars, support = build_barycenter_lp(distributions)

    # 若想打印更多 PDLP 进度，可打开日志
    solver.EnableOutput()

    status = solver.Solve()

    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"求解器返回非最优状态: {status}")

    weights = np.array([v.solution_value() for v in w_vars])
    
    # 提取运输矩阵
    transport_matrices = []
    for t, (pts_t, _) in enumerate(distributions):
        m_t = len(pts_t)
        Pi_t = np.zeros((len(support), m_t))
        for i in range(len(support)):
            for j in range(m_t):
                var_name = f"pi_{t}_{i}_{j}"
                var = solver.LookupVariable(var_name)
                if var is not None:
                    Pi_t[i, j] = var.solution_value()
        transport_matrices.append(Pi_t)

    if verbose:
        print("Optimal objective value :", solver.Objective().Value())
        print("Barycenter weights:", weights)
        print("Weight sum sanity check   :", weights.sum())

        top_indices = np.argsort(weights)[:][::-1]
        for i, idx in enumerate(top_indices, 1):
            print(f"第{i}大: 索引 {idx}, 权重值 {weights[idx]:.6f}")

    return support, weights, transport_matrices


def visualize_barycenter(distributions, support, weights, 
                         show_density=True, top_k=10):
    """
    可视化输入分布和 Wasserstein 重心，支持密度图叠加。

    Args:
        distributions: 输入分布列表，每个元素为 (points, weights)
        support: 重心支持点坐标
        weights: 重心权重（数组）
        show_density: 是否添加 kernel density 背景热力图
        top_k: 是否只高亮 top 权重点
    """
    plt.figure(figsize=(8, 8))

    # 热力图背景（可选）
    if show_density:
        all_points = np.vstack([pts for pts, _ in distributions])
        sns.kdeplot(x=all_points[:, 0], y=all_points[:, 1],
                    cmap="Reds", fill=True, alpha=0.4, levels=100, thresh=0.01)

    # 原始分布点
    for i, (pts, _) in enumerate(distributions):
        plt.scatter(pts[:, 0], pts[:, 1], 
                    label=f'输入分布 {i+1}', s=40, alpha=0.5)

    # 所有支持点位置（黑色 x）
    plt.scatter(support[:, 0], support[:, 1],
                c='black', marker='x', label='支持点', alpha=0.3)

    # 重心权重（蓝色点），只显示 top_k 个
    if top_k is not None:
        top_indices = np.argsort(weights)[-top_k:]
    else:
        top_indices = np.arange(len(weights))
    
    plt.scatter(support[top_indices, 0], support[top_indices, 1],
                s=weights[top_indices] * 500, c='blue', alpha=0.6,
                label=f'重心（前 {top_k} 权重）' if top_k else '重心')

    plt.legend()
    plt.title('Wasserstein 重心可视化' + (' + 密度图' if show_density else ''))
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
    N = len(P_locations)
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
    # 例子：3 个高斯分布，每个 50 点
    distributions = make_gaussian_samples(n_dists=3, n_points=50, seed=42)
    # used seed:
    # 42, 40, 2259， 1605
    support, weights, transport_matrices= solve_barycenter_lp(distributions)
    
    # 使用两种不同的可视化方法
    print("\n方法1:基础可视化")
    visualize_barycenter(distributions, support, weights, show_density=True, top_k=12)
    
