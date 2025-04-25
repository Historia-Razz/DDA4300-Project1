import numpy as np
from scipy import sparse
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


def barycenter_lp_matrices(distributions, remove_redundant=True):
    """给定离散分布列表，构造 Wasserstein barycenter 线性规划."""
    # === 预设重心支持点 ===
    # 这里用所有输入分布的并集；也可先做 k‑means 压缩
    """
    Build the (A, b, c) matrices of the pre-specified-support
    Wasserstein barycenter LP in standard form  (min c^T x  s.t.  Ax=b, x>=0).

    Parameters
    ----------
    distributions : list[(pts, a)]
        pts : (m_t, d) ndarray – support of μ_t
        a   : (m_t,)    ndarray – weights, sum(a)=1
    remove_redundant : bool, default True
        If True, drop the (M + k·m + 1)-th rows described in Lemma 3.1
        so that the final matrix has full row-rank.

    Returns
    -------
    A : scipy.sparse.csr_matrix   (n_row, n_col)
    b : (n_row,)  ndarray
    c : (n_col,)  ndarray
    meta : dict   – helpful sizes / index offsets
    """
    # ---------- basic sizes ----------
    support = np.vstack([pts for pts, _ in distributions])
    m = len(support)  # 重心支持点数量
    N = len(distributions)
    m_t = [len(a) for _, a in distributions]     # m_1,…,m_N
    M = sum(m_t)                               # ∑ m_i
    n_col = sum(mi * m for mi in m_t) + m          # |Π| + |w|
    n_row = N * m + M + 1                          # before redundancy removal


    # convenient offsets ----------------------------------------------------
    pi_start = np.zeros(N+1, dtype=int)
    for t in range(N):
        pi_start[t+1] = pi_start[t] + m_t[t]*m       # length of vec(Π^(t))
    w_offset = pi_start[-1]                          # first index of w variables

    # ---------- build c ----------------------------------------------------
    c = np.zeros(n_col)
    for t, (pts_t, _) in enumerate(distributions):
        for i in range(m):
            dists = np.sum((support[i] - pts_t)**2, axis=1)   # (m_t,)
            j_idx = np.arange(m_t[t])
            var_idx = pi_start[t] + i*m_t[t] + j_idx
            c[var_idx] = dists

    # w-part of c is already zero

    # ---------- build sparse A, b ------------------------------------------
    data, rows, cols = [], [], []
    b = np.zeros(n_row)

    row_ptr = 0

    # (1)  Σ_j Π_{ij}^{(t)}  - w_i = 0    ----  N·m rows
    for t in range(N):
        mi = m_t[t]
        for i in range(m):
            # Π block
            j_idx = np.arange(mi)
            cols.extend(pi_start[t] + i*mi + j_idx)
            rows.extend([row_ptr]*mi)
            data.extend([1.0]*mi)
            # -w_i
            cols.append(w_offset + i)
            rows.append(row_ptr)
            data.append(-1.0)
            row_ptr += 1                     # advance to next row

    # (2)  Σ_i Π_{ij}^{(t)} = a_j^(t)       ----  M rows
    for t, (_, a_t) in enumerate(distributions):
        mi = m_t[t]
        for j in range(mi):
            i_idx = np.arange(m)
            cols.extend(pi_start[t] + i_idx*mi + j)
            rows.extend([row_ptr]*m)
            data.extend([1.0]*m)
            b[row_ptr] = a_t[j]
            row_ptr += 1

    # (3)  Σ_i w_i = 1                      ----  1 row
    cols.extend(w_offset + np.arange(m))
    rows.extend([row_ptr]*m)
    data.extend([1.0]*m)
    b[row_ptr] = 1.0
    row_ptr += 1

    # 组装完整约束矩阵
    A_full = sparse.csr_matrix((data, (rows, cols)), shape=(n_row, n_col))

    # ---------- 根据 Lemma 3.1 删除冗余行 ----------
    if remove_redundant:
        M = sum(m_t)  # 质量守恒约束的总行数
        rows_to_remove = [M + t * m for t in range(N)]  # 待删除的行索引
        
        keep = np.ones(n_row, dtype=bool)
        keep[rows_to_remove] = False  # 标记删除行
        
        A = A_full[keep]
        b = b[keep]
    else:
        A = A_full

    meta = dict(N=N, m=m, m_t=m_t, n_row=A.shape[0], n_col=n_col,
                pi_start=pi_start, w_offset=w_offset, support=support)
    return A, b, c, meta


def solve_barycenter_lp(distributions, solver_name="PDLP", verbose=True):
    """求解 Wasserstein barycenter 线性规划问题。
    
    Args:
        distributions: 输入分布列表，每个元素为 (points, weights)
        solver_name: 求解器名称，默认为 "PDLP"
        verbose: 是否打印详细信息
        
    Returns:
        support: 重心支持点坐标
        weights: 重心权重
        transport_matrices: 运输矩阵列表
    """
    # 获取线性规划矩阵
    A, b, c, meta = barycenter_lp_matrices(distributions)
    
    # 创建求解器
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if not solver:
        raise RuntimeError(f"无法创建 {solver_name} 求解器，请确认 OR‑Tools 版本 ≥9.8")
    
    # 设置求解器参数
    pdlp_params = r"""
    termination_criteria {
        eps_optimal_relative: 1e-8
    }
    verbosity_level: 3
    """
    solver.SetSolverSpecificParametersAsString(str(pdlp_params))
    
    # 创建变量
    n_col = A.shape[1]
    x = [solver.NumVar(0.0, solver.infinity(), f"x_{i}") for i in range(n_col)]
    
    # 添加约束 Ax = b
    for i in range(A.shape[0]):
        constraint = solver.Constraint(b[i], b[i])
        for j in range(n_col):
            if A[i,j] != 0:
                constraint.SetCoefficient(x[j], A[i,j])
    
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
        Pi_t = solution[pi_start:pi_start + meta['m']*m_t].reshape(meta['m'], m_t)
        transport_matrices.append(Pi_t)
    
    if verbose:
        print("Optimal objective value :", solver.Objective().Value())
        print("Barycenter weights:", weights)
        print("Weight sum sanity check:", weights.sum())

        top_indices = np.argsort(weights)[:][::-1]
        for i, idx in enumerate(top_indices, 1):
            print(f"第{i}大: 索引 {idx}, 权重值 {weights[idx]:.6f}")
    
    return meta['support'], weights, transport_matrices


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
    distributions = make_gaussian_samples(n_dists=3, n_points=20, seed=42)
    # used seed:
    # 42, 40, 2259， 1605
    support, weights, transport_matrices = solve_barycenter_lp(distributions)

    # 使用两种不同的可视化方法
    print("\n方法1:基础可视化")
    visualize_barycenter(distributions, support, weights, show_density=True, top_k=12)

    #print("\n方法2：带运输流的可视化")
    # 提取位置和权重
    ##P_weights = [w for _, w in distributions]
    #plot_barycenter(P_locations, P_weights, support, weights, Pis=transport_matrices)
