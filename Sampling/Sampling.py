import numpy as np
from scipy import sparse
import h5py
import os


def make_gaussian_samples(N_dists_range=(3, 10),
                          n_points_range=(20, 50),
                          dim=2,
                          seed=0, uniform_weights=True):
    """
    Generate a list of N Gaussian distributions, each with a set of support points and weights.

    Parameters:
    -----------
    N_dists_range : tuple of int
        Range for the number of distributions to generate (inclusive on the lower bound, exclusive on the upper bound + 1).
    n_points_range : tuple of int
        Range for the number of points in each distribution.
    dim : int
        Dimensionality of each point.
    seed : int
        Random seed for reproducibility.
    uniform_weights : bool
        Whether to assign uniform weights to each distribution's points.

    Returns:
    --------
    dists : list of tuples
        Each tuple contains:
        - pts: ndarray of shape (m_t, dim), coordinates of support points
        - w: ndarray of shape (m_t,), weights summing to 1
    """
    rng = np.random.default_rng(seed)  # Initialize random number generator
    dists = []

    # Randomly determine the number of distributions to generate
    N_dists = rng.integers(N_dists_range[0], N_dists_range[1] + 1)

    for _ in range(N_dists):
        # Randomly choose the number of points for this distribution
        m_t = rng.integers(n_points_range[0], n_points_range[1] + 1)

        # Randomly generate a mean and std deviation per coordinate
        mean = rng.uniform(-5, 5, size=dim)
        std = rng.uniform(0.3, 1.0, size=dim)

        # Sample points from a multivariate normal distribution
        pts = rng.normal(loc=mean, scale=std, size=(m_t, dim))

        # Assign weights
        if uniform_weights:
            # Equal weight for all points
            w = np.full(m_t, 1.0 / m_t)
        else:
            # Random weights normalized to sum to 1
            w = rng.uniform(0.1, 1.0, size=m_t)
            w /= w.sum()

        # Append the distribution to the list
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
        pts : (m_t, d) ndarray - support of μ_t
        a   : (m_t,)    ndarray - weights, sum(a)=1
    remove_redundant : bool, default True
        If True, drop the (M + k·m + 1)-th rows described in Lemma 3.1
        so that the final matrix has full row-rank.

    Returns
    -------
    A : scipy.sparse.csr_matrix   (n_row, n_col)
    b : (n_row,)  ndarray
    c : (n_col,)  ndarray
    meta : dict   - helpful sizes / index offsets
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


# Save sample data to an HDF5 file
def save_sample_hdf5(filename, distributions, A, b, c, meta):
    """
    Save the distribution data, LP matrices, and metadata to an HDF5 file.

    Parameters:
    -----------
    filename : str
        Name of the HDF5 file (under 'sample/' folder).
    distributions : list of tuples
        Each tuple contains (points, weights) for a distribution.
    A : scipy.sparse matrix
        Constraint matrix in sparse CSR format.
    b : ndarray
        Constraint bounds.
    c : ndarray
        Objective coefficients.
    meta : dict
        Additional metadata (e.g., sample ID, generation info, etc.).
    """
    # Ensure the output folder exists
    os.makedirs('sample', exist_ok=True)

    # Full file path under the 'sample' directory
    filepath = os.path.join('sample', filename)

    with h5py.File(filepath, 'w') as f:
        # === Save distributions ===
        grp_dist = f.create_group("distribution")
        for idx, (pts, wts) in enumerate(distributions):
            grp_dist.create_dataset(f"points_{idx}", data=pts)
            grp_dist.create_dataset(f"weights_{idx}", data=wts)

        # === Save LP matrices (sparse A matrix stored in CSR format) ===
        grp_lp = f.create_group("lp")
        A_csr = A.tocsr()
        grp_lp.create_dataset("A_data", data=A_csr.data)
        grp_lp.create_dataset("A_indices", data=A_csr.indices)
        grp_lp.create_dataset("A_indptr", data=A_csr.indptr)
        grp_lp.create_dataset("A_shape", data=A_csr.shape)
        grp_lp.create_dataset("b", data=b)
        grp_lp.create_dataset("c", data=c)

        # === Save metadata ===
        grp_meta = f.create_group("meta")
        for k, v in meta.items():
            grp_meta.create_dataset(k, data=v)


# Load sample data from an HDF5 file
def load_sample_hdf5(filename):
    """
    Load distribution data, LP matrices, and metadata from an HDF5 file.

    Parameters:
    -----------
    filename : str
        Name of the HDF5 file to load (under 'sample/' folder).

    Returns:
    --------
    distributions : list of tuples
        List of (points, weights) for each distribution.
    A : scipy.sparse.csr_matrix
        Loaded sparse constraint matrix A.
    b : ndarray
        Constraint bounds vector.
    c : ndarray
        Objective coefficients vector.
    meta : dict
        Loaded metadata dictionary.
    """
    filepath = os.path.join('sample', filename)

    with h5py.File(filepath, 'r') as f:
        # === Load distributions ===
        dists = []
        i = 0
        while f"distribution/points_{i}" in f:
            pts = f[f"distribution/points_{i}"][:]
            wts = f[f"distribution/weights_{i}"][:]
            dists.append((pts, wts))
            i += 1

        # === Load LP matrices ===
        lp_grp = f["lp"]
        A = sparse.csr_matrix((
            lp_grp["A_data"][:],
            lp_grp["A_indices"][:],
            lp_grp["A_indptr"][:]
        ), shape=tuple(lp_grp["A_shape"][:]))

        b = lp_grp["b"][:]
        c = lp_grp["c"][:]

        # === Load metadata ===
        meta = {k: v[()] for k, v in f["meta"].items()}

    return dists, A, b, c, meta

'''
目前已生成和存储以下样本
Gaussian_distributions = make_gaussian_samples(n_dists=3, n_points=20, seed=42)
A, b, c, meta = barycenter_lp_matrices(Gaussian_distributions)
save_sample_hdf5("sample_000.h5", Gaussian_distributions, A, b, c, meta)
Gaussian_distributions = make_gaussian_samples(n_dists=3, n_points=50, seed=42)
A, b, c, meta = barycenter_lp_matrices(Gaussian_distributions)
save_sample_hdf5("sample_001.h5", Gaussian_distributions, A, b, c, meta)
'''

'''
Gaussian_distributions = make_gaussian_samples(n_dists=3, n_points=20, seed=42)
A, b, c, meta = barycenter_lp_matrices(Gaussian_distributions)

# 输出结果
print("A shape:", A.shape)  # (n_row_kept, n_col)
print(A)
print("b shape:", b.shape)  # (n_row_kept,)
print("c shape:", c.shape)  # (n_col,)
print("x size:", meta["n_col"])  # ∑(m_t * m) + m
'''