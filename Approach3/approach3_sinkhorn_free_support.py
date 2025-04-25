import numpy as np
import ot
import matplotlib.pyplot as plt

def project_simplex(x):
    """Project Simplex

    Projects an arbitrary vector :math:`\mathbf{x}` into the probability simplex, such that,

    .. math:: \tilde{\mathbf{x}}_{i} = \dfrac{\mathbf{x}_{i}}{\sum_{j=1}^{n}\mathbf{x}_{j}}

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Numpy array of shape (n,)

    Returns
    -------
    y : :class:`numpy.ndarray`
        numpy array lying on the probability simplex of shape (n,)
    """
    x_sorted = np.sort(x)[::-1]
    cumsum = np.cumsum(x_sorted)
    k = np.where(x_sorted > (cumsum - 1) / (np.arange(len(x)) + 1))[0][-1]
    theta = (cumsum[k] - 1) / (k + 1)
    return np.maximum(x - theta, 0)

def fixed_support_barycenter(B_list, M_list, weights=None, eta=10, numItermax=100, stopThr=1e-9, verbose=False):
    """Fixed Support Wasserstein Barycenter

    We follow the Algorithm 1. of [1], into calculating the Wasserstein barycenter of N measures over a pre-defined
    grid :math:`\mathbf{X}`. These measures, of course, have variable sample weights :math:`\mathbf{b}_{i}`.
    
    Parameters
    ----------
    B : :class:`numpy.ndarray`
        Numpy array of shape (N, d), for N histograms, and d dimensions.
    M : :class:`numpy.ndarray`
        Numpy array of shape (d, d), containing the pairwise distances for the support of B
    weights : :class:`numpy.ndarray`
        Numpy array or None. If None, weights are uniform. Otherwise, weights each measure in the barycenter
    eta : float
        Mirror descent step size
    numItermax : integer
        Maximum number of descent steps
    stopThr : float
        Threshold for stopping mirror descent iterations
    verbose : bool
        If true, display information about each descent step

    Returns
    -------
    a : :class:`numpy.ndarray`
        Array of shape (d,) containing the barycenter of the N measures.
    """
    a = ot.unif(M_list[0].shape[0])
    a_prev = a.copy()
    weights = ot.unif(len(B_list)) if weights is None else weights

    for k in range(numItermax):
        potentials = []
        for i in range(len(B_list)):
            _, ret = ot.emd(a, B_list[i], M_list[i], log=True)
            potentials.append(ret['u'])
        
        # Calculates the gradient
        f_star = sum(potentials) / len(potentials)

        # Mirror Descent
        a = a * np.exp(- eta * f_star)

        # Projection
        a = project_simplex(a)

        # Calculate change in a
        da = sum(np.abs(a - a_prev))
        if da < stopThr: return a
        if verbose: print('[{}, {}] |da|: {}'.format(k, numItermax, da))

        # Update previous a
        a_prev = a.copy()
    return a

def free_support_barycenter(Y_list, b_list, X_init, a_init, numItermax=100, stopThr=1e-9, eta=0.1, theta=0.5, verbose=False):
    """Free Support Wasserstein Barycenter with Algorithm 2

    We follow the Algorithm 2. of [1], into calculating the Wasserstein barycenter of N measures with free support set.

    Parameters
    ----------
    Y_list : list of :class:`numpy.ndarray`
        List of numpy arrays of shape (d_i, ), for N supports.
    b_list : list of :class:`numpy.ndarray`
        List of numpy arrays of shape (d_i, ), for N histograms.
    X_init : :class:`numpy.ndarray`
        Initial support set of shape (d, )
    a_init : :class:`numpy.ndarray`
        Initial weights of shape (d, )
    numItermax : integer
        Maximum number of iterations
    stopThr : float
        Threshold for stopping iterations
    eta : float
        Mirror descent step size
    theta : float
        Update step size for support set
    verbose : bool
        If true, display information about each iteration

    Returns
    -------
    X : :class:`numpy.ndarray`
        Final support set of shape (d, )
    a : :class:`numpy.ndarray`
        Final weights of shape (d, )
    """
    X = X_init.copy()
    a = a_init.copy()

    for k in range(numItermax):
        X_prev = X.copy()
        a_prev = a.copy()
        # Step 1: Optimize weights with fixed support set
        # 计算每个点之间的距离矩阵
        # Step 2: Update support set with fixed weights
        transports = []

        # 计算当前支持集X和Y_list[i]之间的距离矩阵
        M_list = [ot.dist(X, Y, metric='euclidean') for Y in Y_list]
        a = fixed_support_barycenter(b_list, M_list, eta=eta, numItermax=100, stopThr=1e-5)
        for i in range(len(Y_list)):
            # 计算最优传输矩阵
            T_i = ot.emd(a, b_list[i], M_list[i])
            transports.append(T_i)

        # Update support set using Newton step
        # 计算Y_i T_i^T 的平均值
        Y_sum = np.zeros_like(X)
        for i in range(len(Y_list)):
            Y_sum += transports[i] @ Y_list[i]
        Y_avg = Y_sum / len(Y_list)

        # Update X
        X = (1 - theta) * X + theta *np.diag(1.0 / a) @ Y_avg 

        # Calculate change in X and a
        dX = np.linalg.norm(X - X_prev)
        da = np.linalg.norm(a - a_prev)

        if verbose:
            print('[{}, {}] |dX|: {}, |da|: {}'.format(k, numItermax, dX, da))

        if dX < stopThr and da < stopThr:
            break

        # Update previous X and a
        X_prev = X.copy()
        a_prev = a.copy()

    return X, a

# Example usage:
'''
# Y_list contains the support sets of the input measures
Y1 = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
b1 = np.array([0.4, 0.3, 0.2, 0.1])

Y2 = np.array([[0.5, 0.5], [0.6, 0.6], [0.7, 0.7], [0.8, 0.8]])
b2 = np.array([0.1, 0.2, 0.3, 0.4])

Y3 = np.array([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]])
b3 = np.array([0.25, 0.25, 0.25, 0.25])

Y_list = [Y1, Y2, Y3]
b_list = [b1, b2, b3]

# Initialize X and a
X_init = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
a_init = np.array([0.33, 0.33, 0.34])

# Compute Wasserstein Barycenter
X, a = free_support_barycenter(Y_list, b_list, X_init, a_init, numItermax=1000, stopThr=1e-6, theta=0.5, verbose=True)
print("Final support set X:")
print(X)
print(f"Final weights a:{a}")
# Plotting
plt.figure(figsize=(12, 10))

# Plot input support sets Y_list
for i, Y in enumerate(Y_list):
    b = b_list[i]
    plt.scatter(Y[:, 0], Y[:, 1], s=b * 1000, label=f'Input Support Y_{i+1}', alpha=0.6)

# Plot final support set X
sizes = a * 1000
sc = plt.scatter(X[:, 0], X[:, 1], s=sizes, c=a, cmap='viridis', label='Wasserstein Barycenter', alpha=0.8, edgecolors='k')

# Add a color bar to indicate the weights
cbar = plt.colorbar(sc)
cbar.set_label('Weights')

# Set plot labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Wasserstein Barycenter of Three Complex Distributions')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
'''
# 随机生成四个二元正态分布的样本
np.random.seed(10)  # 设置随机种子以确保结果可重复

# 定义四个不同的二元正态分布
distributions = []
for _ in range(10):
    mean = np.random.uniform(-5, 5, size=2)
    cov = np.random.randint(1, 5)*np.array([[1.0, np.random.uniform(-0.9, 0.9)], [np.random.uniform(-0.9, 0.9), 1.0]])
    #cov = np.random.uniform(1, 5, size=(2, 2))
    #cov = np.dot(cov, cov.T)  # 确保协方差矩阵是正定的
    n_samples = np.random.randint(50, 80)
    Y = np.random.multivariate_normal(mean, cov, size=n_samples)
    #b = np.ones(n_samples) / n_samples  # 权重均匀分布
    b = np.random.dirichlet(np.ones(n_samples), size=1)[0]  # 权重服从Dirichlet分布
    distributions.append((Y, b))

Y_list, b_list = zip(*distributions)
# 初始化支持集和权重
X_init = np.array([[0.0, 0.0], [2.0, 2.0], [-2.0, -2.0], [2.0, -2.0], 
                   [-2.0, 2.0], [0.0, 2.0], [0.0, -2.0], [2.0, 0.0], 
                   [-2.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], 
                   [-1.0, 1.0], [1.0, -2.0], [-1.0, 2.0], [1.0, 2.0], 
                   [-1.0, -2.0], [2.0, 1.0], [-2.0, -1.0], [2.0, -1.0]])
a_init = np.array([0.05] * 20)

# 计算 Wasserstein Barycenter
X, a = free_support_barycenter(Y_list, b_list, X_init, a_init, numItermax=100, stopThr=1e-6, theta=0.2, verbose=True)
print("Final support set X:")
print(X)
print(f"Final weights a:{a}")
# 绘制结果
plt.figure(figsize=(12, 10))

# 绘制输入支持集
for i, Y in enumerate(Y_list):
    b = b_list[i]
    plt.scatter(Y[:, 0], Y[:, 1], s=b * 1000, label=f'Input Distribution {i+1}', alpha=0.7)

# 绘制最终支持集
sizes = a * 1000
sc = plt.scatter(X[:, 0], X[:, 1], s=sizes, c=a, cmap='viridis', label='Wasserstein Barycenter', alpha=1.0, edgecolors='k', marker='*')

# 添加颜色条表示权重
cbar = plt.colorbar(sc)
cbar.set_label('Weights')

# 设置图表标签和标题
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Wasserstein Barycenter of Four Random Bivariate Normal Distributions')
plt.legend()
plt.grid(True)

# 显示图表
plt.show()