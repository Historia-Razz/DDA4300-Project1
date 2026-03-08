import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

# 引入自定义模块中的函数
from Sampling import barycenter_lp_matrices
from PDLP import solve_barycenter_lp, plot_barycenter


def extract_distributions(dataset, label, n_samples=3, topk=None, verbose=True):
    """
    从数据集中提取指定标签的图像，并转换为 (pts, weights) 格式的支持分布。

    Args:
        dataset: torchvision 数据集对象（如 MNIST 或 Fashion-MNIST）
        label: 要提取的标签（如 3 表示 Dress）
        n_samples: 提取的分布数量
        topk: 若指定，则只使用灰度值最大的 top-k 像素作为支持点（int 或 None）
        verbose: 是否打印调试信息

    Returns:
        distributions: [(pts, weights)] 列表
        img_arrays: 原始图像数组（用于可视化）
    """
    distributions = []
    img_arrays = []
    count = 0

    for idx, (img, lbl) in enumerate(dataset):
        if lbl != label:
            continue

        img_array = np.array(img, dtype=float)
        total_intensity = img_array.sum()

        if total_intensity == 0:
            continue  # 跳过全黑图像

        if topk is not None:
            flat = img_array.ravel()
            nonzero_mask = flat > 0
            if nonzero_mask.sum() < topk:
                continue  # 非零像素不足，跳过

            topk_indices = np.argpartition(flat, -topk)[-topk:]
            rows, cols = np.unravel_index(topk_indices, img_array.shape)
            pts = np.column_stack((cols, rows)).astype(float)
            weights = (img_array[rows, cols] / img_array[rows, cols].sum()).astype(float)
        else:
            rows, cols = np.nonzero(img_array)
            pts = np.column_stack((cols, rows)).astype(float)
            weights = (img_array[rows, cols] / total_intensity).astype(float)

        distributions.append((pts, weights))
        img_arrays.append(img_array)
        count += 1

        if count >= n_samples:
            break

    if verbose:
        print(f"Found {count} valid samples for label {label} (requested {n_samples})")

    if count < n_samples:
        raise ValueError(f"Only {count} valid samples found for label {label}. Try increasing dataset size or lowering filtering threshold.")

    return distributions, img_arrays


# Step 1: 加载 MNIST 数据集并选取指定数字的图像
digit = 3            # 要提取的数字类别，例如 3
n_samples = 2        # 使用的样本图像数量，可增大此值以使用多张图像
# dataset = datasets.MNIST(root='./data', train=True, download=True)  # 下载MNIST训练集
dataset = datasets.FashionMNIST(root='./data', train=True, download=True)
# 注: 如需使用 Fashion-MNIST 数据集，请将上面一行中的 datasets.MNIST 改为 datasets.FashionMNIST。

# 加载 Fashion MNIST
dataset = datasets.FashionMNIST(root='./data', train=True, download=True)

# 提取 n 个 Dress 图像（label=3），使用前 top-150 像素
distributions, img_arrays = extract_distributions(dataset, label=digit, n_samples=n_samples, topk=150)


# 提示: 如果想使用多张图像求重心，只需将 n_samples 设为大于1，以上循环会提取前 n_samples 张标签为digit的图像。
# distributions 列表此时包含每张图像的支持点坐标数组和权重数组，它们将一起用于计算Wasserstein重心。

# Step 2: 构造 Wasserstein Barycenter 的线性规划模型 (A, b, c)
A, b, c, meta = barycenter_lp_matrices(distributions, remove_redundant=True)
# 以上使用 Sampling.py 中的函数，根据输入的分布列表构造线性规划的矩阵表示:
# min c^T x  s.t. A x = b, x >= 0

# Step 3: 调用 OR-Tools PDLP 求解线性规划以获得重心分布
support, bary_weights, transport_matrices = solve_barycenter_lp(
    distributions, A, b, c, meta, solver_name="PDLP", verbose=True
)
# support: 重心支持点坐标 (二维数组)
# bary_weights: 重心对应的权重 (一维数组，和为1)
# transport_matrices: 列表，包含每个输入分布到重心的最优运输矩阵

# Step 4: 可视化结果

# Step 4a. 原始图像可视化（显示所有 n_samples 张图像）
# 多图拼接显示（横向排列）
plt.figure(figsize=(3 * n_samples, 3))
for i, img_array in enumerate(img_arrays):
    plt.subplot(1, n_samples, i + 1)
    plt.imshow(img_array, cmap='gray')
    plt.title(f"#{i+1}")
    plt.axis('off')
plt.suptitle(f"{n_samples} Samples of Digit {digit}")
plt.show()

# 4b. 图像预处理后的二维分布热力图（使用像素灰度作为强度）
plt.figure(figsize=(3 * n_samples, 3))
for i, img_array in enumerate(img_arrays):
    plt.subplot(1, n_samples, i + 1)
    plt.imshow(img_array, cmap='hot')   # 使用热力图颜色显示权重分布
    # 可选：叠加散点以显示支持点位置（非零像素点）
    rows, cols = np.nonzero(img_array)
    plt.scatter(cols, rows, s=10, facecolors='none', edgecolors='cyan', label='Support points')
    plt.title(f"#{i+1}")
    plt.axis('off')
    plt.legend(loc='upper right')
plt.suptitle(f"{n_samples} Samples of Digit {digit}")
plt.show()

# 所有图像的非零支持点位置合并显示
all_rows = []
all_cols = []
for img_array in img_arrays:
    r, c = np.nonzero(img_array)
    all_rows.extend(r)
    all_cols.extend(c)

# 求多个样本图像的平均图像（像素叠加）
sum_img = np.sum(img_arrays, axis=0)     # 多张图像逐像素相加
avg_img = sum_img / len(img_arrays)      # 或者取平均

# 显示平均热力图和所有支持点位置
plt.figure(figsize=(4, 4))
plt.imshow(avg_img, cmap='hot')
plt.scatter(all_cols, all_rows, s=5, facecolors='none', edgecolors='cyan', label='All support points')
plt.title("Averaged Heatmap with All Supports")
plt.axis('off')
plt.legend(loc='upper right')
plt.show()


# 4c. 最终 Wasserstein 重心的二维可视化
# 1. 镜像 support 和 input distributions 的 y 轴：y -> 27 - y
def flip_and_mirror(pts_list):
    return [np.column_stack((pts[:, 0], 27 - pts[:, 1])) for pts in pts_list]
# 使用 PDLP.py 提供的 plot_barycenter 函数，绘制输入分布和重心（包括可选的运输流连线）。
P_locations = [pts for pts, _ in distributions]   # 输入分布的支持点坐标列表
P_weights = [wts for _, wts in distributions]   # 输入分布的权重列表
# 翻转输入分布位置
P_locations_flipped = flip_and_mirror(P_locations)
support_flipped = np.column_stack((support[:, 0], 27 - support[:, 1]))

# 调用绘图函数，使用翻转后的坐标
# 绘制重心；如果输入了多张图像，plot_barycenter 会以不同颜色显示每个输入分布，并以红色显示重心。
plot_barycenter(P_locations_flipped, P_weights, support_flipped, bary_weights,
                Pis=None, flow_thresh=1e-3, cmap="Set2")
# 提示: 若不想显示运输流连线，可将 Pis=None/transport_matrices 或增大 flow_thresh 以只显示较大流量。

# 创建空图像
bary_img = np.zeros((28, 28), dtype=float)

# 将每个支持点（x, y）四舍五入映射到像素格子
for (x, y), w in zip(support, bary_weights):
    ix, iy = int(round(x)), int(round(y))
    if 0 <= iy < 28 and 0 <= ix < 28:
        bary_img[iy, ix] += w

# 归一化（可选）
bary_img /= bary_img.max()

# 可视化
plt.figure(figsize=(4, 4))
plt.imshow(bary_img, cmap='gray')
plt.title("Reconstructed Barycenter Image (as 28x28)")
plt.axis('off')
plt.show()
