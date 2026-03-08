import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

# Import functions from custom modules
from Sampling import barycenter_lp_matrices
from PDLP import solve_barycenter_lp, plot_barycenter


def extract_distributions(dataset, label, n_samples=3, topk=None, verbose=True):
    """
    Extracts images with a given label from a dataset and converts them into (pts, weights) format.

    Args:
        dataset: A torchvision dataset object (e.g., MNIST or Fashion-MNIST).
        label: The label to extract (e.g., 3 for Dress).
        n_samples: Number of distributions to extract.
        topk: If specified, only use the top-k pixels with the highest grayscale values.
        verbose: Whether to print debug information.

    Returns:
        distributions: A list of tuples (pts, weights).
        img_arrays: Original image arrays for visualization.
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
            continue  # Skip fully black images

        if topk is not None:
            flat = img_array.ravel()
            nonzero_mask = flat > 0
            if nonzero_mask.sum() < topk:
                continue  # Skip if not enough non-zero pixels

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
        raise ValueError(f"Only {count} valid samples found for label {label}. "
                         f"Try increasing dataset size or lowering filtering threshold.")

    return distributions, img_arrays


# Step 1: Load dataset and extract images of the specified label
digit = 3            # Label to extract, e.g., 3
n_samples = 2        # Number of images to use
# dataset = datasets.MNIST(root='./data', train=True, download=True)  # Use MNIST
dataset = datasets.FashionMNIST(root='./data', train=True, download=True)
# Note: For Fashion-MNIST, use datasets.FashionMNIST instead of datasets.MNIST.

# Extract Dress images (label = 3) using top-150 pixels
distributions, img_arrays = extract_distributions(dataset, label=digit, n_samples=n_samples, topk=150)

# Note: To use more images, just increase n_samples.
# distributions contains [(pts, weights)] for each image, used in Wasserstein barycenter.

# Step 2: Construct the linear program for Wasserstein barycenter
A, b, c, meta = barycenter_lp_matrices(distributions, remove_redundant=True)
# This constructs the LP matrices: min c^T x  s.t. A x = b, x >= 0

# Step 3: Solve the LP using PDLP to obtain the barycenter
support, bary_weights, transport_matrices = solve_barycenter_lp(
    distributions, A, b, c, meta, solver_name="PDLP", verbose=True
)
# support: Coordinates of barycenter support points
# bary_weights: Barycenter weights (sum to 1)
# transport_matrices: Optimal transport matrices from each input distribution

# Step 4: Visualization

# Step 4a. Display original image arrays
plt.figure(figsize=(3 * n_samples, 3))
for i, img_array in enumerate(img_arrays):
    plt.subplot(1, n_samples, i + 1)
    plt.imshow(img_array, cmap='gray')
    plt.title(f"#{i+1}")
    plt.axis('off')
plt.suptitle(f"{n_samples} Samples of Digit {digit}")
plt.show()

# Step 4b. Heatmap visualization of pixel intensities and support points
plt.figure(figsize=(3 * n_samples, 3))
for i, img_array in enumerate(img_arrays):
    plt.subplot(1, n_samples, i + 1)
    plt.imshow(img_array, cmap='hot')   # Use heatmap colormap
    rows, cols = np.nonzero(img_array)
    plt.scatter(cols, rows, s=10, facecolors='none', edgecolors='cyan', label='Support points')
    plt.title(f"#{i+1}")
    plt.axis('off')
    plt.legend(loc='upper right')
plt.suptitle(f"{n_samples} Samples of Digit {digit}")
plt.show()

# Merge support point locations across all images
all_rows = []
all_cols = []
for img_array in img_arrays:
    r, c = np.nonzero(img_array)
    all_rows.extend(r)
    all_cols.extend(c)

# Compute average image by summing and normalizing
sum_img = np.sum(img_arrays, axis=0)
avg_img = sum_img / len(img_arrays)

# Display averaged heatmap with all support points
plt.figure(figsize=(4, 4))
plt.imshow(avg_img, cmap='hot')
plt.scatter(all_cols, all_rows, s=5, facecolors='none', edgecolors='cyan', label='All support points')
plt.title("Averaged Heatmap with All Supports")
plt.axis('off')
plt.legend(loc='upper right')
plt.show()


# Step 4c. 2D visualization of final Wasserstein barycenter
# Flip the y-axis for visualization: y -> 27 - y
def flip_and_mirror(pts_list):
    return [np.column_stack((pts[:, 0], 27 - pts[:, 1])) for pts in pts_list]

# Prepare data for plotting
P_locations = [pts for pts, _ in distributions]   # Support points of input distributions
P_weights = [wts for _, wts in distributions]     # Weights of input distributions
P_locations_flipped = flip_and_mirror(P_locations)
support_flipped = np.column_stack((support[:, 0], 27 - support[:, 1]))

# Plot barycenter with flipped coordinates
plot_barycenter(P_locations_flipped, P_weights, support_flipped, bary_weights,
                Pis=None, flow_thresh=1e-3, cmap="Set2")
# Tip: Set Pis=None or increase flow_thresh to hide small transport lines

# Reconstruct barycenter image from support points
bary_img = np.zeros((28, 28), dtype=float)
for (x, y), w in zip(support, bary_weights):
    ix, iy = int(round(x)), int(round(y))
    if 0 <= iy < 28 and 0 <= ix < 28:
        bary_img[iy, ix] += w

# Normalize image (optional)
bary_img /= bary_img.max()

# Display reconstructed barycenter image
plt.figure(figsize=(4, 4))
plt.imshow(bary_img, cmap='gray')
plt.title("Reconstructed Barycenter Image (as 28x28)")
plt.axis('off')
plt.show()
