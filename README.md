# DDA4300-Project1: Computing Wasserstein Barycenter via Linear Programming

> **Course:** DDA4300 - Optimization Methods in Machine Learning
>
> **Institution:** CUHKSZ
>
> **Topic:** Wasserstein Barycenter Computation

---

## Project Background

The Wasserstein barycenter problem is a fundamental challenge in optimal transport theory with wide-ranging applications in:

- **Image Processing:** Averaging multiple images while preserving geometric structure
- **Machine Learning:** Data aggregation, clustering, and distribution matching
- **Statistics:** Robust averaging of probability distributions
- **Computer Vision:** Shape averaging and morphological analysis

### Problem Statement

Given $N$ discrete probability measures $\mu_1, \dots, \mu_N$ (discrete probability distributions) with weights $\lambda_1, \dots, \lambda_N$ (where $\sum_t \lambda_t = 1$), the **Wasserstein barycenter** $\mu$ solves:

$$
\min_\mu \sum_{t=1}^N \lambda_t W_2^2(\mu, \mu_t)
$$

where $W_2$ is the 2-Wasserstein distance induced by a ground cost function $c(x,y) = \|x-y\|^2$.

### Project Requirements

This project explores three main algorithmic approaches for computing Wasserstein barycenters:

1. **Pre-specified Support:** The barycenter support is predetermined (union of input supports)
2. **Free Support:** The barycenter support is optimized alongside weights
3. **Fixed Support:** The barycenter support is fixed to a given set of points

Each approach presents unique computational challenges and trade-offs in terms of:
- **Accuracy:** Exact LP solution vs. regularized approximation
- **Speed:** First-order vs. second-order methods
- **Scalability:** Performance with increasing problem dimensions

---

## Methodologies Implemented

### Approach 1 & 5: Primal-Dual Linear Programming (PDLP) + MNIST

**Pre-specified Support Setting**

- **Mathematical Modeling:** Transformed the barycenter problem into a standard linear programming form
- **Algorithm:** Implemented PDLP (Primal-Dual Hybrid Gradient) solver using Google's OR-Tools
- **Key Features:**
  - Modular sampling module with metadata storage
  - Visualizable convergence metrics (heatmap supported)
  - Warm-start capabilities
  - Termination criteria: `rel_prim_res`, `rel_dual_res`, `rel_gap` < tolerance
- **MNIST Application:** Applied to digit image barycenter computation

**Mathematical Formulation:**

$$
\min_{\pi^{(1)},\dots,\pi^{(N)}, w} \sum_{t=1}^N \langle \pi^{(t)}, C^{(t)} \rangle \quad
\text{s.t. } \pi^{(t)} \mathbf{1} = a^{(t)},\ (\pi^{(t)})^\top \mathbf{1} = w,\ \sum w_i = 1
$$

**Location:** `approach1&5_PDLP&MNIST_Code/`

### Approach 2 & 4: Interior-Point Methods (IPM)

**Fixed Support Setting with Advanced IPM Variants**

- **Methods Implemented:**
  - **MAAIPM** (Modified Augmented Algorithmic Interior Point Method): Uses Bregman distance and preconditioning
  - **SLRM** (Symmetric Low-Rank Method): Exploits matrix structure for efficiency
  - **DLRM** (Dual Low-Rank Method): Works in dual space with rank-based optimization
  - **Vanilla IPM:** Standard interior-point method as baseline
  - **Sinkhorn:** Entropy-regularized for comparison

- **Experimental Design:**
  - Systematic evaluation on synthetic data
  - Varying parameters: $N \in \{3,4,6,8,10,12,15\}$, $m_t \in \{5,10,20,40,60,80,100\}$
  - Custom `.d2` format for efficient data storage

- **Data Structure (`.d2` format):**
  - Stores dimension $d$ and support points count $m$ for each distribution
  - Contains support point coordinates and normalized weights

**Location:** `approach2&4_code&data/`

### Approach 3: Sinkhorn Algorithm

**Free Support Setting**

- **Algorithm:** Entropy-regularized barycenter using Sinkhorn iterations
- **Optimization Problem:**
  $$
  \min_\pi \langle \pi, C \rangle - \epsilon H(\pi)
  $$
  where $H(\pi)$ is entropy and $\epsilon$ is regularization parameter

- **Implementation:** Python Optimal Transport (POT) library
- **Advantages:**
  - Fast convergence (seconds vs minutes for IPM)
  - GPU-friendly implementation
  - Scalable to larger problem sizes
  - 2D visualization capabilities
- **Trade-offs:** Accuracy limited by regularization parameter $\epsilon$

**Location:** `approach3_code/`

---

## Project Structure

```
Project1_HW_Submission/
├── approach1&5_PDLP&MNIST_Code/
│   └── Project1_PDLP&MNIST_Code/
│       ├── PDLP/
│       │   ├── PDLP.py                      # Core PDLP solver
│       │   ├── PDLP_MNIST_Implement.py       # MNIST experiments
│       │   ├── Sampling.py                   # Sampling module
│       │   ├── cuPDLP copy 3.py             # CuPDLP implementation
│       │   └── Experiment_result/            # Visualization outputs
│       └── Sampling/
│           └── Sampling.py
│
├── approach2&4_code&data/
│   └── approach2&4_code&data/
│       ├── Code/
│       │   ├── centroid_sphBregman.m          # MAAIPM implementation
│       │   ├── centroid_sphFreeMAAIPM.m       # Free support variant
│       │   ├── sinkhorn_barycenter_*.m        # Sinkhorn variants
│       │   ├── Vanilla_IPM.m                  # Baseline IPM
│       │   ├── generate_data.m                # Data generation
│       │   ├── loaddata.m                     # Data loader
│       │   ├── test_data/                     # Generated test datasets
│       │   └── resultfile.txt                # Experimental results
│       └── data_result/
│           ├── fixed/                        # Fixed support results
│           │   ├── data_result.xlsx
│           │   ├── DLRM/
│           │   ├── SLRM/
│           │   └── mixed/
│           └── free/                         # Free support results
│               ├── result.xlsx
│               ├── maa/
│               └── shk/
│
├── approach3_code/
│   ├── maa_mean_max_min.m
│   └── sinkhorn_barycenter_project_adaptive(1).m
│
└── .gitignore
```

---

## Algorithm Comparison

| Method | Support Type | Order | Speed | Accuracy | Scalability | GPU Support |
|--------|--------------|-------|-------|----------|-------------|-------------|
| **PDLP** | Pre-specified | First-order | Slow (minutes) | High (exact LP) | Medium | Limited |
| **Sinkhorn** | Free | First-order | Fast (seconds) | Medium (smoothed) | High | Yes |
| **MAAIPM** | Fixed | Second-order | Medium (sec-min) | High | Medium | No |
| **SLRM** | Fixed | Second-order | Medium | High | Medium | No |
| **DLRM** | Fixed | Second-order | Medium | High | Medium | No |

---

## Experimental Results

### Key Findings

#### 1. PDLP Performance
- Convergence highly dependent on support point selection
- Works well for small-scale problems (200-500 support points) but scales poorly
- Termination criteria: All of `rel_prim_res`, `rel_dual_res`, `rel_gap` must be < tolerance
- Key performance metrics: `iter#`, `time`, `kkt_pass`, `prim_obj`, `dual_obj`

#### 2. Interior-Point Methods
- **MAAIPM** with Bregman preconditioning shows superior convergence for fixed support
- **SLRM/DLRM** provide good accuracy but computational cost increases rapidly with problem size
- Mixed strategies combining MAAIPM and Sinkhorn show promising results
- Iteration complexity: $O(\sqrt{n} \log(1/\epsilon))$ for second-order methods

#### 3. Sinkhorn Algorithm
- Dramatically faster than IPM methods (10-100x speedup)
- Trade-off between regularization parameter $\epsilon$ and accuracy
- Excellent for visualization and rapid prototyping
- Linear convergence rate

### Visual Results

- **MNIST Barycenters:** Successfully computed average digit representations
- **2D Distribution Barycenters:** Clear visualization using heatmaps
- **Convergence Plots:** Demonstrated iteration-wise improvement in primal/dual objectives
- **Gap Analysis:** Objective gap decreases monotonically with iterations

---

## Requirements

### Python (Approaches 1, 3, 5)
```
numpy >= 1.21.0
ortools >= 9.0.0
POT >= 0.8.0
matplotlib >= 3.4.0
scipy >= 1.7.0
```

### MATLAB (Approaches 2, 4)
- MATLAB R2020a or later
- Optimization Toolbox
- Optional: Gurobi Solver (for improved IPM performance)

---

## Usage

### Approach 1: PDLP (Pre-specified Support)

```python
from PDLP.PDLP_MNIST_Implement import compute_barycenter

# Load your distributions
distributions = load_distributions(...)

# Compute barycenter
barycenter, result = compute_barycenter(
    distributions=distributions,
    method='pdlp',
    verbose=True
)
```

### Approach 3: Sinkhorn (Free Support)

```python
import ot

# Using POT library
barycenter = ot.barycenter(
    distributions,
    reg=0.1,  # Regularization parameter
    weights=lambda_weights,
    method='sinkhorn'
)
```

### Approach 2/4: MATLAB IPM Methods

```matlab
% Load data
[dist_data, metadata] = loaddata('test_N10_mt20.d2');

% Compute barycenter using MAAIPM
[bary, iter_data] = centroid_sphBregman(dist_data, metadata);

% Or using Sinkhorn
[bary, iter_data] = sinkhorn_barycenter_project_adaptive(dist_data, metadata);
```

---

## Team & Contributions

| Member | Responsibilities |
|--------|------------------|
| **Li Jian** | Approach 1 & 5: PDLP implementation, sampling module, MNIST experiments |
| **Jiang Boyuan** | Approach 4: MAAIPM implementation, mathematical modeling |
| **Chen Supeng** | Approach 2: SLRM/DLRM implementation, IPM comparisons |
| **Guo Jinxin** | Approach 3: Sinkhorn algorithm, free support methods, visualization |

---

## References

### Main References

1. **Agueh, M., & Carlier, G.** (2011). Barycenters in the Wasserstein space. *SIAM Journal on Mathematical Analysis*, 43(2), 904-924.
   - **Location:** `Reference/`

2. **Cuturi, M.** (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances. *ICML*.
   - **Location:** `Reference/Sinkhorn/Cuturi - 2013 - Sinkhorn Distances Lightspeed Computation of Optimal Transportation Distances.pdf`

3. **Cuturi, M., & Doucet, A.** (2014). Fast Computation of Wasserstein Barycenters. *ICML*.
   - **Location:** `Reference/Sinkhorn/Cuturi和Doucet - 2014 - Fast Computation of Wasserstein Barycenters.pdf`

### Interior-Point Methods

4. **Ge, R., Jin, C., Jin, Y., Netrapalli, P., & Yin, W.** (2020). Interior-Point Methods Strike Back: Solving the Wasserstein Barycenter Problem via Optimization. *NeurIPS*.
   - **Location:** `Reference/MAAIPM/Ge 等 - 2020 - Interior-Point Methods Strike Back Solving the Wasserstein Barycenter Problem.pdf`
   - **Location:** `Reference/IPM-SLRM/Ge 等 - 2020 - Interior-Point Methods Strike Back Solving the Wasserstein Barycenter Problem.pdf`
   - **Location:** `Reference/PDLP/Ge 等 - 2020 - Interior-Point Methods Strike Back Solving the Wasserstein Barycenter Problem.pdf`

### Primal-Dual Methods

5. **Applegate, D., Cook, W., Dash, S., Espinoza, D., Goycoolea, M., & Johnson, E.** (2025). PDLP: A Practical First-Order Method for Large-Scale Linear Programming. *Mathematical Programming*.
   - **Location:** `Reference/PDLP/Applegate 等 - 2025 - PDLP A Practical First-Order Method for Large-Scale Linear Programming.pdf`

6. **Applegate, D., Cook, W., Dash, S., Espinoza, D., Goycoolea, M., & Johnson, E.** (2022). Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient.
   - **Location:** `Reference/PDLP/Applegate 等 - 2022 - Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient.pdf`

### Additional References

7. **Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.** (2015). Iterative Bregman Projections for Regularized Transportation Problems. *SIAM Journal on Scientific Computing*, 37(2), A1111-A1138.

8. **Peyré, G., & Cuturi, M.** (2019). Computational Optimal Transport. *Foundations and Trends® in Machine Learning*, 11(5-6), 355-607.

---

## License

This project is submitted for academic purposes. Please refer to individual component licenses for third-party libraries.

---

## Acknowledgments

- Google OR-Tools for the PDLP solver
- Python Optimal Transport (POT) library
- Lingnan University DDA4300 Course
- All referenced authors for their groundbreaking work in optimal transport and optimization
