# DDA4300-Project1: Computing Wasserstein Barycenter via Linear Programming

> **Course:** DDA4300 - Optimization Methods in Machine Learning
>
> **Institution:** CUHKSZ
>
> **Topic:** Wasserstein Barycenter Computation

---

## Overview

This project implements and compares multiple optimization-based approaches for computing Wasserstein barycenters under different settings. The Wasserstein barycenter problem aims to find a probability distribution that minimizes the average Wasserstein distance to a set of given distributions. This has important applications in image processing, data aggregation, and machine learning.

### Problem Statement

Given $N$ probability measures $\mu_1, \dots, \mu_N$ with weights $\lambda_1, \dots, \lambda_N$ (where $\sum_t \lambda_t = 1$), the **Wasserstein barycenter** $\mu$ solves:

$$
\min_\mu \sum_{t=1}^N \lambda_t W_2^2(\mu, \mu_t)
$$

where $W_2$ is the 2-Wasserstein distance induced by a ground cost function.

---

## Methodologies Implemented

### Approach 1 & 5: Primal-Dual Linear Programming (PDLP) + MNIST

**Pre-specified Support Setting**

- **Mathematical Modeling:** Transformed the barycenter problem into a standard linear programming form
- **Algorithm:** Implemented PDLP (Primal-Dual Hybrid Gradient) solver using Google's OR-Tools
- **Features:**
  - Modular sampling module with metadata storage
  - Visualizable convergence metrics (heatmap supported)
  - Warm-start capabilities
- **MNIST Application:** Applied to digit image barycenter computation

**Location:** `approach1&5_PDLP&MNIST_Code/`

### Approach 2 & 4: Interior-Point Methods (IPM)

**Fixed Support Setting with Advanced IPM Variants**

- **Methods Implemented:**
  - **SLRM** (Symmetric Low-Rank Method)
  - **DLRM** (Dual Low-Rank Method)
  - **MAAIPM** (Modified Augmented Algorithmic Interior Point Method)
  - **Vanilla IPM** (baseline)
  - **Sinkhorn** (entropy-regularized, for comparison)
- **Experimental Results:** Comprehensive evaluation on synthetic data with varying parameters (N=3,4,6,8,10,12,15; m_t=5,10,20,40,60,80,100)
- **Data Structure:** Custom `.d2` format for storing probability distributions with support points and weights

**Location:** `approach2&4_code&data/`

### Approach 3: Sinkhorn Algorithm

**Free Support Setting**

- **Algorithm:** Entropy-regularized barycenter using Sinkhorn iterations
- **Implementation:** Python Optimal Transport (POT) library
- **Advantages:**
  - Fast convergence (seconds vs minutes for IPM)
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
│       │   ├── PDLP.py              # Core PDLP solver
│       │   ├── PDLP_MNIST_Implement.py  # MNIST experiments
│       │   ├── Sampling.py           # Sampling module
│       │   ├── cuPDLP copy 3.py      # CuPDLP implementation
│       │   └── Experiment_result/    # Visualization outputs
│       └── Sampling/
│           └── Sampling.py
│
├── approach2&4_code&data/
│   └── approach2&4_code&data/
│       ├── Code/
│       │   ├── centroid_sphBregman.m      # MAAIPM implementation
│       │   ├── centroid_sphFreeMAAIPM.m  # Free support variant
│       │   ├── sinkhorn_barycenter_*.m    # Sinkhorn variants
│       │   ├── Vanilla_IPM.m              # Baseline IPM
│       │   ├── generate_data.m            # Data generation
│       │   ├── loaddata.m                 # Data loader
│       │   ├── test_data/                  # Generated test datasets
│       │   └── resultfile.txt             # Experimental results
│       └── data_result/
│           ├── fixed/                     # Fixed support results
│           │   ├── data_result.xlsx
│           │   ├── DLRM/
│           │   ├── SLRM/
│           │   └── mixed/
│           └── free/                      # Free support results
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

| Method | Support Type | Speed | Accuracy | Scalability |
|--------|--------------|-------|----------|-------------|
| **PDLP** | Pre-specified | Slow (minutes) | High (exact LP) | Medium |
| **Sinkhorn** | Free | Fast (seconds) | Medium (smoothed) | High |
| **MAAIPM** | Fixed | Medium (seconds-minutes) | High | Medium |
| **SLRM** | Fixed | Medium | High | Medium |
| **DLRM** | Fixed | Medium | High | Medium |

---

## Experimental Results

### Key Findings

1. **PDLP Performance:**
   - Convergence highly dependent on support point selection
   - Works well for small-scale problems but scales poorly
   - Termination criteria: `rel_prim_res`, `rel_dual_res`, `rel_gap` all < tolerance

2. **Interior-Point Methods:**
   - MAAIPM with preconditioning shows superior convergence for fixed support
   - SLRM/DLRM provide good accuracy but computational cost increases rapidly with problem size
   - Mixed strategies combining MAAIPM and Sinkhorn show promise

3. **Sinkhorn Algorithm:**
   - Dramatically faster than IPM methods
   - Trade-off between regularization parameter $\epsilon$ and accuracy
   - Excellent for visualization and rapid prototyping

### Visual Results

- **MNIST Barycenters:** Successfully computed average digit representations
- **2D Distribution Barycenters:** Clear visualization using heatmaps
- **Convergence Plots:** Demonstrated iteration-wise improvement in primal/dual objectives

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

1. Agueh, M., & Carlier, G. (2011). Barycenters in the Wasserstein space. *SIAM Journal on Mathematical Analysis*, 43(2), 904-924.

2. Cuturi, M., & Doucet, A. (2014). Fast Computation of Wasserstein Barycenters. *ICML*.

3. Ge, R., Jin, C., & Netrapalli, P. (2017). Efficient Algorithms for Large-scale Wasserstein Barycenter. *NeurIPS*.

4. PDLP: Primal-Dual Hybrid Gradient for Large-Scale Linear Programming. Google Research.

5. Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G. (2015). Iterative Bregman Projections for Regularized Transportation Problems. *SIAM Journal on Scientific Computing*.

---

## License

This project is submitted for academic purposes. Please refer to individual component licenses for third-party libraries.

---

## Acknowledgments

- Google OR-Tools for the PDLP solver
- Python Optimal Transport (POT) library
- Lingnan University DDA4300 Course
