# DDA4300 Course Project I
**Computing 2-Wasserstein Barycenter via Linear Programming**

---

## 🔍 Project Overview
We study the computation of the 2-Wasserstein barycenter of a collection of discrete probability measures. Given empirical distributions \(P^{(t)}\) (from samples of \(N(\mu_t,\sigma_t^2)\)), our goal is to find:

\[ \min_{P}\sum_{t=1}^N W_2^2\bigl(P,P^{(t)}\bigr), \]

which can be cast as a linear program when support is fixed. Our focus is on implementing and comparing multiple algorithmic strategies from recent literature.

---

## 🚩 Project Plan & Approaches

### Approach 1: Pre-specified Support + PDLP
- Formulate LP model of Wasserstein barycenter
- Implement Primal-Dual LP solver (PDLP) [1,3]
- Tune optimization parameters (step size, termination)
- Visualize convergence metrics

### Approach 2: Interior-Point with Low-Rank (SLRM/DLRM)
- Implement SLRM & DLRM Newton solvers [2]
- Compare with standard IPM (e.g., CVXPY)
- Evaluate runtime & objective value

### Approach 3: Sinkhorn + Free Support
- Implement Algorithm 3 from [5] using POT
- Entropic regularization with \(\lambda\)
- Visualize in 2D space

### Approach 4: MAAIPM for Free Support
- Alternate LP/QP steps + analytical QP update [2]
- Use warm-start & jump heuristics to escape local minima
- Compare with Sinkhorn in terms of convergence & cost

### Approach 6: MNIST / Fashion-MNIST Visualization
- Use real image data to visualize barycenters
- Compare learned supports via pixels

---

## 📅 Current Progress (as of April 24)

| Approach | Status       | Summary                                                  |
|----------|--------------|----------------------------------------------------------|
| 1        | ✅ Done       | PDLP implemented with LP modeling + visualization       |
| 2        | ⏳ Ongoing    | Vanilla IPM done; SLRM/DLRM under study                 |
| 3        | ✅ Done       | Sinkhorn free-support implemented + visualized          |
| 4        | ⏳ Designing  | Alternating LP/QP outlined; jump strategies discussed    |
| 6        | ⏳ Skeleton   | MNIST loading script started                            |

📓 See [progress_log.md](./progress_log.md) for detailed logs & planning whiteboard.

---

## 📁 Repository Structure

```
Project1_Code/
├── Sampling/
│   ├── sample/                       # Generated data
│   ├── Sampling.py                  # Sample generator
│   └── what_sample.md              # Sampling description
├── PDLP/                            # Approach 1
│   ├── PDLP.py
│   ├── visualization_utils.py
│   └── PDLP_plots/
├── Approach3/                       # Sinkhorn
│   └── approach3_sinkhorn_free_support.py
├── progress_log.md                 # Daily logs
├── run_guide.md                    # How to run modules
└── README.md                       # ← This file
```

---

## ✨ How to Contribute
1. Fork this repository and clone locally
2. Install dependencies (see `run_guide.md`)
3. Create a branch per feature or experiment:
   ```bash
   git checkout -b approach2-slrm
   ```
4. Add your code, push and open a PR

---

## 🎯 Upcoming Milestones

- **Apr 25–28**: Finalize Approach 1 report; advance Approach 2
- **May 1–4**: Compare Approach 3 vs 4 on synthetic data
- **May 5–10**: Migrate experiments to MNIST + prepare slides

---

## 📚 References

[1] Applegate et al., "Faster first-order primal-dual methods for LP using restarts and sharpness"  
[2] Ge et al., "IPM strike back: solving the Wasserstein barycenter problem"  
[3] Lu et al., "cuPDLP-C: GPU-accelerated PDLP"  
[4] Cuturi, "Sinkhorn distances"  
[5] Cuturi & Doucet, "Fast computation of Wasserstein barycenters"

