# DDA4300-Project1: Optimal Transport Methods for Wasserstein Barycenter

## 🔍 Project Overview
This project explores optimization-based approaches for computing Wasserstein barycenters under different settings. We aim to implement and compare multiple algorithmic strategies such as:
- Primal-Dual Linear Programming (PDLP)
- Sinkhorn algorithm (Free support)
- Interior-point methods (e.g., IPM, DLRM, SLRM)

Our work focuses on both theoretical understanding and practical experiments.

---

## 📅 Current Progress (as of April 24)

### Approach 1 (Pre-specified support)
- ✅ Mathematical modeling completed
- ✅ LP transformation (standard form) implemented
- ✅ PDLP code implemented with visualizable convergence metrics (heatmap supported)
- ✅ Sampling module supports custom dataset generation and metadata storage

### Approach 3 (Free support using Sinkhorn)
- ✅ Implemented with POT library and visualized in 2D
- ✅ Supports larger scale problems with better runtime compared to classical IPM

### Approach 2 & 4 (In Progress)
- ⏳ Literature review and SLRM/DLRM theoretical understanding
- ⏳ Initial IPM-based experiments conducted (via cvxpy)

> Detailed daily progress and task log is available in [progress_log.md](./progress_log.md)

---

## 📄 Project Structure

```
Project1_Code/
├── Sampling/
│   ├── sample/
│   ├── Sampling.py
│   └── what_sample.md
├── PDLP/
│   └── PDLP照片/  (Visual evaluation resources)
├── Approach3/
│   └── approach3_skinkhorn_free_support.py
├── progress_log.md
├── README.md
```

---

## ✨ How to Contribute
1. Clone the repository
2. Activate your Python environment and install dependencies
3. Use the provided sampling code to generate datasets
4. Run and compare results across multiple approaches

Please push all contributions to **separate branches** and open a Pull Request.

---

## 📈 Goals
- ✅ Validate multiple algorithms on common synthetic data
- ✅ Explore differences in runtime, memory, accuracy
- ✅ Present final comparison plots, tables, and summary

Final report and visual presentation will be prepared in early May.

---

📅 For task details and whiteboard plans, refer to [progress_log.md](./progress_log.md).

