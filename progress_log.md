# 📘 项目日志记录：DDA4300 Wasserstein Barycenter Project

## ✅ 白板项目任务拆解（4月中旬计划）


【分工内容】
- Approach1
  - i) Model / Notations. [2]
  - ii) pre-specified / free [2]
  - iii) PDLP [1,3]
  - iv) 研究算法 (GitHub)
  - v) 做实验 (1人)
  - vi) 写报告 (preview) (1人)
- Approach 2/3/4 分别按照个人分工进行

【关注内容】
1. 理论方向：建模 / 算法分解 / 结果评估
2. 实验方向：代码使用 / 图片 / 报告

---

## 🗓️ 2024-04-23

**Li Jian**
- 完成 PDLP 的理论了解
- 调试已有代码：
  - 了解 OR-tools 可以查看优化进程
  - 了解 termination 可调节
  - 了解对效果有影响的参数（样本 / support 选择）
  - 增加 visualization 温度图 (heatmap)
  - 思考如何选择 samples
- 找到 Approach2 应该和文献二的 interior point 方法对比，非 PDLP

**Jiang Boyuan**
- 完成基本建模数学公式，将项目文档转化至 test 中
- 理解 Approach4: MAAIPM 首次使用 PDPCIP 优化 specific 问题，后期用 IPM
- PDPCIP 参考: https://github.com/martinResearch/PySparseLP/blob/master/pysparselp/MehrotraPDIP.py

**Future Work：**
- 样本设计
- warmstart 技术
- IPM 中探索解规方程方法（SLRM/DLRM）

---

## 🗓️ 2024-04-24

**Li Jian**
- 基本完善 Approach1 代码架构
- 完成样本生成 + 存储，包括 distribution + LP 参数 Abc + meta 信息

**Future Work**
- 实现模块化样本生成/读写
- PDLP 模块独立化
- 完成 Approach1 相关文章 + 实验 + 图片效果
- 思考如何对比多种算法？如何表现结果？

**Chen Supeng**
- 阅读 SLRM/DLRM 的 motivation 和原理
- 调用 cvxpy 实现经典 IPM 方法，对比发现统计维度大时效率低，甚至无法求解

**Future Work**
- 实现 SLRM 和 DLRM

**Guo Jinxin**
- 阅读文献5，了解 dual solution 作为 subgradient 实现 projected gradient method
- 进行 free support 算法的代码实现，并通过 POT 库完成二维可视化
- 相比 interior-point 方法，实验后效率更高，对比时需紧控参数一致性

**Future Work**
- 完成 Approach3 的原理讲解，包括数学推导 + 优化方法代码 + 假代码

