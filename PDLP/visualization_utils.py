import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

def plot_convergence_metrics(iterations: List[int], 
                           rel_prim_res: List[float],
                           rel_dual_res: List[float],
                           rel_gap: List[float],
                           save_path: str = None):
    """
    绘制收敛指标随迭代次数的变化
    
    Args:
        iterations: 迭代次数列表
        rel_prim_res: 原始残差列表
        rel_dual_res: 对偶残差列表
        rel_gap: 相对间隙列表
        save_path: 保存图像的路径（可选）
    """
    plt.figure(figsize=(10, 6))
    
    plt.semilogy(iterations, rel_prim_res, 'b-', label='Relative Primal Residual', linewidth=2)
    plt.semilogy(iterations, rel_dual_res, 'r-', label='Relative Dual Residual', linewidth=2)
    plt.semilogy(iterations, np.abs(rel_gap), 'g-', label='Relative Gap', linewidth=2)
    
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Residuals (log scale)', fontsize=12)
    plt.title('Convergence Metrics vs Iterations', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_kkt_passes(iterations: List[int],
                   kkt_passes: List[float],
                   save_path: str = None):
    """
    绘制KKT passes随迭代次数的变化
    
    Args:
        iterations: 迭代次数列表
        kkt_passes: KKT passes列表
        save_path: 保存图像的路径（可选）
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(iterations, kkt_passes, 'b-', linewidth=2)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Normalized KKT Passes', fontsize=12)
    plt.title('KKT Passes vs Iterations', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_termination_comparison(iterations: List[int],
                              data_1e4: Dict[str, List[float]],
                              data_1e8: Dict[str, List[float]],
                              save_path: str = None):
    """
    比较不同终止条件下的性能
    
    Args:
        iterations: 迭代次数列表
        data_1e4: epsilon=1e-4的数据字典
        data_1e8: epsilon=1e8的数据字典
        save_path: 保存图像的路径（可选）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制duality gap比较
    ax1.semilogy(iterations, data_1e4['duality_gap'], 'b-', label='ε=1e-4', linewidth=2)
    ax1.semilogy(iterations, data_1e8['duality_gap'], 'r--', label='ε=1e8', linewidth=2)
    ax1.set_xlabel('Iterations', fontsize=12)
    ax1.set_ylabel('Duality Gap (log scale)', fontsize=12)
    ax1.set_title('Duality Gap Comparison', fontsize=14)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(fontsize=10)
    
    # 绘制KKT passes比较
    ax2.plot(iterations, data_1e4['kkt_passes'], 'b-', label='ε=1e-4', linewidth=2)
    ax2.plot(iterations, data_1e8['kkt_passes'], 'r--', label='ε=1e8', linewidth=2)
    ax2.set_xlabel('Iterations', fontsize=12)
    ax2.set_ylabel('Normalized KKT Passes', fontsize=12)
    ax2.set_title('KKT Passes Comparison', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_runtime_analysis(iterations: List[int],
                         runtime: List[float],
                         save_path: str = None):
    """
    绘制运行时间分析
    
    Args:
        iterations: 迭代次数列表
        runtime: 运行时间列表
        save_path: 保存图像的路径（可选）
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(iterations, runtime, 'b-', linewidth=2)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Runtime Analysis', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_objective_convergence(iterations: List[int],
                             primal_obj: List[float],
                             dual_obj: List[float],
                             save_path: str = None):
    """
    绘制目标函数收敛过程
    
    Args:
        iterations: 迭代次数列表
        primal_obj: 原始目标函数值列表
        dual_obj: 对偶目标函数值列表
        save_path: 保存图像的路径（可选）
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(iterations, primal_obj, 'b-', label='Primal Objective', linewidth=2)
    plt.plot(iterations, dual_obj, 'r-', label='Dual Objective', linewidth=2)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Objective Value', fontsize=12)
    plt.title('Objective Function Convergence', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show() 