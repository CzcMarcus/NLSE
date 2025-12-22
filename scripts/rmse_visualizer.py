"""
计算并可视化数值解与解析解之间的均方根误差（RMSE）

计算经典方法和归一化经典方法的数值解与解析解之间的RMSE，
并可视化各时间步的误差变化
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Tuple

# 设置matplotlib参数
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 12

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from func_models.analytical_solutions import soliton_analytical_folded


def calculate_rmse(
    classic_data_path: str = "/public/home/caozicheng/NLSE/result/data/classic_nlse_data.npz",
    nclassic_data_path: str = "/public/home/caozicheng/NLSE/result/data/nclassic_nlse_data.npz",
    output_dir: str = "/public/home/caozicheng/NLSE/result"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算两个数值解与解析解之间的均方根误差（RMSE）
    
    Args:
        classic_data_path: 经典方法数据文件路径
        nclassic_data_path: 归一化经典方法数据文件路径
        output_dir: 输出目录，用于保存可视化图像
    
    Returns:
        time_steps_array: 时间步数组（1到100）
        classic_rmse: 经典方法各时间步的RMSE
        nclassic_rmse: 归一化经典方法各时间步的RMSE
    """
    # 加载数据
    print("Loading data files...")
    classic_data = np.load(classic_data_path, allow_pickle=True)
    nclassic_data = np.load(nclassic_data_path, allow_pickle=True)
    
    # 提取波函数和配置
    classic_psi = classic_data['psi']  # shape: (time_steps+1, 2^n)
    nclassic_psi = nclassic_data['psi']  # shape: (time_steps+1, 2^n)
    classic_config = classic_data['config'].item()
    nclassic_config = nclassic_data['config'].item()
    
    # 验证配置参数是否一致
    assert classic_config.time_steps == nclassic_config.time_steps, "时间步数不一致"
    assert classic_config.n == nclassic_config.n, "空间分辨率不一致"
    assert classic_config.x0 == nclassic_config.x0, "初始位置不一致"
    assert classic_config.a == nclassic_config.a, "孤子宽度参数不一致"
    assert classic_config.v == nclassic_config.v, "传播速度不一致"
    
    # 使用经典方法的配置（两者应该相同）
    config = classic_config
    time_steps = config.time_steps
    n = config.n
    x0 = config.x0
    a = config.a
    v = config.v
    dt = config.dt
    
    # 空间设置
    N = 2**n
    L = 2 * np.pi
    dx = L / N
    x = np.linspace(-L/2, L/2 - dx, N)  # [-π, π)
    
    # 计算初始解析解的连续 L2 范数（守恒量）
    psi0_analytical = np.array([soliton_analytical_folded(0, xi, x0, a, v) for xi in x])
    norm0_continuous = np.linalg.norm(psi0_analytical) * np.sqrt(dx)  # ≈ sqrt(∫|ψ|² dx)
    
    # 初始化RMSE数组（只计算1到100的时间步，不包括初始时刻0）
    classic_rmse = np.zeros(time_steps)
    nclassic_rmse = np.zeros(time_steps)
    
    print(f"Computing RMSE for {time_steps} time steps...")
    print("=" * 60)
    
    # 对每个时间步计算RMSE
    for t_idx in range(1, time_steps + 1):
        t = t_idx * dt
        
        # 计算当前时刻的解析解（未归一化，物理真实解）
        psi_analytical = np.array([soliton_analytical_folded(t, xi, x0, a, v) for xi in x])
        
        # === Classic 方法：直接与解析解比较（两者均为物理尺度）===
        diff_classic = classic_psi[t_idx, :] - psi_analytical
        classic_rmse[t_idx - 1] = np.sqrt(np.mean(np.abs(diff_classic)**2))
        
        # === Normalized Classic 方法：先恢复物理尺度，再比较 ===
        # 假设 nclassic_psi 是单位向量（离散 l2 范数为1），需缩放至物理 L2 范数 norm0_continuous
        # 注意：离散向量范数 ||ψ||_discrete = sqrt(sum |ψ_i|^2)
        # 连续范数 ≈ ||ψ||_discrete * sqrt(dx)，所以要让连续范数 = norm0_continuous，
        # 则缩放因子 = norm0_continuous / sqrt(dx)
        scale_factor = norm0_continuous / np.sqrt(dx)
        nclassic_psi_physical = nclassic_psi[t_idx, :] * scale_factor
        
        diff_nclassic = nclassic_psi_physical - psi_analytical
        nclassic_rmse[t_idx - 1] = np.sqrt(np.mean(np.abs(diff_nclassic)**2))
        
        if t_idx % 20 == 0:
            print(f"Completed {t_idx}/{time_steps} time steps")
    
    print("=" * 60)
    print("RMSE computation completed!")
    
    # 生成时间步数组（1到100）
    time_steps_array = np.arange(1, time_steps + 1)
    
    return time_steps_array, classic_rmse, nclassic_rmse


def visualize_rmse(
    time_steps_array: np.ndarray,
    classic_rmse: np.ndarray,
    nclassic_rmse: np.ndarray,
    output_dir: str = "/public/home/caozicheng/NLSE/result"
) -> str:
    """
    可视化RMSE随时间的变化
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(time_steps_array, classic_rmse, 'b-', linewidth=2.5, 
             label='Classic Method', marker='o', markersize=4, alpha=0.8, markevery=5)
    plt.plot(time_steps_array, nclassic_rmse, 'r--', linewidth=2.5, 
             label='Normalized Classic Method', marker='s', markersize=4, alpha=0.8, markevery=5)
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Root Mean Square Error (RMSE)', fontsize=12)
    plt.title('RMSE between Numerical and Analytical Solutions vs Time Steps', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, len(time_steps_array) + 1)
    plt.ylim(bottom=0)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rmse_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()
    
    return output_path


def compute_and_visualize_rmse(
    classic_data_path: str = "/public/home/caozicheng/NLSE/result/data/classic_nlse_data.npz",
    nclassic_data_path: str = "/public/home/caozicheng/NLSE/result/data/nclassic_nlse_data.npz",
    output_dir: str = "/public/home/caozicheng/NLSE/result"
) -> str:
    """
    主函数：计算并可视化RMSE
    """
    time_steps_array, classic_rmse, nclassic_rmse = calculate_rmse(
        classic_data_path, nclassic_data_path, output_dir
    )
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("RMSE Statistics:")
    print(f"Classic Method - Mean RMSE: {np.mean(classic_rmse):.6e}")
    print(f"Classic Method - Max RMSE: {np.max(classic_rmse):.6e}")
    print(f"Normalized Classic Method - Mean RMSE: {np.mean(nclassic_rmse):.6e}")
    print(f"Normalized Classic Method - Max RMSE: {np.max(nclassic_rmse):.6e}")
    print("=" * 60)
    
    output_path = visualize_rmse(time_steps_array, classic_rmse, nclassic_rmse, output_dir)
    return output_path


if __name__ == "__main__":
    output_path = compute_and_visualize_rmse()
    print(f"\nDone! Results saved to: {output_path}")
