"""
计算并可视化三种算法（经典、NC、量子）与解析解之间的 RMSE 对比（按模长差）。
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from func_models.config import SimulationConfig

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def compute_rmse_classic(
    num_path: str,
    ana_path: str,
    config: SimulationConfig,
):
    """
    计算经典数值解与解析解之间的 RMSE
    
    Args:
        num_path: 经典数值解数据路径（.npz格式）
        ana_path: 解析解数据路径（.npz格式）
        config: 模拟配置
    
    Returns:
        t_axis: 时间轴
        rmse: RMSE数组
    """
    num = np.load(num_path, allow_pickle=True)
    ana = np.load(ana_path, allow_pickle=True)

    dt = float(config.dt)
    psi_num = num["psi"]      # 物理波函数 Ψ(x): ∫|Ψ|² dx = 2a
    psi_ana = ana["psi"]      # 物理波函数 Ψ(x): ∫|Ψ|² dx = 2a

    assert psi_num.shape == psi_ana.shape, "Shape mismatch!"

    steps = psi_num.shape[0] - 1
    t_axis = np.arange(1, steps + 1) * dt
    rmse = np.zeros(steps)
    
    for i in range(1, steps + 1):
        # 直接比较物理波函数模长
        diff = np.abs(psi_num[i]) - np.abs(psi_ana[i])
        rmse[i - 1] = np.sqrt(np.mean(diff ** 2))

    return t_axis, rmse


def compute_rmse_nc(
    num_path: str,
    ana_path: str,
    config: SimulationConfig,
):
    """
    计算NC（非经典）数值解与解析解之间的 RMSE
    
    Args:
        num_path: NC数值解数据路径（.npz格式）
        ana_path: 解析解数据路径（.npz格式）
        config: 模拟配置
    
    Returns:
        t_axis: 时间轴
        rmse: RMSE数组
    """
    num = np.load(num_path, allow_pickle=True)
    ana = np.load(ana_path, allow_pickle=True)

    dt = float(config.dt)
    psi_num = num["psi"]      # 物理波函数 Ψ(x): ∫|Ψ|² dx = 2a
    psi_ana = ana["psi"]      # 物理波函数 Ψ(x): ∫|Ψ|² dx = 2a

    assert psi_num.shape == psi_ana.shape, "Shape mismatch!"

    steps = psi_num.shape[0] - 1
    t_axis = np.arange(1, steps + 1) * dt
    rmse = np.zeros(steps)
    
    for i in range(1, steps + 1):
        # 直接比较物理波函数模长
        diff = np.abs(psi_num[i]) - np.abs(psi_ana[i])
        rmse[i - 1] = np.sqrt(np.mean(diff ** 2))

    return t_axis, rmse


def compute_rmse_quantum(
    num_path: str,
    ana_path: str,
    config: SimulationConfig,
):
    """
    计算量子数值解与解析解之间的 RMSE
    
    Args:
        num_path: 量子数值解数据路径（.npz格式）
        ana_path: 解析解数据路径（.npz格式）
        config: 模拟配置
    
    Returns:
        t_axis: 时间轴
        rmse: RMSE数组
    """
    num = np.load(num_path, allow_pickle=True)
    ana = np.load(ana_path, allow_pickle=True)

    dt = float(config.dt)
    psi_num = num["psi"]      # 归一化量子态: sum(|psi_num|^2) = 1
    psi_ana = ana["psi"]      # 物理波函数 Ψ(x): ∫|Ψ|² dx = 2a

    # 网格使用与生成时一致的均匀步长：x = linspace(-π, π, 2^n, endpoint=False)
    x_grid = np.linspace(-np.pi, np.pi, 2 ** config.n, endpoint=False)
    dx = x_grid[1] - x_grid[0]
    
    # 正确还原物理波函数模长
    total_norm = 2 * config.a / config.s        # ∫|Ψ|² dx = 2a/s
    scale = np.sqrt(total_norm / dx)            # 因为 ∑|Ψ_i|² = total_norm / dx
    psi_num_phys_mod = np.abs(psi_num) * scale  # 还原后的 |Ψ_num(x)|

    assert psi_num.shape == psi_ana.shape, "Shape mismatch!"

    steps = psi_num.shape[0] - 1
    t_axis = np.arange(1, steps + 1) * dt
    rmse = np.zeros(steps)
    
    for i in range(1, steps + 1):
        # 使用物理波函数模长与解析解比较
        diff = psi_num_phys_mod[i] - np.abs(psi_ana[i])
        rmse[i - 1] = np.sqrt(np.mean(diff ** 2))

    return t_axis, rmse


def plot_combined_rmse(
    classic_path: str,
    nc_path: str,
    quantum_path: str,
    ana_path: str,
    output_dir: str = ".",
    config: SimulationConfig | None = None,
):
    """
    计算并可视化三种算法的 RMSE 对比图
    
    Args:
        classic_path: 经典数值解数据路径
        nc_path: NC数值解数据路径
        quantum_path: 量子数值解数据路径
        ana_path: 解析解数据路径
        output_dir: 输出目录
        config: 模拟配置
    """
    if config is None:
        raise ValueError("请提供 SimulationConfig，用于获取 dt 等信息。")

    print("计算经典算法 RMSE...")
    t_classic, rmse_classic = compute_rmse_classic(classic_path, ana_path, config)
    
    print("计算NC算法 RMSE...")
    t_nc, rmse_nc = compute_rmse_nc(nc_path, ana_path, config)
    
    print("计算量子算法 RMSE...")
    t_quantum, rmse_quantum = compute_rmse_quantum(quantum_path, ana_path, config)

    # 可视化
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 7))
    
    # 绘制三条曲线，使用不同的颜色和样式
    plt.plot(t_classic, rmse_classic, "b-", linewidth=2.2, marker="o", 
             markersize=4, alpha=0.85, markevery=5, label="Classic")
    plt.plot(t_nc, rmse_nc, "r-", linewidth=2.2, marker="s", 
             markersize=4, alpha=0.85, markevery=5, label="NC")
    plt.plot(t_quantum, rmse_quantum, "g-", linewidth=2.2, marker="^", 
             markersize=4, alpha=0.85, markevery=5, label="Quantum")
    
    plt.xlabel("Time (t)", fontsize=14)
    plt.ylabel("Root Mean Square Error", fontsize=14)
    plt.title("RMSE vs Time: Comparison of Three Algorithms", fontsize=16)
    
    # 设置 y 轴刻度间隔为 0.001，格式为 3 位小数
    ax = plt.gca()
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    ax.yaxis.set_major_locator(MultipleLocator(0.001))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.3f}'))
    
    plt.legend(loc='best', fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_png = os.path.join(output_dir, "rmse_combined.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n合并RMSE图已保存到: {out_png}")
    print("\n=== RMSE 统计信息 ===")
    print(f"经典算法 - Min: {np.min(rmse_classic):.6e}, Mean: {np.mean(rmse_classic):.6e}, Max: {np.max(rmse_classic):.6e}")
    print(f"NC算法    - Min: {np.min(rmse_nc):.6e}, Mean: {np.mean(rmse_nc):.6e}, Max: {np.max(rmse_nc):.6e}")
    print(f"量子算法  - Min: {np.min(rmse_quantum):.6e}, Mean: {np.mean(rmse_quantum):.6e}, Max: {np.max(rmse_quantum):.6e}")


if __name__ == "__main__":
    # 配置参数
    config = SimulationConfig(
        n=6,
        d=12,
        x0=-1,
        a=2,
        v=10,
        s=1,  # 对应论文目标，非线性强度为1
        time_steps=100,
        dt=3e-3,
        initial_runs=10,
        maxiter=20000,
        maxfun=100000,
        ftol=1e-14,
    )
    
    classic_path = "/public/home/caozicheng/NLSE/result/data/classic_nlse_data.npz"
    nc_path = "/public/home/caozicheng/NLSE/result/data/nclassic_nlse_data.npz"
    quantum_path = "/public/home/caozicheng/NLSE/result/data/q_nlse_data.npz"
    ana_path = "/public/home/caozicheng/NLSE/result/data/analytical_nlse_data.npz"
    
    plot_combined_rmse(
        classic_path, nc_path, quantum_path, ana_path,
        "/public/home/caozicheng/NLSE/result", config=config
    )
