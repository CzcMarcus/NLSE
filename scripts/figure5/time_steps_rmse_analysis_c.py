"""
研究时间步数对RMSE的影响 - 经典算法

固定dt=1e-3，x0=0，测试不同时间步数，记录RMSE并可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# 确保能导入项目根目录的模块
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "../..")
sys.path.insert(0, os.path.abspath(project_root))

from func_models.config import SimulationConfig
from scripts.c_nlse import PseudoSpectralSplitStepSolver, soliton_analytical_folded

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def compute_rmse_at_t(psi_num: np.ndarray, psi_ana: np.ndarray) -> float:
    """
    计算经典算法数值解与解析解的RMSE（按模长差）
    
    Args:
        psi_num: 经典算法数值解，形状为 (N,)
        psi_ana: 解析解，形状为 (N,)
    
    Returns:
        RMSE值
    """
    diff = np.abs(psi_num) - np.abs(psi_ana)
    rmse = np.sqrt(np.mean(diff ** 2))
    return rmse


def run_single_test(time_steps: int, dt: float, config_base: SimulationConfig) -> float:
    """
    运行单次测试：使用经典算法，固定dt，指定时间步数运行模拟，计算最终时刻的RMSE
    
    Args:
        time_steps: 时间步数
        dt: 固定时间步长
        config_base: 基础配置
    
    Returns:
        最终时刻的RMSE值（经典算法数值解与解析解的误差）
    """
    # 创建经典伪谱分裂步求解器，使用固定的dt
    solver = PseudoSpectralSplitStepSolver(
        n=config_base.n,
        s=config_base.s,
        dt=dt
    )
    
    # 初始化：t=0时刻的解析解
    x = solver.x
    psi0 = soliton_analytical_folded(0, x, config_base.x0, config_base.a, config_base.v)
    
    # 运行模拟
    current_psi = psi0.copy()
    for _ in range(time_steps):
        current_psi = solver.step(current_psi)
    
    # 计算最终时刻的解析解
    t_final = time_steps * dt
    psi_analytical = soliton_analytical_folded(t_final, x, config_base.x0, config_base.a, config_base.v)
    
    # 计算RMSE
    rmse = compute_rmse_at_t(current_psi, psi_analytical)
    
    return rmse


def analyze_time_steps_effect(
    config: SimulationConfig,
    dt: float = 1e-3,
    min_steps: int = 1,
    max_steps: int = 100,
    output_dir: str = "."
):
    """
    分析时间步数对经典算法RMSE的影响（固定dt）
    
    Args:
        config: 基础配置参数
        dt: 固定时间步长
        min_steps: 最小时间步数
        max_steps: 最大时间步数
        output_dir: 输出目录
    """
    print("=" * 60)
    print("时间步数对RMSE影响分析 - 经典算法（固定dt）")
    print(f"固定时间步长: dt = {dt}")
    print(f"测试范围: {min_steps} - {max_steps} 步")
    print(f"对应总时间范围: {min_steps*dt:.3f}s - {max_steps*dt:.3f}s")
    print(f"配置参数: n={config.n}, a={config.a}, x0={config.x0}, v={config.v}, s={config.s}")
    print("=" * 60)
    
    # 存储结果
    time_steps_list = np.arange(min_steps, max_steps + 1)
    rmse_list = []
    
    # 对每个时间步数进行测试
    for steps in tqdm(time_steps_list, desc="测试进度"):
        try:
            rmse = run_single_test(steps, dt, config)
            rmse_list.append(rmse)
        except Exception as e:
            print(f"\n警告: 时间步数 {steps} 测试失败: {e}")
            rmse_list.append(np.nan)
    
    rmse_array = np.array(rmse_list)
    
    # 计算对应的时间数组
    time_array = time_steps_list * dt
    
    # 可视化
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # RMSE vs 时间
    plt.plot(time_array, rmse_array, 'k--', linewidth=1.5, alpha=0.8)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.title(f"RMSE vs Time (Classical Algorithm, dt={dt}, x0={config.x0})", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度：每0.01s一个刻度，每0.02s显示一个标签
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    ax = plt.gca()
    ax.set_xlim(0, 0.1)  # 设置x轴范围为0-0.1s
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}' if round(x * 100) % 2 == 0 else ''))
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(output_dir, f"time_steps_rmse_analysis_c_dt{dt}_x0{config.x0}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"可视化结果已保存到: {plot_path}")
    plt.close()
    
    # 打印统计信息
    valid_rmse = rmse_array[~np.isnan(rmse_array)]
    if len(valid_rmse) > 0:
        min_rmse = np.min(valid_rmse)
        min_idx = np.argmin(valid_rmse)
        min_steps_val = time_steps_list[min_idx]
        print("\n" + "=" * 60)
        print("统计信息:")
        print(f"  最小RMSE: {min_rmse:.6e} (时间步数: {min_steps_val}, 总时间: {min_steps_val*dt:.3f}s)")
        print(f"  最大RMSE: {np.max(valid_rmse):.6e}")
        print(f"  平均RMSE: {np.mean(valid_rmse):.6e}")
        print(f"  中位数RMSE: {np.median(valid_rmse):.6e}")
        print("=" * 60)


if __name__ == "__main__":
    # 参数配置
    config = SimulationConfig(
        n=6,
        d=12,
        x0=0,  # 用户指定
        a=2,
        v=10,
        s=1,
        time_steps=100,
        dt=1e-3,  # 用户指定，固定使用
        initial_runs=10,
        maxiter=20000,
        maxfun=100000,
        ftol=1e-13,  # 用户指定（对经典算法不适用，但保留在config中）
    )
    
    # 固定时间步长
    dt = 1e-3
    
    # 输出目录
    output_dir = "/public/home/caozicheng/NLSE/scripts/figure5"
    
    # 运行分析
    analyze_time_steps_effect(
        config=config,
        dt=dt,
        min_steps=1,
        max_steps=100,
        output_dir=output_dir
    )
