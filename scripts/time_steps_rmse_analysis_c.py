"""
研究时间步数对RMSE的影响 - 经典算法

在t=0.3s时刻，测试时间步数从1到300，记录RMSE并可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from func_models.config import SimulationConfig
from scripts.c_nlse import PseudoSpectralSplitStepSolver, soliton_analytical_folded

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def compute_rmse_at_t(psi_num: np.ndarray, psi_ana: np.ndarray) -> float:
    """
    计算经典算法数值解与解析解在t=0.3s时刻的RMSE（按模长差）
    
    Args:
        psi_num: 经典算法数值解，形状为 (N,)
        psi_ana: 解析解，形状为 (N,)
    
    Returns:
        RMSE值
    """
    diff = np.abs(psi_num) - np.abs(psi_ana)
    rmse = np.sqrt(np.mean(diff ** 2))
    return rmse


def run_single_test(time_steps: int, config_base: SimulationConfig) -> float:
    """
    运行单次测试：使用经典算法，指定时间步数运行模拟到t=0.3s，计算RMSE
    
    Args:
        time_steps: 时间步数
        config_base: 基础配置（将根据time_steps调整dt）
    
    Returns:
        t=0.3s时刻的RMSE值（经典算法数值解与解析解的误差）
    """
    # 计算时间步长：总时间为0.3s
    total_time = 0.3
    dt = total_time / time_steps
    
    # 创建经典伪谱分裂步求解器
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
    
    # 计算t=0.3s时刻的解析解
    t_final = 0.3
    psi_analytical = soliton_analytical_folded(t_final, x, config_base.x0, config_base.a, config_base.v)
    
    # 计算RMSE
    rmse = compute_rmse_at_t(current_psi, psi_analytical)
    
    return rmse


def analyze_time_steps_effect(
    config: SimulationConfig,
    min_steps: int = 1,
    max_steps: int = 300,
    output_dir: str = "."
):
    """
    分析时间步数对经典算法RMSE的影响
    
    Args:
        config: 基础配置参数
        min_steps: 最小时间步数
        max_steps: 最大时间步数
        output_dir: 输出目录
    """
    print("=" * 60)
    print("时间步数对RMSE影响分析 - 经典算法")
    print(f"测试范围: {min_steps} - {max_steps} 步")
    print(f"目标时间: t = 0.3s")
    print(f"配置参数: n={config.n}, a={config.a}, x0={config.x0}, v={config.v}, s={config.s}")
    print("=" * 60)
    
    # 存储结果
    time_steps_list = np.arange(min_steps, max_steps + 1)
    rmse_list = []
    
    # 对每个时间步数进行测试
    for steps in tqdm(time_steps_list, desc="测试进度"):
        try:
            rmse = run_single_test(steps, config)
            rmse_list.append(rmse)
        except Exception as e:
            print(f"\n警告: 时间步数 {steps} 测试失败: {e}")
            rmse_list.append(np.nan)
    
    rmse_array = np.array(rmse_list)
    
    # 可视化
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # RMSE vs 时间步数
    plt.plot(time_steps_list, rmse_array, 'b-', linewidth=1.5, alpha=0.8)
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("RMSE at t=0.3s", fontsize=14)
    plt.title("RMSE vs Time Steps at t=0.3s", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度：每10一个刻度，每50显示一个标签
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(round(x))}' if round(x) % 50 == 0 else ''))
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(output_dir, "time_steps_rmse_analysis_c.png")
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
        print(f"  最小RMSE: {min_rmse:.6e} (时间步数: {min_steps_val})")
        print(f"  最大RMSE: {np.max(valid_rmse):.6e}")
        print(f"  平均RMSE: {np.mean(valid_rmse):.6e}")
        print(f"  中位数RMSE: {np.median(valid_rmse):.6e}")
        print("=" * 60)


if __name__ == "__main__":
    # 参数配置
    config = SimulationConfig(
        n=6,
        d=12,
        x0=-1,
        a=2,
        v=10,
        s=1,
        time_steps=100,  # 测试中忽略
        dt=3e-3,  # 测试中忽略
        initial_runs=10,
        maxiter=20000,
        maxfun=100000,
        ftol=1e-14,
    )
    
    # 输出目录
    output_dir = "/public/home/caozicheng/NLSE/result"
    
    # 运行分析
    analyze_time_steps_effect(
        config=config,
        min_steps=10,
        max_steps=300,
        output_dir=output_dir
    )
