"""
研究时间步数对RMSE的影响 - 量子算法

在t=0.3s时刻，测试指定的时间步数，记录量子算法数值解与解析解的RMSE并可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from func_models.config import SimulationConfig
from scripts.c_nlse import soliton_analytical_folded
from scripts.q_nlse import run_simulation

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def compute_rmse_at_t_quantum(psi_num: np.ndarray, psi_ana: np.ndarray, config: SimulationConfig) -> float:
    """
    计算量子算法数值解与解析解在t=0.3s时刻的RMSE（按模长差）
    
    量子态是归一化的：sum(|psi_num|^2) = 1
    物理波函数需要满足：∫|Ψ|² dx = 2a/s
    转换公式：psi_num_phys_mod = |psi_num| * sqrt((2a/s) / dx)
    
    Args:
        psi_num: 量子算法数值解（归一化量子态），形状为 (N,)
        psi_ana: 解析解（物理波函数），形状为 (N,)
        config: 模拟配置，用于计算归一化因子
    
    Returns:
        RMSE值
    """
    # 网格使用与生成时一致的均匀步长：x = linspace(-π, π, 2^n, endpoint=False)
    x_grid = np.linspace(-np.pi, np.pi, 2 ** config.n, endpoint=False)
    dx = x_grid[1] - x_grid[0]
    
    # 正确还原物理波函数模长
    total_norm = 2 * config.a  # ∫|Ψ|² dx = 2a/s = 2a
    scale = np.sqrt(total_norm / dx)  # 因为 ∑|Ψ_i|² = total_norm / dx
    psi_num_phys_mod = np.abs(psi_num) * scale  # 还原后的 |Ψ_num(x)|
    
    # 使用物理波函数模长与解析解比较
    diff = psi_num_phys_mod - np.abs(psi_ana)
    rmse = np.sqrt(np.mean(diff ** 2))
    return rmse


def run_single_test_quantum(time_steps: int, config_base: SimulationConfig) -> tuple:
    """
    运行单次测试：使用量子算法，指定时间步数运行模拟到t=0.3s，计算RMSE
    
    Args:
        time_steps: 时间步数
        config_base: 基础配置（将根据time_steps调整dt）
    
    Returns:
        tuple: (rmse, psi_num, config_used)
            - rmse: t=0.3s时刻的RMSE值（量子算法数值解与解析解的误差）
            - psi_num: t=0.3s时刻的量子算法数值解（归一化量子态）
            - config_used: 使用的配置对象
    """
    # 计算时间步长：总时间为0.3s
    total_time = 0.3
    dt = total_time / time_steps
    
    # 创建配置，使用指定的时间步数和dt
    config = SimulationConfig(
        n=config_base.n,
        d=config_base.d,
        x0=config_base.x0,
        a=config_base.a,
        v=config_base.v,
        s=config_base.s,
        time_steps=time_steps,
        dt=dt,
        initial_runs=config_base.initial_runs,
        maxiter=config_base.maxiter,
        maxfun=config_base.maxfun,
        ftol=config_base.ftol,
    )
    
    # 运行量子算法模拟
    parameters, psi, config_used = run_simulation(config, output_dir=".")
    
    # 获取t=0.3s时刻的数值解（最后一个时间步）
    psi_num = psi[-1, :]
    
    # 计算t=0.3s时刻的解析解
    x = np.linspace(-np.pi, np.pi, 2**config.n, endpoint=False)
    t_final = 0.3
    psi_analytical = soliton_analytical_folded(t_final, x, config.x0, config.a, config.v)
    
    # 计算RMSE
    rmse = compute_rmse_at_t_quantum(psi_num, psi_analytical, config)
    
    return rmse, psi_num, config_used


def analyze_time_steps_effect(
    config: SimulationConfig,
    time_steps_list: list,
    output_dir: str = "."
):
    """
    分析时间步数对量子算法RMSE的影响
    
    Args:
        config: 基础配置参数
        time_steps_list: 要测试的时间步数列表
        output_dir: 输出目录
    """
    print("=" * 60)
    print("时间步数对RMSE影响分析 - 量子算法")
    print(f"测试时间步数: {time_steps_list}")
    print(f"目标时间: t = 0.3s")
    print(f"配置参数: n={config.n}, d={config.d}, a={config.a}, x0={config.x0}, v={config.v}, s={config.s}")
    print("=" * 60)
    
    # 存储结果：只保存RMSE和t=0.3s时刻的波函数值
    rmse_list = []
    psi_data_dict = {}  # 用于保存t=0.3s时刻的波函数值
    
    # 对每个时间步数进行测试
    for steps in tqdm(time_steps_list, desc="量子算法测试进度"):
        try:
            print(f"\n正在测试时间步数: {steps}")
            rmse, psi_num, config_used = run_single_test_quantum(steps, config)
            rmse_list.append(rmse)
            psi_data_dict[f'psi_t0.3s_steps_{steps}'] = psi_num  # 保存t=0.3s时刻的波函数值
            print(f"时间步数 {steps}: RMSE = {rmse:.6e}")
        except Exception as e:
            print(f"\n警告: 时间步数 {steps} 测试失败: {e}")
            rmse_list.append(np.nan)
    
    rmse_array = np.array(rmse_list)
    time_steps_array = np.array(time_steps_list)
    
    # 保存t=0.3s时刻的波函数值和RMSE到.npz文件，方便复现
    os.makedirs(output_dir, exist_ok=True)
    psi_data_path = os.path.join(output_dir, "time_steps_psi_quantum_t0.3s.npz")
    # 只保存RMSE和t=0.3s时刻的波函数值
    save_dict = {
        'rmse': rmse_array,
    }
    # 添加所有波函数数据
    save_dict.update(psi_data_dict)
    
    np.savez(psi_data_path, **save_dict)
    print(f"t=0.3s时刻的波函数数据和RMSE已保存到: {psi_data_path}")
    print(f"  包含 {len(psi_data_dict)} 个时间步数在t=0.3s的波函数值和RMSE值")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    
    # RMSE vs 时间步数
    plt.plot(time_steps_array, rmse_array, 'g-', linewidth=1.5, alpha=0.8, marker='o', markersize=6)
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("RMSE at t=0.3s", fontsize=14)
    plt.title("RMSE vs Time Steps at t=0.3s (Quantum Algorithm)", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度：每10一个刻度，每50显示一个标签
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(round(x))}' if round(x) % 50 == 0 else ''))
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(output_dir, "time_steps_rmse_analysis_quantum.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"可视化结果已保存到: {plot_path}")
    plt.close()
    
    # 打印统计信息
    valid_rmse = rmse_array[~np.isnan(rmse_array)]
    if len(valid_rmse) > 0:
        min_rmse = np.min(valid_rmse)
        min_idx = np.argmin(valid_rmse)
        min_steps_val = time_steps_array[min_idx]
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
        s=4,
        time_steps=100,  # 测试中忽略
        dt=3e-3,  # 测试中忽略
        initial_runs=10,
        maxiter=20000,
        maxfun=100000,
        ftol=1e-14,
    )
    
    # 指定的时间步数列表
    time_steps_list = [10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300]
    
    # 输出目录
    output_dir = "/public/home/caozicheng/NLSE/result"
    
    # 运行分析
    analyze_time_steps_effect(
        config=config,
        time_steps_list=time_steps_list,
        output_dir=output_dir
    )
