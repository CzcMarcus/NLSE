"""
研究时间步数对RMSE的影响 - 量子算法

固定dt=1e-3，x0=0，测试不同d值（8,9,10,11,12,13）和时间步数，记录RMSE并可视化
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
from scripts.c_nlse import soliton_analytical_folded
from scripts.q_nlse import run_simulation

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def compute_rmse_at_t_quantum(psi_num: np.ndarray, psi_ana: np.ndarray, config: SimulationConfig) -> float:
    """
    计算量子算法数值解与解析解的RMSE（按模长差）
    
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


def run_single_test_quantum(time_steps: int, dt: float, config_base: SimulationConfig) -> tuple:
    """
    运行单次测试：使用量子算法，固定dt，指定时间步数运行模拟，计算最终时刻的RMSE
    
    Args:
        time_steps: 时间步数
        dt: 固定时间步长
        config_base: 基础配置
    
    Returns:
        tuple: (rmse, psi_num, config_used)
            - rmse: 最终时刻的RMSE值（量子算法数值解与解析解的误差）
            - psi_num: 最终时刻的量子算法数值解（归一化量子态）
            - config_used: 使用的配置对象
    """
    # 创建配置，使用固定的dt和指定的时间步数
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
    
    # 获取最终时刻的数值解（最后一个时间步）
    psi_num = psi[-1, :]
    
    # 计算最终时刻的解析解
    x = np.linspace(-np.pi, np.pi, 2**config.n, endpoint=False)
    t_final = time_steps * dt
    psi_analytical = soliton_analytical_folded(t_final, x, config.x0, config.a, config.v)
    
    # 计算RMSE
    rmse = compute_rmse_at_t_quantum(psi_num, psi_analytical, config)
    
    return rmse, psi_num, config_used


def analyze_time_steps_effect_for_d(
    config: SimulationConfig,
    d: int,
    dt: float = 1e-3,
    min_steps: int = 0,
    max_steps: int = 100,
    output_dir: str = "."
) -> dict:
    """
    分析特定d值下时间步数对量子算法RMSE的影响（固定dt）
    
    Args:
        config: 基础配置参数
        d: 要测试的d值
        dt: 固定时间步长
        min_steps: 最小时间步数
        max_steps: 最大时间步数
        output_dir: 输出目录
    
    Returns:
        包含RMSE数组、时间数组等数据的字典
    """
    print("=" * 60)
    print(f"时间步数对RMSE影响分析 - 量子算法（d={d}，固定dt）")
    print(f"固定时间步长: dt = {dt}")
    print(f"测试范围: {min_steps} - {max_steps} 步")
    print(f"对应总时间范围: {min_steps*dt:.3f}s - {max_steps*dt:.3f}s")
    print(f"配置参数: n={config.n}, d={d}, a={config.a}, x0={config.x0}, v={config.v}, s={config.s}")
    print("=" * 60)
    
    # 更新配置的d值
    config.d = d
    
    # 存储结果
    time_steps_list = np.arange(min_steps, max_steps + 1)
    rmse_list = []
    psi_data_dict = {}  # 用于保存最终时刻的波函数值
    
    # 对每个时间步数进行测试
    for steps in tqdm(time_steps_list, desc=f"d={d} 测试进度"):
        try:
            rmse, psi_num, config_used = run_single_test_quantum(steps, dt, config)
            rmse_list.append(rmse)
            psi_data_dict[f'psi_steps_{steps}'] = psi_num  # 保存最终时刻的波函数值
        except Exception as e:
            print(f"\n警告: d={d}, 时间步数 {steps} 测试失败: {e}")
            rmse_list.append(np.nan)
            psi_data_dict[f'psi_steps_{steps}'] = np.full(2**config.n, np.nan, dtype=complex)
    
    rmse_array = np.array(rmse_list)
    time_array = time_steps_list * dt
    
    # 保存数据到.npz文件
    os.makedirs(output_dir, exist_ok=True)
    data_path = os.path.join(output_dir, f"time_steps_rmse_quantum_d{d}_dt{dt}_x0{config.x0}.npz")
    save_dict = {
        'rmse': rmse_array,
        'time': time_array,
        'time_steps': time_steps_list,
        'd': d,
        'dt': dt,
        'x0': config.x0,
    }
    save_dict.update(psi_data_dict)
    np.savez(data_path, **save_dict)
    print(f"\n数据已保存到: {data_path}")
    print(f"  包含 {len(psi_data_dict)} 个时间步数的波函数值和RMSE值")
    
    # 打印统计信息
    valid_rmse = rmse_array[~np.isnan(rmse_array)]
    if len(valid_rmse) > 0:
        min_rmse = np.min(valid_rmse)
        min_idx = np.argmin(valid_rmse)
        min_steps_val = time_steps_list[min_idx]
        print("\n" + "=" * 60)
        print(f"d={d} 统计信息:")
        print(f"  最小RMSE: {min_rmse:.6e} (时间步数: {min_steps_val}, 总时间: {min_steps_val*dt:.3f}s)")
        print(f"  最大RMSE: {np.max(valid_rmse):.6e}")
        print(f"  平均RMSE: {np.mean(valid_rmse):.6e}")
        print(f"  中位数RMSE: {np.median(valid_rmse):.6e}")
        print("=" * 60)
    
    return {
        'rmse': rmse_array,
        'time': time_array,
        'time_steps': time_steps_list,
        'd': d
    }


def analyze_multiple_d_values(
    config: SimulationConfig,
    d_list: list,
    dt: float = 1e-3,
    min_steps: int = 0,
    max_steps: int = 100,
    output_dir: str = "."
):
    """
    分析多个d值下时间步数对量子算法RMSE的影响
    
    Args:
        config: 基础配置参数
        d_list: 要测试的d值列表
        dt: 固定时间步长
        min_steps: 最小时间步数
        max_steps: 最大时间步数
        output_dir: 输出目录
    """
    print("\n" + "=" * 60)
    print("多d值时间步数对RMSE影响分析 - 量子算法")
    print(f"测试d值: {d_list}")
    print(f"固定时间步长: dt = {dt}")
    print(f"测试时间步数范围: {min_steps} - {max_steps}")
    print("=" * 60 + "\n")
    
    # 存储所有d值的结果
    all_results = {}
    
    # 对每个d值进行分析
    for d in d_list:
        result = analyze_time_steps_effect_for_d(
            config=config,
            d=d,
            dt=dt,
            min_steps=min_steps,
            max_steps=max_steps,
            output_dir=output_dir
        )
        all_results[d] = result
        print("\n")
    
    # 可视化：在同一张图上显示所有d值的结果
    plt.figure(figsize=(10, 6))
    
    # 为每个d值绘制曲线（黑色虚线）
    for d in d_list:
        result = all_results[d]
        plt.plot(result['time'], result['rmse'], 
                'k--', linewidth=1.5, alpha=0.8, 
                label=f'd={d}')
    
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.title(f"RMSE vs Time (Quantum Algorithm, dt={dt}, x0={config.x0})", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度：每0.01s一个刻度，每0.02s显示一个标签
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    ax = plt.gca()
    ax.set_xlim(0, 0.1)  # 设置x轴范围为0-0.1s
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}' if round(x * 100) % 2 == 0 else ''))
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(output_dir, f"time_steps_rmse_analysis_quantum_multid_dt{dt}_x0{config.x0}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"可视化结果已保存到: {plot_path}")
    plt.close()
    
    # 保存汇总数据
    summary_path = os.path.join(output_dir, f"time_steps_rmse_quantum_summary_dt{dt}_x0{config.x0}.npz")
    summary_dict = {}
    for d in d_list:
        result = all_results[d]
        summary_dict[f'rmse_d{d}'] = result['rmse']
        summary_dict[f'time_d{d}'] = result['time']
    summary_dict['time_steps'] = all_results[d_list[0]]['time_steps']
    summary_dict['d_list'] = np.array(d_list)
    summary_dict['dt'] = dt
    summary_dict['x0'] = config.x0
    np.savez(summary_path, **summary_dict)
    print(f"汇总数据已保存到: {summary_path}")


if __name__ == "__main__":
    # 参数配置
    config = SimulationConfig(
        n=6,
        d=12,  # 基础值，会在测试中被覆盖
        x0=0,  # 用户指定
        a=2,
        v=10,
        s=4,
        time_steps=100,
        dt=1e-3,  # 用户指定，固定使用
        initial_runs=10,
        maxiter=20000,
        maxfun=100000,
        ftol=1e-13,  # 用户指定
    )
    
    # 固定时间步长
    dt = 1e-3
    
    # 要测试的d值列表
    d_list = [8, 9, 10, 11, 12, 13]
    
    # 输出目录
    output_dir = "/public/home/caozicheng/NLSE/scripts/figuere5"
    
    # 运行分析
    analyze_multiple_d_values(
        config=config,
        d_list=d_list,
        dt=dt,
        min_steps=0,
        max_steps=100,
        output_dir=output_dir
    )
