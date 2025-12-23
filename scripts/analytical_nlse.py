"""
计算并存储 NLSE 解析解

计算 t=0 到 t=100*dt 这 101 个时间点的 NLSE 解析解值并存储
"""


import numpy as np
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from func_models.config import SimulationConfig
from func_models.analytical_solutions import soliton_analytical_folded


def calculate_analytical_solution(
    config: SimulationConfig,
    output_dir: str = "."
) -> np.ndarray:
    """
    计算 NLSE 解析解在所有时间步的值
    
    Args:
        config: 模拟配置参数
        output_dir: 输出目录路径
    
    Returns:
        psi_analytical: 解析解数组，形状为 (time_steps+1, 2^n)，类型为 complex
    """
    print("Calculating analytical solution for NLSE...")
    print(f"Time steps: {config.time_steps}, dt: {config.dt}")
    print(f"Spatial points: {2**config.n}")
    print("=" * 60)
    
    # 生成空间坐标（在 [-π, π] 区间上的物理坐标）
    x_physical = np.linspace(-np.pi, np.pi, 2**config.n, endpoint=False)
    
    # 存储所有时间步的解析解（含 t=0 初始态），形状: (time_steps + 1, 2**n)
    psi_analytical = np.zeros((config.time_steps+1, 2**config.n), dtype=complex)
    
    # 计算每个时间步的解析解
    for i in range(config.time_steps + 1):
        # 计算当前时间
        t = i * config.dt
        
        if i % 10 == 0 or i == config.time_steps:
            print(f"Calculating t={t:.6f} (step {i}/{config.time_steps})...", end=" ", flush=True)
        
        # 在每个空间点上计算解析解
        for j in range(2**config.n):
            psi_analytical[i, j] = soliton_analytical_folded(
                t, x_physical[j], config.x0, config.a, config.v
            )
        
        if i % 10 == 0 or i == config.time_steps:
            print("Done")
    
    print("=" * 60)
    print("Analytical solution calculation completed!")
    
    return psi_analytical


def save_analytical_solution(
    config: SimulationConfig,
    output_dir: str = "."
) -> str:
    """
    计算并保存 NLSE 解析解
    
    Args:
        config: 模拟配置参数
        output_dir: 输出目录路径
    
    Returns:
        保存的数据文件路径
    """
    # 计算解析解
    psi_analytical = calculate_analytical_solution(config, output_dir)
    
    # 保存数据
    os.makedirs(output_dir, exist_ok=True)
    data_path = os.path.join(output_dir, "analytical_nlse_data.npz")
    np.savez(data_path, psi=psi_analytical, config=config)
    print(f"\nAnalytical solution saved to: {data_path}")
    print(f"Data shape: {psi_analytical.shape}")
    
    return data_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算并存储 NLSE 解析解")
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="输出目录路径（默认当前目录）"
    )
    
    args = parser.parse_args()
    
    # 使用与 q_nlse.py 相同的配置参数
    config = SimulationConfig(
        n=6,
        d=12,
        x0=-1,
        a=2,
        v=10,
        s=4,
        time_steps=100,
        dt=3e-3,
        initial_runs=10,
        maxiter=20000,
        maxfun=100000,
        ftol=1e-14,
    )
    
    # 计算并保存解析解
    save_analytical_solution(config, args.output)
