"""
LSE 的量子变分算法模拟

使用变分量子算法求解 LSE，通过优化找到每个时间步的最优参数
"""


import numpy as np
import sys
import os
from typing import Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qiskit.algorithms.optimizers import L_BFGS_B
from func_models import (
    ansatz,
    stateFromParameters,
    cost_function_fast,
)
from func_models.config import SimulationConfig


def run_simulation(
    config: Optional[SimulationConfig] = None,
    output_dir: str = "."
) -> Tuple[np.ndarray, np.ndarray, SimulationConfig]:
    """
    LSE 的量子变分算法模拟
    
    Args:
        config: 模拟配置参数，如果为 None 则使用默认配置
        output_dir: 输出目录，用于保存结果图像和数据
    
    Returns:
        parameters: 优化后的变分参数数组，形状为 (time_steps, 2*n*(d+1))，类型为 float
        psi: 各时间步的波函数数组，形状为 (time_steps+1, 2^n)，类型为 complex
        config: 使用的模拟配置对象
    """
    if config is None:
        config = SimulationConfig()
    
    # ===== 生成初始条件 =====
    print("Generating initial condition...")
    initial_condition = config.generate_initial_condition()
    x = np.linspace(-np.pi, np.pi, 2**config.n, endpoint=False)
    
    # ===== 初始化优化器 =====
    print("Initializing optimizer...")
    optimizer = L_BFGS_B(maxiter=config.maxiter, maxfun=config.maxfun, ftol=config.ftol)
    
    # ===== 存储每个时间步的最优参数 =====
    # 形状：(时间步数, 参数数量)，即 (time_steps, 2*n*(d+1))
    parameters = np.zeros((config.time_steps, 2*config.n*(config.d+1)))
    
    # ===== 第一步：从初始条件开始 =====
    print(f"\nStarting simulation with {config.time_steps} time steps...")
    print("=" * 60)
    
    # 定义第一步的成本函数
    # startFromStateVector=True 表示 parameters_0 是状态向量而不是参数
    cost_function = lambda x: cost_function_fast(
        x, initial_condition, ansatz, config.n, config.d, config.s, config.dt, True
    )
    
    # 记录最佳成本函数值，初始化为正无穷
    fvalue_opt = float("Inf")
    
    # 多次随机初始化以缓解局部最优问题，保留最优解
    print(f"Step 1/{config.time_steps}: Trying {config.initial_runs} random initializations...")
    for i in range(config.initial_runs):
        print(f"  Run {i+1}/{config.initial_runs}...", end=" ", flush=True)
        
        # 第一步无先验参数，故在 [0, 2π) 均匀采样随机初始点（匹配旋转门的周期性）
        initial_params = 2*np.pi*np.random.random(2*config.n*(config.d+1))
        
        # 执行优化，找到最小化成本函数的参数（即找到最优参数）
        sol = optimizer.minimize(cost_function, initial_params)
        
        # 评估优化后的成本函数值
        fvalue_temp = cost_function(sol.x)
        print(f"Cost: {fvalue_temp:.6e}")
        
        # 保留成本最低的解，作为后续时间步的初始参数
        if fvalue_temp < fvalue_opt:
            fvalue_opt = fvalue_temp
            parameters[0, :] = sol.x
    
    print(f"Step 1 completed. Best cost: {fvalue_opt:.6e}")
    
    # ===== 后续步骤：从前一步的参数开始 =====
    # 基于波函数在相邻时间步的高度相似性，以前一步的最优参数初始化当前优化
    for i in range(1, config.time_steps):
        print(f"Step {i+1}/{config.time_steps}...", end=" ", flush=True)
        
        # 定义当前时间步的成本函数，以前一步的最优参数态作为参考态
        cost_function = lambda x: cost_function_fast(
            x, parameters[i-1, :], ansatz, config.n, config.d, config.s, config.dt
        )
        
        # 以前一步的最优参数热启动优化（通常仅需微调），显著加速收敛
        sol = optimizer.minimize(cost_function, parameters[i-1, 0:2*config.n*(config.d+1)])
        
        # 保存优化后的参数
        parameters[i, :] = sol.x
        
        # 评估成本函数值
        fvalue = cost_function(sol.x)
        print(f"Cost: {fvalue:.6e}")
    
    print("=" * 60)
    print("Simulation completed!")
    
    # ===== 从优化后的参数提取量子态 =====
    print("\nExtracting wave functions from optimized parameters...")
    
    # 存储所有时间步的波函数（含 t=0 初始态），形状: (time_steps + 1, 2**n)
    psi = np.zeros((config.time_steps+1, 2**config.n), dtype=complex)
    
    # 第一个时刻：使用初始条件
    psi[0, :] = initial_condition
    
    # 从每个时间步的参数生成对应的量子态
    for i in range(config.time_steps):
        # 从参数生成状态向量，给出优化后的参数对应的实际波函数
        psi[i+1, :] = stateFromParameters(
            ansatz, parameters[i, 0:2*config.n*(config.d+1)], config.n, config.d
        )
    
    # ===== 保存数值结果 =====
    data_path = os.path.join(output_dir, "q_lse_data.npz")
    np.savez(data_path, parameters=parameters, psi=psi, config=config)
    print(f"Data saved to: {data_path}")
    
    return parameters, psi, config


if __name__ == "__main__":
    # 自定义使用量子算法求解 LSE 的配置
    config = SimulationConfig(
        n=6,
        d=12,
        x0=-1,
        a=2,
        v=10,
        s=0,
        time_steps=100,
        dt=3e-3,
        initial_runs=10,
        maxiter=20000,
        maxfun=100000,
        ftol=1e-14,
    )
    
    # 运行模拟
    parameters, psi, config = run_simulation(config)
