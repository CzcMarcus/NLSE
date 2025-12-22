"""
主模拟脚本

使用变分量子算法求解NLSE，通过优化找到每个时间步的最优参数。
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Optional,Tuple

# 添加项目根目录到路径，以便导入func_models包
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
    运行完整的NLSE模拟。
    
    Args:
        config: 模拟配置参数，如果为None则使用默认配置
        output_dir: 输出目录，用于保存结果图像和数据
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
    # 形状：(时间步数, 参数数量)，即(time_steps, 2*n*(d+1))
    parameters = np.zeros((config.time_steps, 2*config.n*(config.d+1)))
    
    # ===== 第一步：从初始条件开始 =====
    print(f"\nStarting simulation with {config.time_steps} time steps...")
    print("=" * 60)
    
    # 定义第一步的成本函数
    # startFromStateVector=True表示parameters_0是状态向量而不是参数
    cost_function = lambda x: cost_function_fast(
        x, initial_condition, ansatz, config.n, config.d, config.s, config.dt, True
    )
    
    # 记录最佳成本函数值
    fvalue_opt = float("Inf")
    
    # 尝试多个随机初始值（可以避免陷入局部最优解），选择最好的
    print(f"Step 1/{config.time_steps}: Trying {config.initial_runs} random initializations...")
    for i in range(config.initial_runs):
        print(f"  Run {i+1}/{config.initial_runs}...", end=" ", flush=True)
        
        # 使用随机初始参数进行优化（因为第一步没有好的初始猜测，需要多次尝试）
        # 参数范围：[0, 2π]，因为旋转角度是周期性的
        # 随机初始化可以探索参数空间的不同区域
        initial_params = 2*np.pi*np.random.random(2*config.n*(config.d+1))
        
        # 执行优化，找到最小化成本函数的参数（即找到最优参数）
        sol = optimizer.minimize(cost_function, initial_params)
        
        # 评估优化后的成本函数值
        fvalue_temp = cost_function(sol.x)
        print(f"Cost: {fvalue_temp:.6e}")
        
        # 如果找到更好的解，保存它
        # 这确保我们使用最好的初始参数进行后续步骤
        if fvalue_temp < fvalue_opt:
            fvalue_opt = fvalue_temp
            parameters[0, :] = sol.x
    
    print(f"Step 1 completed. Best cost: {fvalue_opt:.6e}")
    
    # ===== 后续步骤：从前一步的参数开始 =====
    # 对于后续步骤，我们可以使用前一步的参数作为初始猜测
    # 这通常能大大加快收敛速度，因为相邻时间步的波函数应该相似
    for i in range(1, config.time_steps):
        print(f"Step {i+1}/{config.time_steps}...", end=" ", flush=True)
        
        # 定义当前步的成本函数
        # 使用前一步的参数作为参考态
        cost_function = lambda x: cost_function_fast(
            x, parameters[i-1, :], ansatz, config.n, config.d, config.s, config.dt
        )
        
        # 从前一步的参数开始优化（通常只需要微调）
        # 使用前一步的参数作为初始猜测，可以大大加快收敛
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
    
    # 存储每个时间步的波函数
    # 形状：(时间步数+1, 空间点数)
    # +1是因为包括初始条件（t=0时刻）
    psi = np.zeros((config.time_steps+1, 2**config.n), dtype=complex)
    
    # 第一个时刻：使用初始条件
    psi[0, :] = initial_condition
    
    # 从每个时间步的参数生成对应的量子态
    for i in range(config.time_steps):
        # 从参数生成状态向量，给出优化后的参数对应的实际波函数
        psi[i+1, :] = stateFromParameters(
            ansatz, parameters[i, 0:2*config.n*(config.d+1)], config.n, config.d
        )
    
    # ===== 可视化结果 =====
    print("Visualizing results...")
    
    # 绘制波包随时间的演化
    # 每10个时间步绘制一次，避免图像过于拥挤
    plt.figure(figsize=(10, 6))
    for i in range(0, config.time_steps+1, 10):
        # 绘制概率密度 |ψ|² 随位置的变化
        # x/np.pi 将位置归一化到 [-1, 1]（单位是π）
        # |ψ|²是概率密度，表示在位置x找到粒子的概率
        plt.plot(x/np.pi, np.square(np.abs(psi[i, :])), label=f"t={i}")
    
    # 添加图例和标签
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.xlabel("$x$ [$\pi$]")  # x轴：位置（单位是π）
    plt.ylabel("$|\Psi|^2$")   # y轴：概率密度
    plt.title("Wave Packet Evolution")
    plt.tight_layout()         # 自动调整布局，确保所有元素都可见
    
    # 保存图像
    image_path = os.path.join(output_dir, "image.png")
    plt.savefig(image_path, dpi=150)
    print(f"Image saved to: {image_path}")
    
    # 保存数据（参数和波函数）供后续分析
    data_path = os.path.join(output_dir, "data.npz")
    np.savez(data_path, parameters=parameters, psi=psi, config=config)
    print(f"Data saved to: {data_path}")
    
    # 显示图像
    plt.show()
    
    return parameters, psi, config


if __name__ == "__main__":
    # 可以在这里自定义配置
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
    
    # 运行模拟
    parameters, psi, config = run_simulation(config)
