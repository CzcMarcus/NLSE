"""
NLSE 的经典伪谱分裂步法模拟

使用经典的一阶伪谱分裂步法求解 NLSE
"""


import numpy as np
import sys
import os
from typing import Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from func_models.config import SimulationConfig


def classical_nlse_step(
    psi_prev: np.ndarray,
    s: float,
    delta_t: float
) -> np.ndarray:
    """
    经典一阶 split-step 方法求解 NLSE: i ∂ψ/∂t = -1/2 ∂²ψ/∂x² - s |ψ|² ψ
    
    步骤:
      1. 线性步（动能项）: ψ_tilde = IFFT[exp(-i Δt/2 * k²) * FFT[ψ_prev]]
      2. 非线性步（欧拉显式）: ψ_next = ψ_tilde + i s Δt |ψ_tilde|² ψ_tilde
    
    Args:
        psi_prev: 上一时刻的波函数 (复数数组, shape=(N,))
        s: 非线性系数 (s>0: 聚焦, s<0: 散焦)
        delta_t: 时间步长
    
    Returns:
        psi_next: 当前时刻的波函数
    """
    # ===== Step 1: 线性演化（动能项） via FFT =====
    # 转换到动量空间（傅里叶变换）
    # fftshift 将零频率移到中心，便于处理
    psi_tilde = np.fft.fftshift(np.fft.fft(psi_prev))
    
    # 在动量空间应用拉普拉斯算子的演化
    # 演化算子：exp(-i*dt/2 * k²)
    # 这里 k 的范围是 -N/2 到 N/2-1，其中 N = len(psi_prev)
    N = len(psi_prev)
    k_values = np.linspace(-N/2, N/2-1, N)
    psi_tilde *= np.exp(-1j * delta_t/2 * np.square(k_values))
    
    # 校正因 fftshift 和离散波数定义引入的全局相位偏移，确保数值解与解析解的相位一致
    psi_tilde *= np.exp(1j * delta_t/2 * (-N/2)**2)
    
    # 转换回位置空间（逆傅里叶变换）
    psi_tilde = np.fft.ifft(np.fft.ifftshift(psi_tilde))
    
    # ===== Step 2: 非线性演化（显式欧拉步） =====
    # NLSE: i ∂ψ/∂t = ... - s |ψ|² ψ  ⇒  ∂ψ/∂t = i s |ψ|² ψ + ...
    # 显式欧拉: ψ_{n+1} = ψ_n + Δt * (i s |ψ_n|² ψ_n)
    nonlinear_increment = 1j * s * np.abs(psi_tilde)**2 * psi_tilde
    psi_next = psi_tilde + delta_t * nonlinear_increment
    
    return psi_next


def run_simulation(
    config: Optional[SimulationConfig] = None,
    output_dir: str = "."
) -> Tuple[None, np.ndarray, SimulationConfig]:
    """
    NLSE 的经典伪谱分裂步法模拟
    
    Args:
        config: 模拟配置参数，如果为 None 则使用默认配置
        output_dir: 输出目录，用于保存结果图像和数据
    
    Returns:
        parameters: None（经典方法不使用参数）
        psi: 各时间步的波函数数组，形状为 (time_steps+1, 2^n)，类型为 complex
        config: 使用的模拟配置对象
    """
    if config is None:
        config = SimulationConfig()
    
    # ===== 生成初始条件 =====
    print("Generating initial condition...")
    initial_condition = config.generate_initial_condition()
    x = np.linspace(-np.pi, np.pi, 2**config.n, endpoint=False)
    
    # ===== 存储所有时间步的波函数 =====
    # 形状：(time_steps + 1, 2**n)，包含初始态和所有演化后的态
    psi = np.zeros((config.time_steps+1, 2**config.n), dtype=complex)
    psi[0, :] = initial_condition
    
    # ===== 时间演化循环 =====
    print(f"\nStarting simulation with {config.time_steps} time steps...")
    print("=" * 60)
    
    for i in range(config.time_steps):
        print(f"Step {i+1}/{config.time_steps}...", end=" ", flush=True)
        
        # 执行一个时间步的演化
        psi[i+1, :] = classical_nlse_step(
            psi[i, :], s=config.s, delta_t=config.dt
        )
        
        # 评估归一化（用于监控）
        norm = np.linalg.norm(psi[i+1, :])
        print(f"Norm: {norm:.10f}")
    
    print("=" * 60)
    print("Simulation completed!")
    
    # ===== 保存数值结果 =====
    data_path = os.path.join(output_dir, "classic_nlse_data.npz")
    np.savez(data_path, psi=psi, config=config)
    print(f"Data saved to: {data_path}")
    
    return None, psi, config


if __name__ == "__main__":
    # 自定义使用经典伪谱分裂步法求解 NLSE 的配置
    config = SimulationConfig(
        n=6,
        x0=-1,
        a=2,
        v=10,
        s=4,
        time_steps=100,
        dt=3e-3,
    )
    
    # 运行模拟
    parameters, psi, config = run_simulation(config)
