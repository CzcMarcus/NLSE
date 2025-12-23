"""
经典伪谱分裂步法求解非线性薛定谔方程
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from numpy.typing import NDArray

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from func_models.config import SimulationConfig


def soliton_analytical(t: float, x: NDArray, x0: float, a: float, v: float) -> NDArray:
    """
    非线性薛定谔方程的解析孤子解（在整个实数轴上）。
    
    数学表达式：
        ψ(x,t) = a / cosh(a*(x-x0-v*t)) * exp(i*v*(x-x0) + i*(a²-v²)*t/2)
    
    Args:
        t: 时间
        x: 空间位置数组
        x0: 初始位置
        a: 孤子宽度参数
        v: 孤子传播速度
    
    Returns:
        孤子解的复振幅值
    """
    return a / np.cosh(a*(x - x0 - v*t)) * np.exp(1j*v*(x - x0) + 1j*(a**2 - v**2)*t/2.0)


def soliton_analytical_folded(t: float, x: NDArray, x0: float, a: float, v: float) -> NDArray:
    """
    将解析孤子解折叠到周期区间 [-π, π] 上。
    """
    candidates = [soliton_analytical(t, x + i*2*np.pi, x0, a, v) for i in range(-2, 3)]
    result = np.zeros_like(x, dtype=complex)
    for i in range(len(x)):
        max_idx = np.argmax([np.abs(cand[i]) for cand in candidates])
        result[i] = candidates[max_idx][i]
    return result


class PseudoSpectralSplitStepSolver:
    """
    经典伪谱分裂步算法（严格效仿参考代码中公式12的实现）
    
    算法步骤：
    1. 线性步（动量空间）：
        ψ̃ = F⁻¹[ exp(-i/2 * k² * dt) * F(ψ) ] （含 fftshift 和相位校正）
    2. 非线性 Euler 步：
        ψ(t+dt) = ψ̃ + i s dt |ψ̃|² ψ̃
    """

    def __init__(self, n: int = 6, s: float = 1.0, dt: float = 0.01):
        self.n = n
        self.N = 2**n
        self.s = s
        self.dt = dt
        
        # 空间网格：x ∈ [-π, π)
        self.x = np.linspace(-np.pi, np.pi, self.N, endpoint=False)
        
        # 波数：k_j = -N/2 + j, j = 0, ..., N-1
        self.k = np.arange(-self.N//2, self.N//2)  # shape (N,)

    def step(self, psi: NDArray) -> NDArray:
        """
        执行一个时间步
        """
        # --- Step 1: 线性演化（动量空间）---
        psi_k = np.fft.fft(psi)                     # FFT
        psi_k_shifted = np.fft.fftshift(psi_k)      # shift zero freq to center
        
        # 应用线性相位：exp(-i * 0.5 * k^2 * dt)
        phase = np.exp(-1j * 0.5 * self.k**2 * self.dt)
        psi_k_evolved_shifted = psi_k_shifted * phase
        
        # 相位校正（补偿 k 起点为 -N/2 引入的全局相位）
        correction = np.exp(1j * 0.5 * self.dt * (self.N//2)**2)
        psi_k_evolved_shifted *= correction
        
        # 逆变换回位置空间
        psi_k_evolved = np.fft.ifftshift(psi_k_evolved_shifted)
        psi_tilde = np.fft.ifft(psi_k_evolved)
        
        # --- Step 2: 非线性 Euler 显式步 ---
        nonlinear_term = 1j * self.s * self.dt * np.abs(psi_tilde)**2 * psi_tilde
        psi_next = psi_tilde + nonlinear_term
        
        return psi_next

    def propagate(self, psi0: NDArray, num_steps: int, save_every: int = 1) -> Tuple[NDArray, NDArray]:
        """
        传播波函数多个时间步。
        """
        psi = psi0.copy()
        save_indices = sorted(set(list(range(0, num_steps, save_every)) + [num_steps]))
        psi_history = []
        times = []
        current_save_idx = 0

        for step in range(num_steps + 1):
            t = step * self.dt
            if current_save_idx < len(save_indices) and step == save_indices[current_save_idx]:
                psi_history.append(psi.copy())
                times.append(t)
                current_save_idx += 1
            if step < num_steps:
                psi = self.step(psi)

        return np.array(psi_history), np.array(times)


def run_simulation(output_dir: str = "."):
    """
    运行经典伪谱分裂步模拟
    """
    # ========== 参数设置 ==========
    n = 6               # 2^6 = 64 points
    a = 2.0             # 孤子宽度
    x0 = -1.0           # 初始位置
    v = 10.0            # 速度
    s = 1               # 非线性系数
    total_time = 0.3
    num_steps = 100
    dt = total_time / num_steps

    # ========== 初始化求解器和初始条件 ==========
    solver = PseudoSpectralSplitStepSolver(n=n, s=s, dt=dt)
    x = solver.x
    psi0 = soliton_analytical_folded(0, x, x0, a, v)

    print(f"Starting classical simulation:")
    print(f"  n={n}, N={solver.N}, dt={dt:.6f}, steps={num_steps}")
    print(f"  Soliton: a={a}, x0={x0}, v={v}, s={s}")

    # ========== 时间演化 ==========
    psi = np.zeros((num_steps + 1, solver.N), dtype=complex)
    psi[0, :] = psi0
    current_psi = psi0.copy()

    for i in range(num_steps):
        current_psi = solver.step(current_psi)
        psi[i + 1, :] = current_psi

    print("Simulation completed!")

    # ========== 可视化 ==========
    plt.figure(figsize=(10, 6))
    for i in range(0, num_steps + 1, 10):
        plt.plot(x / np.pi, np.abs(psi[i, :])**2, label=f"t={i*dt:.3f}")

    plt.xlabel("$x$ [$\\pi$]")
    plt.ylabel("$|\\Psi|^2$")
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "image_classic.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")

    # 保存数据
    config = SimulationConfig(
        n=n,
        x0=x0,
        a=a,
        v=v,
        s=s,
        time_steps=num_steps,
        dt=dt,
    )
    data_path = os.path.join(output_dir, "classic_nlse_data.npz")
    np.savez(data_path, psi=psi, config=config, x=x, t=np.linspace(0, total_time, num_steps+1))
    print(f"Data saved to: {data_path}")

    plt.show()

    return solver, psi


if __name__ == "__main__":
    run_simulation()
