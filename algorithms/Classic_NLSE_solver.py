import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from numpy.typing import NDArray


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
    return a / np.cosh(a*(x-x0-v*t)) * np.exp(1j*v*(x-x0) + 1j*(a**2 - v**2)*t/2.0)


def soliton_analytical_folded(t: float, x: NDArray, x0: float, a: float, v: float) -> NDArray:
    """
    将解析孤子解折叠到周期区间 [-π, π] 上。
    
    由于使用周期性边界条件，需要考虑多个周期的贡献。
    """
    # 考虑5个周期
    candidates = [soliton_analytical(t, x + i*2*np.pi, x0, a, v) for i in range(-2, 3)]
    
    # 选择幅度最大的
    result = np.zeros_like(x, dtype=complex)
    for i in range(len(x)):
        # 找出当前位置x_i处幅度最大的候选值
        max_idx = np.argmax([np.abs(cand[i]) for cand in candidates])
        result[i] = candidates[max_idx][i]
    
    return result


class PseudoSpectralSplitStepSolver:
    """
    经典的伪谱分裂步算法实现（参考论文中的公式12和13）
    
    算法步骤（对应公式12和13）：
    1. Step 1（隐式步）：处理线性项（拉普拉斯算子）
        ψ̃(x_j, t) = exp(-iVΔt) F⁻¹[exp(-i/2 * k_j² Δt) F(ψ(x_j, t))]
        
    2. Step 2（显式步）：处理非线性项（Euler步）
        ψ(x_j, t+Δt) = ψ̃(x_j, t) + i s Δt |ψ̃(x_j, t)|² ψ̃(x_j, t)
    
    参数说明：
    - N: 空间离散点数
    - L: 空间域长度（默认2π）
    - gamma: 非线性系数（s）
    - dt: 时间步长
    """
    
    def __init__(self, N: int = 64, L: float = 2*np.pi, 
                 gamma: float = 1.0, dt: float = 0.01):
        """
        初始化伪谱分裂步算法。
        
        Args:
            N: 空间离散点数（2的幂次）
            L: 空间域长度
            gamma: 非线性系数（s）
            dt: 时间步长
        """
        self.N = N
        self.L = L
        self.beta = -1.0  # 固定为-1，对应论文中的-1/2因子
        self.gamma = gamma
        self.dt = dt
        
        # 空间网格
        self.dx = L / N
        self.x = np.linspace(-L/2, L/2, N, endpoint=False)
        
        # 波数网格（动量空间）
        self.k = 2 * np.pi * np.fft.fftfreq(N, self.dx)
        
        # 线性算子的相位因子（动量空间）
        # 注意：论文中拉普拉斯算子是 -1/2 * ∂²/∂x²
        # 在动量空间中，这对应 -1/2 * (ik)² = 1/2 * k²
        # 但β已经包含了这个因子
        self.linear_phase = np.exp(1j * self.beta * self.k**2 * dt / 2)
        
        # 势能V（默认为0）
        self.V = 0.0
    
    def step(self, psi: NDArray) -> NDArray:
        """
        执行一个时间步（标准的对称split-step方法）。
        
        标准的split-step方法步骤：
        1. 线性步（dt/2）：处理线性项（拉普拉斯算子）
        2. 非线性步（dt）：处理非线性项（精确指数形式）
        3. 线性步（dt/2）：再次处理线性项
        
        这种对称方法对于保持孤子解的稳定性很重要。
        
        Args:
            psi: 当前时间步的波函数
            
        Returns:
            下一个时间步的波函数
        """
        # Step 1: 线性步（dt/2）- 处理线性部分
        # 转到动量空间
        psi_k = np.fft.fft(psi)
        
        # 应用线性演化（dt/2）
        psi_k = psi_k * self.linear_phase
        
        # 转回位置空间
        psi_tilde = np.fft.ifft(psi_k)
        
        # 应用势能项（如果V不为0）
        if np.any(self.V != 0):
            psi_tilde = psi_tilde * np.exp(-1j * self.V * self.dt)
        
        # Step 2: 非线性步（dt）- 处理非线性部分
        # 使用精确的指数形式：ψ = exp(i s dt |ψ|²) * ψ
        # 这是非归一化版本的精确解
        psi_mid = np.exp(1j * self.gamma * self.dt * np.abs(psi_tilde)**2) * psi_tilde
        
        # Step 3: 线性步（dt/2）- 再次处理线性部分
        # 转到动量空间
        psi_k = np.fft.fft(psi_mid)
        
        # 应用线性演化（dt/2）
        psi_k = psi_k * self.linear_phase
        
        # 转回位置空间
        psi_next = np.fft.ifft(psi_k)
        
        return psi_next
    
    def propagate(self, psi0: NDArray, num_steps: int, 
                  save_every: int = 1) -> Tuple[NDArray, NDArray]:
        """
        传播波函数多个时间步。
        
        Args:
            psi0: 初始波函数
            num_steps: 时间步数
            save_every: 每多少步保存一次结果
        
        Returns:
            psi_history: 保存的波函数历史
            times: 对应的时间点
        """
        psi = psi0.copy()
        
        # 保存结果
        save_indices = list(range(0, num_steps, save_every)) + [num_steps]
        save_indices = list(dict.fromkeys(save_indices))  # 去重并保持顺序
        save_indices.sort()
        
        psi_history = []
        times = []
        
        current_save_idx = 0
        
        for step in range(num_steps + 1):
            current_time = step * self.dt
            
            # 如果是需要保存的时间步
            if current_save_idx < len(save_indices) and step == save_indices[current_save_idx]:
                psi_history.append(psi.copy())
                times.append(current_time)
                current_save_idx += 1
            
            # 执行一步（除了最后一步）
            if step < num_steps:
                psi = self.step(psi)
        
        return np.array(psi_history), np.array(times)
    
    def compute_energy(self, psi: NDArray) -> float:
        """
        计算系统的总能量（守恒量）。
        
        E = ∫ (β|∇ψ|² + γ/2 |ψ|⁴ + V|ψ|²) dx
        
        Args:
            psi: 波函数
            
        Returns:
            总能量
        """
        # 动能项
        psi_k = np.fft.fft(psi)
        kinetic = self.beta * np.trapz(np.abs(self.k * psi_k)**2 / (2*np.pi), self.k)
        
        # 势能项
        potential = np.trapz(self.V * np.abs(psi)**2, self.x)
        
        # 非线性项
        nonlinear = self.gamma/2 * np.trapz(np.abs(psi)**4, self.x)
        
        return kinetic + potential + nonlinear
    
    def compute_mass(self, psi: NDArray) -> float:
        """
        计算总质量（粒子数，守恒量）。
        
        M = ∫ |ψ|² dx
        
        Args:
            psi: 波函数
            
        Returns:
            总质量
        """
        return np.trapz(np.abs(psi)**2, self.x)


def run_simulation():
    """
    运行伪谱分裂步算法的演示。
    """
    # ========== 参数设置 ==========
    # 与论文中的参数保持一致
    n = 6  # 对应2^6=64个点
    N = 2**n
    
    # 孤子参数
    a = 2.0
    x0 = -1.0
    v = 10.0
    
    # 非线性系数
    s = 1  # 非线性系数
    
    # 时间步参数
    total_time = 0.3  # 总模拟时间
    num_steps = 100  # 时间步数
    dt = total_time / num_steps  # 时间步长
    
    # ========== 创建求解器 ==========
    solver = PseudoSpectralSplitStepSolver(N=N, L=2*np.pi, 
                                           gamma=s, dt=dt)
    
    # ========== 初始条件 ==========
    x = solver.x
    psi0 = soliton_analytical_folded(0, x, x0, a, v)
    
    # 直接使用解析解作为初始条件（非归一化版本）
    print(f"Starting simulation: N={N}, dt={dt:.6f}, num_steps={num_steps}")
    print(f"Soliton parameters: a={a}, x0={x0}, v={v}, s={s}")
    
    # ========== 存储所有时间步的波函数 ==========
    # 形状：(时间步数+1, 空间点数)
    # +1是因为包括初始条件（t=0时刻）
    psi = np.zeros((num_steps+1, N), dtype=complex)
    
    # 第一个时刻：使用初始条件
    psi[0, :] = psi0
    
    # 传播波函数
    current_psi = psi0.copy()
    for i in range(num_steps):
        # 执行一个时间步
        current_psi = solver.step(current_psi)
        # 保存当前时间步的波函数
        psi[i+1, :] = current_psi
    
    print(f"\nSimulation completed!")
    
    # ========== 可视化结果 ==========
    # 绘制波包随时间的演化
    # 每10个时间步绘制一次，避免图像过于拥挤
    for i in range(0, num_steps+1, 10):
        # 绘制概率密度 |ψ|² 随位置的变化
        # x/np.pi 将位置归一化到 [-1, 1]（单位是π）
        # |ψ|²是概率密度，表示在位置x找到粒子的概率
        plt.plot(x/np.pi, np.square(np.abs(psi[i, :])), label=f"t={i*dt:.3f}")
    
    # 添加图例和标签
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.xlabel("$x$ [$\pi$]")  # x轴：位置（单位是π）
    plt.ylabel("$|\Psi|^2$")   # y轴：概率密度
    plt.tight_layout()         # 自动调整布局，确保所有元素都可见
    
    # 保存图像
    plt.savefig("./image.png", dpi=150)  # 保存为PNG格式的图像文件
    
    # 保存数据（波函数）供后续分析
    np.savez("./data.npz", psi=psi)
    
    # 显示图像
    plt.show()
    
    return solver, psi, None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pseudo-Spectral Split-Step NLSE Solver')
    args = parser.parse_args()
    
    # 运行主模拟
    solver, psi, _ = run_simulation()
