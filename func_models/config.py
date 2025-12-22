"""
配置参数模块

定义量子电路参数、初始条件、时间演化参数和优化器参数
"""


import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    """模拟配置参数类"""
    # 量子电路参数
    n: int = 6   # 量子比特数量，用于编码波函数
                 # 2^n = 64个空间点，将[-π, π]区间离散化为64个点
                 # 更多的量子比特可以提供更高的空间分辨率，但计算成本也更高
    d: int = 12  # Ansatz电路的深度（CNOT层的数量）
                 # 更深的电路可以表示更复杂的波函数，但优化也更困难
                 # 需要在表达能力和优化难度之间找到平衡
    
    # 初始条件：孤子波包的参数
    x0: float = -1  # 初始位置（在[-π, π]区间内）
                    # 孤子波包的中心位置，负值表示在区间左侧
    a: float = 2    # 孤子宽度参数（宽度与a成反比，a越大波包越窄）
                    # a控制波包的宽度，较大的a产生更窄的波包
    v: float = 10   # 孤子传播速度（向右传播）
                    # 正数表示向右传播，速度越大传播越快
    
    # 非线性强度参数
    # 注意：这里的s=4对应论文中的s=1，因为量子态总是归一化的
    # 实际非线性强度 = s / (2a)，所以s=4对应论文中的s=1（当a=2时）
    # s控制NLSE中非线性项的强度，较大的s产生更强的非线性效应
    s: float = 4
    
    # 时间演化参数
    time_steps: int = 100  # 时间步数（总共演化100步）
                           # 更多的步数可以观察更长时间的演化，但计算时间也更长
    dt: float = 3e-3       # 每个时间步的长度
                           # 较小的时间步可以提高精度，但需要更多的步数
                           # 需要满足稳定性条件：dt足够小以保证数值稳定性
    
    # 优化器参数
    initial_runs: int = 10  # 第一步的重复次数
                           # 由于第一步没有好的初始猜测，我们尝试多个随机初始值
                           # 选择最好的结果用于后续步骤
                           # 更多的重复次数可以提高找到全局最优解的概率，但计算时间也更长
    maxiter: int = 20000    # 优化器的最大迭代次数
                           # L-BFGS-B算法的最大迭代次数，防止优化过程无限进行
    maxfun: int = 100000    # 成本函数的最大评估次数
                           # 限制成本函数的调用次数，防止计算时间过长
    ftol: float = 1e-14     # 收敛容差（函数值变化的阈值）
                           # 当成本函数值的变化小于ftol时，认为优化已收敛
                           # 较小的值提供更高的精度，但可能需要更多的迭代
    
    def generate_initial_condition(self) -> np.ndarray:
        """
        从NLSE的解析解生成归一化的初始条件。
        
        Returns:
            np.ndarray: 归一化的初始波函数，长度为2^n
        """
        from .analytical_solutions import soliton_analytical_folded
        
        # 在[-π, π]区间上采样解析孤子解，作为初始波函数
        initial_condition = np.zeros(2**self.n, dtype=complex)
        x = np.linspace(-np.pi, np.pi, 2**self.n, endpoint=False)
        
        # 在每个网格点上计算解析解
        for i in range(2**self.n):
            # 计算位置：从-π开始，每个点间隔2π/(2^n)
            x_pos = -np.pi + 2*np.pi*i/(2**self.n)
            # 使用折叠的解析解（处理周期性边界条件）
            initial_condition[i] = soliton_analytical_folded(0, x_pos, self.x0, self.a, self.v)
        
        # 归一化：确保量子态的总概率为1（量子力学的基本要求：⟨ψ|ψ⟩ = 1）
        initial_condition /= np.linalg.norm(initial_condition)
        
        return initial_condition
