"""
混合伪谱-变分量子算法求解非线性薛定谔方程的成本/代价函数模块

用于变分优化，衡量当前参数化的量子态与理论演化后的态之间的差异。
"""


from typing import Callable, Union, Tuple
from numpy.typing import NDArray
import numpy as np
from qiskit import QuantumCircuit, transpile, execute, BasicAer

from .quantum_circuits import stateFromParameters
from .quantum_operations import qft, iqft, quadraticExponential


def cost_function_fast(
    parameters: NDArray[np.floating], 
    parameters_0: Union[NDArray[np.floating], NDArray[np.complexfloating]], 
    ansatz: Callable[[QuantumCircuit, int, int, int, NDArray[np.floating], bool], None], 
    n: int, 
    d: int, 
    s: float, 
    delta_t: float, 
    startFromStateVector: bool = False
) -> float:
    """
    快速成本函数（使用经典计算，除了ansatz使用qiskit）-- 即公式19的实现。
    此成本函数即为论文所提出的量子算法的最终待优化的代价函数
    
    这是变分量子算法的核心：成本函数衡量当前参数化的量子态与理论演化后的态之间的差异。
    通过最小化这个成本函数，我们找到能够准确描述NLSE时间演化的最优参数。
    
    成本函数包含两个项：
    1. 重叠项（overlap term）：-Re⟨ψ|φ⟩，衡量当前态|ψ⟩与演化后态|φ⟩的重叠
    2. 非线性项（nonlinear term）：(s*Δt/2π) * Im⟨ψ| |φ|²φ⟩，处理NLSE中的非线性项|ψ|²ψ
    
    时间演化通过以下步骤实现（使用伪谱方法）：
    - 将波函数转换到动量空间（FFT）
    - 应用动能算符的演化（在动量空间是简单的乘法：exp(-i*dt/2 * k²)，其中k²是动量空间的动能算符）
    - 转换回位置空间（IFFT）
    
    注意：这个函数使用经典FFT而不是量子傅里叶变换，因此计算速度快得多。
    这是论文中推荐的方法，因为FFT在经典计算机上已经非常高效。
    
    Args:
        parameters: 当前时刻的参数（要优化的参数），长度为2*n*(d+1)
        parameters_0: 前一时刻的参数（或初始态向量）
                      - 如果是参数：长度为2*n*(d+1)的数组
                      - 如果是状态向量：长度为2^n的复数数组
        ansatz: ansatz函数，用于从参数生成量子态
        n: 量子比特数量
        d: 电路深度
        s: 非线性强度参数，控制NLSE中非线性项的强度
        delta_t: 时间步长，每个时间步的长度
        startFromStateVector: 如果为True，parameters_0是状态向量而不是参数
    
    Returns:
        float: 成本函数值（越小越好，理想情况下应该接近0）

    Note:
        - 为什么代码中不存在如公式（19）中的参数a？
          - 论文公式 (19) 是在物理波函数空间 Ψ(x,t)，带有特定归一化 ∫∥Ψ∥dx=2a
          - 但代码实现是在归一化量子态空间 ∥ψ(t)⟩，满足 ⟨ψ∥ψ⟩=1 ； 并且在公式（14）中就已经体现了向量子态空间转化
          - 代码中假设输入的 psi_0 已经是一个正确的、归一化的量子态（这个态的制备过程会用到 a）
          - 而成本函数本身只需要在这个抽象的希尔伯特空间里工作，因此不再需要显式的 a 参数
    """    
    # ===== 步骤1：获取前一时刻的态 =====
    if startFromStateVector:
        # 如果直接提供了状态向量，直接使用
        psi_0 = parameters_0
    else:
        # 否则从参数生成量子态
        psi_0 = np.array(stateFromParameters(ansatz, parameters_0, n, d))
    
    # ===== 步骤2：获取当前试探态 =====
    # 从当前参数生成试探态|ψ⟩，这是我们要优化的态
    psi = np.array(stateFromParameters(ansatz, parameters, n, d))

    # ===== 步骤3：线性演化：e^{-i Δt/2 T} |psi_0> =====
    # 在动量空间应用拉普拉斯算子的演化
    # T是动能算符（拉普拉斯算子）
    # 注意：在位置空间，拉普拉斯算子是 ∂²/∂x²，在动量空间对应 -k²
    # 但在这个实现中，动能算符 T 在动量空间被定义为 k²（而不是标准的 ½k²）
    # 因此演化算子 exp(-i*dt/2 * T) 在动量空间是 exp(-i*dt/2 * k²)
    
    # 转换到动量空间（傅里叶变换）
    # fftshift将零频率移到中心，便于处理
    psi_0 = np.fft.fftshift(np.fft.fft(psi_0))
    
    # 在动量空间应用拉普拉斯算子的演化
    # 演化算子：exp(-i*dt/2 * k²)
    # 这里k的范围是 -2^n/2 到 2^n/2-1
    k_values = np.linspace(-2**n/2, 2**n/2-1, 2**n)
    psi_0 *= np.exp( -1j * delta_t/2 * np.square(k_values) )

    # 修正相位（由于FFT的索引偏移）
    # 校正因 fftshift 和离散波数定义引入的全局相位偏移，确保数值解与解析解的相位一致
    psi_0 *= np.exp( 1j * delta_t/2 * (-2**n/2)**2 )
    
    # 转换回位置空间（逆傅里叶变换）
    # 得到演化后的态|φ⟩ = e^{-i Δt/2 T} |psi_0>
    # 这边需要注意的是没有存在对于exp(-iVdelta_t)的校正，因为这里使用的是伪谱方法，不需要考虑势能项(该部分为单位阵在此处无影响)
    psi_0 = np.fft.ifft(np.fft.ifftshift(psi_0))
    
    # ===== 步骤4：计算成本函数 =====
    # 成本函数 = 重叠项 + 非线性项
    
    # 重叠项：-Re⟨ψ|φ⟩
    # 衡量当前试探态|ψ⟩与演化后态|φ⟩的重叠
    overlap_term = -np.real( np.vdot(psi, psi_0) )
    
    # 非线性项：(s Δt / 2π) * Im⟨ψ| |φ|²φ⟩
    # 处理NLSE中的非线性项|ψ|²ψ
    # |φ|²是概率密度，|φ|²φ是非线性项
    # 虚部是因为非线性项在NLSE中是纯虚数
    nonlinear_term = s * delta_t / (2*np.pi) * (2**n) * np.imag( np.vdot( psi, np.square(np.abs(psi_0)) * psi_0 ) )
    
    return overlap_term + nonlinear_term


def cost_function_qiskit(
    parameters: NDArray[np.floating], 
    parameters_0: Union[NDArray[np.floating], NDArray[np.complexfloating]], 
    ansatz: Callable[[QuantumCircuit, int, int, int, NDArray[np.floating], bool], None], 
    n: int, 
    d: int, 
    s: float, 
    delta_t: float, 
    startFromStateVector: bool = False, 
    return_circuits: bool = False
) -> Union[float, Tuple[float, QuantumCircuit, QuantumCircuit]]:
    """
    完整的量子成本函数（完全使用qiskit实现）。
    
    这个函数使用量子电路直接计算成本函数，而不是使用经典FFT。
    它构建了复杂的量子电路来测量重叠项和非线性项。
    
    注意：此函数比cost_function_fast慢得多，通常不会被使用。
    论文明确说明在经典端用FFT完成，这个程序主要用于演示如何在真正的量子计算机上实现算法。
    
    实现原理：
    1. 使用量子傅里叶变换（QFT）代替经典FFT
    2. 使用Hadamard测试（Hadamard test）来测量两个态的重叠
    3. 使用复杂的多寄存器电路来测量非线性项
    
    Args:
        parameters: 当前时刻的参数（要优化的参数）
        parameters_0: 前一时刻的参数或状态向量
        ansatz: ansatz函数
        n: 量子比特数量
        d: 电路深度
        s: 非线性强度
        delta_t: 时间步长
        startFromStateVector: 如果为True，parameters_0是状态向量
        return_circuits: 如果为True，返回使用的量子电路（用于调试和可视化）
    
    Returns:
        float或tuple: 成本函数值，或(成本函数值, 重叠电路, 非线性电路)
    """
    # ===== 步骤1：准备前一时刻的量子态并应用时间演化 =====
    qc_old = QuantumCircuit(n)
    if startFromStateVector:
        # 如果直接提供了状态向量，直接准备该态
        qc_old.prepare_state(parameters_0)
    else:
        # 否则从参数生成量子态
        ansatz(qc_old, n, d, 0, parameters_0)
    
    # 应用时间演化：QFT -> 动能算符 -> IQFT
    qft(qc_old, n)  # 量子傅里叶变换
    qc_old.x(n-1)   # X门用于调整索引偏移（因为QFT的索引顺序）
    quadraticExponential(qc_old, -delta_t/2, n)  # 应用动能算符的演化 exp(-i*dt/2 * k²)
    qc_old.x(n-1)   # 再次X门恢复
    iqft(qc_old, n) # 逆量子傅里叶变换

    # ===== 步骤2：将演化后的态转换为受控门 =====
    # 将整个电路编译为单个门，以便在更大的电路中使用
    unitary_old_noControl = transpile(qc_old).to_gate(label="old_nc")  # 无控制版本
    unitary_old = transpile(qc_old).to_gate(label="prev").control()   # 受控版本（用于重叠计算）

    # ===== 步骤3：准备当前时刻的试验态 =====
    qc_new = QuantumCircuit(n)
    ansatz(qc_new, n, d, 0, parameters)  # 从当前参数生成试验态
    unitary_new = transpile(qc_new).to_gate(label="trial").control()  # 转换为受控门

    # ===== 步骤4：准备试验态的共轭 =====
    # 非线性项需要计算 <ψ| |ψ_0|² |ψ_0>，这需要共轭态
    qc_new_conj = QuantumCircuit(n)
    ansatz(qc_new_conj, n, d, 0, parameters, True)  # conj=True使用负参数生成共轭态
    unitary_new_conj = transpile(qc_new_conj).to_gate(label="trial*").control()

    # ===== 步骤5：计算重叠项 <ψ|ψ_0> =====
    # 使用Hadamard测试（Hadamard test）来测量两个态的重叠
    # 原理：通过辅助量子比特和受控门来测量 <ψ|U|ψ_0>
    qc_overlap = QuantumCircuit(n+1)  # n个数据量子比特 + 1个辅助量子比特

    # Hadamard测试的标准结构
    qc_overlap.h(0)  # 在辅助量子比特上应用H门，创建叠加态 (|0⟩+|1⟩)/√2

    # 应用受控的旧态和试验态的逆演化
    qc_overlap.append(unitary_old, range(n+1))  # 受控的旧态演化
    qc_overlap.append(unitary_new.inverse(), range(n+1))  # 受控的试验态逆演化

    qc_overlap.h(0)  # 再次应用H门

    # 执行电路并测量
    backend = BasicAer.get_backend("statevector_simulator")
    job = execute(qc_overlap, backend)
    counts = job.result().get_counts()

    # 计算辅助量子比特为|0⟩的概率
    # 这个概率与重叠的实部相关：P(0) = (1 + Re<ψ|ψ_0>)/2
    prob = 0
    for key, value in counts.items():
        if key[-1] == "0":  # 检查辅助量子比特（最后一个）是否为0
            prob += value
    
    # 从概率提取重叠：Re<ψ|ψ_0> = 2*P(0) - 1
    # 我们想要 -Re<ψ|ψ_0>（因为要最小化）
    val_overlap = (-1) * (2*prob-1)

    # ===== 步骤6：计算非线性项 =====
    # 非线性项 <ψ| |ψ_0|² |ψ_0> 需要更复杂的电路
    # 需要3个n量子比特的寄存器来存储 |ψ_0⟩, |ψ*⟩, |ψ_0⟩
    qc_NL = QuantumCircuit(3*n+1)  # 3个n量子比特寄存器 + 1个辅助量子比特

    # 准备辅助量子比特的特定叠加态（用于测量虚部）
    qc_NL.h(0)
    qc_NL.sdg(0)  # S†门，用于测量虚部而不是实部

    qc_NL.barrier()  # 可视化分隔

    # 在第一个寄存器中准备演化后的旧态
    qc_NL.append(unitary_old_noControl, range(1, n+1))
    
    qc_NL.barrier()

    # 在第二个和第三个寄存器中准备受控的态
    # 第二个寄存器：试验态的共轭
    qc_NL.append(unitary_new_conj, [0] + list(range(n+1, 2*n+1)))
    # 第三个寄存器：旧态
    qc_NL.append(unitary_old, [0] + list(range(2*n+1, 3*n+1)))

    qc_NL.barrier()

    # 计算 |ψ_0|²：通过受控的Toffoli门（CCX）来实现
    # 这需要将第一个寄存器的内容复制到第二和第三个寄存器
    # 当辅助量子比特为|1⟩时，执行复制操作
    for i in range(n):
        qc_NL.ccx(0, 1+i, n+1+i)  # 复制到第二个寄存器

    for i in range(n):
        qc_NL.ccx(0, 1+i, 2*n+1+i)  # 复制到第三个寄存器

    qc_NL.barrier()

    # 再次应用H门完成测量
    qc_NL.h(0)

    # 执行电路并测量
    backend = BasicAer.get_backend("statevector_simulator")
    job = execute(qc_NL, backend)
    counts = job.result().get_counts()

    # 计算概率并提取非线性项
    prob = 0
    for key, value in counts.items():
        if key[-1] == "0":
            prob += value

    # 非线性项的计算（包含归一化因子）
    val_NL = delta_t * (2*prob-1) / (2*np.pi) * (2**n) * s

    # ===== 返回结果 =====
    if return_circuits:
        # 如果请求，返回电路用于调试
        return val_overlap+val_NL, qc_overlap, qc_NL
    else:
        return val_overlap+val_NL
