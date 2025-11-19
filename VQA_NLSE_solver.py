"""
混合伪谱-变分量子算法求解非线性薛定谔方程（Hybrid Pseudo-Spectral VQA-NLSE Solver）

项目说明：
==========
本程序使用混合伪谱-变分量子算法（Hybrid Pseudo-Spectral Variational Quantum Algorithm）来数值求解
非线性薛定谔方程（Nonlinear Schrödinger Equation, NLSE）。NLSE是描述波包在非线性介质中演化的重要
物理方程，在光学、凝聚态物理、等离子体物理等领域有广泛应用。

核心思想：
----------
1. 波函数编码：将波函数编码到量子态中，使用n个量子比特编码2^n维的波函数
2. 参数化表示：使用参数化的量子电路（ansatz）来表示波函数
3. 变分优化：通过优化算法找到最优参数，使得量子态能够准确描述NLSE的演化
4. 混合方法：时间演化采用混合方法
   - 线性部分（拉普拉斯算子）：使用伪谱方法（FFT/IFFT）在动量空间精确处理
   - 非线性部分：在位置空间处理，通过变分优化找到最优量子态表示

算法流程：
----------
1. 初始化：从解析孤子解（soliton solution）开始，生成归一化的初始波函数
2. 时间步进：对每个时间步
   a) 使用FFT将前一步的态转换到动量空间
   b) 在动量空间应用线性演化算子 exp(-i*dt/2 * k²)
   c) 使用IFFT转换回位置空间
   d) 通过变分优化（最小化成本函数）找到最优参数，使得量子态最接近演化后的态
3. 状态提取：从优化后的参数中提取量子态
4. 可视化：绘制波包随时间的演化

混合方法说明：
--------------
本算法结合了两种数值方法的优势：
- 伪谱方法（Pseudo-Spectral Method）：利用FFT高效处理线性微分算子，在动量空间精确计算
- 变分量子算法（VQA）：使用参数化量子电路紧凑表示高维波函数，通过优化找到最优表示
这种混合方法既保持了伪谱方法的高精度，又利用了量子算法的压缩表示优势。
"""

# ============================================================================
# 导入必要的库
# ============================================================================
import numpy as np  
import matplotlib.pyplot as plt 

from qiskit.algorithms.optimizers import L_BFGS_B  # L-BFGS-B优化器，用于寻找最优参数
from qiskit import QuantumCircuit, transpile, execute, BasicAer  # 量子电路相关模块
from qiskit.quantum_info import Statevector  # 量子态向量类


# ============================================================================
# 第一部分：解析解函数
# 用于生成初始条件和验证数值结果的正确性
# ============================================================================

def soliton_analytical(t, x, x0, a, v):
    """
    非线性薛定谔方程的解析孤子解（在整个实数轴上）。
    
    孤子解是一种特殊的波包解，它在传播过程中保持形状不变。这是NLSE的一个精确解，可以用来验证数值方法的正确性。
    
    数学表达式：
        ψ(x,t) = a / cosh(a*(x-x0-v*t)) * exp(i*v*(x-x0) + i*(a²-v²)*t/2)，即论文中公式3
    
    Args:
        t: 时间（标量或数组）
        x: 空间位置（标量或数组）
        x0: 初始位置，孤子波包的中心位置
        a: 孤子宽度参数，宽度与a成反比（a越大，波包越窄）
        v: 孤子传播速度，正数表示向右传播
    
    Returns:
        complex: 孤子解的复振幅值，形状与输入x相同
    """
    # 计算包络部分：a / cosh(a*(x-x0-v*t))
    envelope = a / np.cosh(a*(x-x0-v*t))
    
    # 计算相位部分：exp(i*v*(x-x0) + i*(a²-v²)*t/2)
    phase = np.exp(1j*v*(x-x0) + 1j*(a*a-v*v)*t/2.0)
    
    return envelope * phase


def soliton_analytical_folded(t, x, x0, a, v):
    """
    将解析孤子解折叠到周期区间 [-π, π] 上。
    
    由于我们使用量子比特编码，空间被离散化为有限个点，并且采用周期性边界条件。
    为了处理周期性边界条件，我们需要将无限域上的解折叠到有限区间[-π, π]。
    
    方法：考虑多个周期（x-4π, x-2π, x, x+2π, x+4π）的候选值，选择幅度最大的那个。
    这是因为孤子解在远离中心时快速衰减，所以最接近真实孤子中心的那个周期应该具有最大的幅度。
    
    Args:
        t: 时间（标量或数组）
        x: 空间位置（在[-π, π]区间内）
        x0: 初始位置
        a: 孤子宽度参数
        v: 孤子传播速度
    
    Returns:
        complex: 折叠后的孤子解值，选择幅度最大的周期
    """
    # 考虑5个周期：x-4π, x-2π, x, x+2π, x+4π，这样可以覆盖孤子可能出现的所有位置
    candidates = [soliton_analytical(t, x+i*2*np.pi, x0, a, v) for i in range(-2, 3)]
    
    # 按幅度大小排序，选择最大的（最接近真实值的）
    # 使用np.abs的平方作为排序键，因为幅度越大表示越接近孤子中心
    candidates.sort(key=lambda val: np.square(np.abs(val)))
    
    # 返回幅度最大的候选值
    return candidates[-1]


# ============================================================================
# 第二部分：量子电路构建函数
# 用于构建参数化的ansatz电路，这是变分量子算法的核心
# ============================================================================

def ansatz(qc, n, d, offset, parameters, conj=False):
    """
    构建参数化的ansatz量子电路。
    
    Ansatz是变分量子算法的核心，它是一个参数化的量子电路，用于表示波函数。
    这个函数构建了一个由旋转门（RX, RZ）和CNOT门组成的电路。
    
    电路结构：
    - 每一层包含：对所有量子比特应用RX和RZ旋转门（参数可调）
    - 层与层之间用CNOT门连接，形成纠缠
    - 总共d+1层旋转门，d层CNOT连接
    
    参数索引规则：
    - 参数总数为 2 * n * (d+1)
    - 对于第i层（i=0,1,...,d）的第j个量子比特（j=0,1,...,n-1）：
      * RX旋转角度：parameters[2*(i*n+j)]
      * RZ旋转角度：parameters[2*(i*n+j)+1]
    
    共轭态构建：
    - 当conj=True时，使用参数的负值来构建共轭态
    - 这是因为旋转门的共轭是反向旋转：R(θ)† = R(-θ)
    
    Args:
        qc: 量子电路对象（会被修改，直接在该电路上添加门）
        n: 量子比特数量（编码2^n维波函数）
        d: 电路深度（CNOT层的数量，总共有d+1层旋转门）
        offset: 量子比特的起始索引（用于多量子比特系统，允许在指定位置开始构建电路）
        parameters: 参数数组，长度为2*n*(d+1)，包含所有旋转角度
        conj: 如果为True，使用参数的负值（用于构建共轭态|ψ*⟩）
    
    Note:
        参数的总数 = 2 * n * (d+1)
        因为每个量子比特在每一层需要2个旋转角度（RX和RZ）
    """
    # 根据是否需要共轭，决定参数的符号
    # 共轭态使用负参数，因为旋转门的共轭是反向旋转
    sign = -1 if conj else 1

    # 构建d+1层旋转门
    for i in range(d+1):
        # 对每个量子比特应用旋转门
        for j in range(n):
            # RX旋转：绕X轴旋转，参数索引为 2*(i*n+j)
            # RX(θ) = exp(-i*θ*σ_x/2)，其中σ_x是Pauli-X矩阵
            qc.rx(sign * parameters[2*(i*n+j)], j+offset)
            
            # RZ旋转：绕Z轴旋转，参数索引为 2*(i*n+j)+1
            # RZ(θ) = exp(-i*θ*σ_z/2)，其中σ_z是Pauli-Z矩阵
            qc.rz(sign * parameters[2*(i*n+j)+1], j+offset)

        # 在层之间添加CNOT门来创建纠缠（最后一层不需要）
        if i != d:
            # 创建链式CNOT：0->1->2->...->(n-2)->(n-1)
            # 这是线性链结构，无闭环，用于创建相邻量子比特之间的纠缠
            for k in range(n - 1):
                # CNOT门：控制比特k，目标比特k+1
                # 当控制比特为|1⟩时，翻转目标比特
                qc.cx(k + offset, k + 1 + offset)


def stateFromParameters(ansatz, parameters, n, d):
    """
    从给定的ansatz和参数生成理想的状态向量，即当前参数所定义的试探波函数|ψ(λ)⟩。
    
    这个函数构建量子电路，应用参数，然后提取最终的量子态。
    这是将参数化电路转换为实际量子态的关键步骤，用于：
    1. 计算成本函数（需要知道当前参数对应的量子态）
    2. 提取优化后的波函数（用于可视化和分析）
    
    工作流程：
    1. 创建n个量子比特的空电路
    2. 应用ansatz电路（从索引0开始）
    3. 提取理想的状态向量（无噪声的完美量子态）
    
    Args:
        ansatz: ansatz函数（上面定义的函数），用于构建参数化电路
        parameters: 参数数组，长度为2*n*(d+1)，包含所有旋转角度
        n: 量子比特数量
        d: 电路深度
    
    Returns:
        Statevector: 2^n维的量子态向量（复数数组），表示当前参数对应的波函数
    """
    # 创建n个量子比特的电路
    qc = QuantumCircuit(n)
    
    # 应用ansatz电路（从索引0开始）
    ansatz(qc, n, d, 0, parameters)
    
    # 返回理想的状态向量（无噪声的完美量子态）
    # Statevector会计算电路作用在|00...0⟩上的结果
    return Statevector(qc)


# ============================================================================
# 第三部分：成本/代价函数
# 用于变分优化，衡量当前参数化的量子态与理论演化后的态之间的差异
# ============================================================================

def cost_function_fast(parameters, parameters_0, ansatz, n, d, s, delta_t, startFromStateVector=False):
    """
    快速成本函数（使用经典计算，除了ansatz使用qiskit）-- 即公式19的实现。
    
    这是变分量子算法的核心：成本函数衡量当前参数化的量子态与理论演化后的态之间的差异。
    通过最小化这个成本函数，我们找到能够准确描述NLSE时间演化的最优参数。
    
    成本函数包含两个项：
    1. 重叠项（overlap term）：-Re⟨ψ|φ⟩，衡量当前态|ψ⟩与演化后态|φ⟩的重叠
    2. 非线性项（nonlinear term）：(s*Δt/2π) * Im⟨ψ| |φ|²φ⟩，处理NLSE中的非线性项|ψ|²ψ
    
    时间演化通过以下步骤实现（使用伪谱方法）：
    - 将波函数转换到动量空间（FFT）
    - 应用拉普拉斯算子的演化（在动量空间是简单的乘法：exp(-i*dt/2 * k²)）
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
    """
    N = 2**n  # 空间离散化点数
    
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
    # T是动能算符（拉普拉斯算子），在动量空间中是k²
    
    # 3.1 转换到动量空间（FFT）
    psi_k = np.fft.fft(psi_0)
    
    # 3.2 构造动量网格k
    # FFT的默认顺序是[0, 1, 2, ..., N/2-1, -N/2, ..., -1]
    # 我们需要重新排列为[-N/2, ..., -1, 0, 1, ..., N/2-1]
    k = np.concatenate([np.arange(0, N//2), np.arange(-N//2, 0)])
    
    # 3.3 在动量空间应用演化算子：exp(-i*dt/2 * k²)
    # 这是拉普拉斯算子在动量空间的表示
    psi_k_evolved = psi_k * np.exp(-1j * delta_t / 2 * k**2)
    
    # 3.4 转换回位置空间（IFFT）
    # 得到演化后的态|φ⟩ = e^{-i Δt/2 T} |psi_0>
    phi = np.fft.ifft(psi_k_evolved)

    # ===== 步骤4：计算成本函数 =====
    # 成本函数 = 重叠项 + 非线性项
    
    # 4.1 重叠项：-Re⟨ψ|φ⟩
    # 衡量当前试探态|ψ⟩与演化后态|φ⟩的重叠
    # 负号是因为我们要最小化，而重叠越大越好
    overlap_term = -np.real(np.vdot(psi, phi))
    
    # 4.2 非线性项：(s Δt / 2π) * Im⟨ψ| |φ|²φ⟩
    # 处理NLSE中的非线性项|ψ|²ψ
    # |φ|²是概率密度，|φ|²φ是非线性项
    # 虚部是因为非线性项在NLSE中是纯虚数
    nonlinear_term = (s * delta_t / (2 * np.pi)) * np.imag(np.vdot(psi, np.abs(phi)**2 * phi))
    
    return overlap_term + nonlinear_term


def cost_function_qiskit(parameters, parameters_0, ansatz, n, d, s, delta_t, startFromStateVector=False, return_circuits=False):
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
    
    # 应用时间演化：QFT -> 拉普拉斯算子 -> IQFT
    # 这相当于在动量空间应用演化，然后转换回位置空间
    qft(qc_old, n)  # 量子傅里叶变换（转换到动量空间）
    qc_old.x(n-1)   # X门用于调整索引偏移（因为QFT的索引顺序）
    quadraticExponential(qc_old, -delta_t/2, n)  # 应用拉普拉斯算子的演化 exp(-i*dt/2 * k²)
    qc_old.x(n-1)   # 再次X门恢复
    iqft(qc_old, n) # 逆量子傅里叶变换（转换回位置空间）

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

    # 应用受控的旧态和试验态的逆
    qc_overlap.append(unitary_old, range(n+1))        # 受控的旧态演化
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


# ============================================================================
# 第四部分：辅助函数
# 用于构建量子电路的量子傅里叶变换和拉普拉斯算子
# ============================================================================

def qft(circuit, n, indexOffset=0):
    """
    量子傅里叶变换（Quantum Fourier Transform, QFT）。
    
    QFT是经典FFT的量子版本，用于在位置空间和动量空间之间转换。
    在NLSE求解中，我们使用QFT将波函数转换到动量空间，在那里拉普拉斯算子
    的演化变得简单（只是相位旋转）。
    
    算法实现：
    1. 对每个量子比特应用Hadamard门和受控相位门
    2. 交换量子比特的顺序（因为QFT的输出是位反转的）
    
    注意：虽然这个函数在代码中定义，但实际使用的是经典FFT（在cost_function_fast中），
    因为FFT在经典计算机上已经非常高效。这个函数主要用于cost_function_qiskit。
    
    Args:
        circuit: 量子电路对象（会被修改）
        n: 量子比特数量
        indexOffset: 起始索引（用于多量子比特系统，允许在指定位置开始构建电路）
    """
    # 第一遍：应用Hadamard和受控相位门
    # 从最高位到最低位（反向遍历）
    for i in reversed(range(n)):
        circuit.h(i+indexOffset)  # Hadamard门，创建叠加态
        # 对之前的每个量子比特应用受控相位门
        for j in range(i):
            # 相位 = π / 2^(i-j)
            # 这是QFT的核心：相位门实现傅里叶变换的相位因子
            circuit.cp(np.pi/2**(i-j), j+indexOffset, i+indexOffset)

    # 第二遍：交换量子比特以纠正位反转
    # QFT的输出是位反转的，需要交换量子比特来纠正
    for i in range(n//2):
        circuit.swap(i+indexOffset, n-1-i+indexOffset)


def iqft(circuit, n, indexOffset=0):
    """
    逆量子傅里叶变换（Inverse Quantum Fourier Transform, IQFT）。
    
    IQFT是QFT的逆操作，用于从动量空间转换回位置空间。
    实现上，它是QFT的逆序操作，相位取负。
    
    算法实现：
    1. 首先交换量子比特（与QFT相反的顺序）
    2. 然后应用Hadamard和受控相位门（相位取负）
    
    注意：虽然这个函数在代码中定义，但实际使用的是经典IFFT（在cost_function_fast中）。
    这个函数主要用于cost_function_qiskit。
    
    Args:
        circuit: 量子电路对象（会被修改）
        n: 量子比特数量
        indexOffset: 起始索引（用于多量子比特系统）
    """
    # 首先交换量子比特（与QFT相反的顺序）
    for i in range(n//2):
        circuit.swap(i+indexOffset, n-1-i+indexOffset)

    # 然后应用Hadamard和受控相位门（相位取负）
    # 从最低位到最高位（正向遍历）
    for i in range(n):
        for j in reversed(range(i)):
            # 负相位：-π / 2^(i-j)
            # 这是IQFT的核心：负相位实现逆傅里叶变换
            circuit.cp(-np.pi/2**(i-j), j+indexOffset, i+indexOffset)
        circuit.h(i+indexOffset)


def U_ij(circuit, i, j, gamma, indexOffset=0):
    """
    实现 x² 的交叉项相位：exp(i * gamma * 2 * 2^(i+j) * b_i * b_j)。
    
    这个函数用于实现二次型相位算子的交叉项。当我们需要实现exp(i*gamma*x²)时，
    x²可以展开为对角项和交叉项的和。这个函数处理交叉项。
    
    数学原理：
    - x = -2^(n-1) + Σ b_k * 2^k（二进制表示）
    - x²的交叉项：b_i * b_j * 2^(i+j+1)
    - 相位：gamma * 2 * 2^(i+j) = gamma * 2^(i+j+1)
    
    注意：此函数假设 i < j，且只调用一次（避免重复计算）。
    
    Args:
        circuit: 量子电路对象（会被修改）
        i, j: 量子比特索引，满足 i < j
        gamma: 相位参数（例如 gamma = -delta_t / 2）
        indexOffset: 寄存器起始偏移（用于多量子比特系统）
    """
    # 计算相位：gamma * 2^(i+j+1)
    phase = 2 * gamma * (2 ** (i + j))
    
    # 应用受控相位门：当b_i和b_j都为1时，应用相位
    circuit.cp(phase, i + indexOffset, j + indexOffset)


def U_ii(circuit, i, beta, gamma, indexOffset=0):
    """
    实现 x² 的对角项相位：exp(i * gamma * (2^(2i) + 2*beta*2^i) * b_i)。
    
    这个函数用于实现二次型相位算子的对角项。当我们需要实现exp(i*gamma*x²)时，
    x²可以展开为对角项和交叉项的和。这个函数处理对角项。
    
    数学原理：
    - x = beta + Σ b_k * 2^k，其中beta = -2^(n-1)是偏移量
    - x²的对角项：b_i² * 2^(2i) + 2*beta*b_i * 2^i
    - 由于b_i是二进制（0或1），b_i² = b_i
    - 相位：gamma * (2^(2i) + 2*beta*2^i)
    
    Args:
        circuit: 量子电路对象（会被修改）
        i: 量子比特索引
        beta: 偏移量，beta = -2^(n-1)，用于处理有符号整数表示
        gamma: 相位参数（例如 gamma = -delta_t / 2）
        indexOffset: 寄存器起始偏移（用于多量子比特系统）
    """
    # 计算相位：gamma * (2^(2i) + 2*beta*2^i)
    phase = gamma * (2 ** (2 * i) + 2 * beta * (2 ** i))
    
    # 应用相位门：当b_i为1时，应用相位
    circuit.p(phase, i + indexOffset)


def quadraticExponential(circuit, gamma, n, indexOffset=0):
    """
    实现 exp(i * gamma * x²)，其中 x = -2^(n-1) + sum_{k=0}^{n-1} b_k * 2^k。
    
    这对应于动量空间中的拉普拉斯演化：exp(-i * (delta_t / 2) * k²)
    此时应传入 gamma = -delta_t / 2。
    
    实现原理：
    - x²可以展开为对角项和交叉项的和
    - 对角项：使用U_ii函数实现
    - 交叉项：使用U_ij函数实现（i < j，避免重复）
    
    编码说明：
    - 计算基态 |b_{n-1}...b_0> 表示整数 x ∈ [-2^(n-1), 2^(n-1) - 1]
    - 这是二进制补码表示，用于表示有符号整数
    
    Args:
        circuit: 量子电路对象（会被修改）
        gamma: 相位系数（实数），例如 gamma = -delta_t / 2
        n: 量子比特数量
        indexOffset: 寄存器起始偏移（默认 0）
    """
    # 计算偏移量beta，使得 x = beta + Σ b_i 2^i
    # beta = -2^(n-1) 使得x的范围是[-2^(n-1), 2^(n-1)-1]
    beta = - (2 ** (n - 1))

    # 对角项：i == j
    # 对每个量子比特应用对角项相位
    for i in range(n):
        U_ii(circuit, i, beta, gamma, indexOffset)

    # 交叉项：i < j（避免重复，因为U_ij已经处理了i<j的情况）
    # 对所有量子比特对应用交叉项相位
    for i in range(n):
        for j in range(i + 1, n):
            U_ij(circuit, i, j, gamma, indexOffset)


# ============================================================================
# 第五部分：模拟参数设置
# 定义量子电路参数、初始条件、时间演化参数和优化器参数
# ============================================================================

# ----- 量子电路参数 -----
n = 6   # 量子比特数量，用于编码波函数
        # 2^n = 64个空间点，将[-π, π]区间离散化为64个点
        # 更多的量子比特可以提供更高的空间分辨率，但计算成本也更高
d = 12  # Ansatz电路的深度（CNOT层的数量）
        # 更深的电路可以表示更复杂的波函数，但优化也更困难
        # 需要在表达能力和优化难度之间找到平衡

# ----- 初始条件：孤子波包的参数 -----
x0 = -1  # 初始位置（在[-π, π]区间内）
         # 孤子波包的中心位置，负值表示在区间左侧
a = 2    # 孤子宽度参数（宽度与a成反比，a越大波包越窄）
         # a控制波包的宽度，较大的a产生更窄的波包
v = 10   # 孤子传播速度（向右传播）
         # 正数表示向右传播，速度越大传播越快

# ----- 非线性强度参数 -----
# 注意：这里的s=4对应论文中的s=1，因为量子态总是归一化的
# 实际非线性强度 = s / (2a)，所以s=4对应论文中的s=1（当a=2时）
# s控制NLSE中非线性项的强度，较大的s产生更强的非线性效应
s = 4

# ----- 时间演化参数 -----
time_steps = 100  # 时间步数（总共演化100步）
                  # 更多的步数可以观察更长时间的演化，但计算时间也更长
dt = 3e-3         # 每个时间步的长度
                  # 较小的时间步可以提高精度，但需要更多的步数
                  # 需要满足稳定性条件：dt足够小以保证数值稳定性

# ----- 优化器参数 -----
initial_runs = 10  # 第一步的重复次数
                   # 由于第一步没有好的初始猜测，我们尝试多个随机初始值
                   # 选择最好的结果用于后续步骤
                   # 更多的重复次数可以提高找到全局最优解的概率，但计算时间也更长
maxiter = 20000    # 优化器的最大迭代次数
                   # L-BFGS-B算法的最大迭代次数，防止优化过程无限进行
maxfun = 100000    # 成本函数的最大评估次数
                   # 限制成本函数的调用次数，防止计算时间过长
ftol = 1e-14       # 收敛容差（函数值变化的阈值）
                   # 当成本函数值的变化小于ftol时，认为优化已收敛
                   # 较小的值提供更高的精度，但可能需要更多的迭代

# ===== 从解析解生成归一化的初始条件 =====
# 在[-π, π]区间上采样解析孤子解，作为初始波函数
initial_condition = np.zeros(2**n, dtype=complex)  # 初始化复数数组，存储初始波函数
x = np.linspace(-np.pi, np.pi, 2**n, endpoint=False)  # 空间网格点，均匀分布在[-π, π)区间

# 在每个网格点上计算解析解
for i in range(2**n):
    # 计算位置：从-π开始，每个点间隔2π/(2^n)
    x_pos = -np.pi + 2*np.pi*i/(2**n)
    # 使用折叠的解析解（处理周期性边界条件）
    # 这确保了解在周期性边界条件下是正确的
    initial_condition[i] = soliton_analytical_folded(0, x_pos, x0, a, v)

# 归一化：确保量子态的总概率为1
# 这是量子力学的基本要求：⟨ψ|ψ⟩ = 1
initial_condition /= np.linalg.norm(initial_condition)


# ============================================================================
# 第六部分：主模拟循环
# 使用变分量子算法求解NLSE，通过优化找到每个时间步的最优参数
# ============================================================================
# 
# 注意：如果想使用完整的量子电路实现（而不是快速经典方法），
# 可以将下面两处 "cost_function_fast" 替换为 "cost_function_qiskit"
# 但这样会慢很多，因为需要构建和执行大量量子电路
# ============================================================================

# 初始化L-BFGS-B优化器
# L-BFGS-B是一种准牛顿优化算法，适合处理有界约束的优化问题
# 它是L-BFGS算法的有界版本，在变分量子算法中广泛使用
optimizer = L_BFGS_B(maxiter=maxiter, maxfun=maxfun, ftol=ftol)

# 存储每个时间步的最优参数
# 形状：(时间步数, 参数数量)
# 参数数量 = 2 * n * (d+1)（每个量子比特每层需要2个旋转角度）
parameters = np.zeros((time_steps, 2*n*(d+1)))

# ===== 第一步：从初始条件开始 =====
# 定义第一步的成本函数
# startFromStateVector=True表示parameters_0是状态向量而不是参数
cost_function = lambda x: cost_function_fast(x, initial_condition, ansatz, n, d, s, dt, True)

# 记录最佳成本函数值
fvalue_opt = float("Inf")

# 尝试多个随机初始值，选择最好的
# 这是因为第一步没有好的初始猜测，需要多次尝试
# 使用随机初始值可以避免陷入局部最优解
for i in range(initial_runs):
    print("Step 1/{time_steps} : {i}/{initial_runs}".format(time_steps=time_steps, i=i+1, initial_runs=initial_runs))

    # 使用随机初始参数进行优化
    # 参数范围：[0, 2π]，因为旋转角度是周期性的
    # 随机初始化可以探索参数空间的不同区域
    initial_params = 2*np.pi*np.random.random(2*n*(d+1))
    
    # 执行优化，找到最小化成本函数的参数
    sol = optimizer.minimize(cost_function, initial_params)
    
    # 评估优化后的成本函数值
    fvalue_temp = cost_function(sol.x)

    # 如果找到更好的解，保存它
    # 这确保我们使用最好的初始参数进行后续步骤
    if fvalue_temp < fvalue_opt:
        fvalue_opt = fvalue_temp
        parameters[0, :] = sol.x

# ===== 后续步骤：从前一步的参数开始 =====
# 对于后续步骤，我们可以使用前一步的参数作为初始猜测
# 这通常能大大加快收敛速度，因为相邻时间步的波函数应该相似
for i in range(1, time_steps):
    print("Step {i}/{time_steps}".format(i=i+1, time_steps=time_steps))

    # 定义当前步的成本函数
    # 使用前一步的参数作为参考态
    cost_function = lambda x: cost_function_fast(x, parameters[i-1, :], ansatz, n, d, s, dt)
    
    # 从前一步的参数开始优化（通常只需要微调）
    # 使用前一步的参数作为初始猜测，可以大大加快收敛
    sol = optimizer.minimize(cost_function, parameters[i-1, 0:2*n*(d+1)])
    
    # 保存优化后的参数
    parameters[i, :] = sol.x


# ============================================================================
# 第七部分：从优化后的参数提取量子态
# 将优化后的参数转换为实际的波函数，用于可视化和分析
# ============================================================================

# 存储每个时间步的波函数
# 形状：(时间步数+1, 空间点数)
# +1是因为包括初始条件（t=0时刻）
psi = np.zeros((time_steps+1, 2**n), dtype=complex)

# 第一个时刻：使用初始条件
psi[0, :] = initial_condition

# 从每个时间步的参数生成对应的量子态
# 这允许我们查看波函数随时间的演化
for i in range(time_steps):
    # 从参数生成状态向量
    # 这给出了优化后的参数对应的实际波函数
    psi[i+1, :] = stateFromParameters(ansatz, parameters[i, 0:2*n*(d+1)], n, d)


# ============================================================================
# 第八部分：可视化结果
# 绘制波包随时间的演化，并保存结果数据
# ============================================================================

# 绘制波包随时间的演化
# 每10个时间步绘制一次，避免图像过于拥挤
# 这样可以清楚地看到波包的演化过程
for i in range(0, time_steps+1, 10):
    # 绘制概率密度 |ψ|² 随位置的变化
    # x/np.pi 将位置归一化到 [-1, 1]（单位是π）
    # |ψ|²是概率密度，表示在位置x找到粒子的概率
    plt.plot(x/np.pi, np.square(np.abs(psi[i, :])), label="i={i}".format(i=i))

# 添加图例和标签
plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))  # 图例放在左上角
plt.xlabel("$x$ [$\pi$]")  # x轴：位置（单位是π）
plt.ylabel("$|\Psi|^2$")   # y轴：概率密度
plt.tight_layout()         # 自动调整布局，确保所有元素都可见

# 保存图像
plt.savefig("./image.png")  # 保存为PNG格式的图像文件

# 保存数据（参数和波函数）供后续分析
# 这允许我们在不重新运行模拟的情况下进行进一步分析
np.savez("./data.npz", parameters=parameters, psi=psi)

# 显示图像
plt.show()  # 在屏幕上显示图像