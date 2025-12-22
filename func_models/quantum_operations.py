"""
量子操作函数模块

用于构建量子电路的量子傅里叶变换和拉普拉斯算子
"""


import numpy as np
from qiskit import QuantumCircuit


def qft(circuit: QuantumCircuit, n: int, indexOffset: int = 0) -> None:
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


def iqft(circuit: QuantumCircuit, n: int, indexOffset: int = 0) -> None:
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


def U_ij(
    circuit: QuantumCircuit, 
    i: int, 
    j: int, 
    beta: float, 
    gamma: float, 
    indexOffset: int = 0
) -> None:
    """
    实现二次相位门 U_ij：在量子比特i和j之间应用受控相位门
    
    这是实现 exp(i*gamma*x²) 的一部分，其中x²的交叉项（i≠j）通过
    受控相位门实现。
    
    参数：
        circuit: 量子电路对象
        i, j: 量子比特索引（i≠j）
        beta: 偏移参数（用于处理负索引）
        gamma: 相位参数
        indexOffset: 起始索引
    """
    # 交叉项的相位：gamma * 2^(i+j)
    circuit.cp(gamma*2**(i+j), i+indexOffset, j+indexOffset)


def U_ii(
    circuit: QuantumCircuit, 
    i: int, 
    beta: float, 
    gamma: float, 
    indexOffset: int = 0
) -> None:
    """
    实现二次相位门 U_ii：在量子比特i上应用相位门
    
    这是实现 exp(i*gamma*x²) 的一部分，其中x²的对角项（i=j）通过
    单量子比特相位门实现。
    
    参数：
        circuit: 量子电路对象
        i: 量子比特索引
        beta: 偏移参数（用于处理负索引）
        gamma: 相位参数
        indexOffset: 起始索引
    """
    # 对角项的相位：gamma * (2^(2i) + 2*beta*2^i)
    # 这对应于 x² 项，其中 x = -N/2 + sum(2^j * bit_j)
    circuit.p(gamma*( 2**(2*i)+2*beta*2**i ), i+indexOffset)


def quadraticExponential(
    circuit: QuantumCircuit, 
    gamma: float, 
    n: int, 
    indexOffset: int = 0
) -> None:
    """
    实现二次指数演化算子：exp(i*gamma*x²)
    
    这个函数实现了动能算符在动量空间的演化。
    注意：在这个实现中，动能算符 T 在动量空间被定义为 k²（而不是标准的 ½k²），
    因此演化算子 exp(-i*dt/2 * T) 在动量空间是 exp(-i*dt/2 * k²)。
    当 gamma = -delta_t/2 时，实现的就是 exp(-i*dt/2 * k²)。
    
    实现原理：
    - 将x表示为二进制：x = -N/2 + sum(2^i * bit_i)
    - x²可以展开为：x² = sum_i sum_j 2^(i+j) * bit_i * bit_j + 对角项
    - 使用受控相位门实现交叉项，使用相位门实现对角项
    
    参数：
        circuit: 量子电路对象
        gamma: 相位参数（通常是 -delta_t/2）
        n: 量子比特数量
        indexOffset: 起始索引
    
    注意：
        x的范围是 -N/2 到 N/2-1，其中 N = 2^n
    """
    beta = -2**n/2  # 偏移量，使得x从-N/2开始

    # 对所有量子比特对应用相位门
    for i in range(n):
        for j in range(n):
            if i == j:
                # 对角项：使用单量子比特相位门
                U_ii(circuit, i, beta, gamma, indexOffset)
            else:
                # 交叉项：使用受控相位门
                U_ij(circuit, i, j, beta, gamma, indexOffset)
