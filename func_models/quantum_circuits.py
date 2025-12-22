"""
量子电路构建函数模块

用于构建参数化的ansatz电路，这是变分量子算法的核心
"""


from typing import Callable
from numpy.typing import NDArray
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def ansatz(
    qc: QuantumCircuit, 
    n: int, 
    d: int, 
    offset: int, 
    parameters: NDArray[np.floating], 
    conj: bool = False
) -> None:
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
            # 创建链式CNOT：0->1->2->...->(n-1)->0（形成环）
            for k in range(n-1):
                qc.cx(k+offset, k+1 + offset)
            # 最后一个CNOT连接回第一个量子比特，形成周期边界条件
            qc.cx(n-1 + offset, offset)


def stateFromParameters(
    ansatz_func: Callable[[QuantumCircuit, int, int, int, NDArray[np.floating], bool], None], 
    parameters: NDArray[np.floating], 
    n: int, 
    d: int
) -> Statevector:
    """
    从给定的ansatz和参数生成理想的状态向量，即当前参数所定义的试探波函数|ψ(λ)⟩。
    
    这个函数构建量子电路，应用参数，然后提取最终的量子态。
    这是将参数化电路转换为实际量子态的关键步骤，用于：
    1. 计算成本函数（需要知道当前参数对应的量子态）
    2. 提取优化后的波函数（用于可视化和分析）
    
    工作流程：
    1. 创建n个量子比特的空电路（从零状态开始）
    2. 应用ansatz电路（从索引0开始）
    3. 提取理想的状态向量（无噪声的完美量子态-使用的是Qiskit中的(1)）
    
    Args:
        ansatz_func: ansatz函数（上面定义的函数），用于构建参数化电路
        parameters: 参数数组，长度为2*n*(d+1)，包含所有旋转角度
        n: 量子比特数量
        d: 电路深度
    
    Returns:
        Statevector: 2^n维的量子态向量（复数数组），表示当前参数对应的波函数
    """
    # 创建n个量子比特的电路
    qc = QuantumCircuit(n)
    
    # 应用ansatz电路（从索引0开始）
    ansatz_func(qc, n, d, 0, parameters)
    
    # 返回理想的状态向量（无噪声的完美量子态）
    return Statevector(qc)
