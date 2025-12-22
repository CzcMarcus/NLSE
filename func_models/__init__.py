"""
混合伪谱-变分量子算法求解非线性薛定谔方程（Hybrid Pseudo-Spectral VQA-NLSE Solver）

项目说明：
==========
本程序使用混合伪谱-变分量子算法（Hybrid Pseudo-Spectral Variational Quantum Algorithm）来数值求解
非线性薛定谔方程（Nonlinear Schrödinger Equation, NLSE）

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
这种混合方法既保持了伪谱方法的高精度，又利用了量子算法的压缩表示优势
"""

# 导出主要函数和类
from .analytical_solutions import soliton_analytical, soliton_analytical_folded
from .quantum_circuits import ansatz, stateFromParameters
from .cost_functions import cost_function_fast, cost_function_qiskit
from .quantum_operations import qft, iqft, quadraticExponential, U_ij, U_ii

__all__ = [
    # 解析解函数
    'soliton_analytical',
    'soliton_analytical_folded',
    # 量子电路函数
    'ansatz',
    'stateFromParameters',
    # 成本函数
    'cost_function_fast',
    'cost_function_qiskit',
    # 量子操作函数
    'qft',
    'iqft',
    'quadraticExponential',
    'U_ij',
    'U_ii',
]
