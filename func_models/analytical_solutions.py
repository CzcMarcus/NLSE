"""
NLSE的解析解函数模块

用于生成初始条件和验证数值结果的正确性。
"""


import numpy as np
from typing import Union
from numpy.typing import NDArray


def soliton_analytical(
    t: Union[float, NDArray[np.floating]], 
    x: Union[float, NDArray[np.floating]], 
    x0: float, 
    a: float, 
    v: float
) -> Union[complex, NDArray[np.complexfloating]]:
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
    # 返回一维无势聚焦型NLSE的精确解析解，即公式3
    return a / np.cosh(a*(x-x0-v*t)) * np.exp(1j*v*(x-x0)+1j*(a*a-v*v)*t/2.0)


def soliton_analytical_folded(
    t: Union[float, NDArray[np.floating]], 
    x: Union[float, NDArray[np.floating]], 
    x0: float, 
    a: float, 
    v: float
) -> Union[complex, NDArray[np.complexfloating]]:
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
    
    # 按幅度大小排序，幅度越大表示越接近孤子中心，选择最大的（最接近真实值的）
    candidates.sort(key=lambda a: np.square(np.abs(a)))
    
    # 返回幅度最大的候选值
    return candidates[-1]
