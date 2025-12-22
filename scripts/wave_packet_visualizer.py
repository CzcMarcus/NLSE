"""
NLSE/LSE 算法模拟结果的可视化模块

本模块提供波函数时间演化结果的可视化功能，支持从计算结果或保存的数据文件
进行可视化分析，用于展示求解非线性薛定谔方程（NLSE）和线性薛定谔方程（LSE）的数值结果
"""


import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from func_models.config import SimulationConfig

# 可视化配置参数
PLOT_STEPS = [1, 20, 40, 60, 80, 100]  # 要绘制的时间步
SHOW_FIG = True   # 是否显示图像
SAVE_FIG = True   # 是否保存图像
IMAGE_NAME = "image.png"


def visualize_wave_packet(
    psi: np.ndarray,
    config: SimulationConfig,
    output_dir: str = ".",
) -> Optional[str]:
    """
    根据给定的波函数演化结果进行可视化。
    
    Args:
        psi: 波函数数组，形状为 (time_steps+1, 2^n)，类型为 complex
        config: 模拟配置对象，用于获取 n、time_steps 等参数
        output_dir: 输出目录路径，用于保存图像文件
    
    Returns:
        保存的图像文件路径；若未保存则返回 None
    """
    print("Visualizing results...")
    
    # 在 [-π, π] 区间上采样了 64 个点，索引为 0-63
    x = np.arange(0, 2**config.n)
    
    # 计算归一化因子，将归一化量子态恢复为物理波函数的概率密度
    # 物理波函数的归一化条件：∫|Ψ(x)|²dx = 2a
    # 离散化形式：∑|Ψ_i|² * Δx = 2a，其中空间步长 Δx = 2π/(2^n)
    # 量子态的归一化条件：∑|ψ_i|² = 1，其中 |ψ_i|² = |Ψ_i|² / (∑|Ψ_j|²)
    # 因此转换关系为：|Ψ_i|² = |ψ_i|² * (2a / Δx)
    dx = 2 * np.pi / (2**config.n)  # 空间离散化步长
    normalization_factor = 2 * config.a / dx  # 量子态到物理波函数的归一化因子
    
    # 创建图形窗口
    plt.figure(figsize=(10, 6))
    
    # 选择绘制时间步
    steps = [s for s in PLOT_STEPS if 0 <= s <= config.time_steps]
    if not steps:
        steps = [0, config.time_steps]
    
    # 根据时间步范围生成颜色映射
    min_step = min(steps)
    max_step = max(steps)
    step_range = max_step - min_step if max_step > min_step else 1
    
    for idx, i in enumerate(steps):
        # 计算物理波函数的概率密度
        prob_density = np.square(np.abs(psi[i, :])) * normalization_factor
        # 时间步越小，颜色越深（从 0.8 到 0.5 ）
        color_intensity = 0.8 - 0.3 * (i - min_step) / step_range
        color = plt.cm.Blues(color_intensity)
        # 实线和虚线交替
        linestyle = '-' if idx % 2 == 0 else '--'
        linewidth = 2.5
        plt.plot(x, prob_density, 
                color=color,
                linestyle=linestyle, 
                linewidth=linewidth,
                label=f"m={i}")
    
    # 添加图例和标签
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9, framealpha=0.9)
    plt.xlabel("Position", fontsize=11)  # x 轴：位置
    plt.ylabel("$|\\Psi|^2$", fontsize=11)   # y 轴：概率密度
    plt.title("Wave Packet Evolution", fontsize=12, fontweight='bold')
    
    # 在左上角添加 s 参数标签（显示归一化后的值）
    s_normalized = config.s / (2 * config.a)
    plt.text(0.02, 0.98, f"s={s_normalized:.0f}", transform=plt.gca().transAxes,
            fontsize=11, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 设置 x 轴刻度：数据点对应 0-63，x轴刻度范围是 0-64
    n_points = 2**config.n
    x_ticks = list(range(0, n_points, 8)) + [n_points-1]  
    x_tick_labels = [str(tick) for tick in x_ticks]
    plt.xticks(x_ticks, labels=x_tick_labels)  # 显式设置刻度位置和标签
    plt.xlim(0, n_points - 1)  # x 轴范围从 0 到 63
    
    # 开启网格显示，使用浅灰色
    plt.grid(True, alpha=0.5, linestyle='-', linewidth=0.8, color='gray')
    
    # 自动调整布局
    plt.tight_layout()  
    
    # 保存图像
    image_path: Optional[str] = None
    if SAVE_FIG:
        os.makedirs(output_dir, exist_ok=True)
        image_path = os.path.join(output_dir, IMAGE_NAME)
        plt.savefig(image_path, dpi=150)
        print(f"Image saved to: {image_path}")
    
    # 显示图像
    if SHOW_FIG:
        plt.show()
    else:
        plt.close()
    
    return image_path


def visualize_from_file(
    data_path: str,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    从保存的数据文件中加载结果并进行可视化。
    
    Args:
        data_path: 数据文件路径（.npz格式），包含 parameters、psi
        output_dir: 输出目录路径，用于保存图像文件。若为 None，则使用数据文件所在目录
    
    Returns:
        保存的图像文件路径；若未保存则返回 None
    """
    # 加载数据
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    
    psi = data['psi']
    
    # 使用固定的配置参数
    config = SimulationConfig(
        n=6,
        d=12,
        x0=-1,
        a=2,
        v=10,
        s=4,
        time_steps=100,
        dt=3e-3,
        initial_runs=10,
        maxiter=20000,
        maxfun=100000,
        ftol=1e-14,
    )
    
    # 如果未指定输出目录，使用数据文件所在目录
    if output_dir is None:
        output_dir = os.path.dirname(data_path)
    
    # 调用可视化函数
    return visualize_wave_packet(psi, config, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化 NLSE/NLSE 模拟结果")
    parser.add_argument(
        "--data", 
        type=str, 
        default="data.npz",
        help="数据文件路径（.npz格式）"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="输出目录路径（默认使用数据文件所在目录）"
    )
    
    args = parser.parse_args()
    
    visualize_from_file(
        args.data,
        args.output,
    )
