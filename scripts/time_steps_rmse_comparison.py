"""
比较经典算法和归一化经典算法的时间步数对RMSE的影响

在同一张图上绘制两种算法的RMSE vs 时间步数曲线
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from func_models.config import SimulationConfig
from scripts.time_steps_rmse_analysis_c import (
    run_single_test as run_single_test_c
)
from scripts.time_steps_rmse_analysis_nc import (
    run_single_test as run_single_test_nc
)

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def compare_time_steps_effect(
    config: SimulationConfig,
    min_steps: int = 10,
    max_steps: int = 300,
    output_dir: str = "."
):
    """
    比较经典算法和归一化经典算法的时间步数对RMSE的影响
    
    Args:
        config: 基础配置参数
        min_steps: 最小时间步数
        max_steps: 最大时间步数
        output_dir: 输出目录
    """
    print("=" * 60)
    print("时间步数对RMSE影响分析 - 经典算法 vs 归一化经典算法")
    print(f"测试范围: {min_steps} - {max_steps} 步")
    print(f"目标时间: t = 0.3s")
    print(f"配置参数: n={config.n}, a={config.a}, x0={config.x0}, v={config.v}, s={config.s}")
    print("=" * 60)
    
    # 存储结果
    time_steps_list = np.arange(min_steps, max_steps + 1)
    rmse_list_c = []  # 经典算法
    rmse_list_nc = []  # 归一化经典算法
    
    # 对每个时间步数进行测试
    print("\n运行经典算法测试...")
    for steps in tqdm(time_steps_list, desc="经典算法进度"):
        try:
            rmse = run_single_test_c(steps, config)
            rmse_list_c.append(rmse)
        except Exception as e:
            print(f"\n警告: 经典算法 - 时间步数 {steps} 测试失败: {e}")
            rmse_list_c.append(np.nan)
    
    print("\n运行归一化经典算法测试...")
    for steps in tqdm(time_steps_list, desc="归一化经典算法进度"):
        try:
            rmse = run_single_test_nc(steps, config)
            rmse_list_nc.append(rmse)
        except Exception as e:
            print(f"\n警告: 归一化经典算法 - 时间步数 {steps} 测试失败: {e}")
            rmse_list_nc.append(np.nan)
    
    rmse_array_c = np.array(rmse_list_c)
    rmse_array_nc = np.array(rmse_list_nc)
    
    # 可视化
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # 绘制两条曲线
    plt.plot(time_steps_list, rmse_array_c, '-', linewidth=1.5, alpha=0.8, 
             label='Classical Algorithm', color='blue')
    plt.plot(time_steps_list, rmse_array_nc, '-', linewidth=1.5, alpha=0.8, 
             label='Normalized Classical Algorithm', color='red')
    
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("RMSE at t=0.3s", fontsize=14)
    plt.title("RMSE vs Time Steps at t=0.3s", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 设置x轴刻度：每10一个刻度，每50显示一个标签
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(round(x))}' if round(x) % 50 == 0 else ''))
    
    # 设置y轴范围上限为0.02
    plt.ylim(top=0.02)
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(output_dir, "time_steps_rmse_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n可视化结果已保存到: {plot_path}")
    plt.close()
    
    # 打印统计信息
    valid_rmse_c = rmse_array_c[~np.isnan(rmse_array_c)]
    valid_rmse_nc = rmse_array_nc[~np.isnan(rmse_array_nc)]
    
    if len(valid_rmse_c) > 0:
        min_rmse_c = np.min(valid_rmse_c)
        min_idx_c = np.argmin(valid_rmse_c)
        min_steps_val_c = time_steps_list[min_idx_c]
        
        print("\n" + "=" * 60)
        print("经典算法统计信息:")
        print(f"  最小RMSE: {min_rmse_c:.6e} (时间步数: {min_steps_val_c})")
        print(f"  最大RMSE: {np.max(valid_rmse_c):.6e}")
        print(f"  平均RMSE: {np.mean(valid_rmse_c):.6e}")
        print(f"  中位数RMSE: {np.median(valid_rmse_c):.6e}")
    
    if len(valid_rmse_nc) > 0:
        min_rmse_nc = np.min(valid_rmse_nc)
        min_idx_nc = np.argmin(valid_rmse_nc)
        min_steps_val_nc = time_steps_list[min_idx_nc]
        
        print("\n归一化经典算法统计信息:")
        print(f"  最小RMSE: {min_rmse_nc:.6e} (时间步数: {min_steps_val_nc})")
        print(f"  最大RMSE: {np.max(valid_rmse_nc):.6e}")
        print(f"  平均RMSE: {np.mean(valid_rmse_nc):.6e}")
        print(f"  中位数RMSE: {np.median(valid_rmse_nc):.6e}")
        print("=" * 60)


if __name__ == "__main__":
    # 参数配置
    config = SimulationConfig(
        n=6,
        d=12,
        x0=-1,
        a=2,
        v=10,
        s=1,
        time_steps=100,  # 测试中忽略
        dt=3e-3,  # 测试中忽略
        initial_runs=10,
        maxiter=20000,
        maxfun=100000,
        ftol=1e-14,
    )
    
    # 输出目录
    output_dir = "/public/home/caozicheng/NLSE/result"
    
    # 运行比较分析
    compare_time_steps_effect(
        config=config,
        min_steps=10,
        max_steps=300,
        output_dir=output_dir
    )
