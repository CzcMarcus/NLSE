"""
可视化整合后的时间步数RMSE数据 - 量子算法

从整合后的npz文件中加载RMSE数据并进行可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def visualize_merged_rmse(
    data_path: str,
    output_dir: str = "."
):
    """
    从整合后的数据文件中加载RMSE数据并可视化
    
    Args:
        data_path: 整合后的数据文件路径（.npz格式）
        output_dir: 输出目录
    """
    print("=" * 60)
    print("可视化整合后的时间步数RMSE数据 - 量子算法")
    print("=" * 60)
    
    if not os.path.exists(data_path):
        print(f"错误: 文件不存在: {data_path}")
        return
    
    # 加载数据
    print(f"\n正在加载数据: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    keys = list(data.keys())
    
    # 提取时间步数和RMSE
    psi_keys = [k for k in keys if k.startswith('psi_t0.3s_steps_')]
    time_steps = sorted([int(k.replace('psi_t0.3s_steps_', '')) for k in psi_keys])
    
    if 'rmse' not in data:
        print("错误: 文件中未找到rmse数组")
        return
    
    rmse_array = data['rmse']
    
    print(f"找到 {len(time_steps)} 个时间步数")
    print(f"时间步数: {time_steps}")
    print(f"RMSE数组长度: {len(rmse_array)}")
    
    # 检查数据一致性
    if len(time_steps) != len(rmse_array):
        print(f"警告: 时间步数数量({len(time_steps)})与RMSE数组长度({len(rmse_array)})不匹配")
        # 尝试匹配
        if len(rmse_array) == len(time_steps):
            print("使用现有的RMSE数组")
        else:
            print("错误: 无法匹配数据")
            return
    else:
        # 确保RMSE数组按时间步数顺序排列
        rmse_dict = {}
        for i, step in enumerate(time_steps):
            if i < len(rmse_array):
                rmse_dict[step] = rmse_array[i]
        
        # 按时间步数顺序重新排列
        sorted_steps = sorted(rmse_dict.keys())
        rmse_array = np.array([rmse_dict[step] for step in sorted_steps])
        time_steps = sorted_steps
    
    time_steps_array = np.array(time_steps)
    
    # 可视化
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # RMSE vs 时间步数
    plt.plot(time_steps_array, rmse_array, 'g-', linewidth=1.5, alpha=0.8, marker='o', markersize=6)
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("RMSE at t=0.3s", fontsize=14)
    plt.title("RMSE vs Time Steps at t=0.3s (Quantum Algorithm)", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度：每10一个刻度，每50显示一个标签
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(round(x))}' if round(x) % 50 == 0 else ''))
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(output_dir, "time_steps_rmse_analysis_quantum_merged.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n可视化结果已保存到: {plot_path}")
    plt.close()
    
    # 打印统计信息
    valid_rmse = rmse_array[~np.isnan(rmse_array)]
    if len(valid_rmse) > 0:
        min_rmse = np.min(valid_rmse)
        min_idx = np.argmin(valid_rmse)
        min_steps_val = time_steps_array[min_idx]
        print("\n" + "=" * 60)
        print("统计信息:")
        print(f"  最小RMSE: {min_rmse:.6e} (时间步数: {min_steps_val})")
        print(f"  最大RMSE: {np.max(valid_rmse):.6e}")
        print(f"  平均RMSE: {np.mean(valid_rmse):.6e}")
        print(f"  中位数RMSE: {np.median(valid_rmse):.6e}")
        print("=" * 60)
    
    # 打印每个时间步数的RMSE值
    print("\n详细数据:")
    print("时间步数\tRMSE")
    print("-" * 30)
    for step, rmse in zip(time_steps_array, rmse_array):
        print(f"{step:8d}\t{rmse:.6e}")


if __name__ == "__main__":
    # 整合后的数据文件路径
    data_path = "/public/home/caozicheng/NLSE/result/data/time_steps_psi_quantum_t0.3s_merged.npz"
    
    # 输出目录
    output_dir = "/public/home/caozicheng/NLSE/result"
    
    # 运行可视化
    visualize_merged_rmse(
        data_path=data_path,
        output_dir=output_dir
    )
