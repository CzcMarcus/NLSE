"""
研究时间步数对RMSE的影响 - 经典算法与量子算法对比

固定dt=1e-3，x0=0，在一张图中绘制左右两张子图：
- 左图：经典算法（c）和量子算法d=8,9,10
- 右图：经典算法（c）和量子算法d=11,12,13
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# 确保能导入项目根目录的模块
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "../..")
sys.path.insert(0, os.path.abspath(project_root))

from func_models.config import SimulationConfig
from scripts.c_nlse import PseudoSpectralSplitStepSolver, soliton_analytical_folded

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def compute_rmse_at_t(psi_num: np.ndarray, psi_ana: np.ndarray) -> float:
    """
    计算经典算法数值解与解析解的RMSE（按模长差）
    """
    diff = np.abs(psi_num) - np.abs(psi_ana)
    rmse = np.sqrt(np.mean(diff ** 2))
    return rmse


def run_single_test(time_steps: int, dt: float, config_base: SimulationConfig) -> float:
    """
    运行单次测试：使用经典算法，固定dt，指定时间步数运行模拟，计算最终时刻的RMSE
    """
    # 创建经典伪谱分裂步求解器，使用固定的dt
    solver = PseudoSpectralSplitStepSolver(
        n=config_base.n,
        s=config_base.s,
        dt=dt
    )
    
    # 初始化：t=0时刻的解析解
    x = solver.x
    psi0 = soliton_analytical_folded(0, x, config_base.x0, config_base.a, config_base.v)
    
    # 运行模拟
    current_psi = psi0.copy()
    for _ in range(time_steps):
        current_psi = solver.step(current_psi)
    
    # 计算最终时刻的解析解
    t_final = time_steps * dt
    psi_analytical = soliton_analytical_folded(t_final, x, config_base.x0, config_base.a, config_base.v)
    
    # 计算RMSE
    rmse = compute_rmse_at_t(current_psi, psi_analytical)
    
    return rmse


def analyze_classical_algorithm(
    config: SimulationConfig,
    dt: float = 1e-3,
    min_steps: int = 1,
    max_steps: int = 100
) -> dict:
    """
    分析时间步数对经典算法RMSE的影响（固定dt）
    """
    print("=" * 60)
    print("时间步数对RMSE影响分析 - 经典算法（固定dt）")
    print(f"固定时间步长: dt = {dt}")
    print(f"测试范围: {min_steps} - {max_steps} 步")
    print(f"对应总时间范围: {min_steps*dt:.3f}s - {max_steps*dt:.3f}s")
    print(f"配置参数: n={config.n}, a={config.a}, x0={config.x0}, v={config.v}, s={config.s}")
    print("=" * 60)
    
    # 存储结果
    time_steps_list = np.arange(min_steps, max_steps + 1)
    rmse_list = []
    
    # 对每个时间步数进行测试
    for steps in tqdm(time_steps_list, desc="经典算法测试进度"):
        try:
            rmse = run_single_test(steps, dt, config)
            rmse_list.append(rmse)
        except Exception as e:
            print(f"\n警告: 时间步数 {steps} 测试失败: {e}")
            rmse_list.append(np.nan)
    
    rmse_array = np.array(rmse_list)
    time_array = time_steps_list * dt
    
    # 打印统计信息
    valid_rmse = rmse_array[~np.isnan(rmse_array)]
    if len(valid_rmse) > 0:
        min_rmse = np.min(valid_rmse)
        min_idx = np.argmin(valid_rmse)
        min_steps_val = time_steps_list[min_idx]
        print("\n" + "=" * 60)
        print("经典算法统计信息:")
        print(f"  最小RMSE: {min_rmse:.6e} (时间步数: {min_steps_val}, 总时间: {min_steps_val*dt:.3f}s)")
        print(f"  最大RMSE: {np.max(valid_rmse):.6e}")
        print(f"  平均RMSE: {np.mean(valid_rmse):.6e}")
        print(f"  中位数RMSE: {np.median(valid_rmse):.6e}")
        print("=" * 60)
    
    return {
        'rmse': rmse_array,
        'time': time_array,
        'time_steps': time_steps_list,
    }


def load_quantum_data(data_dir: str, d: int, dt: float = 1e-3, x0: int = 0) -> dict:
    """
    加载量子算法的数据文件
    """
    data_path = os.path.join(data_dir, f"time_steps_rmse_quantum_d{d}_dt{dt}_x0{x0}.npz")
    if not os.path.exists(data_path):
        return None
    
    data = np.load(data_path, allow_pickle=True)
    return {
        'rmse': data['rmse'],
        'time': data['time'],
        'time_steps': data['time_steps'],
        'd': int(data['d'])
    }


def plot_comparison(
    classical_result: dict,
    quantum_results: dict,
    dt: float = 1e-3,
    x0: int = 0,
    output_dir: str = "."
):
    """
    绘制经典算法与量子算法对比图：左右两个子图
    左图：经典算法（c）和量子算法d=8,9,10
    右图：经典算法（c）和量子算法d=11,12,13
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 设置x轴刻度：每0.005s一个刻度，每0.02s显示一个标签
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    
    # 左图：经典算法 + 量子算法d=8,9,10
    d_left = [8, 9, 10]
    
    # 定义不同d值的颜色映射
    color_map = {
        8: '#1f77b4',   # 蓝色
        9: '#2ca02c',   # 绿色
        10: '#d62728',  # 红色
        11: '#ff7f0e',  # 橙色
        12: '#9467bd',  # 紫色
        13: '#8c564b',  # 棕色
    }
    
    # 绘制经典算法（黑色虚线）
    if classical_result is not None:
        ax1.plot(classical_result['time'], classical_result['rmse'], 
                'k--', linewidth=2.0, alpha=0.9, 
                label='Classical')
    
    # 绘制量子算法d=8,9,10（不同颜色）
    for d in d_left:
        if d in quantum_results and quantum_results[d] is not None:
            result = quantum_results[d]
            color = color_map.get(d, 'gray')
            ax1.plot(result['time'], result['rmse'], 
                    color=color, linewidth=1.5, alpha=0.8, 
                    label=f'Quantum d={d}')
    
    ax1.set_xlabel("Time (s)", fontsize=14)
    ax1.set_ylabel("Root Mean Square Error", fontsize=14)
    ax1.set_title(f"Classical vs Quantum (d=8,9,10)", fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 0.1)  # 设置x轴范围为0-0.1s
    ax1.xaxis.set_major_locator(MultipleLocator(0.005))
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}' if abs(round(x / 0.02) - x / 0.02) < 1e-6 else ''))
    
    # 右图：经典算法 + 量子算法d=11,12,13
    d_right = [11, 12, 13]
    
    # 绘制经典算法（黑色虚线）
    if classical_result is not None:
        ax2.plot(classical_result['time'], classical_result['rmse'], 
                'k--', linewidth=2.0, alpha=0.9, 
                label='Classical')
    
    # 绘制量子算法d=11,12,13（不同颜色）
    for d in d_right:
        if d in quantum_results and quantum_results[d] is not None:
            result = quantum_results[d]
            color = color_map.get(d, 'gray')
            ax2.plot(result['time'], result['rmse'], 
                    color=color, linewidth=1.5, alpha=0.8, 
                    label=f'Quantum d={d}')
    
    ax2.set_xlabel("Time (s)", fontsize=14)
    ax2.set_ylabel("Root Mean Square Error", fontsize=14)
    ax2.set_title(f"Classical vs Quantum (d=11,12,13)", fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 0.1)  # 设置x轴范围为0-0.1s
    ax2.xaxis.set_major_locator(MultipleLocator(0.005))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}' if abs(round(x / 0.02) - x / 0.02) < 1e-6 else ''))
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "figure5.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n可视化结果已保存到: {plot_path}")
    plt.close()


def analyze_comparison(
    config: SimulationConfig,
    d_list: list,
    dt: float = 1e-3,
    min_steps: int = 1,
    max_steps: int = 100,
    data_dir: str = ".",
    output_dir: str = ".",
    use_existing_data: bool = True
):
    """
    分析经典算法与量子算法的对比
    """
    print("\n" + "=" * 60)
    print("经典算法与量子算法对比分析")
    print(f"量子算法d值: {d_list}")
    print(f"固定时间步长: dt = {dt}")
    print(f"测试时间步数范围: {min_steps} - {max_steps}")
    print("=" * 60 + "\n")
    
    # 计算经典算法数据（每次运行都重新计算，因为跑得很快）
    print("计算经典算法数据...")
    classical_result = analyze_classical_algorithm(
        config=config,
        dt=dt,
        min_steps=min_steps,
        max_steps=max_steps
    )
    
    # 加载量子算法数据
    print("\n处理量子算法数据...")
    quantum_results = {}
    for d in d_list:
        if use_existing_data:
            quantum_data = load_quantum_data(data_dir, d, dt, config.x0)
            if quantum_data is not None:
                print(f"加载已有量子算法数据: d={d}")
                quantum_results[d] = quantum_data
            else:
                print(f"警告: 未找到量子算法数据 d={d}")
                quantum_results[d] = None
        else:
            quantum_data = load_quantum_data(data_dir, d, dt, config.x0)
            if quantum_data is not None:
                quantum_results[d] = quantum_data
            else:
                print(f"警告: 未找到量子算法数据 d={d}")
                quantum_results[d] = None
    
    # 绘制左右两个子图
    plot_comparison(
        classical_result=classical_result,
        quantum_results=quantum_results,
        dt=dt,
        x0=config.x0,
        output_dir=output_dir
    )


if __name__ == "__main__":
    # 参数配置
    config = SimulationConfig(
        n=6,
        d=12,  # 对经典算法不使用，但保留在config中
        x0=0,  # 用户指定
        a=2,
        v=10,
        s=1,
        time_steps=100,
        dt=1e-3,  # 用户指定，固定使用
        initial_runs=10,
        maxiter=20000,
        maxfun=100000,
        ftol=1e-13,  # 用户指定（对经典算法不适用，但保留在config中）
    )
    
    # 固定时间步长
    dt = 1e-3
    
    # 要测试的量子算法d值列表
    d_list = [8, 9, 10, 11, 12, 13]
    
    # 数据目录和输出目录
    data_dir = "/public/home/caozicheng/NLSE/result/data/figure5"
    output_dir = "/public/home/caozicheng/NLSE/scripts/figure5"
    
    # 运行对比分析
    analyze_comparison(
        config=config,
        d_list=d_list,
        dt=dt,
        min_steps=1,
        max_steps=100,
        data_dir=data_dir,
        output_dir=output_dir,
        use_existing_data=True  # 如果数据已存在则直接加载，否则重新计算
    )
