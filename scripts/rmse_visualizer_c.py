"""
计算并可视化经典数值解与解析解之间的 RMSE（按模长差）。
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from func_models.config import SimulationConfig

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def compute_rmse_classic(
    num_path: str,
    ana_path: str,
    output_dir: str = ".",
    config: SimulationConfig | None = None,
):
    """
    加载经典数值解与解析解，逐时间步计算模长 RMSE 并可视化。
    
    Args:
        num_path: 经典数值解数据路径（.npz格式），仅读取 psi（已经是物理波函数）
        ana_path: 解析解数据路径（.npz格式），仅读取 psi
        output_dir: 输出目录
        config: 模拟配置，提供 dt 等信息；用于时间轴
    """
    num = np.load(num_path, allow_pickle=True)
    ana = np.load(ana_path, allow_pickle=True)

    if config is None:
        raise ValueError("请提供 SimulationConfig，用于获取 dt 等信息。")

    dt = float(config.dt)

    psi_num = num["psi"]      # 物理波函数 Ψ(x): ∫|Ψ|² dx = 2a
    psi_ana = ana["psi"]      # 物理波函数 Ψ(x): ∫|Ψ|² dx = 2a

    assert psi_num.shape == psi_ana.shape, "Shape mismatch!"

    steps = psi_num.shape[0] - 1
    t_axis = np.arange(1, steps + 1) * dt
    rmse = np.zeros(steps)
    
    for i in range(1, steps + 1):
        # 直接比较物理波函数模长
        diff = np.abs(psi_num[i]) - np.abs(psi_ana[i])
        rmse[i - 1] = np.sqrt(np.mean(diff ** 2))

    # 可视化
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(t_axis, rmse, "b-", linewidth=2.2, marker="o", markersize=4, alpha=0.85, markevery=5)
    plt.xlabel("Time (t)")
    plt.ylabel("Root Mean Square Error")
    plt.title("RMSE vs Time")
    # 设置 y 轴刻度间隔为 0.001，格式为 3 位小数
    ax = plt.gca()
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    ax.yaxis.set_major_locator(MultipleLocator(0.001))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.3f}'))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(output_dir, "rmse_classic.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"RMSE plot saved to: {out_png}")
    print(f"Min RMSE: {np.min(rmse):.6e}")
    print(f"Mean RMSE: {np.mean(rmse):.6e}")
    print(f"Max  RMSE: {np.max(rmse):.6e}")


if __name__ == "__main__":
    # 配置参数
    config = SimulationConfig(
        n=6,
        d=12,
        x0=-1,
        a=2,
        v=10,
        s=1,  # 对应论文目标，非线性强度为1
        time_steps=100,
        dt=3e-3,
        initial_runs=10,
        maxiter=20000,
        maxfun=100000,
        ftol=1e-14,
    )
    
    num_path = "/public/home/caozicheng/NLSE/result/data/classic_nlse_data.npz"
    ana_path = "/public/home/caozicheng/NLSE/result/data/analytical_nlse_data.npz"
    compute_rmse_classic(num_path, ana_path, "/public/home/caozicheng/NLSE/result", config=config)
