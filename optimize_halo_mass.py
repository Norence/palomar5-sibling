#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化 NFW Halo 质量参数
遍历 1e11 到 2e12 的质量范围，找到使星流成员与 Pal 5 过去位置最接近的最优质量
"""

import numpy as np
import matplotlib.pyplot as plt
import agama
from scipy.interpolate import interp1d
from tqdm import tqdm
import pandas as pd

# ============================================================
# 配置参数
# ============================================================

# 质量范围配置
MASS_MIN = 1e11  # 最小 halo 质量
MASS_MAX = 2e12  # 最大 halo 质量
N_STEPS = 20     # 遍历步数（可以调整以平衡精度和计算时间）

# ============================================================
# 辅助函数
# ============================================================

def create_potential(mass_halo):
    """
    创建给定 halo 质量的总势能
    
    Parameters:
    -----------
    mass_halo : float
        NFW halo 的质量（太阳质量单位）
    
    Returns:
    --------
    pot_total : agama.Potential
        总势能
    """
    pot_bulge = agama.Potential(type='Dehnen', scaleRadius=1.0, gamma=1.0, mass=2e10)
    pot_disk = agama.Potential(type='MiyamotoNagai', scaleRadius=3.0, scaleHeight=0.3, mass=5e10)
    pot_halo = agama.Potential(type='NFW', scaleRadius=15.0, mass=mass_halo)
    pot_total = agama.Potential(pot_bulge, pot_disk, pot_halo)
    return pot_total


def compute_min_distance(pot_total, ic_pal5, ic_stream, integration_time=2.0):
    """
    计算星流成员与 Pal 5 在过去轨道上的最小距离
    
    Parameters:
    -----------
    pot_total : agama.Potential
        总势能
    ic_pal5 : array
        Pal 5 的初始条件 [x, y, z, vx, vy, vz]
    ic_stream : array
        星流成员的初始条件 [x, y, z, vx, vy, vz]
    integration_time : float
        积分时间（单位：轨道周期数）
    
    Returns:
    --------
    min_dist : float
        最小 3D 距离（kpc）
    min_time : float
        最小距离对应的时间（Gyr）
    """
    # 计算 Pal 5 的后向轨道
    t_gc_m, orb_gc_m = agama.orbit(potential=pot_total, ic=ic_pal5,
                                    time=-integration_time*pot_total.Tcirc(ic_pal5), 
                                    trajsize=10000)
    
    # 计算星流成员的后向轨道
    t_stream_m, orb_stream_m = agama.orbit(potential=pot_total, ic=ic_stream,
                                           time=-integration_time*pot_total.Tcirc(ic_stream), 
                                           trajsize=10000)
    
    # 反转时间序列
    t_gc_past = t_gc_m[::-1]
    orb_gc_past = orb_gc_m[::-1]
    t_stream_past = t_stream_m[::-1]
    orb_stream_past = orb_stream_m[::-1]
    
    # 在共同时间网格上插值
    t_min = max(t_gc_past[0], t_stream_past[0])
    t_max = min(t_gc_past[-1], t_stream_past[-1])
    
    if t_min >= t_max:
        return np.inf, np.nan
    
    t_common = np.linspace(t_min, t_max, 1000)
    
    # 插值 Pal 5
    gc_x_interp = interp1d(t_gc_past, orb_gc_past[:,0], kind='cubic')(t_common)
    gc_y_interp = interp1d(t_gc_past, orb_gc_past[:,1], kind='cubic')(t_common)
    gc_z_interp = interp1d(t_gc_past, orb_gc_past[:,2], kind='cubic')(t_common)
    
    # 插值星流成员
    stream_x_interp = interp1d(t_stream_past, orb_stream_past[:,0], kind='cubic')(t_common)
    stream_y_interp = interp1d(t_stream_past, orb_stream_past[:,1], kind='cubic')(t_common)
    stream_z_interp = interp1d(t_stream_past, orb_stream_past[:,2], kind='cubic')(t_common)
    
    # 计算3D距离
    dist_3d = np.sqrt((gc_x_interp - stream_x_interp)**2 + 
                      (gc_y_interp - stream_y_interp)**2 + 
                      (gc_z_interp - stream_z_interp)**2)
    
    # 找到最小距离
    min_idx = np.argmin(dist_3d)
    min_dist = dist_3d[min_idx]
    min_time = t_common[min_idx]
    
    return min_dist, min_time


def optimize_halo_mass(ic_pal5, ic_streams, mass_range, n_steps=20):
    """
    遍历 halo 质量，找到最优值
    
    Parameters:
    -----------
    ic_pal5 : array
        Pal 5 的初始条件
    ic_streams : array (n_members, 6)
        所有星流成员的初始条件
    mass_range : tuple
        质量范围 (min, max)
    n_steps : int
        遍历步数
    
    Returns:
    --------
    results : dict
        包含质量、平均最小距离、标准差等信息
    """
    masses = np.logspace(np.log10(mass_range[0]), np.log10(mass_range[1]), n_steps)
    
    results = {
        'masses': masses,
        'mean_min_dist': [],
        'std_min_dist': [],
        'median_min_dist': [],
        'all_min_dists': [],
        'all_min_times': []
    }
    
    print(f"\n{'='*80}")
    print(f"开始遍历 Halo 质量范围: {mass_range[0]:.2e} - {mass_range[1]:.2e}")
    print(f"共 {n_steps} 步，每步计算 {len(ic_streams)} 个星流成员")
    print(f"{'='*80}\n")
    
    for mass in tqdm(masses, desc="优化 Halo 质量"):
        pot_total = create_potential(mass)
        
        min_dists = []
        min_times = []
        
        for ic_stream in ic_streams:
            min_dist, min_time = compute_min_distance(pot_total, ic_pal5, ic_stream)
            min_dists.append(min_dist)
            min_times.append(min_time)
        
        min_dists = np.array(min_dists)
        min_times = np.array(min_times)
        
        # 过滤掉无效值
        valid_mask = np.isfinite(min_dists)
        valid_dists = min_dists[valid_mask]
        
        if len(valid_dists) > 0:
            results['mean_min_dist'].append(np.mean(valid_dists))
            results['std_min_dist'].append(np.std(valid_dists))
            results['median_min_dist'].append(np.median(valid_dists))
        else:
            results['mean_min_dist'].append(np.inf)
            results['std_min_dist'].append(np.inf)
            results['median_min_dist'].append(np.inf)
        
        results['all_min_dists'].append(min_dists)
        results['all_min_times'].append(min_times)
    
    # 转换为数组
    for key in ['mean_min_dist', 'std_min_dist', 'median_min_dist']:
        results[key] = np.array(results[key])
    
    return results


def plot_optimization_results(results, save_path=None):
    """
    绘制优化结果
    
    Parameters:
    -----------
    results : dict
        optimize_halo_mass 的返回结果
    save_path : str, optional
        保存图片的路径
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    masses = results['masses']
    mean_dist = results['mean_min_dist']
    std_dist = results['std_min_dist']
    median_dist = results['median_min_dist']
    
    # 子图1: 平均最小距离
    ax1 = axes[0]
    ax1.plot(masses, mean_dist, 'o-', color='#2E86AB', lw=2.5, ms=8, label='Mean')
    ax1.fill_between(masses, mean_dist - std_dist, mean_dist + std_dist, 
                     alpha=0.3, color='#2E86AB', label='±1σ')
    ax1.plot(masses, median_dist, 's--', color='#A23B72', lw=2, ms=6, label='Median')
    
    # 标记最优质量
    optimal_idx = np.argmin(mean_dist)
    optimal_mass = masses[optimal_idx]
    optimal_dist = mean_dist[optimal_idx]
    
    ax1.plot(optimal_mass, optimal_dist, '*', color='#F18F01', ms=20, 
            markeredgecolor='white', markeredgewidth=2, 
            label=f'Optimal: {optimal_mass:.2e} M☉', zorder=10)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Halo Mass [M☉]', fontsize=18)
    ax1.set_ylabel('Mean Min Distance from Pal 5 [kpc]', fontsize=18)
    ax1.legend(fontsize=14, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=16)
    
    # 子图2: 所有成员的最小距离分布
    ax2 = axes[1]
    
    # 创建箱线图数据
    all_dists = [dists[np.isfinite(dists)] for dists in results['all_min_dists']]
    
    bp = ax2.boxplot(all_dists, positions=range(len(masses)), widths=0.6,
                     patch_artist=True, showfliers=False)
    
    for patch in bp['boxes']:
        patch.set_facecolor('#6FCDCD')
        patch.set_alpha(0.7)
    
    # 标记最优位置
    ax2.axvline(optimal_idx, color='#F18F01', linestyle='--', lw=2.5, alpha=0.7,
               label=f'Optimal Mass: {optimal_mass:.2e} M☉')
    
    ax2.set_xlabel('Halo Mass', fontsize=18)
    ax2.set_ylabel('Min Distance Distribution [kpc]', fontsize=18)
    ax2.set_xticks(range(0, len(masses), max(1, len(masses)//10)))
    ax2.set_xticklabels([f'{m:.1e}' for m in masses[::max(1, len(masses)//10)]], 
                        rotation=45, ha='right')
    ax2.legend(fontsize=14, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n图片已保存至: {save_path}")
    
    plt.show()
    
    return fig


def print_optimization_summary(results):
    """
    打印优化结果摘要
    
    Parameters:
    -----------
    results : dict
        optimize_halo_mass 的返回结果
    """
    masses = results['masses']
    mean_dist = results['mean_min_dist']
    
    optimal_idx = np.argmin(mean_dist)
    optimal_mass = masses[optimal_idx]
    optimal_dist = mean_dist[optimal_idx]
    
    print(f"\n{'='*80}")
    print("优化结果摘要")
    print(f"{'='*80}")
    print(f"最优 Halo 质量: {optimal_mass:.4e} M☉")
    print(f"对应的平均最小距离: {optimal_dist:.4f} kpc")
    print(f"对应的中位数最小距离: {results['median_min_dist'][optimal_idx]:.4f} kpc")
    print(f"对应的距离标准差: {results['std_min_dist'][optimal_idx]:.4f} kpc")
    print(f"{'='*80}\n")
    
    # 创建结果表格
    df_results = pd.DataFrame({
        'Halo Mass [M☉]': [f'{m:.3e}' for m in masses],
        'Mean Min Dist [kpc]': mean_dist,
        'Median Min Dist [kpc]': results['median_min_dist'],
        'Std Dev [kpc]': results['std_min_dist']
    })
    
    print("详细结果表格:")
    print(df_results.to_string(index=False))
    print(f"\n{'='*80}\n")
    
    return optimal_mass, optimal_dist


# ============================================================
# 主函数
# ============================================================

def main():
    """
    主函数 - 需要传入实际数据
    
    使用方法:
    1. 从你的主脚本中导入或定义 ic_pal5 和 sample_A
    2. 调用此函数进行优化
    """
    
    # 示例：这里需要替换为实际的初始条件
    # ic_pal5 = np.array([x, y, z, vx, vy, vz])  # Pal 5 的初始条件
    # ic_streams = sample_A[:, :6]  # 星流成员的初始条件
    
    print("注意: 此脚本需要从主脚本中传入以下变量:")
    print("  - ic_pal5: Pal 5 的初始条件 [x, y, z, vx, vy, vz]")
    print("  - sample_A: 星流成员的初始条件数组")
    print("\n请在主脚本中导入此模块并调用相应函数。")
    print("\n示例用法:")
    print("  from optimize_halo_mass import optimize_halo_mass, plot_optimization_results")
    print("  results = optimize_halo_mass(ic_pal5, sample_A[:, :6], (1e11, 2e12), n_steps=20)")
    print("  plot_optimization_results(results, 'optimization_results.png')")
    

if __name__ == "__main__":
    main()