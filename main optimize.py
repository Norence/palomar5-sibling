#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主脚本 - 优化 Halo 质量参数并可视化 Pal 5 与星流成员的轨道相位
"""

import numpy as np
import matplotlib.pyplot as plt
import agama
import galpy.util.coords as guc
from astropy.io import fits
from astropy import constants as c
from scipy.interpolate import interp1d
import pandas as pd

# 导入优化模块
from optimize_halo_mass import (
    optimize_halo_mass, 
    plot_optimization_results,
    print_optimization_summary,
    create_potential
)

# ============================================================
# 基础参数设置
# ============================================================
u_sun = 11.1
v_sun = 12.24
w_sun = 7.25
x_sun = 8.3
v_lsr = 232.
Msun = 1.9884e30
Mhalo = 1.25e12 * Msun
Mdisk = 0.065 * Mhalo
Mbulge = 0.01 * Mhalo
rh = 38.35 * 1.0e3 * c.pc.value
b0 = 3.5 * 1.0e3 * c.pc.value
c0 = 0.15 * b0

# ============================================================
# 加载球状星团数据
# ============================================================
print("加载球状星团数据...")
GCs = fits.open("/home/zyh/stream/S2808stream/GC_PM_catalog_Vasiliev.fits")
GCs_data = GCs[1].data
name_gc = GCs_data["Name"]
ra_gc = np.array(GCs_data["RAdeg"])
dec_gc = np.array(GCs_data["DEdeg"])
dist_gc = np.array(GCs_data["Dist"])
HRV_gc = np.array(GCs_data["HRV"])
pmRA_gc = np.array(GCs_data["pmRA"])
pmDE_gc = np.array(GCs_data["pmDE"])

# 匹配目标球状星团
target_gcs = [
    "E 3", "Rup 106", "Pal 5",
    "NGC 4590", "NGC 5024", "NGC 5053", "NGC 5272", "NGC 6981",
    "NGC 5634", "NGC 5904"
]

extracted_gcs = []
for target in target_gcs:
    matches = [i for i, name in enumerate(name_gc) 
               if name.strip().upper() == target.strip().upper()]
    
    if matches:
        idx = matches[0]
        extracted_gcs.append({
            'Name': name_gc[idx],
            'RA': ra_gc[idx],
            'Dec': dec_gc[idx],
            'Distance': dist_gc[idx],
            'HRV': HRV_gc[idx],
            'pmRA': pmRA_gc[idx],
            'pmDE': pmDE_gc[idx]
        })

df_extracted = pd.DataFrame(extracted_gcs)

print("\n" + "="*80)
print("提取的球状星团参数:")
print("="*80)
print(df_extracted.to_string(index=False))

# ============================================================
# 准备星流成员数据
# ============================================================
# 注意: 这里需要从你的主代码中获取实际的星流成员数据
# ra_all, dec_all, pmra_all, pmdec_all, RV_all 和 sample_A

print("\n" + "="*80)
print("准备星流成员数据...")
print("="*80)

# 这里假设你已经有了以下变量（从你的主代码中）:
# ra_all, dec_all, pmra_all, pmdec_all, RV_all
# sample_A - 包含所有星流成员的初始条件

# 示例: 如果这些变量还未定义，你需要从主代码中复制相关部分
# mask_in_L = ~mask_out_L
# mask_in_D = ~mask_out_D
# mask_in_S = ~mask_out_S
# 
# ra_all = np.concatenate([ra_L[mask_in_L], ra_D[mask_in_D], ra_S[mask_in_S]])
# dec_all = np.concatenate([dec_L[mask_in_L], dec_D[mask_in_D], dec_S[mask_in_S]])
# ... 等等

# ============================================================
# 获取 Pal 5 的初始条件
# ============================================================
# 从提取的球状星团中找到 Pal 5
pal5_idx = [i for i, gc in enumerate(extracted_gcs) if 'Pal' in gc['Name'].upper()][0]
pal5_data = extracted_gcs[pal5_idx]

print(f"\nPal 5 参数:")
print(f"  RA: {pal5_data['RA']:.4f}°")
print(f"  Dec: {pal5_data['Dec']:.4f}°")
print(f"  Distance: {pal5_data['Distance']:.2f} kpc")
print(f"  RV: {pal5_data['HRV']:.2f} km/s")
print(f"  pmRA: {pal5_data['pmRA']:.4f} mas/yr")
print(f"  pmDE: {pal5_data['pmDE']:.4f} mas/yr")

# 将 Pal 5 的观测数据转换为笛卡尔坐标初始条件
# 这里需要根据你的坐标转换代码来设置
# ic_pal5 = convert_to_cartesian(pal5_data)  # 你需要实现这个函数

# ============================================================
# 优化 Halo 质量
# ============================================================
print("\n" + "="*80)
print("开始优化 Halo 质量参数")
print("="*80)

# 注意: 在实际使用前，确保 ic_pal5 和 sample_A 已正确定义
# 这里我们先用占位符
try:
    # 执行优化
    results = optimize_halo_mass(
        ic_pal5=ic_pal5,              # Pal 5 的初始条件
        ic_streams=sample_A[:, :6],   # 星流成员的初始条件
        mass_range=(1e11, 2e12),      # 质量范围
        n_steps=20                    # 遍历步数
    )
    
    # 打印优化结果
    optimal_mass, optimal_dist = print_optimization_summary(results)
    
    # 绘制优化结果
    plot_optimization_results(
        results, 
        save_path="/home/zyh/stream/palomar5_sibling/plot/halo_mass_optimization.png"
    )
    
    print(f"\n推荐使用的最优 Halo 质量: {optimal_mass:.4e} M☉")
    
except NameError as e:
    print(f"\n警告: {e}")
    print("请确保已定义以下变量:")
    print("  - ic_pal5: Pal 5 的初始条件")
    print("  - sample_A: 星流成员的初始条件数组")
    print("\n继续使用默认质量 1e12...")
    optimal_mass = 1e12

# ============================================================
# 第二步: 使用最优质量绘制轨道相位图
# ============================================================
print("\n" + "="*80)
print("使用最优质量绘制轨道相位图")
print("="*80)

# 使用最优质量创建势能
pot_total = create_potential(optimal_mass)

# 计算 Pal 5 的轨道
t_gc, orb_gc = agama.orbit(potential=pot_total, ic=ic_pal5,
                           time=2.*pot_total.Tcirc(ic_pal5), trajsize=10000)
t_gc_m, orb_gc_m = agama.orbit(potential=pot_total, ic=ic_pal5,
                               time=-2.*pot_total.Tcirc(ic_pal5), trajsize=10000)

# 绘制距离演化图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# 前2个：LAMOST - 红色系
# 中间3个：DESI - 紫色系  
# 最后2个：S5 - 蓝色系
colors_sample = [
    '#DC3545',  # GC #1 - LAMOST 
    '#DC3545',  # GC #2 - LAMOST 
    '#8B00FF',  # GC #3 - DESI 
    '#8B00FF',  # GC #4 - DESI 
    '#8B00FF',  # GC #5 - DESI 
    '#6B98DC',  # GC #6 - S5 
    '#6B98DC',  # GC #7 - S5 
]

print("\n计算星流成员与 Pal 5 的距离演化...")

for member_idx in range(len(ra_all)):
    init_condition = sample_A[member_idx, :6]
    
    # 只计算后向轨道（过去）
    t_m, orb_m = agama.orbit(potential=pot_total, ic=init_condition,
                             time=-2.*pot_total.Tcirc(init_condition), trajsize=10000)
    
    # 反转时间序列
    t_stream = t_m[::-1]
    orb_stream = orb_m[::-1]
    
    # 计算与 Pal 5 的距离
    t_gc_past = t_gc_m[::-1]
    orb_gc_past = orb_gc_m[::-1]
    
    # 在共同时间网格上插值
    t_min = max(t_gc_past[0], t_stream[0])
    t_max = min(t_gc_past[-1], t_stream[-1])
    t_common = np.linspace(t_min, t_max, 1000)
    
    # 插值 Pal 5
    gc_x_interp = interp1d(t_gc_past, orb_gc_past[:,0], kind='cubic')(t_common)
    gc_y_interp = interp1d(t_gc_past, orb_gc_past[:,1], kind='cubic')(t_common)
    gc_z_interp = interp1d(t_gc_past, orb_gc_past[:,2], kind='cubic')(t_common)
    
    # 插值星流成员
    stream_x_interp = interp1d(t_stream, orb_stream[:,0], kind='cubic')(t_common)
    stream_y_interp = interp1d(t_stream, orb_stream[:,1], kind='cubic')(t_common)
    stream_z_interp = interp1d(t_stream, orb_stream[:,2], kind='cubic')(t_common)
    
    # 计算3D距离
    dist_3d = np.sqrt((gc_x_interp - stream_x_interp)**2 + 
                      (gc_y_interp - stream_y_interp)**2 + 
                      (gc_z_interp - stream_z_interp)**2)
    
    # 绘制距离演化曲线 
    ax.plot(t_common, dist_3d, '-', color=colors_sample[member_idx], lw=2.5, 
            alpha=0.85)
    
    # 标记最小距离点 
    min_idx = np.argmin(dist_3d)
    ax.plot(t_common[min_idx], dist_3d[min_idx], 'o', 
            color=colors_sample[member_idx], ms=12, mec='white', mew=2, zorder=10)

# 标记参考线
ax.axhline(5, color='#FFA500', ls=':', lw=2.5, alpha=0.6, label='5 kpc', zorder=5)

# 手动添加图例
ax.plot([0], [0], '-', color='#DC3545', lw=2.5, alpha=0.85, label='LAMOST')
ax.plot([0], [0], '-', color='#8B00FF', lw=2.5, alpha=0.85, label='DESI')
ax.plot([0], [0], '-', color='#6B98DC', lw=2.5, alpha=0.85, label='S5')

# 设置坐标轴
ax.set_xlim([t_common[0], 0])
ax.set_ylim([0, None])

ax.set_xlabel('Time [Gyr]', fontsize=22)
ax.set_ylabel('3D Distance from Pal 5 [kpc]', fontsize=22)
ax.set_title(f'Optimal Halo Mass: {optimal_mass:.2e} M☉', fontsize=20, pad=15)

# 图例优化
ax.legend(fontsize=16, loc="lower right")

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()

save_path = "/home/zyh/stream/palomar5_sibling/plot/distance_evolution_past_optimized.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n图片已保存至: {save_path}")
plt.show()

print("\n" + "="*80)
print("分析完成!")
print("="*80)