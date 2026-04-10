import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, shapiro
from typing import Dict, List, Tuple, Any

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.animation as animation
from IPython.display import HTML

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

import tsm_config as config


# ======================================================================
# ЭТАП 5: ВИЗУАЛИЗАЦИЯ
# ======================================================================

# 8.0 Scatter plot: z_lex vs z_sem с зонами диагностики
def scatter_plot(adaptive_thresholds, df_final):
    fig, ax = plt.subplots(figsize=(10, 7))

    # Отрисовка точек по типам диагнозов
    for i, (diag_type, color) in enumerate(config.DIAGNOSIS_COLORS.items()):
        subset = df_final[df_final['diagnosis_type'] == diag_type]
        if len(subset) > 0:
            ax.scatter(subset['z_lex'], subset['z_sem'], 
                    c=color, 
                    label=config.DIAGNOSIS_ZONES[i],
                    alpha=0.7, s=40, edgecolors='white', linewidth=0.5)

    # Оси
    ax.axhline(y=0, color='#34495e', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.axvline(x=0, color='#34495e', linestyle='-', linewidth=1.5, alpha=0.7)

    # Границы зон (τ = 2)
    # tau = 2.0
    tau_lex_upper = adaptive_thresholds.get('tau_lex_upper')
    tau_lex_lower = adaptive_thresholds.get('tau_lex_lower')
    tau_sem_upper = adaptive_thresholds.get('tau_sem_upper')
    tau_sem_lower = adaptive_thresholds.get('tau_sem_lower')

    ax.axhline(y=tau_sem_lower, color='#34495e', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=tau_sem_upper, color='#34495e', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=tau_lex_lower, color='#34495e', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=tau_lex_upper, color='#34495e', linestyle='--', linewidth=1.5, alpha=0.7)

    # Прямоугольник "хорошей зоны"
    from matplotlib.patches import Rectangle
    good_zone = Rectangle((tau_lex_lower, tau_sem_lower), tau_lex_upper-tau_lex_lower, tau_sem_upper-tau_sem_lower, 
                        linewidth=2, edgecolor='#2ecc71', 
                        facecolor='#2ecc71', alpha=0.1)
    ax.add_patch(good_zone)

    ax.axhline(y=2, color='#34495e', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axhline(y=-2, color='#34495e', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=2, color='#34495e', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=-2, color='#34495e', linestyle=':', linewidth=1.5, alpha=0.7)

    # Прямоугольник "хорошей зоны" τ = 2
    from matplotlib.patches import Rectangle
    good_zone = Rectangle((-2, -2), 4, 4, 
                        linewidth=2, edgecolor='#2ecc71', 
                        facecolor='#2ecc71', alpha=0.15)
    ax.add_patch(good_zone)

    # Аннотации зон с улучшенным стилем
    bbox_style = dict(boxstyle='round,pad=0.5', facecolor='white', 
                    edgecolor='gray', alpha=0.9, linewidth=1.5)

    # Центр целевой зоны
    center_x = (tau_lex_lower + tau_lex_upper) / 2
    center_y = (tau_sem_lower + tau_sem_upper) / 2

    # Аннотации зон с улучшенным стилем
    ax.text(center_x, 0, 'Целевая зона', ha='center', fontsize=10, 
            fontweight='bold', bbox=bbox_style, color='#2ecc71')
    ax.text(tau_lex_upper + 1.4, 0, 'Избыточное\nопирование', ha='center', fontsize=10,# rotation=90,
            fontweight='bold', bbox=bbox_style, color='#3498db')

    ax.text(center_x, -3.2, 'Семантическая\nнеполнота', ha='center', fontsize=10,
            fontweight='bold', bbox=bbox_style, color='#9b59b6')
    ax.text(center_x, tau_sem_upper + 1.4, 'Неоднозначный\nпаттерн', ha='center', fontsize=10,
            fontweight='bold', bbox=bbox_style, color='gray')
    if 'low_lexical' in list(df_final['diagnosis_type']):
        ax.text(tau_lex_lower - 1.4, center_y, 'Низкое лекс.\nсходство', ha='center', fontsize=10,# rotation=90,
                fontweight='bold', bbox=bbox_style, color='#f39c12') # #f1c40f #3498db


    if 'under_compressed' in list(df_final['diagnosis_type']):
        ax.text(tau_lex_upper + 1.4, tau_sem_upper + 1.4, 'Недостоточное\nсжатие', ha='center', fontsize=10,# rotation=90,
                fontweight='bold', bbox=bbox_style, color='#e74c3c')
    if 'over_compressed' in list(df_final['diagnosis_type']):
        ax.text(tau_lex_upper + 1.4, -3.2, 'Избыточное\nсжатие', ha='center', fontsize=10,# rotation=90,
                fontweight='bold', bbox=bbox_style, color='#008f8f')


    ax.set_xlabel(f'Лекс. отклон. ({config.LEX_NAME})', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Сем. отклон. ({config.SEM_NAME})', fontsize=12, fontweight='bold')
    ax.set_title('Двумерная диагностическая классификация рефератов', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11, framealpha=0.95, 
            edgecolor='gray', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)

    margin = 0.75
    x_min, x_max = min(df_final['z_lex']), max(df_final['z_lex'])
    y_min, y_max = min(df_final['z_sem']), max(df_final['z_sem'])
    ax.set_xlim(x_min-margin+1, x_max+margin)
    ax.set_ylim(y_min-margin, y_max+margin)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_path = f'{config.MODE}/tsm-figures/diagnostic_scatter.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"✓ График 1 сохранён: {save_path}")
    # plt.show()



# 8.1 3D Scatter plot: z_lex vs z_sem vs z_comp с проекциями на плоскости
def scatter_plot_3d(adaptive_thresholds, df_final):
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111, projection='3d')

    # Пороги
    tau_lex_upper = adaptive_thresholds.get('tau_lex_upper')
    tau_lex_lower = adaptive_thresholds.get('tau_lex_lower')
    tau_sem_upper = adaptive_thresholds.get('tau_sem_upper')
    tau_sem_lower = adaptive_thresholds.get('tau_sem_lower')
    tau_comp_upper = adaptive_thresholds.get('tau_comp_upper', 2.0)
    tau_comp_lower = adaptive_thresholds.get('tau_comp_lower', -2.0)

    # Пределы осей (с запасом)
    margin = 1.5
    x_lo = min(df_final['z_lex'].min(), tau_lex_lower) - margin + 1.0
    x_hi = max(df_final['z_lex'].max(), tau_lex_upper) + margin + 2.0
    y_lo = min(df_final['z_sem'].min(), tau_sem_lower) - margin
    y_hi = max(df_final['z_sem'].max(), tau_sem_upper) + margin + 1.0
    z_lo = min(df_final['z_comp'].min(), tau_comp_lower) - margin - 1.2
    z_hi = max(df_final['z_comp'].max(), tau_comp_upper) + margin

    # --- 1. Проекции на стенки (рисуем первыми, чтобы были на заднем плане) ---
    proj_alpha = 0.20
    proj_size = 5

    for diag_type, color in config.DIAGNOSIS_COLORS.items():
        subset = df_final[df_final['diagnosis_type'] == diag_type]
        if len(subset) > 0:
            zl = subset['z_lex'].values
            zs = subset['z_sem'].values
            zc = subset['z_comp'].values

            # Проекция на плоскость XY (z = z_lo) — «пол»
            ax.scatter(zl, zs, np.full_like(zl, z_lo), c=color,
                    alpha=0.2, marker='o', edgecolors='none', s=proj_size) # 

            # Проекция на плоскость XZ (y = y_hi) — «задняя стена»
            ax.scatter(zl, np.full_like(zs, y_hi), zc, c=color,
                    alpha=proj_alpha, s=proj_size)

            # Проекция на плоскость YZ (x = x_lo) — «левая стена»
            ax.scatter(np.full_like(zl, x_hi), zs, zc, c=color,
                    alpha=proj_alpha, s=proj_size)

    labels = [
        'Целевая зона',
        'Избыточное копирование',
        'Семантическая неполнота',
        'Низкая лекс. сходство',
        'Неоднозначный паттерн',
        'Недостаточное сжатие',
        'Избыточное сжатие',
        ]
    # --- 2. Основные точки (поверх проекций) ---
    i = 0
    for diag_type, color in config.DIAGNOSIS_COLORS.items():
        subset = df_final[df_final['diagnosis_type'] == diag_type]
        if len(subset) > 0:
            ax.scatter(subset['z_lex'], subset['z_sem'], subset['z_comp'],
                    c=color, label=labels[i], # diag_type.replace('_', ' ').capitalize(),
                    #    alpha=0.75 if diag_type != 'under_compressed' else 0.2, s=50, edgecolors='white', linewidth=0.3, depthshade=True)
                    alpha=0.75, s=15, edgecolors='white', linewidth=0.3, depthshade=True)
            i+=1

    # --- 3. Целевая зона: полупрозрачный параллелепипед ---
    x0, x1 = tau_lex_lower, tau_lex_upper
    y0, y1 = tau_sem_lower, tau_sem_upper
    z0, z1 = tau_comp_lower, tau_comp_upper

    faces = [
        [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],
        [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
        [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],
        [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],
        [[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]],
        [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]],
    ]
    poly = Poly3DCollection(faces, alpha=0.18, facecolor='#2ecc71', edgecolor='#2ecc71', linewidth=0.8)
    ax.add_collection3d(poly)
    faces = [
        [[-2,-2,-2],[2,-2,-2],[2,2,-2],[-2,2,-2]],
        [[-2,-2,2],[2,-2,2],[2,2,2],[-2,2,2]],
        [[-2,-2,-2],[2,-2,-2],[2,-2,2],[-2,-2,2]],
        [[-2,2,-2],[2,2,-2],[2,2,2],[-2,2,2]],
        [[-2,-2,-2],[-2,2,-2],[-2,2,2],[-2,-2,2]],
        [[2,-2,-2],[2,2,-2],[2,2,2],[2,-2,2]],
    ]
    poly = Poly3DCollection(faces, alpha=0.08, facecolor='#2ecc71', edgecolor='#00754a', linewidth=1.2)
    ax.add_collection3d(poly)


    # --- 4.1 Проекции целевой зоны на стенки (прямоугольники-тени) ---
    # На пол (XY, z=z_lo)
    floor_rect = [[x0,y0,z_lo],[x1,y0,z_lo],[x1,y1,z_lo],[x0,y1,z_lo]]
    poly_floor = Poly3DCollection([floor_rect], alpha=0.15, facecolor='#2ecc71', edgecolor='#2ecc71', linewidth=0.5, linestyle='--')
    ax.add_collection3d(poly_floor)
    floor_rect = [[-2,-2,z_lo],[2,-2,z_lo],[2,2,z_lo],[-2,2,z_lo]]
    poly_floor = Poly3DCollection([floor_rect], alpha=0.15, facecolor='#2ecc71', edgecolor='#2ecc71', linewidth=0.5, linestyle='--')
    ax.add_collection3d(poly_floor)

    # На заднюю стену (XZ, y=y_hi)
    back_rect = [[x0,y_hi,z0],[x1,y_hi,z0],[x1,y_hi,z1],[x0,y_hi,z1]]
    poly_back = Poly3DCollection([back_rect], alpha=0.3, facecolor='#2ecc71', edgecolor='#2ecc71', linewidth=0.5, linestyle='--')
    ax.add_collection3d(poly_back)
    back_rect = [[-2,y_hi,-2],[2,y_hi,-2],[2,y_hi,2],[-2,y_hi,2]]
    poly_back = Poly3DCollection([back_rect], alpha=0.15, facecolor='#2ecc71', edgecolor='#2ecc71', linewidth=0.5, linestyle='--')
    ax.add_collection3d(poly_back)

    # На левую стену (YZ, x=x_lo)
    left_rect = [[x_hi,y0,z0],[x_hi,y1,z0],[x_hi,y1,z1],[x_hi,y0,z1]]
    poly_left = Poly3DCollection([left_rect], alpha=0.15, facecolor='#2ecc71', edgecolor='#2ecc71', linewidth=0.5, linestyle='--')
    ax.add_collection3d(poly_left)
    left_rect = [[x_hi,-2,-2],[x_hi,2,-2],[x_hi,2,2],[x_hi,-2,2]]
    poly_left = Poly3DCollection([left_rect], alpha=0.15, facecolor='#2ecc71', edgecolor='#2ecc71', linewidth=0.5, linestyle='--')
    ax.add_collection3d(poly_left)

    # --- Оси ---
    x_coords = np.array([x_lo, x_hi])
    y_coords = np.array([y_hi, y_hi])
    z_coords = np.array([0, 0])
    ax.plot(x_coords, y_coords, z_coords, color='#34495e', linestyle='--', linewidth=1.5, alpha=0.5)
    x_coords = np.array([x_hi, x_hi])
    y_coords = np.array([y_lo, y_hi])
    z_coords = np.array([0, 0])
    ax.plot(x_coords, y_coords, z_coords, color='#34495e', linestyle='--', linewidth=1.5, alpha=0.5)
    x_coords = np.array([x_lo, x_hi])
    y_coords = np.array([0, 0])
    z_coords = np.array([z_lo, z_lo])
    ax.plot(x_coords, y_coords, z_coords, color='#34495e', linestyle='--', linewidth=1.5, alpha=0.5)

    x_coords = np.array([0, 0])
    y_coords = np.array([y_hi, y_hi])
    z_coords = np.array([z_lo, z_hi])
    ax.plot(x_coords, y_coords, z_coords, color='#34495e', linestyle='--', linewidth=1.5, alpha=0.5)
    x_coords = np.array([x_hi, x_hi])
    y_coords = np.array([0, 0])
    z_coords = np.array([z_lo, z_hi])
    ax.plot(x_coords, y_coords, z_coords, color='#34495e', linestyle='--', linewidth=1.5, alpha=0.5)
    x_coords = np.array([0, 0])
    y_coords = np.array([y_lo, y_hi])
    z_coords = np.array([z_lo, z_lo])
    ax.plot(x_coords, y_coords, z_coords, color='#34495e', linestyle='--', linewidth=1.5, alpha=0.5)

    x_coords = np.array([x_lo, x_hi])
    y_coords = np.array([0, 0])
    z_coords = np.array([0, 0])
    ax.plot(x_coords, y_coords, z_coords, color="#34495e", linestyle='-', linewidth=1.5, alpha=1)
    x_coords = np.array([0, 0])
    y_coords = np.array([0, 0])
    z_coords = np.array([z_lo, z_hi])
    ax.plot(x_coords, y_coords, z_coords, color='#34495e', linestyle='-', linewidth=1.5, alpha=1)
    x_coords = np.array([0, 0])
    y_coords = np.array([y_lo, y_hi])
    z_coords = np.array([0, 0])
    ax.plot(x_coords, y_coords, z_coords, color='#34495e', linestyle='-', linewidth=1.5, alpha=1)

    # --- 5. Оформление ---
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)

    ax.set_xlabel(f'\nЛекс. отклон. ({config.LEX_NAME})', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel(f'\nСем. отклон. ({config.SEM_NAME})', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel('\nОтклон. сжат. (Compression)', fontsize=11, fontweight='bold', labelpad=10)

    ax.set_title(f'Трёхмерная диагностическая классификация рефератов', # \nПороги: {THRESHOLD_MODE_LABEL}
                fontsize=14, fontweight='bold', pad=-20)

    ax.legend(loc='upper left', fontsize=9, framealpha=0.95,
            edgecolor='gray', fancybox=True, shadow=True,
            bbox_to_anchor=(0.0, 0.95))

    # Угол обзора
    ax.view_init(elev=90-65, azim=-90-40, roll=0)

    # Панели
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    save_path = f'{config.MODE}/tsm-figures/diagnostic_scatter_3d.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.5)
    print(f"✓ График 1 сохранён: {save_path}")
    # plt.show()



# 8.1 Scatter plots: z_rouge vs z_bert - ОТДЕЛЬНО ДЛЯ КАЖДОЙ МОДЕЛИ
def plot_model_diagnostic(df_model, model_name, adaptive_thresholds):
    """
    Создаёт диагностический scatter plot для одной модели с АДАПТИВНЫМИ порогами.
    """
    from matplotlib.patches import Rectangle
    
#     fig, ax = plt.subplots(figsize=(10, 9))
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Извлекаем адаптивные пороги
    tau_lex_upper = adaptive_thresholds.get('tau_lex_upper')
    tau_lex_lower = adaptive_thresholds.get('tau_lex_lower')
    tau_sem_upper = adaptive_thresholds.get('tau_sem_upper')
    tau_sem_lower = adaptive_thresholds.get('tau_sem_lower')
    
    # Отрисовка точек по типам диагнозов
    for diag_type, color in config.DIAGNOSIS_COLORS.items():
        subset = df_model[df_model['diagnosis_type'] == diag_type]
        if len(subset) > 0:
            ax.scatter(subset['z_lex'], subset['z_sem'], 
                      c=color, label=config.DIAG_TRANSLATE.get(diag_type), #diag_type.capitalize().replace('_', ' '), 
                      alpha=0.8, s=100, edgecolors='white', linewidth=0.5)
    
    # АДАПТИВНЫЕ границы зон (не симметричные!)
    ax.axvline(x=tau_lex_upper, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7,)
            #    label=f'τ_{LEX_MODE}+ = {tau_lex_upper:.2f}')
    ax.axvline(x=tau_lex_lower, color='#3498db', linestyle='--', linewidth=2, alpha=0.7,)
            #    label=f'τ_{LEX_MODE}+ = {tau_lex_lower:.2f}')
    
    ax.axhline(y=tau_sem_upper, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7,)
            #    label=f'τ_{SEM_MODE}+ = {tau_sem_upper:.2f}')
    ax.axhline(y=tau_sem_lower, color='#3498db', linestyle='--', linewidth=2, alpha=0.7, )
            #    label=f'τ_{SEM_MODE}- = {tau_sem_lower:.2f}')

    ax.axhline(y=0, color='#7f8c8d', linestyle='-', linewidth=0.8, alpha=0.4)
    ax.axvline(x=0, color='#7f8c8d', linestyle='-', linewidth=0.8, alpha=0.4)
    
    # АДАПТИВНЫЙ прямоугольник "хорошей зоны" (не квадрат!)
    width_good = tau_lex_upper - tau_lex_lower
    height_good = tau_sem_upper - tau_sem_lower
    good_zone = Rectangle((tau_lex_lower, tau_sem_lower), width_good, height_good, 
                          linewidth=2.5, edgecolor='#2ecc71', 
                          facecolor='#2ecc71', alpha=0.15)
    ax.add_patch(good_zone)
    
    # Аннотации зон
    bbox_style = dict(boxstyle='round,pad=0.5', facecolor='white', 
                      edgecolor='gray', alpha=0.9, linewidth=1.5)
    
    # Центр целевой зоны
    center_x = (tau_lex_lower + tau_lex_upper) / 2
    center_y = (tau_sem_lower + tau_sem_upper) / 2

    # Статистика по модели
    stats_text_pct = f"n = {len(df_model)}, пороги: ±τ\n"
    for diag_type in config.DIAGNOSIS_COLORS.keys():
        count = (df_model['diagnosis_type'] == diag_type).sum()
        pct = 100 * count / len(df_model) if len(df_model) > 0 else 0
        if count > 0:
            # stats_text_pct += f"{diag_type.capitalize().replace('_', ' ')}: {count} ({pct:.1f}%)\n"
            stats_text_pct += f"{config.DIAG_TRANSLATE.get(diag_type)}: {count} ({pct:.1f}%)\n"

    stats_text_tau = f"n = {len(df_model)}, пороги: ±τ\n"
    for diag_type in config.DIAGNOSIS_COLORS.keys():
        count = (df_model['diagnosis_type'] == diag_type).sum()
        pct = 100 * count / len(df_model) if len(df_model) > 0 else 0
        if count > 0:
            stats_text_tau += f"{config.DIAG_TRANSLATE.get(diag_type)}: {count} ({pct:.1f}%)\n"
    
    ax.text(0.02, 0.98, stats_text_pct, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.text(0.02, 0.98, stats_text_tau, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_xlabel(f'Z-score ({config.LEX_NAME})', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Z-score ({config.SEM_NAME})', fontsize=12, fontweight='bold')
    
    # Красивое название модели
    model_display_name = model_name.replace('summary_', '').upper()
    ax.set_title(f'Диагностика модели: {config.INTER_LABELS.get(model_name)}', 
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95, 
              edgecolor='gray', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    # Оси
    ax.axhline(y=0, color='#34495e', linestyle='-', linewidth=1.5, alpha=0.3)
    ax.axvline(x=0, color='#34495e', linestyle='-', linewidth=1.5, alpha=0.3)
    
    # Динамические пределы осей (чуть шире зоны данных)
    margin = 1.75
    x_min = min(df_model['z_lex'].min(), tau_lex_lower) - margin
    x_max = max(df_model['z_lex'].max(), tau_lex_upper) + margin*2
    y_min = min(df_model['z_sem'].min(), tau_sem_lower) - margin
    y_max = max(df_model['z_sem'].max(), tau_sem_upper) + margin*2
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig, ax



# Создаём отдельные графики для каждой модели
def scatter_plot_each_models(adaptive_thresholds, df_final):
    models_list = df_final['model'].unique()
    print(f"\nСоздание индивидуальных диагностических графиков для {len(models_list)} моделей...")
    print(f"  (с адаптивными порогами)")

    for model_name in sorted(models_list):
        df_model = df_final[df_final['model'] == model_name]
        fig, ax = plot_model_diagnostic(df_model, model_name, adaptive_thresholds)
        
        filename = f'diagnostic_scatter_{model_name[model_name.find("/")+1:]}.png'
        save_path = f'{config.MODE}/tsm-figures/model-diagnosis/{filename}'
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Сохранён: {save_path}")
        plt.close()



# 8.1.2 Сводный график (все модели на одном + эталонные АР)
def scatter_plot_all_models(adaptive_thresholds, df_final, df_final_ref):
    print(f"\nСоздание сводного графика (режим порогов: {config.THRESHOLD_MODE})...")

    models_list = df_final['model'].unique()

    # fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.flatten()

    from matplotlib.patches import Rectangle

    # Пороги из выбранного режима
    tau_lex_upper = adaptive_thresholds['tau_lex_upper']
    tau_lex_lower = adaptive_thresholds['tau_lex_lower']
    tau_sem_upper = adaptive_thresholds['tau_sem_upper']
    tau_sem_lower = adaptive_thresholds['tau_sem_lower']
    tau_comp_upper = adaptive_thresholds['tau_comp_upper']
    tau_comp_lower = adaptive_thresholds['tau_comp_lower']

    # =============================================================================
    # ПЕРВЫЙ SUBPLOT: Эталонные авторские рефераты (ОТ-АР)
    # =============================================================================
    ax_ref = axes[0]

    # Получаем данные ОТ-АР и вычисляем z-scores
    # df_ot_ar = df_raw[df_raw['comparison'] == 'OT-AR'].copy()
    # df_ot_ar = df_ot_ar[~df_ot_ar['doc_id'].isin(calibration['outlier_doc_ids'])]

    # df_ot_ar['z_lex'] = (df_ot_ar['lexical'] - calibration['mu_lex']) / calibration['sigma_lex']
    # df_ot_ar['z_sem'] = (df_ot_ar['semantic'] - calibration['mu_sem']) / calibration['sigma_sem']

    df_ot_ar = df_final_ref

    # LEX [-1.352, 1.370], SEM [-1.293, 1.268], COMP [-0.762, 1.466]

    # Отрисовка точек АР
    for diag_type, color in config.DIAGNOSIS_COLORS.items():
        subset = df_ot_ar[df_ot_ar['diagnosis_type'] == diag_type]
        if len(subset) > 0:
            ax_ref.scatter(subset['z_lex'], subset['z_sem'], 
                        c=color, alpha=0.7, s=40, edgecolors='white', linewidth=0.5)


    # Оси
    ax_ref.axhline(y=0, color='#34495e', linestyle='-', linewidth=1.5, alpha=0.3)
    ax_ref.axvline(x=0, color='#34495e', linestyle='-', linewidth=1.5, alpha=0.3)

    # Границы зон
    ax_ref.axhline(y=tau_sem_lower, color='#34495e70', linestyle='--', linewidth=1, alpha=0.7)
    ax_ref.axhline(y=tau_sem_upper, color='#34495e70', linestyle='--', linewidth=1, alpha=0.7)
    ax_ref.axvline(x=tau_lex_lower, color='#34495e70', linestyle='--', linewidth=1, alpha=0.7)
    ax_ref.axvline(x=tau_lex_upper, color='#34495e70', linestyle='--', linewidth=1, alpha=0.7)

    ax_ref.axhline(y=-2, color='#34495e70', linestyle=':', linewidth=1, alpha=0.7)
    ax_ref.axhline(y=2,  color='#34495e70', linestyle=':', linewidth=1, alpha=0.7)
    ax_ref.axvline(x=-2, color='#34495e70', linestyle=':', linewidth=1, alpha=0.7)
    ax_ref.axvline(x=2,  color='#34495e70', linestyle=':', linewidth=1, alpha=0.7)

    # Целевая зона
    width_good = tau_lex_upper - tau_lex_lower
    height_good = tau_sem_upper - tau_sem_lower
    good_zone = Rectangle((tau_lex_lower, tau_sem_lower), width_good, height_good, 
                        linewidth=2, edgecolor='#2ecc71', 
                        facecolor='#2ecc71', alpha=0.15)
    good_zone = Rectangle((-2, -2), 4, 4, 
                        linewidth=2, edgecolor='#2ecc71', 
                        facecolor='#2ecc71', alpha=0.1)
    ax_ref.add_patch(good_zone)

    # ax_ref.set_title('АВТОРСКИЕ РЕФЕРАТЫ', fontsize=11, fontweight='bold', pad=8, color='#d35400')
    # ax_ref.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Пределы для первого графика
    margin = 0.5
    # x_lim = max(abs(tau_lex_lower), abs(tau_lex_upper), 3) + margin
    # y_lim = max(abs(tau_sem_lower), abs(tau_sem_upper), 3) + margin
    # ax_ref.set_xlim(-x_lim, x_lim)
    # ax_ref.set_ylim(-y_lim, y_lim)
    ax_ref.set_xlim(-4-margin, 4+margin)
    ax_ref.set_ylim(-4-margin, 4+margin)

    ax_ref.spines['top'].set_visible(False)
    ax_ref.spines['right'].set_visible(False)
    ax_ref.set_xlabel('Лекс. отклон.', fontsize=12)
    ax_ref.set_ylabel('Сем. отклон.', fontsize=12)

    # Статистика для АР
    good_ar_pct = 100 * len(df_ot_ar[df_ot_ar['diagnosis_type'] == 'good']) / len(df_ot_ar)
    good_ar_tau = 100 * len(df_ot_ar[(df_ot_ar['z_lex'] >=  -2) & (df_ot_ar['z_lex'] <=  2) & 
                                    (df_ot_ar['z_sem'] >=  -2) & (df_ot_ar['z_sem'] <=  2) & 
                                    (df_ot_ar['z_comp'] >= -2) & (df_ot_ar['z_comp'] <= 2)]) / len(df_ot_ar)

    # ax_ref.text(0.05, 0.95, f'n={len(df_ot_ar)}\nв зоне ±2σ: {pct_ar_good_tau:.0f}%\nв зоне ±τ: {pct_ar_good_ptc:.0f}%', 
    #             transform=ax_ref.transAxes, fontsize=9, verticalalignment='top',
    #             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    ax_ref.set_title(f'Авторские рефераты\nв зоне ±2σ: {good_ar_tau:.0f}%\nв зоне ±τ :{good_ar_pct:.0f}%', fontsize=12, fontweight='bold', pad=5)

    # =============================================================================
    # ОСТАЛЬНЫЕ SUBPLOTS: Модели суммаризации
    # =============================================================================
    # for idx, model_name in enumerate(sorted(models_list)):
    for idx, model_name in enumerate(models_list):
        ax = axes[idx+1]
        df_model = df_final[df_final['model'] == model_name]
        
        # Отрисовка точек
        for diag_type, color in config.DIAGNOSIS_COLORS.items():
            subset = df_model[df_model['diagnosis_type'] == diag_type]
            if len(subset) > 0:
                ax.scatter(subset['z_lex'], subset['z_sem'], 
                        c=color, alpha=0.7, s=40, edgecolors='white', linewidth=0.25)
        
        # Оси
        ax.axhline(y=0, color='#34495e', linestyle='-', linewidth=1.5, alpha=0.3)
        ax.axvline(x=0, color='#34495e', linestyle='-', linewidth=1.5, alpha=0.3)
        
        # Границы зон
        ax.axhline(y=tau_sem_lower, color='#34495e', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=tau_sem_upper, color='#34495e', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=tau_lex_lower, color='#34495e', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=tau_lex_upper, color='#34495e', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.axhline(y=-2, color='#34495e', linestyle=':', linewidth=1, alpha=0.3)
        ax.axhline(y=2, color='#34495e', linestyle=':', linewidth=1, alpha=0.3)
        ax.axvline(x=-2, color='#34495e', linestyle=':', linewidth=1, alpha=0.3)
        ax.axvline(x=2, color='#34495e', linestyle=':', linewidth=1, alpha=0.5)
        
        # Целевая зона
        good_zone = Rectangle((tau_lex_lower, tau_sem_lower), width_good, height_good, 
                            linewidth=1.5, edgecolor='#2ecc71', 
                            facecolor='#2ecc71', alpha=0.15)
        ax.add_patch(good_zone)

        good_zone = Rectangle((-2, -2), 4, 4, 
                            linewidth=1.5, edgecolor='#2ecc71', 
                            facecolor='#2ecc71', alpha=0.1)
        ax.add_patch(good_zone)
        
        # Название модели + % в целевой зоне
        good_tau = 100 * len(df_model[(df_model['z_lex'] >=  -2) & (df_model['z_lex'] <=  2) & 
                                    (df_model['z_sem'] >=  -2) & (df_model['z_sem'] <=  2) & 
                                    (df_model['z_comp'] >= -2) & (df_model['z_comp'] <= 2)]) / len(df_model)
        good_pct = 100 * len(df_model[df_model['diagnosis_type'] == 'good']) / len(df_model) if len(df_model) > 0 else 0

        ax.set_title(f'{config.INTER_LABELS.get(model_name)}\nв зоне ±2σ: {good_tau:.0f}%\nв зоне ±τ :{good_pct:.0f}%', fontsize=12, fontweight='bold', pad=5)
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Динамические пределы
        margin = 0.5
        # x_min = min(df_model['z_lex'].min(), tau_lex_lower) - margin
        # x_max = max(df_model['z_lex'].max(), tau_lex_upper) + margin
        # y_min = min(df_model['z_sem'].min(), tau_sem_lower) - margin
        # y_max = max(df_model['z_sem'].max(), tau_sem_upper) + margin

        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)

        if idx not in [1, 2]:
            ax.set_xlim(-4.2-margin, 4.2+margin)
            ax.set_ylim(-4.2-margin, 4.2+margin)

        if idx in [1, 2]:
            ax.set_xlim(-4.2-margin, 4.2+margin)
            # ax.set_ylim(-8-margin, 6+margin)
            ax.set_ylim(-4.2-margin, 4.2+margin)
        
        # else:
        #     ax.set_xlim(-1-margin, 13+margin)
        #     ax.set_ylim(-4-margin, 8+margin)
        

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Подписи осей
        # if idx >= 3:
        # if idx % 4 == 3:
        ax.set_xlabel('Лекс. отклон.', fontsize=12)
        ax.set_ylabel('Сем. отклон.', fontsize=12)

        # if idx > 4:
        #     ax.axis("off")

    plt.legend(loc='best')

    # Заголовок с информацией о режиме
    fig.suptitle(f'Диагностика: {config.LEX_NAME}/{config.SEM_NAME}', #  | Пороги: {THRESHOLD_MODE_LABEL}
                fontsize=13, fontweight='bold', y=0.995)

    save_path = f'{config.MODE}/tsm-figures/diagnostic_scatter_for_all_models.png'
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"✓ График сохранён: {save_path}")
    print(f"  Пороги: LEX [{tau_lex_lower:.3f}, {tau_lex_upper:.3f}], SEM [{tau_sem_lower:.3f}, {tau_sem_upper:.3f}], COMP [{tau_comp_lower:.3f}, {tau_comp_upper:.3f}]")
    print(f"  Авторские рефераты в целевой зоне:\n\tв зоне (±2σ): {good_ar_tau:.0f}%\n\tв зоне (±τ): {good_ar_pct:.1f}%")
    # plt.show()



# 8.2 Box plots: Q по моделям
def box_plots(df_final):
    fig, ax = plt.subplots(figsize=(10, 4))

    models_sorted = df_final.groupby('model')['Q'].mean().sort_values(ascending=False).index
    df_final['model_sorted'] = pd.Categorical(df_final['model'], categories=models_sorted, ordered=True)

    # Создаём палитру цветов для моделей в порядке сортировки
    model_palette = [config.MODEL_COLORS.get(m, '#95a5a6') for m in models_sorted]

    bp = ax.boxplot([df_final[df_final['model'] == m]['Q'].dropna().values for m in models_sorted],
                    # labels=[m.replace('summary_', '') for m in models_sorted],
                    labels=[config.MODEL_SHORT.get(m) for m in models_sorted],
                    patch_artist=True, widths=0.6,
                    boxprops=dict(linewidth=1.5, edgecolor='#34495e'),
                    whiskerprops=dict(linewidth=1.5, color='#34495e'),
                    capprops=dict(linewidth=1.5, color='#34495e'),
                    meanline=True, showmeans=True, 
                    medianprops=dict(linewidth=1.5, color="#3532f0ab"),
                    meanprops=dict(linewidth=2, color="#e42b16"),
                    flierprops=dict(marker='o', markerfacecolor='#95a5a6', 
                                markersize=6, markeredgecolor='#34495e', alpha=0.5))

    # Заливка box-ов цветами
    for patch, color in zip(bp['boxes'], model_palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Модель', fontsize=12, fontweight='bold')
    ax.set_ylabel('Метрика Q', fontsize=12, fontweight='bold')
    ax.set_title('Распределение качества Q по моделям', fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(axis='x', rotation=0, labelsize='small')
    ax.grid(axis='y', alpha=0.4, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_path = f'{config.MODE}/tsm-figures/quality_boxplot.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ График 2 сохранён: {save_path}")
    # plt.show()


def boxplot_animation(h_values, h_to_data, model_labels, model_palette, x_func, k_func):
    # Создаём фигуру с двумя осями: слева boxplot, справа — график функций
    fig, (ax, ax_func) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]})

    
    def s_func(x, k):
        return 1 / (1 + np.exp(-k * x))

    def g_func(x, h, k):
        return x / (1 + (h - 1) * s_func(x, k))

    def f_func(x, h, k):
        return np.exp(-g_func(x, h, k) ** 2 / 2)

    def f0_func(x):
        return np.exp(-x ** 2 / 2)

    # Функция инициализации (пустой график)
    def init():
        ax.clear()
        ax.set_xlabel('Модель', fontsize=12, fontweight='bold')
        ax.set_ylabel('Метрика Q', fontsize=12, fontweight='bold')
        ax.set_title('Распределение качества Q по моделям', fontsize=14, fontweight='bold', pad=15)

    # Функция обновления для каждого кадра
    def animate(frame_idx):
        h = h_values[frame_idx]
        data_for_models = h_to_data[h]

        # ── Boxplot ─────────────────────────────────────────────────────────────
        ax.clear()

        bp = ax.boxplot(data_for_models,
                        labels=model_labels,
                        patch_artist=True, widths=0.6,
                        boxprops=dict(linewidth=1.5, edgecolor='#34495e'),
                        whiskerprops=dict(linewidth=1.5, color='#34495e'),
                        capprops=dict(linewidth=1.5, color='#34495e'),
                        meanline=True, showmeans=True,
                        medianprops=dict(linewidth=1.5, color="#3532f0ab"),
                        meanprops=dict(linewidth=2, color="#e42b16"),
                        flierprops=dict(marker='o', markerfacecolor='#95a5a6',
                                        markersize=6, markeredgecolor='#34495e', alpha=0.5))

        for patch, color in zip(bp['boxes'], model_palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel('Модель', fontsize=12, fontweight='bold')
        ax.set_ylabel('Метрика Q', fontsize=12, fontweight='bold')
        ax.set_title(f'Распределение качества Q по моделям (h = {h:.2f})', fontsize=14, fontweight='bold', pad=15)
        ax.tick_params(axis='x', rotation=0, labelsize='small')
        ax.set_ylim(0, 0.73)
        ax.grid(axis='y', alpha=0.4, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ── График функций ───────────────────────────────────────────────────────
        ax_func.clear()

        ax_func.plot(x_func, f0_func(x_func),            '--', lw=2, color='#000000',  label=r'$f_0(x)=e^{-x^2/2}$')
        ax_func.plot(x_func, g_func(x_func, h, k_func),        lw=2, color='#3498db',  label=r'$g(x)=x\,/\,(1+(h{-}1)s(x))$')
        ax_func.plot(x_func, s_func(x_func, k_func),           lw=2, color='#9b59b6',  label=r'$s(x)=1\,/\,(1+e^{-kx})$')
        ax_func.plot(x_func, f_func(x_func, h, k_func),        lw=2, color='#2ecc71',  label=r'$f(x)=e^{-g(x)^2/2}$')

        ax_func.set_xlabel('Отключение (τ)', fontsize=12, fontweight='bold')
        ax_func.set_ylabel(r'$f(x)$', fontsize=12, fontweight='bold')
        ax_func.set_title(f'Функции рапределения  (h = {h:.2f},  k = {k_func:.1f})', fontsize=14, fontweight='bold', pad=15)
        ax_func.legend(fontsize=9, loc='upper left')
        ax_func.set_xlim(-4, 8)
        ax_func.set_ylim(-0.3, 2.5)
        ax_func.axhline(0, color='black', linewidth=0.6, alpha=0.3)
        ax_func.grid(alpha=0.35, linestyle='-', linewidth=0.5)
        ax_func.spines['top'].set_visible(False)
        ax_func.spines['right'].set_visible(False)

        plt.tight_layout()

    # Создание анимации
    anim = animation.FuncAnimation(fig, animate, frames=len(h_values), init_func=init, interval=200, repeat=True)

    # Для отображения в Jupyter Notebook
    HTML(anim.to_jshtml())
    anim.save(f'{config.MODE}/boxplot+gauss_h_animation.gif', writer='pillow', dpi=300, fps=15)

    # Сохранение
    # anim.save('boxplot+gauss_h_animation.gif', writer='pillow', fps=15)
    # anim.save('boxplot_h_animation.mp4', writer='ffmpeg', fps=15)



# 8.4 Гистограммы распределений ОТ-АР (эталонных)
def histogramm(df_raw, calibration):
    ot_ar_data = df_raw[df_raw['comparison'] == 'OT-AR']
    ot_ar_clean = ot_ar_data[~ot_ar_data['doc_id'].isin(calibration['outlier_doc_ids'])]

    has_comp_hist = 'compression_ratio' in ot_ar_clean.columns and ot_ar_clean['compression_ratio'].notna().any()
    n_plots = 3 if has_comp_hist else 2

    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6))

    # LEXICAL
    axes[0].hist(ot_ar_clean['lexical'], bins=20, alpha=0.75,
                color='#3498db', edgecolor='#2c3e50', linewidth=1.2)
    axes[0].axvline(calibration['mu_lex'], color='#e74c3c',
                    linestyle='--', linewidth=2.5,
                    label=f"μ = {calibration['mu_lex']:.3f}")
    axes[0].axvline(calibration['mu_lex'] - 2*calibration['sigma_lex'],
                    color='#f39c12', linestyle=':', linewidth=2, label='μ ± 2σ')
    axes[0].axvline(calibration['mu_lex'] + 2*calibration['sigma_lex'],
                    color='#f39c12', linestyle=':', linewidth=2)
    axes[0].set_xlabel(f'Метрика {config.LEX_NAME} (ОТ-АР)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Частота', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Распределение {config.LEX_NAME} для эталонных АР',
                    fontsize=13, fontweight='bold', pad=10)
    axes[0].legend(framealpha=0.95, edgecolor='gray', fancybox=True)
    axes[0].grid(alpha=0.4, linestyle='-', linewidth=0.5)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # SEMANTIC
    axes[1].hist(ot_ar_clean['semantic'], bins=20, alpha=0.75,
                color='#2ecc71', edgecolor='#2c3e50', linewidth=1.2)
    axes[1].axvline(calibration['mu_sem'], color='#e74c3c',
                    linestyle='--', linewidth=2.5,
                    label=f"μ = {calibration['mu_sem']:.3f}")
    axes[1].axvline(calibration['mu_sem'] - 2*calibration['sigma_sem'],
                    color='#f39c12', linestyle=':', linewidth=2, label='μ ± 2σ')
    axes[1].axvline(calibration['mu_sem'] + 2*calibration['sigma_sem'],
                    color='#f39c12', linestyle=':', linewidth=2)
    axes[1].set_xlabel(f'Метрика {config.SEM_NAME} (ОТ-АР)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Частота', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Распределение {config.SEM_NAME} для эталонных АР',
                    fontsize=13, fontweight='bold', pad=10)
    axes[1].legend(framealpha=0.95, edgecolor='gray', fancybox=True)
    axes[1].grid(alpha=0.4, linestyle='-', linewidth=0.5)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # COMPRESSION RATIO
    if has_comp_hist:
        axes[2].hist(ot_ar_clean['compression_ratio'], bins=20, alpha=0.75,
                    color='#e67e22', edgecolor='#2c3e50', linewidth=1.2)
        axes[2].axvline(calibration['mu_comp'], color='#e74c3c',
                        linestyle='--', linewidth=2.5,
                        label=f"μ = {calibration['mu_comp']:.3f}")
        axes[2].axvline(calibration['mu_comp'] - 2*calibration['sigma_comp'],
                        color='#f39c12', linestyle=':', linewidth=2, label='μ ± 2σ')
        axes[2].axvline(calibration['mu_comp'] + 2*calibration['sigma_comp'],
                        color='#f39c12', linestyle=':', linewidth=2)
        axes[2].set_xlabel('Степень сжатия (ОТ-АР)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Частота', fontsize=12, fontweight='bold')
        axes[2].set_title('Распределение CR для эталонных АР',
                        fontsize=13, fontweight='bold', pad=10)
        axes[2].legend(framealpha=0.95, edgecolor='gray', fancybox=True)
        axes[2].grid(alpha=0.4, linestyle='-', linewidth=0.5)
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)

    save_path = f'{config.MODE}/tsm-figures/reference_distributions.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ График 4 сохранён: {save_path}")
    # plt.show()



# 8.5 Boxplot: Compression Ratio по моделям (с эталоном АР)
def boxplot_compression(df_raw, df_final, calibration):
    fig, ax = plt.subplots(figsize=(12, 5))

    # Данные АР для эталонной линии
    ar_cr = df_raw[df_raw['comparison'] == 'OT-AR']['compression_ratio']
    ar_cr_clean = ar_cr[~df_raw[df_raw['comparison'] == 'OT-AR']['doc_id'].isin(calibration['outlier_doc_ids'])]

    # Эталонная зона АР
    ax.axhline(ar_cr_clean.mean(), color='#e74c3c', linestyle='--', linewidth=2,
            label=f'АР среднее CR = {ar_cr_clean.mean():.3f}')
    ax.axhspan(ar_cr_clean.mean() - ar_cr_clean.std(), ar_cr_clean.mean() + ar_cr_clean.std(),
            alpha=0.15, color='#e74c3c', label=f'АР ±1σ')

    # Boxplots моделей
    models_sorted = df_final.groupby('model')['compression_ratio'].median().sort_values().index
    bp = ax.boxplot(
        [df_final[df_final['model'] == m]['compression_ratio'].dropna().values for m in models_sorted],
        # labels=[m.replace('summary_', '') for m in models_sorted],
        labels=[config.MODEL_SHORT.get(m) for m in models_sorted],
        patch_artist=True, widths=0.6,
        boxprops=dict(linewidth=1.5, edgecolor='#34495e'),
        whiskerprops=dict(linewidth=1.5, color='#34495e'),
        capprops=dict(linewidth=1.5, color='#34495e'),
        medianprops=dict(linewidth=2, color='#c0392b'),
        flierprops=dict(marker='o', markerfacecolor='#95a5a6', markersize=6,
                        markeredgecolor='#34495e', alpha=0.5)
    )

    model_palette = [config.MODEL_COLORS.get(m, '#95a5a6') for m in models_sorted]
    for patch, color in zip(bp['boxes'], model_palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Модель', fontsize=12, fontweight='bold')
    ax.set_ylabel('CR = len_summary / len_source', fontsize=12, fontweight='bold')
    ax.set_title('Степень сжатия по моделям (с эталоном авторских рефератов)', fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(axis='x', rotation=0, labelsize='large')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(axis='y', alpha=0.4, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_path = f'{config.MODE}/tsm-figures/compression_ratio_boxplot.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ График 5 сохранён: {save_path}")
    # plt.show()



# 8.6 Scatter: z_comp vs z_lex и z_comp vs z_sem
def scatter_lex_sem_comp(adaptive_thresholds, df_final):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    tau_comp_upper = adaptive_thresholds.get('tau_comp_upper', 2.0)
    tau_comp_lower = adaptive_thresholds.get('tau_comp_lower', -2.0)
    tau_lex_upper = adaptive_thresholds.get('tau_lex_upper')
    tau_lex_lower = adaptive_thresholds.get('tau_lex_lower')
    tau_sem_upper = adaptive_thresholds.get('tau_sem_upper')
    tau_sem_lower = adaptive_thresholds.get('tau_sem_lower')

    # z_lex vs z_comp
    for diag_type, color in config.DIAGNOSIS_COLORS.items():
        subset = df_final[df_final['diagnosis_type'] == diag_type]
        if len(subset) > 0:
            axes[0].scatter(subset['z_lex'], subset['z_comp'], c=color,
                        label=diag_type.replace('_', ' ').capitalize(), alpha=0.6, s=50,
                        edgecolors='white', linewidth=0.5)

    axes[0].axhline(y=tau_comp_upper, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[0].axhline(y=tau_comp_lower, color='#3498db', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[0].axvline(x=tau_lex_upper, color='#e74c3c', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[0].axvline(x=tau_lex_lower, color='#3498db', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[0].axhline(y=0, color='#7f8c8d', linestyle='-', linewidth=0.8, alpha=0.4)
    axes[0].axvline(x=0, color='#7f8c8d', linestyle='-', linewidth=0.8, alpha=0.4)

    axes[0].set_xlabel(f'Лекс. отклон. ({config.LEX_NAME})', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Отклон. сжатия (Z Compression)', fontsize=12, fontweight='bold')
    axes[0].set_title('Z-LEX vs Z-COMP', fontsize=14, fontweight='bold', pad=10)
    axes[0].legend(loc='best', fontsize=9, framealpha=0.95)
    axes[0].grid(True, alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # z_sem vs z_comp
    for diag_type, color in config.DIAGNOSIS_COLORS.items():
        subset = df_final[df_final['diagnosis_type'] == diag_type]
        if len(subset) > 0:
            axes[1].scatter(subset['z_sem'], subset['z_comp'], c=color,
                        label=diag_type.replace('_', ' ').capitalize(), alpha=0.6, s=50,
                        edgecolors='white', linewidth=0.5)

    axes[1].axhline(y=tau_comp_upper, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1].axhline(y=tau_comp_lower, color='#3498db', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1].axvline(x=tau_sem_upper, color='#e74c3c', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[1].axvline(x=tau_sem_lower, color='#3498db', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[1].axhline(y=0, color='#7f8c8d', linestyle='-', linewidth=0.8, alpha=0.4)
    axes[1].axvline(x=0, color='#7f8c8d', linestyle='-', linewidth=0.8, alpha=0.4)

    axes[1].set_xlabel(f'Сем. отклон. ({config.SEM_NAME})', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Отклон. сжатия (Z Compression)', fontsize=12, fontweight='bold')
    axes[1].set_title('Z-SEM vs Z-COMP', fontsize=14, fontweight='bold', pad=10)
    axes[1].legend(loc='best', fontsize=9, framealpha=0.95)
    axes[1].grid(True, alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # fig.suptitle(f'Анализ степени сжатия | Пороги: {config.THRESHOLD_MODE_LABEL}', fontsize=14, fontweight='bold', y=1.02)
    
    save_path = f'{config.MODE}/tsm-figures/compression_scatter_z.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ График 6 сохранён: {save_path}")
    # plt.show()



# 8.7 Scatter: z_comp vs Q (влияние компрессии на качество)
def scatter_z_q(adaptive_thresholds, df_final):
    fig, ax = plt.subplots(figsize=(10, 7))

    DIAG_TRANSLATE = {
        'good': 'Целевая зона',
        'copying': 'Избыточное копирование',
        'incomplete': 'Семантическая неполнота',
        'low_lexical': 'Лексическая неполнота',
        'ambiguous': 'Неоднозначный паттерн',

        'under_compressed': 'Недостаточное сжатие',
        'over_compressed': 'Избыточное сжатие'
    }

    for diag_type, color in config.DIAGNOSIS_COLORS.items():
        subset = df_final[df_final['diagnosis_type'] == diag_type]
        if len(subset) > 0:
            ax.scatter(subset['z_comp'], subset['Q'], c=color,
                    #   label=diag_type.replace('_', ' ').capitalize(),
                    label=DIAG_TRANSLATE.get(diag_type), #diag_type.capitalize().replace('_', ' '), 
                    alpha=0.6, s=60, edgecolors='white', linewidth=0.5)

    # Пороги компрессии
    tau_comp_upper = adaptive_thresholds.get('tau_comp_upper', 2.0)
    tau_comp_lower = adaptive_thresholds.get('tau_comp_lower', -2.0)
    ax.axvline(x=tau_comp_upper, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7,
            label=f'Верхний порог τ_comp = {tau_comp_upper:.2f}')
    ax.axvline(x=tau_comp_lower, color='#3498db', linestyle='--', linewidth=1.5, alpha=0.7,
            label=f'Нижняя порог τ_comp = {tau_comp_lower:.2f}')
    ax.axvline(x=0, color='#7f8c8d', linestyle='-', linewidth=0.8, alpha=0.4)

    ax.set_xlabel('Отклон. сжатия (Z Compression)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Метрика Q', fontsize=12, fontweight='bold')
    ax.set_title('Влияние степени сжатия на качество рефератов', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10, framealpha=0.95, edgecolor='gray', fancybox=True)
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Аннотации зон

    center_x = (tau_comp_upper+tau_comp_lower)/2

    bbox_style = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9)
    ax.text(tau_comp_lower - 0.95, ax.get_ylim()[1] * 0.9, 'Избыточное\nсжатие',
            ha='center', fontsize=10, fontweight='bold', bbox=bbox_style, color='#005c5c')
    ax.text(tau_comp_upper + 1.25, ax.get_ylim()[1] * 0.9, 'Недостаточное\nсжатие',
            ha='center', fontsize=10, fontweight='bold', bbox=bbox_style, color='#e74c3c')
    ax.text(center_x, ax.get_ylim()[1] * 0.9, 'Эталонное\nсжатие',
            ha='center', fontsize=10, fontweight='bold', bbox=bbox_style, color='#2ecc71')

    ax.set_xlim((-2.6,11.6))

    save_path = f'{config.MODE}/tsm-figures/compression_vs_quality.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ График 7 сохранён: {save_path}")
    # plt.show()