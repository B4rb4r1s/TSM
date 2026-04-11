import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import tsm_config as config


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

