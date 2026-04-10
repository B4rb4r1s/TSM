# =============================================================================
# 1. ИМПОРТ БИБЛИОТЕК
# =============================================================================

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, shapiro
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Настройка визуализации
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Для корректного отображения кириллицы
plt.rcParams['font.family'] = 'DejaVu Sans'

print("✓ Все библиотеки успешно загружены")

import tsm_config as config
from tsm_db import prepare_dataframe_from_db

import src.visual.tsm_visualization as visualization
from src.calibrate_reference_distribution import calibrate_reference_distribution
from src.compute_z_score import compute_z_scores
from src.calibrate_thresholds import diagnose_all_summaries
from src.composite_metric import compute_quality_score
from src.expert_validation import expert_validation
from src.report import report_generation

from db import Database



if __name__ == "__main__":
    # Подключение к базе данных, формирование Dataframe
    db = Database()
    df_raw = prepare_dataframe_from_db(
        db,
        source=config.SOURCE,                       # '480' или '17k'
        lex_mode=config.LEX_MODE,
        sem_mode=config.SEM_MODE,                   # пока только 'bertscore'
        rouge_measure=config.ROUGE_MEASURE,
        bertscore_measure=config.BERTSCORE_MEASURE,
        compression_unit=config.COMPRESSION_MODE,
    )


    # =============================================================================
    # 4. КАЛИБРОВКА: ВЫЧИСЛЕНИЕ ЭТАЛОННЫХ ПАРАМЕТРОВ
    # =============================================================================
    # Выполнение калибровки
    calibration = calibrate_reference_distribution(df_raw)


    # =============================================================================
    # 5. ВЫЧИСЛЕНИЕ Z-SCORES ДЛЯ ВСЕХ МОДЕЛЕЙ
    # =============================================================================
    # -----------------------------------------------
    #          ВЫПОЛНЕНИЕ КАЛИБРОВКИ ПОРОГОВ         
    # -----------------------------------------------
    df_temp = df_raw[(df_raw['comparison'] == 'OT-SR') & (df_raw['model'] != 'reference')].copy()
    df_z = compute_z_scores(df_temp, calibration)

    df_temp = df_raw[(df_raw['model'] == 'reference')].copy()
    df_z_ref = compute_z_scores(df_temp, calibration)


    print(f"\n{'='*70}")
    print(f"ВЫБРАННЫЙ РЕЖИМ ПОРОГОВ: centered")
    print(f"{'='*70}")

    df_diagnosed, adaptive_thresholds = diagnose_all_summaries(
        df_z,
        df_raw,
        calibration,
        mode='centered',
        tau=config.CENTERED_TAU,
        percentile_low=config.PERCENTILE_LOW,
        percentile_high=config.PERCENTILE_HIGH
    )

    print(f"\n{'='*70}")
    print(f"ВЫБРАННЫЙ РЕЖИМ ПОРОГОВ: {config.THRESHOLD_MODE}")
    print(f"{'='*70}")

    df_diagnosed, adaptive_thresholds = diagnose_all_summaries(
        df_z,
        df_raw,
        calibration,
        mode=config.THRESHOLD_MODE,
        tau=config.CENTERED_TAU,
        percentile_low=config.PERCENTILE_LOW,
        percentile_high=config.PERCENTILE_HIGH
    )
    df_diagnosed_ref, adaptive_thresholds_ref = diagnose_all_summaries(
        df_z_ref,
        df_raw,
        calibration,
        mode=config.THRESHOLD_MODE,
        tau=config.CENTERED_TAU,
        percentile_low=config.PERCENTILE_LOW,
        percentile_high=config.PERCENTILE_HIGH
    )

    THRESHOLD_MODE_LABEL = {
        'centered': f'Центрированные (±{config.CENTERED_TAU}σ)',
        'reference': f'На основе АР ({config.PERCENTILE_LOW}-{config.PERCENTILE_HIGH} процентиль)',
        'adaptive': f'Адаптивные на машинных ({config.PERCENTILE_LOW}-{config.PERCENTILE_HIGH} процентиль)'
    }[config.THRESHOLD_MODE]


    # =============================================================================
    # 7. КОМПОЗИТНАЯ МЕТРИКА КАЧЕСТВА
    # =============================================================================
    # ШАГ 7
    df_final = compute_quality_score(df_diagnosed, df_raw, alpha=0.45, beta=0.25, gamma=0.15, delta=0.15)
    df_final_ref = compute_quality_score(df_diagnosed_ref, df_raw, alpha=0.45, beta=0.25, gamma=0.15, delta=0.15)

    ADAPTIVE_THRESHOLDS = adaptive_thresholds


    # =============================================================================
    # 8. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
    # =============================================================================
    # ШАГ 8
    print("\n" + "="*70)
    print("ЭТАП 5: ВИЗУАЛИЗАЦИЯ")
    print("="*70)

    visualization.scatter_plot(adaptive_thresholds, df_final)
    visualization.scatter_plot_3d(adaptive_thresholds, df_final)
    visualization.scatter_plot_each_models(adaptive_thresholds, df_final)
    visualization.scatter_plot_all_models(adaptive_thresholds, df_final, df_final_ref)
    visualization.box_plots(df_final)

    
    # Диапазон h и шаг
    h_values = np.arange(1.0, 10.05, 0.1)  # шаг 0.1

    # Словарь для быстрого доступа к данным по моделям для каждого h
    # Ключ: значение h, значение: список значений Q для каждой модели в фиксированном порядке
    h_to_data = {}
    # Также определим порядок моделей один раз (например, по первому кадру)
    models_order = None

    for h in h_values:
        df_final = compute_quality_score(df_diagnosed, df_raw,
                                        alpha=  config.ALPHA, 
                                        beta=   config.BETA, 
                                        gamma=  config.GAMMA, 
                                        delta=  config.DELTA,
                                        h=h, verbose=False)  # предполагаем, что h передаётся в функцию
        
        # Определяем порядок моделей по первому кадру (сортировка по убыванию среднего Q)
        if models_order is None:
            models_order = df_final.groupby('model')['Q'].mean().sort_values(ascending=False).index.tolist()
        
        # Для каждой модели из фиксированного порядка собираем значения Q (без пропусков)
        data_for_models = [df_final[df_final['model'] == m]['Q'].dropna().values for m in models_order]
        h_to_data[h] = data_for_models

    # Цветовая палитра (фиксированная, по модели)
    model_palette = [config.MODEL_COLORS.get(m, '#95a5a6') for m in models_order]
    # Короткие названия моделей (если есть словарь MODEL_SHORT)
    model_labels = [config.MODEL_SHORT.get(m, m) for m in models_order]
    # ── Параметры и функции для графика искажения ──────────────────────────────
    k_func = 2.0
    x_func = np.linspace(-4, 8, 2000)

    visualization.boxplot_animation(h_values, h_to_data, model_labels, model_palette, x_func, k_func)
    visualization.histogramm(df_raw, calibration)
    visualization.boxplot_compression(df_raw, df_final, calibration)
    visualization.scatter_lex_sem_comp(adaptive_thresholds, df_final)


    # =============================================================================
    # ЭКСПЕРТНАЯ ВАЛИДАЦИЯ
    # =============================================================================
    expert_validation(df_final)

    # =============================================================================
    # ИТОГОВА СВОДКА И ГЕНЕРАЦИЯ ОТЧЕТА
    # =============================================================================
    report_generation(df_raw, calibration, df_final, adaptive_thresholds, THRESHOLD_MODE_LABEL)

