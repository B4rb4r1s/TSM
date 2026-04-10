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
import tsm_visualization as visualization
from tsm_db import prepare_dataframe_from_db

from db import Database



# =============================================================================
# 4. КАЛИБРОВКА: ВЫЧИСЛЕНИЕ ЭТАЛОННЫХ ПАРАМЕТРОВ
# =============================================================================

def calibrate_reference_distribution(df: pd.DataFrame,
                                    tau_outlier: float = 3.0) -> Dict[str, Any]:
    """
    Этап 1: Калибровка эталонных распределений ОТ-АР.

    Процедура:
    1. Извлекаем значения ОТ-АР (близость авторских рефератов к оригиналу)
    2. Вычисляем μ и σ для каждой метрики (lexical, semantic, compression_ratio)
    3. Детектируем аномальные АР по правилу 3σ
    4. Фильтруем аномалии
    5. Пересчитываем параметры на очищенном корпусе
    """
    ot_ar = df[df['comparison'] == 'OT-AR'].copy()

    has_compression = 'compression_ratio' in ot_ar.columns and ot_ar['compression_ratio'].notna().any()

    # Шаг 1: Первичные параметры
    mu_lexical_initial = ot_ar['lexical'].mean()
    sigma_lexical_initial = ot_ar['lexical'].std(ddof=1)
    mu_semantic_initial = ot_ar['semantic'].mean()
    sigma_semantic_initial = ot_ar['semantic'].std(ddof=1)

    print("="*70)
    print("ЭТАП 1: КАЛИБРОВКА ЭТАЛОННЫХ РАСПРЕДЕЛЕНИЙ")
    print("="*70)
    print(f"\n1.1. Первичные параметры (на полном корпусе n={len(ot_ar)}):")
    print(f"     LEXICAL:      μ = {mu_lexical_initial:.4f}, σ = {sigma_lexical_initial:.4f}")
    print(f"     SEMANTIC:     μ = {mu_semantic_initial:.4f}, σ = {sigma_semantic_initial:.4f}")

    if has_compression:
        mu_comp_initial = ot_ar['compression_ratio'].mean()
        sigma_comp_initial = ot_ar['compression_ratio'].std(ddof=1)
        print(f"     COMPRESSION:  μ = {mu_comp_initial:.4f}, σ = {sigma_comp_initial:.4f}")

    # Шаг 2: Детекция аномалий (правило 3σ)
    ot_ar['z_lex'] = (ot_ar['lexical'] - mu_lexical_initial) / sigma_lexical_initial
    ot_ar['z_sem'] = (ot_ar['semantic'] - mu_semantic_initial) / sigma_semantic_initial

    outlier_condition = (np.abs(ot_ar['z_lex']) > tau_outlier) | \
                        (np.abs(ot_ar['z_sem']) > tau_outlier)

    if has_compression:
        ot_ar['z_comp'] = (ot_ar['compression_ratio'] - mu_comp_initial) / sigma_comp_initial
        outlier_condition = outlier_condition | (np.abs(ot_ar['z_comp']) > tau_outlier)

    ot_ar['is_outlier'] = outlier_condition

    outliers = ot_ar[ot_ar['is_outlier']]
    n_outliers = len(outliers)
    outlier_pct = 100 * n_outliers / len(ot_ar)

    print(f"\n1.2. Детекция аномалий (правило {tau_outlier}σ):")
    print(f"     Обнаружено аномальных АР: {n_outliers} ({outlier_pct:.1f}%)")

    if n_outliers > 0:
        print(f"\n     Статистика аномалий:")
        print(f"     - По LEXICAL:     {(np.abs(outliers['z_lex']) > tau_outlier).sum()} документов")
        print(f"     - По SEMANTIC:    {(np.abs(outliers['z_sem']) > tau_outlier).sum()} документов")
        if has_compression:
            print(f"     - По COMPRESSION: {(np.abs(outliers['z_comp']) > tau_outlier).sum()} документов")
        print(f"\n     Топ-5 аномалий по величине отклонения:")
        z_cols = ['z_lex', 'z_sem'] + (['z_comp'] if has_compression else [])
        outliers['max_z'] = outliers[z_cols].abs().max(axis=1)
        show_cols = ['doc_id', 'lexical', 'semantic'] + (['compression_ratio'] if has_compression else []) + z_cols
        top_outliers = outliers.nlargest(5, 'max_z')[show_cols]
        print(top_outliers.to_string(index=False))

    # Шаг 3: Фильтрация и пересчёт
    ot_ar_clean = ot_ar[~ot_ar['is_outlier']].copy()

    mu_lex_clean = ot_ar_clean['lexical'].mean()
    sigma_lex_clean = ot_ar_clean['lexical'].std(ddof=1)
    mu_sem_clean = ot_ar_clean['semantic'].mean()
    sigma_sem_clean = ot_ar_clean['semantic'].std(ddof=1)

    print(f"\n1.3. Параметры на очищенном корпусе (n={len(ot_ar_clean)}):")
    print(f"     LEXICAL:      μ = {mu_lex_clean:.4f}, σ = {sigma_lex_clean:.4f}")
    print(f"     SEMANTIC:     μ = {mu_sem_clean:.4f}, σ = {sigma_sem_clean:.4f}")

    result = {
        'mu_lex': mu_lex_clean,
        'sigma_lex': sigma_lex_clean,
        'mu_sem': mu_sem_clean,
        'sigma_sem': sigma_sem_clean,
        'outlier_doc_ids': outliers['doc_id'].tolist(),
        'n_clean': len(ot_ar_clean),
    }

    if has_compression:
        mu_comp_clean = ot_ar_clean['compression_ratio'].mean()
        sigma_comp_clean = ot_ar_clean['compression_ratio'].std(ddof=1)
        result['mu_comp'] = mu_comp_clean
        result['sigma_comp'] = sigma_comp_clean
        print(f"     COMPRESSION:  μ = {mu_comp_clean:.4f}, σ = {sigma_comp_clean:.4f}")

    # Тест на нормальность
    _, p_rouge = shapiro(ot_ar_clean['lexical'])
    _, p_bert = shapiro(ot_ar_clean['semantic'])

    print(f"\n1.4. Проверка нормальности (тест Шапиро-Уилка):")
    print(f"     LEXICAL:      p-value = {p_rouge:.4f} {'✓' if p_rouge > 0.05 else '✗ (не нормально)'}")
    print(f"     SEMANTIC:     p-value = {p_bert:.4f} {'✓' if p_bert > 0.05 else '✗ (не нормально)'}")

    normality = {'rouge_pval': p_rouge, 'bert_pval': p_bert}

    if has_compression:
        _, p_comp = shapiro(ot_ar_clean['compression_ratio'])
        normality['comp_pval'] = p_comp
        print(f"     COMPRESSION:  p-value = {p_comp:.4f} {'✓' if p_comp > 0.05 else '✗ (не нормально)'}")

    result['normality'] = normality
    return result



# =============================================================================
# 5. ВЫЧИСЛЕНИЕ Z-SCORES ДЛЯ ВСЕХ МОДЕЛЕЙ
# =============================================================================

def compute_z_scores(df: pd.DataFrame, calibration: Dict) -> pd.DataFrame:
    """
    Этап 2: Вычисление нормализованных отклонений (z-scores).

    Для каждого машинного реферата:
    z_lex  = (s_OS_lex  - μ_OA_lex)  / σ_OA_lex
    z_sem  = (s_OS_sem  - μ_OA_sem)  / σ_OA_sem
    z_comp = (CR_SR     - μ_OA_comp) / σ_OA_comp
    """
    # df_models = df[(df['comparison'] == 'OT-SR') & (df['model'] != 'reference')].copy()
    # df_models = df[(df['comparison'] == 'OT-SR') | (df['model'] == 'reference')].copy()
    df_models = df

    # Z-scores лексики и семантики
    df_models['z_lex'] = (df_models['lexical'] - calibration['mu_lex']) / calibration['sigma_lex']
    df_models['z_sem'] = (df_models['semantic'] - calibration['mu_sem']) / calibration['sigma_sem']

    has_compression = 'mu_comp' in calibration

    if has_compression:
        df_models['z_comp'] = (df_models['compression_ratio'] - calibration['mu_comp']) / calibration['sigma_comp']

    print("\n" + "="*70)
    print("ЭТАП 2: ВЫЧИСЛЕНИЕ Z-SCORES ДЛЯ МАШИННЫХ РЕФЕРАТОВ")
    print("="*70)
    print(f"\nОбработано {len(df_models)} машинных рефератов")
    print(f"Модели: {df_models['model'].unique().tolist()}")

    z_cols = ['z_lex', 'z_sem'] + (['z_comp'] if has_compression else [])
    print(f"\nСредние z-scores по моделям:")
    z_stats = df_models.groupby('model')[z_cols].mean()
    print(z_stats.round(3))

    if has_compression:
        print(f"\nИнтерпретация z_comp:")
        print(f"  z_comp = 0  → сжатие как у авторов (CR ≈ {calibration['mu_comp']:.3f})")
        print(f"  z_comp < 0  → сильнее сжатие (короче авторского)")
        print(f"  z_comp > 0  → слабее сжатие (длиннее авторского)")

    return df_models



# =============================================================================
# 6. АДАПТИВНЫЕ ПОРОГИ И ДИАГНОСТИЧЕСКАЯ КЛАССИФИКАЦИЯ
# =============================================================================

def calibrate_thresholds_centered(tau: float = 2.0) -> Dict[str, float]:
    """Режим 1: Центрированные симметричные пороги ±τσ."""
    thresholds = {
        'tau_lex_upper': +tau,
        'tau_lex_lower': -tau,
        'tau_sem_upper': +tau,
        'tau_sem_lower': -tau,
        'tau_comp_upper': +tau,
        'tau_comp_lower': -tau,
        'correlation': None,
        'mode': 'centered',
        'tau_value': tau
    }

    print(f"\nРежим: ЦЕНТРИРОВАННЫЕ ПОРОГИ (±{tau}σ)")
    print(f"  LEXICAL:      [{-tau:.2f}, +{tau:.2f}]")
    print(f"  SEMANTIC:     [{-tau:.2f}, +{tau:.2f}]")
    print(f"  COMPRESSION:  [{-tau:.2f}, +{tau:.2f}]")
    print(f"  → Ожидается ~{100 - 2*(100-stats.norm.cdf(tau)*100):.1f}% данных в зоне")

    return thresholds


def calibrate_thresholds_reference(df_raw: pd.DataFrame,
                                   calibration: Dict,
                                   percentile_low: float = 10,
                                   percentile_high: float = 90) -> Dict[str, float]:
    """Режим 2: Пороги на основе процентилей АВТОРСКИХ рефератов (ОТ-АР)."""
    df_ot_ar = df_raw[df_raw['comparison'] == 'OT-AR'].copy()
    df_ot_ar = df_ot_ar[~df_ot_ar['doc_id'].isin(calibration['outlier_doc_ids'])]

    df_ot_ar['z_lex'] = (df_ot_ar['lexical'] - calibration['mu_lex']) / calibration['sigma_lex']
    df_ot_ar['z_sem'] = (df_ot_ar['semantic'] - calibration['mu_sem']) / calibration['sigma_sem']

    tau_lex_upper = np.percentile(df_ot_ar['z_lex'], percentile_high)
    tau_lex_lower = np.percentile(df_ot_ar['z_lex'], percentile_low)
    tau_sem_upper = np.percentile(df_ot_ar['z_sem'], percentile_high)
    tau_sem_lower = np.percentile(df_ot_ar['z_sem'], percentile_low)

    correlation = df_ot_ar[['z_lex', 'z_sem']].corr().iloc[0, 1]

    thresholds = {
        'tau_lex_upper': tau_lex_upper,
        'tau_lex_lower': tau_lex_lower,
        'tau_sem_upper': tau_sem_upper,
        'tau_sem_lower': tau_sem_lower,
        'correlation': correlation,
        'mode': 'reference',
        'percentile_low': percentile_low,
        'percentile_high': percentile_high
    }

    has_compression = 'mu_comp' in calibration
    if has_compression:
        df_ot_ar['z_comp'] = (df_ot_ar['compression_ratio'] - calibration['mu_comp']) / calibration['sigma_comp']
        thresholds['tau_comp_upper'] = np.percentile(df_ot_ar['z_comp'], percentile_high)
        thresholds['tau_comp_lower'] = np.percentile(df_ot_ar['z_comp'], percentile_low)

    print(f"\nРежим: ПОРОГИ НА ОСНОВЕ АВТОРСКИХ РЕФЕРАТОВ (ОТ-АР)")
    print(f"  Процентили: {percentile_low}% / {percentile_high}%")
    print(f"  Количество АР (без выбросов): {len(df_ot_ar)}")
    print(f"  LEXICAL:      [{tau_lex_lower:.3f}, {tau_lex_upper:.3f}]")
    print(f"  SEMANTIC:     [{tau_sem_lower:.3f}, {tau_sem_upper:.3f}]")
    if has_compression:
        print(f"  COMPRESSION:  [{thresholds['tau_comp_lower']:.3f}, {thresholds['tau_comp_upper']:.3f}]")
    print(f"  → По определению ~{percentile_high - percentile_low}% авторских рефератов в зоне")

    return thresholds


def calibrate_thresholds_adaptive(df_z: pd.DataFrame,
                                  percentile_low: float = 10,
                                  percentile_high: float = 90) -> Dict[str, float]:
    """Режим 3: Пороги на основе процентилей МАШИННЫХ рефератов."""
    tau_lex_upper = np.percentile(df_z['z_lex'], percentile_high)
    tau_lex_lower = np.percentile(df_z['z_lex'], percentile_low)
    tau_sem_upper = np.percentile(df_z['z_sem'], percentile_high)
    tau_sem_lower = np.percentile(df_z['z_sem'], percentile_low)

    correlation = df_z[['z_lex', 'z_sem']].corr().iloc[0, 1]

    thresholds = {
        'tau_lex_upper': tau_lex_upper,
        'tau_lex_lower': tau_lex_lower,
        'tau_sem_upper': tau_sem_upper,
        'tau_sem_lower': tau_sem_lower,
        'correlation': correlation,
        'mode': 'adaptive',
        'percentile_low': percentile_low,
        'percentile_high': percentile_high
    }

    has_compression = 'mu_comp' in calibration
    if has_compression and 'z_comp' in df_z.columns:
        thresholds['tau_comp_upper'] = np.percentile(df_z['z_comp'], percentile_high)
        thresholds['tau_comp_lower'] = np.percentile(df_z['z_comp'], percentile_low)

    print(f"\nРежим: АДАПТИВНЫЕ ПОРОГИ (на машинных рефератах)")
    print(f"  ⚠️  ВНИМАНИЕ: Пороги смещены относительно авторских рефератов!")
    print(f"  Процентили: {percentile_low}% / {percentile_high}%")
    print(f"  LEXICAL:      [{tau_lex_lower:.3f}, {tau_lex_upper:.3f}]")
    print(f"  SEMANTIC:     [{tau_sem_lower:.3f}, {tau_sem_upper:.3f}]")
    if 'tau_comp_upper' in thresholds:
        print(f"  COMPRESSION:  [{thresholds['tau_comp_lower']:.3f}, {thresholds['tau_comp_upper']:.3f}]")
    print(f"  Корреляция z_lex ↔ z_sem: ρ = {correlation:.3f}")

    return thresholds


def classify_summary(z_lex: float, z_sem: float, z_comp: float, thresholds: Dict[str, float]) -> Dict[str, Any]:
    """
    Классификация реферата по z-scores (трёхмерная).

    Приоритет:
    1. copying — избыток лексики
    2. under_compressed — реферат слишком длинный
    3. incomplete — провал семантики
    4. over_compressed — реферат слишком короткий
    5. low_lexical — низкая лексическая близость
    6. good — все три оси в пределах нормы
    7. ambiguous — неоднозначный паттерн
    """
    tau_L_upper = thresholds['tau_lex_upper']
    tau_L_lower = thresholds['tau_lex_lower']
    tau_S_upper = thresholds['tau_sem_upper']
    tau_S_lower = thresholds['tau_sem_lower']
    tau_C_upper = thresholds.get('tau_comp_upper', float('inf'))
    tau_C_lower = thresholds.get('tau_comp_lower', float('-inf'))

    # 2. Недостаточное сжатие: реферат слишком длинный
    if z_comp > tau_C_upper:
        conf = min((z_comp - tau_C_upper) / max(abs(tau_C_upper), 1.0), 1.0)
        return {'type': 'under_compressed', 'confidence': conf, 'description': 'Недостаточное сжатие (реферат слишком длинный)'}

    # 4. Избыточное сжатие: реферат слишком короткий
    elif z_comp < tau_C_lower:
        conf = min((tau_C_lower - z_comp) / max(abs(tau_C_lower), 1.0), 1.0)
        return {'type': 'over_compressed', 'confidence': conf, 'description': 'Избыточное сжатие (реферат слишком короткий)'}

    # 1. Копирование: избыток лексики
    elif z_lex > tau_L_upper and z_sem > tau_S_lower:
        conf = min((z_lex - tau_L_upper) / max(abs(tau_L_upper), 1.0), 1.0)
        return {'type': 'copying', 'confidence': conf, 'description': 'Избыточное копирование из оригинала'}

    # 3. Неполнота: провал семантики
    elif z_sem < tau_S_lower:
        conf = min((tau_S_lower - z_sem) / max(abs(tau_S_lower), 1.0), 1.0)
        return {'type': 'incomplete', 'confidence': conf, 'description': 'Семантическая неполнота'}

    # 5. Низкая лексика
    elif z_lex < tau_L_lower and z_sem >= tau_S_lower:
        conf = min((tau_L_lower - z_lex) / max(abs(tau_L_lower), 1.0), 1.0)
        return {'type': 'low_lexical', 'confidence': conf, 'description': 'Низкое лексическое сходство'}

    # 6. Хорошо: внутри всех трёх зон
    elif (tau_L_lower <= z_lex <= tau_L_upper) and \
         (tau_S_lower <= z_sem <= tau_S_upper) and \
         (tau_C_lower <= z_comp <= tau_C_upper):
        dist_lex = min(abs(z_lex - tau_L_upper), abs(z_lex - tau_L_lower))
        dist_sem = min(abs(z_sem - tau_S_upper), abs(z_sem - tau_S_lower))
        dist_comp = min(abs(z_comp - tau_C_upper), abs(z_comp - tau_C_lower))
        width_lex = tau_L_upper - tau_L_lower
        width_sem = tau_S_upper - tau_S_lower
        width_comp = tau_C_upper - tau_C_lower
        conf = min(
            dist_lex / (width_lex/2) if width_lex > 0 else 1.0,
            dist_sem / (width_sem/2) if width_sem > 0 else 1.0,
            dist_comp / (width_comp/2) if width_comp > 0 else 1.0
        )
        return {'type': 'good', 'confidence': conf, 'description': 'В пределах эталонного диапазона'}

    # 7. Неоднозначно
    else:
        conf = min((tau_S_upper - z_sem) / max(abs(tau_S_upper), 1.0), 1.0)
        return {'type': 'ambiguous', 'confidence': 0.0, 'description': 'Неоднозначный паттерн'}


def diagnose_all_summaries(df_z: pd.DataFrame,
                           df_raw: pd.DataFrame,
                           calibration: Dict,
                           mode: str = 'centered',
                           tau: float = 2.0,
                           percentile_low: float = 10,
                           percentile_high: float = 90) -> Tuple[pd.DataFrame, Dict]:
    """Применяет диагностическую классификацию с выбранным режимом порогов."""
    if mode == 'centered':
        thresholds = calibrate_thresholds_centered(tau)
    elif mode == 'reference':
        thresholds = calibrate_thresholds_reference(df_raw, calibration, percentile_low, percentile_high)
    elif mode == 'adaptive':
        thresholds = calibrate_thresholds_adaptive(df_z, percentile_low, percentile_high)
    else:
        raise ValueError(f"Неизвестный режим: {mode}")

    # Если z_comp отсутствует, используем 0 (нет штрафа)
    has_z_comp = 'z_comp' in df_z.columns

    diagnoses = df_z.apply(
        lambda row: classify_summary(
            row['z_lex'],
            row['z_sem'],
            row['z_comp'] if has_z_comp else 0.0,
            thresholds
        ),
        axis=1
    )

    df_z['diagnosis_type'] = diagnoses.apply(lambda x: x['type'])
    df_z['diagnosis_confidence'] = diagnoses.apply(lambda x: x['confidence'])
    df_z['diagnosis_description'] = diagnoses.apply(lambda x: x['description'])

    print(f"\n{'='*70}")
    print(f"ДИАГНОСТИЧЕСКАЯ КЛАССИФИКАЦИЯ (режим: {mode.upper()})")
    print(f"{'='*70}")

    type_counts = df_z['diagnosis_type'].value_counts()
    print(f"\nРаспределение по типам диагнозов:")
    for diag_type, count in type_counts.items():
        pct = 100 * count / len(df_z)
        print(f"  {diag_type:20s}: {count:4d} ({pct:5.1f}%)")

    print(f"\nРаспределение диагнозов по моделям (%):")
    diagnosis_by_model = pd.crosstab(df_z['model'], df_z['diagnosis_type'], normalize='index') * 100
    print(diagnosis_by_model.round(1))

    return df_z, thresholds



# =============================================================================
# 7. КОМПОЗИТНАЯ МЕТРИКА КАЧЕСТВА
# =============================================================================

def compute_quality_score(df_diagnosed: pd.DataFrame,
                          df_raw: pd.DataFrame,
                          alpha: float = 0.45,
                          beta: float = 0.25,
                          gamma: float = 0.15,
                          delta: float = 0.15,
                          h: float = 2,
                          k: float = 2) -> pd.DataFrame:
    """
    Вычисление композитной метрики качества Q.

    Q = α·q_sem + β·q_lex + γ·q_align + δ·q_comp

    где:
    q_sem   = s_AS_sem × exp(-(z_OS_sem)²/2)
    q_lex   = s_AS_lex × exp(-(z_OS_lex)²/2)
    q_comp  = exp(-(z_OS_comp)²/2)
    q_align = exp(-(z_OS_lex - z_OS_sem)²/2)

    Компоненты:
    - q_sem, q_lex: прямое сходство с АР, штрафованное за отклонение от эталона
    - q_align: штраф за дисбаланс между лексической и семантической близостью
    - q_comp: штраф за отклонение степени сжатия от авторского эталона
    """
    df_ar_sr = df_raw[(df_raw['comparison'] == 'AR-SR') | (df_raw['comparison'] == 'OT-AR')].copy()

    df_merged = df_diagnosed.merge(
        df_ar_sr[['doc_id', 'model', 'lexical', 'semantic']],
        on=['doc_id', 'model'],
        suffixes=('_os', '_as')
    )

    # Компоненты качества
    df_merged['q_sem'] = df_merged['semantic_as'] * np.exp(-df_merged['z_sem']**2 / 2)
    df_merged['q_lex'] = df_merged['lexical_as'] * np.exp(-df_merged['z_lex']**2 / 2)
    df_merged['q_align'] = np.exp(-(df_merged['z_lex'] - df_merged['z_sem'])**2 / 2)

    has_z_comp = 'z_comp' in df_merged.columns

    if has_z_comp:
        df_merged['q_comp'] = np.exp(-g(df_merged['z_comp'], h, k)**2 / 2)
        df_merged['Q'] = (alpha * df_merged['q_sem'] +
                          beta * df_merged['q_lex'] +
                          gamma * df_merged['q_align'] +
                          delta * df_merged['q_comp'])
        weight_str = f"α={alpha} (семантика), β={beta} (лексика), γ={gamma} (выравнивание), δ={delta} (компрессия)"
    else:
        df_merged['Q'] = alpha * df_merged['q_sem'] + beta * df_merged['q_lex'] + gamma * df_merged['q_align']
        weight_str = f"α={alpha} (семантика), β={beta} (лексика), γ={gamma} (выравнивание)"

    print("\n" + "="*70)
    print("ЭТАП 4: КОМПОЗИТНАЯ МЕТРИКА КАЧЕСТВА")
    print("="*70)
    print(f"\nВеса: {weight_str}")

    print(f"\nСредние значения Q по моделям:")
    q_by_model = df_merged.groupby('model')['Q'].agg(['mean', 'std', 'min', 'max'])
    print(q_by_model.round(4))

    print(f"\n🏆 Рейтинг моделей по Q:")
    ranking = df_merged.groupby('model')['Q'].mean().sort_values(ascending=False)
    for rank, (model, q_score) in enumerate(ranking.items(), 1):
        print(f"   {rank}. {model:20s}: Q = {q_score:.4f}")

    return df_merged

def g(x, h=2, k=2):
    return x/(1+(h-1)*S(x, k))

def S(x, k=2):
    return 1/(1+np.exp(-k*x))



def prepare_expert_sample(df_final: pd.DataFrame, 
                          n_per_diagnosis: int = 10,
                          random_state: int = 42) -> pd.DataFrame:
    """
    Формирует сбалансированную выборку для экспертной оценки.
    """
    samples = []
    
    for diag_type in ['good', 'copying', 'incomplete', 'ambiguous']:
        subset = df_final[df_final['diagnosis_type'] == diag_type]
        if len(subset) >= n_per_diagnosis:
            sample = subset.sample(n=n_per_diagnosis, random_state=random_state)
        else:
            sample = subset
        samples.append(sample)
    
    expert_sample = pd.concat(samples, ignore_index=True)
    
    # Добавляем столбцы для экспертных оценок
    expert_sample['expert1_factuality'] = np.nan
    expert_sample['expert1_coverage'] = np.nan
    expert_sample['expert1_conciseness'] = np.nan
    expert_sample['expert1_coherence'] = np.nan
    
    expert_sample['expert2_factuality'] = np.nan
    expert_sample['expert2_coverage'] = np.nan
    expert_sample['expert2_conciseness'] = np.nan
    expert_sample['expert2_coherence'] = np.nan
    
    expert_sample['expert3_factuality'] = np.nan
    expert_sample['expert3_coverage'] = np.nan
    expert_sample['expert3_conciseness'] = np.nan
    expert_sample['expert3_coherence'] = np.nan
    
    return expert_sample






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

    # Выполнение калибровки
    calibration = calibrate_reference_distribution(df_raw)

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

    # ШАГ 7
    df_final = compute_quality_score(df_diagnosed, df_raw, alpha=0.45, beta=0.25, gamma=0.15, delta=0.15)
    df_final_ref = compute_quality_score(df_diagnosed_ref, df_raw, alpha=0.45, beta=0.25, gamma=0.15, delta=0.15)

    ADAPTIVE_THRESHOLDS = adaptive_thresholds

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
                                        h=h)  # предполагаем, что h передаётся в функцию
        
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

    print("\n" + "="*70)
    print("ЭТАП 6: ПОДГОТОВКА К ЭКСПЕРТНОЙ ВАЛИДАЦИИ")
    print("="*70)

    expert_data = prepare_expert_sample(df_final, n_per_diagnosis=10)

    print(f"\nСформирована выборка для экспертов:")
    print(f"  Всего документов: {len(expert_data)}")
    print(f"  Распределение по типам:")
    print(expert_data['diagnosis_type'].value_counts())

    # Сохраняем для экспертов
    expert_data_export = expert_data[['doc_id', 'model', 'diagnosis_type', 'Q',
                                    'lexical_os', 'semantic_os', 'z_lex', 'z_sem',
                                    'expert1_factuality', 'expert1_coverage', 
                                    'expert1_conciseness', 'expert1_coherence',
                                    'expert2_factuality', 'expert2_coverage',
                                    'expert2_conciseness', 'expert2_coherence',
                                    'expert3_factuality', 'expert3_coverage',
                                    'expert3_conciseness', 'expert3_coherence']]

    expert_data_export.to_csv(f'{config.MODE}/tsm-{config.THRESHOLD_MODE}-expert_validation_template.csv', index=False, encoding='utf-8')
    print(f"\n✓ Файл для экспертной оценки сохранён: expert_validation_template.csv")
    print(f"\nИнструкция для экспертов:")
    print(f"  1. Оцените каждый реферат по 4 критериям (шкала 1-5)")
    print(f"  2. Factuality: точность фактов (1=много ошибок, 5=точен)")
    print(f"  3. Coverage: полнота (1=упущены ключевые моменты, 5=все важное)")
    print(f"  4. Conciseness: краткость (1=многословен, 5=оптимален)")
    print(f"  5. Coherence: связность (1=хаотичен, 5=логичен)")


    print("\n" + "="*70)
    print("ИТОГОВАЯ СВОДКА АНАЛИЗА")
    print("="*70)

    print(f"\n📊 СТАТИСТИКА КОРПУСА:")
    print(f"   Всего документов: {len(df_raw['doc_id'].unique())}")
    print(f"   Моделей суммаризации: {len(df_final['model'].unique())}")
    print(f"   Аномальных АР удалено: {len(calibration['outlier_doc_ids'])}")

    print(f"\n📏 ЭТАЛОННЫЕ ПАРАМЕТРЫ (ОТ-АР, очищенные):")
    print(f"   LEXICAL:      μ = {calibration['mu_lex']:.4f} ± {calibration['sigma_lex']:.4f}")
    print(f"   SEMANTIC:     μ = {calibration['mu_sem']:.4f} ± {calibration['sigma_sem']:.4f}")
    if 'mu_comp' in calibration:
        print(f"   COMPRESSION:  μ = {calibration['mu_comp']:.4f} ± {calibration['sigma_comp']:.4f}")

    print(f"\n🎯 ПОРОГИ (режим: {THRESHOLD_MODE_LABEL}):")
    print(f"   LEXICAL:      [{adaptive_thresholds['tau_lex_lower']:.3f}, {adaptive_thresholds['tau_lex_upper']:.3f}]")
    print(f"   SEMANTIC:     [{adaptive_thresholds['tau_sem_lower']:.3f}, {adaptive_thresholds['tau_sem_upper']:.3f}]")
    if 'tau_comp_lower' in adaptive_thresholds:
        print(f"   COMPRESSION:  [{adaptive_thresholds['tau_comp_lower']:.3f}, {adaptive_thresholds['tau_comp_upper']:.3f}]")
    try:
        print(f"   Корреляция:   ρ = {adaptive_thresholds['correlation']:.3f}")
    except:
        print(f"   Корреляция:   ---")

    print(f"\n🎯 ДИАГНОСТИКА МАШИННЫХ РЕФЕРАТОВ:")
    diagnosis_stats = df_final['diagnosis_type'].value_counts()
    for dtype, count in diagnosis_stats.items():
        pct = 100 * count / len(df_final)
        print(f"   {dtype:20s}: {count:4d} ({pct:5.1f}%)")

    print(f"\n🏆 ТОП-3 МОДЕЛИ ПО МЕТРИКЕ Q:")
    top_models = df_final.groupby('model')['Q'].mean().sort_values(ascending=False)
    for rank, (model, q_score) in enumerate(top_models.head(3).items(), 1):
        print(f"   {rank}. {model:20s}: Q = {q_score:.4f}")

    print(f"\n📐 ФОРМУЛА КАЧЕСТВА:")
    print(f"   Q = 0.45·q_sem + 0.25·q_lex + 0.15·q_align + 0.15·q_comp")
    print(f"   q_comp = exp(-(z_comp)²/2) — штраф за отклонение сжатия от авторского")

    # 11.1 Сравнение с baseline метриками
    print("\n" + "="*70)
    print("БОНУС: СРАВНЕНИЕ С BASELINE МЕТРИКАМИ")
    print("="*70)

    print("\nКорреляция Q с базовыми метриками (AR-SR):")
    print(f"  Q vs LEXICAL (AR-SR):     r = {df_final['Q'].corr(df_final['lexical_as']):.4f}")
    print(f"  Q vs SEMANTIC (AR-SR):    r = {df_final['Q'].corr(df_final['semantic_as']):.4f}")
    if 'compression_ratio' in df_final.columns:
        print(f"  Q vs COMPRESSION RATIO:   r = {df_final['Q'].corr(df_final['compression_ratio']):.4f}")
    if 'q_comp' in df_final.columns:
        print(f"  Q vs Q-COMP:              r = {df_final['Q'].corr(df_final['q_comp']):.4f}")

    print("\nОжидаемые результаты после экспертной валидации:")
    print("  Q vs Expert:       r ~ 0.65-0.75 (целевой показатель)")
    print("  LEXICAL vs Expert: r ~ 0.35 (из литературы)")
    print("  SEMANTIC vs Expert: r ~ 0.42 (из литературы)")

    # 11.2 Экспорт полных результатов
    export_cols = ['doc_id', 'model', 'diagnosis_type',
                'lexical_os', 'semantic_os']

    if 'compression_ratio' in df_final.columns:
        export_cols.append('compression_ratio')

    export_cols += ['lexical_as', 'semantic_as', 'z_lex', 'z_sem']

    if 'z_comp' in df_final.columns:
        export_cols.append('z_comp')

    export_cols += ['q_sem', 'q_lex', 'q_align']

    if 'q_comp' in df_final.columns:
        export_cols.append('q_comp')

    export_cols += ['Q', 'diagnosis_confidence']

    df_final_export = df_final[export_cols]
    save_path = f'{config.MODE}/full_analysis_results.csv'
    df_final_export.to_csv(save_path, index=False, encoding='utf-8')
    print(f"\n✓ Полные результаты сохранены: {save_path} ({len(export_cols)} столбцов)")

    print("\n🎓 Notebook завершён. Готово к использованию в диссертации!")