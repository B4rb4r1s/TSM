# =============================================================================
# 6. АДАПТИВНЫЕ ПОРОГИ И ДИАГНОСТИЧЕСКАЯ КЛАССИФИКАЦИЯ
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro
from typing import Dict, Tuple, Any



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
                                  calibration: Dict,
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
        thresholds = calibrate_thresholds_adaptive(df_z, calibration, percentile_low, percentile_high)
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


