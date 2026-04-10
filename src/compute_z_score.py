# =============================================================================
# 5. ВЫЧИСЛЕНИЕ Z-SCORES ДЛЯ ВСЕХ МОДЕЛЕЙ
# =============================================================================

import pandas as pd
from typing import Dict


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