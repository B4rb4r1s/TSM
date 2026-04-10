# =============================================================================
# 4. КАЛИБРОВКА: ВЫЧИСЛЕНИЕ ЭТАЛОННЫХ ПАРАМЕТРОВ
# =============================================================================

import pandas as pd
import numpy as np
from scipy.stats import shapiro
from typing import Dict, Any


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



# if __name__ == "__main__":
    # Выполнение калибровки
    # calibration = calibrate_reference_distribution(df_raw)