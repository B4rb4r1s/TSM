# =============================================================================
# 7. КОМПОЗИТНАЯ МЕТРИКА КАЧЕСТВА
# =============================================================================

import numpy as np
import pandas as pd



def g_funk(x, h=2, k=2):
    return x/(1+(h-1)*S_funk(x, k))


def S_funk(x, k=2):
    return 1/(1+np.exp(-k*x))


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
        df_merged['q_comp'] = np.exp(-g_funk(df_merged['z_comp'], h, k)**2 / 2)
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