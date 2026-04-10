import tsm_config as config
import numpy as np
import pandas as pd


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


def expert_validation(df_final):
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