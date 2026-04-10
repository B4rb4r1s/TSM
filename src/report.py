

import tsm_config as config


def report_generation(df_raw, calibration, df_final, adaptive_thresholds, THRESHOLD_MODE_LABEL):
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