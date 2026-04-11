import matplotlib.pyplot as plt

import tsm_config as config


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

