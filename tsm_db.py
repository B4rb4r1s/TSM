"""
tsm_db.py — загрузка метрик ТСМ-анализа из БД.

Формирует pandas.DataFrame в том же формате, что и
`prepare_dataframe_extended` в `tsm.py`, но читает данные из БД
(таблицы publications / abstracts / similarity_metrics), а не из JSON.

Использование:
    from tsm_db import prepare_dataframe_from_db
    from db import Database

    db = Database()
    df = prepare_dataframe_from_db(db, source='480',
                                   lex_mode='rougeL', sem_mode='bertscore',
                                   rouge_measure='p',
                                   bertscore_measure='p',
                                   compression_unit='chars')

Результирующие столбцы:
    doc_id, model, comparison ∈ {OT-AR, OT-SR, AR-SR},
    lexical, semantic, compression_ratio, publication_id

Стратегия отбора документов — strict inner join:
    остаются только публикации, у которых есть авторский реферат и
    машинные рефераты ВСЕХ выбранных моделей, причём для всех пар
    посчитаны все три нужные метрики (lexical, semantic, compression).

Список моделей по умолчанию определяется автоматически: берутся все
модели, у которых есть полный набор метрик для выбранного source.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Подключаем модуль БД из соседнего проекта DATA.
_DATA_DIR = Path(__file__).resolve().parent.parent / "DATA"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))
from db import Database  # noqa: E402


# -----------------------------------------------------------------------------
# Сопоставление lex_mode / sem_mode → metric_name в БД
# -----------------------------------------------------------------------------

_ROUGE_MEASURE_SUFFIX = {'f': 'F1', 'p': 'P', 'r': 'R'}
_BERTSCORE_MEASURE_SUFFIX = {'f': 'F1', 'p': 'P', 'r': 'R'}

# Фабрики имён метрик (некоторые зависят от measure: 'p'|'r'|'f')
_LEXICAL_DB_NAME_FACTORY = {
    'rouge1': lambda m: f"ROUGE-1-{_ROUGE_MEASURE_SUFFIX[m]}",
    'rouge2': lambda m: f"ROUGE-2-{_ROUGE_MEASURE_SUFFIX[m]}",
    'rougeL': lambda m: f"ROUGE-L-{_ROUGE_MEASURE_SUFFIX[m]}",
    'bleu':   lambda m: "BLEU",
    'chrf':   lambda m: "chrF++",
    'meteor': lambda m: "METEOR",
    'ter':    lambda m: "TER",
}

_SEMANTIC_DB_NAME = {
    'bertscore': lambda m: f"BERTScore-{_BERTSCORE_MEASURE_SUFFIX[m]}",
}

_COMPRESSION_DB_NAME = {
    'chars': "compression_ratio_chars",
    'words': "compression_ratio_words",
}


def _resolve_metric_names(lex_mode: str, sem_mode: str,
                          rouge_measure: str, bertscore_measure:str,
                          compression_unit: str):
    if lex_mode not in _LEXICAL_DB_NAME_FACTORY:
        raise ValueError(
            f"Неизвестный lex_mode='{lex_mode}'. "
            f"Доступно: {list(_LEXICAL_DB_NAME_FACTORY)}"
        )
    if sem_mode not in _SEMANTIC_DB_NAME:
        raise ValueError(
            f"Неизвестный/отсутствующий в БД sem_mode='{sem_mode}'. "
            f"Доступно: {list(_SEMANTIC_DB_NAME)}"
        )
    if compression_unit not in _COMPRESSION_DB_NAME:
        raise ValueError(
            f"Неизвестный compression_unit='{compression_unit}'. "
            f"Доступно: {list(_COMPRESSION_DB_NAME)}"
        )
    if rouge_measure not in _ROUGE_MEASURE_SUFFIX:
        raise ValueError(
            f"Неизвестный rouge_measure='{rouge_measure}'. "
            f"Доступно: {list(_ROUGE_MEASURE_SUFFIX)}"
        )
    if bertscore_measure not in _BERTSCORE_MEASURE_SUFFIX:
        raise ValueError(
            f"Неизвестный bertscore_measure='{bertscore_measure}'. "
            f"Доступно: {list(_BERTSCORE_MEASURE_SUFFIX)}"
        )

    return (
        _LEXICAL_DB_NAME_FACTORY[lex_mode](rouge_measure),
        _SEMANTIC_DB_NAME[sem_mode](bertscore_measure),
        _COMPRESSION_DB_NAME[compression_unit],
    )


# -----------------------------------------------------------------------------
# Основная загрузка
# -----------------------------------------------------------------------------

def prepare_dataframe_from_db(
    db: Database,
    source: str = '480',
    lex_mode: str = 'rougeL',
    sem_mode: str = 'bertscore',
    rouge_measure: str = 'p',
    bertscore_measure: str = 'f',
    compression_unit: str = 'chars',
    models: Optional[List[str]] = None,
    model_label: str = 'name',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Загрузить метрики из БД и сформировать DataFrame в формате tsm.py.

    Parameters
    ----------
    db : Database
        Экземпляр подключения к БД.
    source : str
        Значение поля publications.source ('480', '17k', ...).
    lex_mode : str
        Лексическая метрика: 'rouge1'|'rouge2'|'rougeL'|'bleu'|'chrf'|'meteor'.
    sem_mode : str
        Семантическая метрика: на данный момент поддерживается 'bertscore'.
    rouge_measure : str
        Мера для ROUGE: 'f'|'p'|'r' (игнорируется для не-ROUGE).
    bertscore_measure : str
        Мера для BERTScore: 'f'|'p'|'r' (игнорируется для не-BERTScore).
    compression_unit : str
        'chars' или 'words' — какой compression_ratio использовать.
    models : list[str] | None
        Явный whitelist моделей. Если None — берутся все модели,
        у которых есть полный набор метрик.
    model_label : str
        Как формировать метку модели: 'name' или 'name_version'.
    verbose : bool
        Печатать ли отчёт о загрузке.

    Returns
    -------
    pandas.DataFrame
        Столбцы: doc_id, model, comparison, lexical, semantic,
                 compression_ratio, publication_id
    """
    lex_name, sem_name, comp_name = _resolve_metric_names(
        lex_mode, sem_mode, rouge_measure, bertscore_measure, compression_unit
    )
    metric_names = [lex_name, sem_name, comp_name]

    if verbose:
        print("=" * 70)
        print(f"Загрузка метрик из БД (source='{source}')")
        print("=" * 70)
        print(f"  Лексическая:   {lex_name}")
        print(f"  Семантическая: {sem_name}")
        print(f"  Сжатие:        {comp_name}")

    # ── 1. Один SQL-запрос: все нужные метрики для source ───────────────────
    # ВАЖНО: для comparison_type='author_vs_machine' в similarity_metrics
    #   abstract_id                → авторский реферат
    #   compared_with_abstract_id  → машинный реферат
    # поэтому информацию о модели AR-SR-строки нужно брать из compared_with-JOIN.
    sql = """
        SELECT
            p.id                AS publication_id,
            a.id                AS abstract_id,
            a.abstract_type     AS abstract_type,
            m.name              AS abstract_model_name,
            m.version           AS abstract_model_version,
            sm.comparison_type,
            sm.compared_with_abstract_id,
            a2.abstract_type    AS cmp_abstract_type,
            m2.name             AS cmp_model_name,
            m2.version          AS cmp_model_version,
            sm.metric_name,
            sm.metric_value
        FROM publications p
        JOIN abstracts a           ON a.publication_id = p.id
        JOIN similarity_metrics sm ON sm.abstract_id = a.id
        LEFT JOIN models m         ON m.id = a.model_id
        LEFT JOIN abstracts a2     ON a2.id = sm.compared_with_abstract_id
        LEFT JOIN models m2        ON m2.id = a2.model_id
        WHERE p.source = %s
          AND sm.metric_name = ANY(%s)
    """
    with db.cursor(commit=False) as cur:
        cur.execute(sql, (source, metric_names))
        rows = cur.fetchall()

    if not rows:
        raise ValueError(
            f"В БД не найдено ни одной метрики для source='{source}' "
            f"и имён {metric_names}"
        )

    raw = pd.DataFrame(rows)

    # Метка модели: для строки, где сравнивается машинный реферат.
    # Для text_vs_machine (OT-SR) машинный реферат = abstract_id → используем
    #   abstract_model_name.
    # Для author_vs_machine (AR-SR) машинный реферат = compared_with_abstract_id
    #   → используем cmp_model_name.
    # Для text_vs_author (OT-AR) машинной модели нет → None.
    def _fmt(name, version):
        if name is None or (isinstance(name, float) and pd.isna(name)):
            return None
        if model_label == 'name_version' and version:
            return f"{name}:{version}"
        return name

    def _make_label(r):
        ct = r['comparison_type']
        if ct == 'text_vs_machine':
            return _fmt(r['abstract_model_name'], r['abstract_model_version'])
        if ct == 'author_vs_machine':
            return _fmt(r['cmp_model_name'], r['cmp_model_version'])
        return None  # text_vs_author

    raw['model_label'] = raw.apply(_make_label, axis=1)

    # Канонический ключ метрики (lexical / semantic / compression_ratio)
    name_to_key = {
        lex_name: 'lexical',
        sem_name: 'semantic',
        comp_name: 'compression_ratio',
    }
    raw['metric_key'] = raw['metric_name'].map(name_to_key)

    needed_triple = ['lexical', 'semantic', 'compression_ratio']
    needed_pair = ['lexical', 'semantic']

    # ── 2. OT-AR: авторские рефераты, text_vs_author ────────────────────────
    ot_ar = raw[(raw['comparison_type'] == 'text_vs_author') &
                (raw['abstract_type'] == 'author')]
    ot_ar_pvt = ot_ar.pivot_table(
        index='publication_id',
        columns='metric_key',
        values='metric_value',
        aggfunc='first',
    )
    missing = [c for c in needed_triple if c not in ot_ar_pvt.columns]
    if missing:
        raise ValueError(
            f"Для OT-AR в БД отсутствуют метрики {missing}. "
            f"Проверьте, что имена метрик в similarity_metrics совпадают с ожидаемыми."
        )
    ot_ar_pvt = ot_ar_pvt.dropna(subset=needed_triple)

    # ── 3. OT-SR: машинные рефераты, text_vs_machine ────────────────────────
    ot_sr = raw[(raw['comparison_type'] == 'text_vs_machine') &
                (raw['abstract_type'] == 'machine')]
    ot_sr_pvt = ot_sr.pivot_table(
        index=['publication_id', 'model_label'],
        columns='metric_key',
        values='metric_value',
        aggfunc='first',
    )
    missing = [c for c in needed_triple if c not in ot_sr_pvt.columns]
    if missing:
        raise ValueError(f"Для OT-SR в БД отсутствуют метрики {missing}")
    ot_sr_pvt = ot_sr_pvt.dropna(subset=needed_triple)

    # ── 4. AR-SR: сравнение автор↔машина, author_vs_machine ─────────────────
    # В БД abstract_id = авторский реферат, compared_with_abstract_id = машинный.
    # Модель для группировки уже вычислена в model_label (из cmp_model_name).
    # compression_ratio здесь не хранится (это свойство одного реферата
    # относительно исходного текста) — будем брать его из OT-SR для той же пары.
    ar_sr = raw[(raw['comparison_type'] == 'author_vs_machine') &
                (raw['abstract_type'] == 'author') &
                (raw['cmp_abstract_type'] == 'machine') &
                (raw['model_label'].notna())]
    ar_sr_pvt = ar_sr.pivot_table(
        index=['publication_id', 'model_label'],
        columns='metric_key',
        values='metric_value',
        aggfunc='first',
    )
    missing = [c for c in needed_pair if c not in ar_sr_pvt.columns]
    if missing:
        raise ValueError(f"Для AR-SR в БД отсутствуют метрики {missing}")
    ar_sr_pvt = ar_sr_pvt.dropna(subset=needed_pair)

    # ── 5. Определить доступные модели ──────────────────────────────────────
    models_ot = set(ot_sr_pvt.index.get_level_values('model_label').unique())
    models_ar = set(ar_sr_pvt.index.get_level_values('model_label').unique())
    auto_models = sorted(m for m in (models_ot & models_ar) if m is not None)

    if models is not None:
        missing_user = sorted(set(models) - set(auto_models))
        if missing_user and verbose:
            print(f"⚠ Пропущены модели (нет полного набора метрик): {missing_user}")
        selected_models = [m for m in models if m in auto_models]
    else:
        selected_models = auto_models

    if not selected_models:
        raise ValueError(
            f"Для source='{source}' не найдено ни одной модели "
            f"с полным набором метрик ({metric_names})"
        )

    # ── 6. Strict inner join по публикациям ─────────────────────────────────
    pub_sets = [set(ot_ar_pvt.index)]
    for mdl in selected_models:
        pubs_ot = {p for (p, m) in ot_sr_pvt.index if m == mdl}
        pubs_ar = {p for (p, m) in ar_sr_pvt.index if m == mdl}
        pub_sets.append(pubs_ot)
        pub_sets.append(pubs_ar)

    common_pubs = sorted(set.intersection(*pub_sets))
    if not common_pubs:
        raise ValueError(
            "Нет публикаций с полным набором метрик по всем выбранным моделям. "
            "Попробуйте сузить список моделей параметром `models`."
        )

    if verbose:
        print(f"  Моделей:       {len(selected_models)}")
        for m in selected_models:
            print(f"    • {m}")
        print(f"  Публикаций:    {len(common_pubs)} "
              f"(strict inner join по всем моделям)")

    pub_to_doc = {p: i for i, p in enumerate(common_pubs)}

    # ── 7. Собрать итоговый DataFrame ───────────────────────────────────────
    rows_out = []

    # OT-AR
    for pub in common_pubs:
        r = ot_ar_pvt.loc[pub]
        rows_out.append({
            'doc_id': pub_to_doc[pub],
            'model': 'reference',
            'comparison': 'OT-AR',
            'lexical': float(r['lexical']),
            'semantic': float(r['semantic']),
            'compression_ratio': float(r['compression_ratio']),
            'publication_id': pub,
        })

    # OT-SR
    for mdl in selected_models:
        for pub in common_pubs:
            r = ot_sr_pvt.loc[(pub, mdl)]
            rows_out.append({
                'doc_id': pub_to_doc[pub],
                'model': mdl,
                'comparison': 'OT-SR',
                'lexical': float(r['lexical']),
                'semantic': float(r['semantic']),
                'compression_ratio': float(r['compression_ratio']),
                'publication_id': pub,
            })

    # AR-SR (compression_ratio переиспользуем из OT-SR — это свойство машинного реферата)
    for mdl in selected_models:
        for pub in common_pubs:
            r = ar_sr_pvt.loc[(pub, mdl)]
            cr = ot_sr_pvt.loc[(pub, mdl), 'compression_ratio']
            rows_out.append({
                'doc_id': pub_to_doc[pub],
                'model': mdl,
                'comparison': 'AR-SR',
                'lexical': float(r['lexical']),
                'semantic': float(r['semantic']),
                'compression_ratio': float(cr),
                'publication_id': pub,
            })

    df = pd.DataFrame(rows_out)

    if verbose:
        print(f"✓ DataFrame сформирован: {len(df)} строк")
        print(df.groupby('comparison').size().to_string())
        print("=" * 70)

    return df


def get_available_models_from_db(
    db: Database,
    source: str = '480',
    lex_mode: str = 'rougeL',
    sem_mode: str = 'bertscore',
    rouge_measure: str = 'p',
    bertscore_measure: str = 'f',
    compression_unit: str = 'chars',
) -> List[str]:
    """
    Вернуть список моделей, у которых есть полный набор метрик
    (lexical + semantic + compression) для данного source.
    """
    df = prepare_dataframe_from_db(
        db,
        source=source,
        lex_mode=lex_mode,
        sem_mode=sem_mode,
        rouge_measure=rouge_measure,
        bertscore_measure=bertscore_measure,
        compression_unit=compression_unit,
        verbose=False,
    )
    return sorted(df[df['comparison'] == 'OT-SR']['model'].unique().tolist())


# -----------------------------------------------------------------------------
# Вспомогательные функции для app.py
# -----------------------------------------------------------------------------

def get_model_names_from_db_fast(db: Database, source: str = '480') -> List[str]:
    """Быстрый запрос: имена моделей с машинными рефератами для source."""
    sql = """
        SELECT DISTINCT m.name
        FROM abstracts a
        JOIN publications p ON p.id = a.publication_id
        JOIN models m ON m.id = a.model_id
        WHERE a.abstract_type = 'machine'
          AND p.source = %s
        ORDER BY m.name
    """
    with db.cursor(commit=False) as cur:
        cur.execute(sql, (source,))
        return [row['name'] for row in cur.fetchall()]


def get_available_metric_modes(db: Database, source: str = '480'):
    """Определить доступные lex_mode и sem_mode из метрик в БД.

    Returns
    -------
    (lex_modes, sem_modes) : tuple[list[str], list[str]]
    """
    sql = """
        SELECT DISTINCT sm.metric_name
        FROM similarity_metrics sm
        JOIN publications p ON p.id = sm.publication_id
        WHERE p.source = %s
    """
    with db.cursor(commit=False) as cur:
        cur.execute(sql, (source,))
        db_names = {row['metric_name'] for row in cur.fetchall()}

    lex_modes = []
    for mode, factory in _LEXICAL_DB_NAME_FACTORY.items():
        for m in ('f', 'p', 'r'):
            if factory(m) in db_names:
                lex_modes.append(mode)
                break

    sem_modes = []
    for mode, factory in _SEMANTIC_DB_NAME.items():
        for m in ('f', 'p', 'r'):
            if factory(m) in db_names:
                sem_modes.append(mode)
                break

    return lex_modes, sem_modes


def load_texts_from_db(db: Database, publication_id: int,
                       model_name: Optional[str] = None) -> dict:
    """Загрузить тексты публикации из БД (аналог eng.load_texts для SQLite).

    Returns
    -------
    dict с ключами: source_text, target_summary, model_summary
    """
    pub = db.get_publication(publication_id)
    if not pub:
        return {}

    result = {
        'source_text': pub.get('clean_text') or pub.get('full_text') or '',
    }

    # Авторский реферат
    author_abs = db.get_abstracts(publication_id, 'author')
    if author_abs:
        result['target_summary'] = author_abs[0].get('text') or ''

    # Машинный реферат конкретной модели
    if model_name:
        machine_abs = db.get_abstracts(publication_id, 'machine')
        for a in machine_abs:
            if a.get('model_name') == model_name:
                result['model_summary'] = a.get('text') or ''
                break

    return result


# -----------------------------------------------------------------------------
# Демо-запуск
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    db = Database()
    try:
        df = prepare_dataframe_from_db(
            db,
            source='480',
            lex_mode='rougeL',
            sem_mode='bertscore',
            rouge_measure='p',
            bertscore_measure='f',
            compression_unit='chars',
        )
        print("\nПример данных:")
        print(df.head(10).to_string(index=False))
        print(f"\nshape={df.shape}")
        print("\nЗаписей по моделям (OT-SR):")
        print(df[df['comparison'] == 'OT-SR'].groupby('model').size().to_string())
    finally:
        db.close()
