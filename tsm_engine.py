"""
tsm_engine.py — Вычислительное ядро трёхфакторной методики оценки рефератов.

Воспроизводит логику из tsm-v3.ipynb в виде чистых функций без UI-зависимостей.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# Константы
# ═══════════════════════════════════════════════════════════════════════

DIAGNOSIS_COLORS = {
    'good': '#2ecc71',
    'copying': '#3498db',
    'incomplete': '#9b59b6',
    'low_lexical': '#f39c12',
    'ambiguous': '#95a5a6',
    'under_compressed': '#e74c3c',
    'over_compressed': '#008f8f',
}

DIAGNOSIS_LABELS_RU = {
    'good': 'Хороший',
    'copying': 'Копирование',
    'incomplete': 'Неполный',
    'low_lexical': 'Низк. лексика',
    'ambiguous': 'Неоднозначный',
    'under_compressed': 'Недост. сжатие',
    'over_compressed': 'Избыт. сжатие',
}

LEXICAL_MODES = {
    'rouge1': ('rouge', 'rouge1'),
    'rouge2': ('rouge', 'rouge2'),
    'rougeL': ('rouge', 'rougeL'),
    'bleu':   ('bleu', None),
    'chrf':   ('chrf', None),
    'meteor': ('meteor', None),
}

SEMANTIC_MODES = {
    'bleurt':    ('bleurt', None),
    'bertscore': ('bertscore', 'f1'),
}

LEXICAL_LABELS = {
    'rouge1': 'ROUGE-1', 'rouge2': 'ROUGE-2', 'rougeL': 'ROUGE-L',
    'bleu': 'BLEU', 'chrf': 'chrF++', 'meteor': 'METEOR',
}

SEMANTIC_LABELS = {
    'bleurt': 'BLEURT', 'bertscore': 'BERTScore (F1)',
}


# ═══════════════════════════════════════════════════════════════════════
# Загрузка данных
# ═══════════════════════════════════════════════════════════════════════

def load_metrics(metrics_dir: str) -> Dict[str, Any]:
    """Загрузить все JSON-метрики из директории."""
    metrics_dir = Path(metrics_dir)
    data = {}

    files = {
        'rouge': 'rouge.json',
        'bleu': 'bleu.json',
        'chrf': 'chrf.json',
        'meteor': 'meteor.json',
        'bertscore': 'bertscore.json',
        'bleurt': 'bleurt.json',
        'embeddings': 'embeddings.json',
        'lengths': 'lengths.json',
    }

    for key, fname in files.items():
        path = metrics_dir / fname
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data[key] = json.load(f)

    return data


def get_available_models(data: Dict) -> List[str]:
    """Получить список моделей, доступных во всех метриках."""
    model_sets = []

    # Из lengths (compression_ratio)
    if 'lengths' in data and 'compression_ratio' in data['lengths']:
        cr = data['lengths']['compression_ratio']
        model_sets.append(set(k for k in cr.keys() if k != 'АР'))

    # Из любой метрики с ОТ-СР
    for key in ('rouge', 'bleu', 'chrf', 'meteor', 'bleurt', 'bertscore'):
        if key in data and 'ОТ-СР' in data[key]:
            sr = data[key]['ОТ-СР']
            if isinstance(sr, dict):
                model_sets.append(set(sr.keys()))

    if not model_sets:
        return []

    # Пересечение — модели, доступные во всех метриках
    common = model_sets[0]
    for s in model_sets[1:]:
        common = common & s

    return sorted(common)


def get_available_lexical_metrics(data: Dict) -> List[str]:
    """Доступные лексические метрики."""
    available = []
    for mode, (source, _) in LEXICAL_MODES.items():
        if source in data:
            available.append(mode)
    return available


def get_available_semantic_metrics(data: Dict) -> List[str]:
    """Доступные семантические метрики."""
    available = []
    for mode, (source, _) in SEMANTIC_MODES.items():
        if source in data:
            available.append(mode)

    # Embedding-модели
    if 'embeddings' in data:
        for emb_model in data['embeddings'].keys():
            available.append(f'emb:{emb_model}')

    return available


def load_texts(db_path: str, doc_id: int, model_col: str) -> Dict[str, str]:
    """Загрузить тексты документа из SQLite."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # doc_id — 0-indexed, rowid — 1-indexed
    # Экранируем имена колонок двойными кавычками (имена LLM-моделей содержат спецсимволы)
    cols = ['"source_text"', '"target_summary"']
    if model_col:
        cols.append(f'"{model_col}"')

    col_str = ', '.join(cols)
    cursor.execute(f'SELECT {col_str} FROM summaries LIMIT 1 OFFSET ?', (doc_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {}

    result = {
        'source_text': row[0] or '',
        'target_summary': row[1] or '',
    }
    if model_col and len(row) > 2:
        result['model_summary'] = row[2] or ''

    return result


# ═══════════════════════════════════════════════════════════════════════
# Подготовка DataFrame
# ═══════════════════════════════════════════════════════════════════════

def _get_lex_value(data: Dict, lex_source: str, lex_key: Optional[str],
                   rouge_measure: str, comparison: str, model: Optional[str],
                   idx: int) -> float:
    """Извлечь лексическое значение из JSON-данных."""
    if model:
        block = data[lex_source][comparison][model]
    else:
        block = data[lex_source][comparison]

    if lex_key is None:
        # bleu, chrf, meteor — плоский список
        if isinstance(block, list):
            return block[idx]
        return block[idx]
    else:
        # rouge — вложенная структура
        if lex_source == 'rouge':
            return block[lex_key][rouge_measure][idx]
        return block[lex_key][idx]


def _get_sem_value(data: Dict, sem_source: str, sem_key: Optional[str],
                   comparison: str, model: Optional[str], idx: int) -> float:
    """Извлечь семантическое значение из JSON-данных."""
    if model:
        block = data[sem_source][comparison][model]
    else:
        block = data[sem_source][comparison]

    if sem_key is None:
        if isinstance(block, list):
            return block[idx]
        return block[idx]
    else:
        return block[sem_key][idx]


def prepare_dataframe(data: Dict, lex_mode: str, sem_mode: str,
                      rouge_measure: str, selected_models: List[str]) -> pd.DataFrame:
    """
    Собрать DataFrame из JSON-данных.

    Структура: [doc_id, model, comparison, lexical, semantic, compression_ratio]
    """
    # Определить источники метрик
    if lex_mode in LEXICAL_MODES:
        lex_source, lex_key = LEXICAL_MODES[lex_mode]
    else:
        raise ValueError(f'Неизвестная лексическая метрика: {lex_mode}')

    if sem_mode in SEMANTIC_MODES:
        sem_source, sem_key = SEMANTIC_MODES[sem_mode]
    elif sem_mode.startswith('emb:'):
        emb_model = sem_mode[4:]
        sem_source = 'embeddings'
        sem_key = emb_model
    else:
        raise ValueError(f'Неизвестная семантическая метрика: {sem_mode}')

    has_compression = 'lengths' in data and 'compression_ratio' in data['lengths']

    # Определить n_docs из ОТ-АР
    if sem_source == 'embeddings':
        n_docs = len(data[sem_source][sem_key]['ОТ-АР'])
    elif sem_key:
        ot_ar = data[sem_source]['ОТ-АР']
        if isinstance(ot_ar, dict) and sem_key in ot_ar:
            n_docs = len(ot_ar[sem_key])
        else:
            n_docs = len(ot_ar)
    else:
        n_docs = len(data[sem_source]['ОТ-АР'])

    rows = []

    # Вспомогательные функции для embedding
    def get_lex(comparison, model, idx):
        if model:
            block = data[lex_source][comparison][model]
        else:
            block = data[lex_source][comparison]

        if lex_key is None:
            return block[idx] if isinstance(block, list) else block[idx]
        if lex_source == 'rouge':
            return block[lex_key][rouge_measure][idx]
        return block[lex_key][idx]

    def get_sem(comparison, model, idx):
        if sem_source == 'embeddings':
            if model:
                return data[sem_source][sem_key][comparison][model][idx]
            else:
                return data[sem_source][sem_key][comparison][idx]
        if model:
            block = data[sem_source][comparison][model]
        else:
            block = data[sem_source][comparison]
        if sem_key is None:
            return block[idx] if isinstance(block, list) else block[idx]
        return block[sem_key][idx]

    # 1. ОТ-АР
    for i in range(n_docs):
        cr = data['lengths']['compression_ratio']['АР'][i] if has_compression else None
        rows.append({
            'doc_id': i,
            'model': 'reference',
            'comparison': 'OT-AR',
            'lexical': get_lex('ОТ-АР', None, i),
            'semantic': get_sem('ОТ-АР', None, i),
            'compression_ratio': cr,
        })

    # 2. ОТ-СР и АР-СР для каждой модели
    for model_name in selected_models:
        for i in range(n_docs):
            cr = data['lengths']['compression_ratio'][model_name][i] if has_compression else None

            # ОТ-СР
            rows.append({
                'doc_id': i,
                'model': model_name,
                'comparison': 'OT-SR',
                'lexical': get_lex('ОТ-СР', model_name, i),
                'semantic': get_sem('ОТ-СР', model_name, i),
                'compression_ratio': cr,
            })

            # АР-СР
            rows.append({
                'doc_id': i,
                'model': model_name,
                'comparison': 'AR-SR',
                'lexical': get_lex('АР-СР', model_name, i),
                'semantic': get_sem('АР-СР', model_name, i),
                'compression_ratio': cr,
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Калибровка
# ═══════════════════════════════════════════════════════════════════════

def calibrate(df: pd.DataFrame, tau_outlier: float = 3.0) -> Dict[str, Any]:
    """
    Калибровка по ОТ-АР: μ, σ, удаление выбросов, пересчёт.
    """
    ot_ar = df[df['comparison'] == 'OT-AR'].copy()
    has_comp = ot_ar['compression_ratio'].notna().any()

    # Первичные параметры
    mu_lex = ot_ar['lexical'].mean()
    sigma_lex = ot_ar['lexical'].std(ddof=1)
    mu_sem = ot_ar['semantic'].mean()
    sigma_sem = ot_ar['semantic'].std(ddof=1)

    # z-scores для выбросов
    z_lex = (ot_ar['lexical'] - mu_lex) / sigma_lex
    z_sem = (ot_ar['semantic'] - mu_sem) / sigma_sem
    outlier_mask = (np.abs(z_lex) > tau_outlier) | (np.abs(z_sem) > tau_outlier)

    if has_comp:
        mu_comp = ot_ar['compression_ratio'].mean()
        sigma_comp = ot_ar['compression_ratio'].std(ddof=1)
        z_comp = (ot_ar['compression_ratio'] - mu_comp) / sigma_comp
        outlier_mask = outlier_mask | (np.abs(z_comp) > tau_outlier)

    outlier_ids = ot_ar.loc[outlier_mask.values, 'doc_id'].tolist()
    clean = ot_ar[~outlier_mask.values]

    result = {
        'mu_lex': clean['lexical'].mean(),
        'sigma_lex': clean['lexical'].std(ddof=1),
        'mu_sem': clean['semantic'].mean(),
        'sigma_sem': clean['semantic'].std(ddof=1),
        'outlier_doc_ids': outlier_ids,
        'n_total': len(ot_ar),
        'n_clean': len(clean),
        'n_outliers': len(outlier_ids),
    }

    if has_comp:
        result['mu_comp'] = clean['compression_ratio'].mean()
        result['sigma_comp'] = clean['compression_ratio'].std(ddof=1)

    return result


# ═══════════════════════════════════════════════════════════════════════
# Z-scores
# ═══════════════════════════════════════════════════════════════════════

def compute_z_scores(df: pd.DataFrame, calibration: Dict) -> pd.DataFrame:
    """Вычислить z-scores для ОТ-СР данных."""
    df = df.copy()
    df['z_lex'] = (df['lexical'] - calibration['mu_lex']) / calibration['sigma_lex']
    df['z_sem'] = (df['semantic'] - calibration['mu_sem']) / calibration['sigma_sem']

    if 'mu_comp' in calibration:
        df['z_comp'] = (df['compression_ratio'] - calibration['mu_comp']) / calibration['sigma_comp']
    else:
        df['z_comp'] = 0.0

    return df


# ═══════════════════════════════════════════════════════════════════════
# Пороги
# ═══════════════════════════════════════════════════════════════════════

def compute_thresholds(mode: str, df_raw: pd.DataFrame, calibration: Dict,
                       percentile_low: float = 10, percentile_high: float = 90,
                       tau: float = 2.0) -> Dict[str, float]:
    """
    Вычислить пороговые значения.

    mode='reference': процентили авторских рефератов
    mode='centered': ±τσ
    """
    if mode == 'centered':
        return {
            'tau_lex_upper': +tau, 'tau_lex_lower': -tau,
            'tau_sem_upper': +tau, 'tau_sem_lower': -tau,
            'tau_comp_upper': +tau, 'tau_comp_lower': -tau,
            'mode': 'centered', 'tau_value': tau,
        }

    # reference mode — процентили по ОТ-АР
    ot_ar = df_raw[df_raw['comparison'] == 'OT-AR'].copy()
    ot_ar = ot_ar[~ot_ar['doc_id'].isin(calibration['outlier_doc_ids'])]

    z_lex = (ot_ar['lexical'] - calibration['mu_lex']) / calibration['sigma_lex']
    z_sem = (ot_ar['semantic'] - calibration['mu_sem']) / calibration['sigma_sem']

    thresholds = {
        'tau_lex_upper': np.percentile(z_lex, percentile_high),
        'tau_lex_lower': np.percentile(z_lex, percentile_low),
        'tau_sem_upper': np.percentile(z_sem, percentile_high),
        'tau_sem_lower': np.percentile(z_sem, percentile_low),
        'mode': 'reference',
        'percentile_low': percentile_low,
        'percentile_high': percentile_high,
    }

    if 'mu_comp' in calibration:
        z_comp = (ot_ar['compression_ratio'] - calibration['mu_comp']) / calibration['sigma_comp']
        thresholds['tau_comp_upper'] = np.percentile(z_comp, percentile_high)
        thresholds['tau_comp_lower'] = np.percentile(z_comp, percentile_low)

    return thresholds


# ═══════════════════════════════════════════════════════════════════════
# Диагностика
# ═══════════════════════════════════════════════════════════════════════

def _classify_one(z_lex: float, z_sem: float, z_comp: float,
                  thresholds: Dict) -> Tuple[str, float]:
    """Классифицировать один реферат. Возвращает (тип, уверенность)."""
    tlu = thresholds['tau_lex_upper']
    tll = thresholds['tau_lex_lower']
    tsu = thresholds['tau_sem_upper']
    tsl = thresholds['tau_sem_lower']
    tcu = thresholds.get('tau_comp_upper', float('inf'))
    tcl = thresholds.get('tau_comp_lower', float('-inf'))

    # 1. Недостаточное сжатие
    if z_comp > tcu:
        conf = min((z_comp - tcu) / max(abs(tcu), 1.0), 1.0)
        return 'under_compressed', conf

    # 2. Копирование
    if z_lex > tlu and z_sem > tsl:
        conf = min((z_lex - tlu) / max(abs(tlu), 1.0), 1.0)
        return 'copying', conf

    # 3. Неполнота
    if z_sem < tsl:
        conf = min((tsl - z_sem) / max(abs(tsl), 1.0), 1.0)
        return 'incomplete', conf

    # 4. Избыточное сжатие
    if z_comp < tcl:
        conf = min((tcl - z_comp) / max(abs(tcl), 1.0), 1.0)
        return 'over_compressed', conf

    # 5. Низкая лексика
    if z_lex < tll and z_sem >= tsl:
        conf = min((tll - z_lex) / max(abs(tll), 1.0), 1.0)
        return 'low_lexical', conf

    # 6. Хороший
    if (tll <= z_lex <= tlu) and (tsl <= z_sem <= tsu) and (tcl <= z_comp <= tcu):
        d_lex = min(abs(z_lex - tlu), abs(z_lex - tll))
        d_sem = min(abs(z_sem - tsu), abs(z_sem - tsl))
        d_comp = min(abs(z_comp - tcu), abs(z_comp - tcl))
        w_lex = tlu - tll
        w_sem = tsu - tsl
        w_comp = tcu - tcl
        conf = min(
            d_lex / (w_lex / 2) if w_lex > 0 else 1.0,
            d_sem / (w_sem / 2) if w_sem > 0 else 1.0,
            d_comp / (w_comp / 2) if w_comp > 0 else 1.0,
        )
        return 'good', conf

    # 7. Неоднозначный
    return 'ambiguous', 0.0


def classify_all(df: pd.DataFrame, thresholds: Dict) -> pd.DataFrame:
    """Диагностика всех рефератов."""
    df = df.copy()
    results = df.apply(
        lambda r: _classify_one(r['z_lex'], r['z_sem'], r['z_comp'], thresholds),
        axis=1, result_type='expand'
    )
    df['diagnosis_type'] = results[0]
    df['diagnosis_confidence'] = results[1]
    return df


# ═══════════════════════════════════════════════════════════════════════
# Интегральная оценка Q
# ═══════════════════════════════════════════════════════════════════════

def compute_quality(df_diagnosed: pd.DataFrame, df_raw: pd.DataFrame,
                    alpha: float = 0.45, beta: float = 0.25,
                    gamma: float = 0.15, delta: float = 0.15) -> pd.DataFrame:
    """
    Вычислить Q = α·q_sem + β·q_lex + γ·q_align + δ·q_comp.
    """
    # Получить AR-SR метрики
    df_ar_sr = df_raw[df_raw['comparison'] == 'AR-SR'][['doc_id', 'model', 'lexical', 'semantic']].copy()
    df_ar_sr.columns = ['doc_id', 'model', 'lexical_as', 'semantic_as']

    df = df_diagnosed.merge(df_ar_sr, on=['doc_id', 'model'], how='left')

    # Заполнить NaN нулями (если AR-SR не найден)
    df['lexical_as'] = df['lexical_as'].fillna(0)
    df['semantic_as'] = df['semantic_as'].fillna(0)

    # Компоненты
    df['q_sem'] = df['semantic_as'] * np.exp(-df['z_sem'] ** 2 / 2)
    df['q_lex'] = df['lexical_as'] * np.exp(-df['z_lex'] ** 2 / 2)
    df['q_align'] = np.exp(-(df['z_lex'] - df['z_sem']) ** 2 / 2)
    df['q_comp'] = np.exp(-df['z_comp'] ** 2 / 2)

    df['Q'] = (alpha * df['q_sem'] + beta * df['q_lex'] +
               gamma * df['q_align'] + delta * df['q_comp'])

    return df


# ═══════════════════════════════════════════════════════════════════════
# Полный пайплайн
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Оценка ручного ввода
# ═══════════════════════════════════════════════════════════════════════

def _extract_metric_value(metrics_block: Dict, lex_mode: str, sem_mode: str,
                          rouge_measure: str, which: str = 'lex') -> Optional[float]:
    """Извлечь значение метрики из блока ot_sr или ar_sr."""
    if which == 'lex':
        if lex_mode in ('rouge1', 'rouge2', 'rougeL'):
            rouge_data = metrics_block.get('rouge', {})
            return rouge_data.get(lex_mode, {}).get(rouge_measure)
        else:
            return metrics_block.get(lex_mode)
    else:  # sem
        if sem_mode == 'bertscore':
            bs = metrics_block.get('bertscore')
            return bs.get('f1') if isinstance(bs, dict) else None
        elif sem_mode == 'bleurt':
            return metrics_block.get('bleurt')
    return None


def evaluate_manual_input(
    metrics_dict: Dict,
    calibration: Dict,
    thresholds: Dict,
    lex_mode: str,
    sem_mode: str,
    rouge_measure: str,
    alpha: float = 0.45,
    beta: float = 0.25,
    gamma: float = 0.15,
    delta: float = 0.15,
) -> Dict:
    """
    Оценить реферат по ручному вводу, используя существующую калибровку.

    Args:
        metrics_dict: результат metrics_compute.compute_all_metrics()
        calibration: результат calibrate() из основного пайплайна
        thresholds: результат compute_thresholds()
        lex_mode: режим лексической метрики ('rouge1', 'bleu', 'chrf', 'meteor')
        sem_mode: режим семантической метрики ('bleurt', 'bertscore')
        rouge_measure: мера ROUGE ('p', 'r', 'f')
        alpha, beta, gamma, delta: коэффициенты Q

    Returns:
        dict с z-scores, диагнозом, Q-score и всеми метриками
    """
    ot_sr = metrics_dict.get('ot_sr', {})

    # 1. Извлечь raw значения метрик
    raw_lex = _extract_metric_value(ot_sr, lex_mode, sem_mode, rouge_measure, 'lex')
    raw_sem = _extract_metric_value(ot_sr, lex_mode, sem_mode, rouge_measure, 'sem')
    raw_comp = metrics_dict.get('compression_ratio', 0.0)

    # Проверка доступности метрик
    if raw_lex is None:
        return {'error': f'Лексическая метрика {lex_mode} не вычислена'}
    if raw_sem is None:
        return {'error': f'Семантическая метрика {sem_mode} не вычислена'}

    # 2. Z-scores
    z_lex = (raw_lex - calibration['mu_lex']) / calibration['sigma_lex']
    z_sem = (raw_sem - calibration['mu_sem']) / calibration['sigma_sem']

    if 'mu_comp' in calibration and calibration['sigma_comp'] > 0:
        z_comp = (raw_comp - calibration['mu_comp']) / calibration['sigma_comp']
    else:
        z_comp = 0.0

    # 3. Диагноз
    diagnosis_type, diagnosis_confidence = _classify_one(z_lex, z_sem, z_comp, thresholds)

    # 4. Q-score
    has_reference = metrics_dict.get('ar_sr') is not None
    if has_reference:
        ar_sr = metrics_dict['ar_sr']
        lex_as = _extract_metric_value(ar_sr, lex_mode, sem_mode, rouge_measure, 'lex') or 0.0
        sem_as = _extract_metric_value(ar_sr, lex_mode, sem_mode, rouge_measure, 'sem') or 0.0
    else:
        # Без авторского реферата — упрощённая формула (множители = 1.0)
        lex_as = 1.0
        sem_as = 1.0

    q_sem = sem_as * np.exp(-z_sem ** 2 / 2)
    q_lex = lex_as * np.exp(-z_lex ** 2 / 2)
    q_align = np.exp(-(z_lex - z_sem) ** 2 / 2)
    q_comp = np.exp(-z_comp ** 2 / 2)
    Q = alpha * q_sem + beta * q_lex + gamma * q_align + delta * q_comp

    return {
        'z_lex': float(z_lex),
        'z_sem': float(z_sem),
        'z_comp': float(z_comp),
        'diagnosis_type': diagnosis_type,
        'diagnosis_confidence': float(diagnosis_confidence),
        'Q': float(Q),
        'q_sem': float(q_sem),
        'q_lex': float(q_lex),
        'q_align': float(q_align),
        'q_comp': float(q_comp),
        'raw_lex': float(raw_lex),
        'raw_sem': float(raw_sem),
        'raw_comp': float(raw_comp),
        'has_reference': has_reference,
        'all_metrics': metrics_dict,
    }


# ═══════════════════════════════════════════════════════════════════════
# Полный пайплайн
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline(data: Dict, lex_mode: str = 'rouge1', sem_mode: str = 'bleurt',
                 rouge_measure: str = 'p', selected_models: Optional[List[str]] = None,
                 threshold_mode: str = 'reference',
                 percentile_low: float = 10, percentile_high: float = 90,
                 tau: float = 2.0,
                 alpha: float = 0.45, beta: float = 0.25,
                 gamma: float = 0.15, delta: float = 0.15) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Полный пайплайн: JSON → DataFrame с диагнозами и Q.

    Returns: (df_result, calibration, thresholds)
    """
    if selected_models is None:
        selected_models = get_available_models(data)

    # 1. Подготовка DataFrame
    df_raw = prepare_dataframe(data, lex_mode, sem_mode, rouge_measure, selected_models)

    # 2. Калибровка
    cal = calibrate(df_raw)

    # 3. Z-scores (только для ОТ-СР)
    df_ot_sr = df_raw[df_raw['comparison'] == 'OT-SR'].copy()
    df_z = compute_z_scores(df_ot_sr, cal)

    # 4. Пороги
    thr = compute_thresholds(threshold_mode, df_raw, cal,
                             percentile_low, percentile_high, tau)

    # 5. Диагностика
    df_diag = classify_all(df_z, thr)

    # 6. Q
    df_result = compute_quality(df_diag, df_raw, alpha, beta, gamma, delta)

    return df_result, cal, thr
