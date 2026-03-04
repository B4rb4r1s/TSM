"""
metrics_compute.py — Вычисление метрик суммаризации для пары текстов в реальном времени.

Используется для ручного ввода текста + реферата на вкладке "Ручной анализ".
Тяжёлые модели (BERTScore, BLEURT) загружаются лениво при первом вызове.
"""

import warnings
warnings.filterwarnings('ignore')

from typing import Dict, Optional

# ═══════════════════════════════════════════════════════════════════════
# Ленивые кеши для тяжёлых моделей
# ═══════════════════════════════════════════════════════════════════════

_rouge_instance = None
_bleurt_scorer = None
_bleurt_available = None  # None = ещё не проверяли


def _get_rouge():
    """Ленивая инициализация Rouge."""
    global _rouge_instance
    if _rouge_instance is None:
        from rouge import Rouge
        _rouge_instance = Rouge()
    return _rouge_instance


# ═══════════════════════════════════════════════════════════════════════
# Лёгкие метрики
# ═══════════════════════════════════════════════════════════════════════

def compute_rouge(reference: str, hypothesis: str) -> Dict:
    """
    ROUGE-1/2/L — precision, recall, f-measure.

    Returns: {"rouge1": {"p": float, "r": float, "f": float}, "rouge2": {...}, "rougeL": {...}}
    """
    if not reference.strip() or not hypothesis.strip():
        zero = {'p': 0.0, 'r': 0.0, 'f': 0.0}
        return {'rouge1': zero.copy(), 'rouge2': zero.copy(), 'rougeL': zero.copy()}

    rouge = _get_rouge()
    scores = rouge.get_scores(hypothesis, reference)
    s = scores[0]
    return {
        'rouge1': {'p': s['rouge-1']['p'], 'r': s['rouge-1']['r'], 'f': s['rouge-1']['f']},
        'rouge2': {'p': s['rouge-2']['p'], 'r': s['rouge-2']['r'], 'f': s['rouge-2']['f']},
        'rougeL': {'p': s['rouge-l']['p'], 'r': s['rouge-l']['r'], 'f': s['rouge-l']['f']},
    }


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Sentence-level BLEU (0–1)."""
    if not reference.strip() or not hypothesis.strip():
        return 0.0

    import sacrebleu
    bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
    return bleu.score / 100.0  # sacrebleu возвращает 0–100


def compute_chrf(reference: str, hypothesis: str) -> float:
    """chrF++ score (0–1)."""
    if not reference.strip() or not hypothesis.strip():
        return 0.0

    import sacrebleu
    chrf = sacrebleu.sentence_chrf(hypothesis, [reference])
    return chrf.score / 100.0  # sacrebleu возвращает 0–100


def compute_meteor(reference: str, hypothesis: str) -> Optional[float]:
    """METEOR score (0–1). Возвращает None при ошибке."""
    if not reference.strip() or not hypothesis.strip():
        return 0.0

    try:
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
        ref_tokens = word_tokenize(reference)
        hyp_tokens = word_tokenize(hypothesis)
        if not ref_tokens or not hyp_tokens:
            return 0.0
        return meteor_score([ref_tokens], hyp_tokens)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Тяжёлые метрики
# ═══════════════════════════════════════════════════════════════════════

def compute_bertscore(reference: str, hypothesis: str,
                      device: str = 'cuda') -> Optional[Dict]:
    """
    BERTScore — precision, recall, f1.
    Возвращает None при ошибке.
    """
    if not reference.strip() or not hypothesis.strip():
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            [hypothesis], [reference],
            lang='ru', device=device, verbose=False,
        )
        return {
            'precision': P[0].item(),
            'recall': R[0].item(),
            'f1': F1[0].item(),
        }
    except Exception:
        # Фоллбэк на CPU
        try:
            from bert_score import score as bert_score_fn
            P, R, F1 = bert_score_fn(
                [hypothesis], [reference],
                lang='ru', device='cpu', verbose=False,
            )
            return {
                'precision': P[0].item(),
                'recall': R[0].item(),
                'f1': F1[0].item(),
            }
        except Exception:
            return None


def compute_bleurt(reference: str, hypothesis: str) -> Optional[float]:
    """
    BLEURT score. Возвращает None если библиотека недоступна.

    Стратегия: пытаемся загрузить через evaluate, если не получается — None.
    Для установки: pip install git+https://github.com/google-research/bleurt.git
    """
    global _bleurt_scorer, _bleurt_available

    if _bleurt_available is False:
        return None

    if not reference.strip() or not hypothesis.strip():
        return 0.0

    if _bleurt_scorer is None:
        try:
            from evaluate import load
            _bleurt_scorer = load('bleurt', 'BLEURT-20')
            _bleurt_available = True
        except Exception:
            _bleurt_available = False
            return None

    try:
        results = _bleurt_scorer.compute(predictions=[hypothesis], references=[reference])
        return results['scores'][0]
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Вспомогательные
# ═══════════════════════════════════════════════════════════════════════

def compute_compression_ratio(source: str, summary: str) -> float:
    """Степень сжатия: len(summary) / len(source)."""
    if len(source) == 0:
        return 0.0
    return len(summary) / len(source)


# ═══════════════════════════════════════════════════════════════════════
# Вычисление всех метрик
# ═══════════════════════════════════════════════════════════════════════

def compute_all_metrics(source: str, summary: str,
                        reference: Optional[str] = None,
                        device: str = 'cuda') -> Dict:
    """
    Вычислить ВСЕ доступные метрики для пары текстов.

    Args:
        source: исходный текст (ОТ)
        summary: реферат (СР)
        reference: авторский реферат (АР), опционально
        device: устройство для BERTScore ('cuda' или 'cpu')

    Returns:
        {
            "ot_sr": {
                "rouge": {"rouge1": {"p":, "r":, "f":}, ...},
                "bleu": float, "chrf": float, "meteor": float | None,
                "bertscore": {"precision":, "recall":, "f1":} | None,
                "bleurt": float | None,
            },
            "ar_sr": { ... } | None,
            "compression_ratio": float,
            "lengths": {"source": int, "summary": int, "reference": int | None},
        }
    """
    # ── ОТ-СР: исходный текст vs реферат ──
    ot_sr = {
        'rouge': compute_rouge(source, summary),
        'bleu': compute_bleu(source, summary),
        'chrf': compute_chrf(source, summary),
        'meteor': compute_meteor(source, summary),
        'bertscore': compute_bertscore(source, summary, device=device),
        'bleurt': compute_bleurt(source, summary),
    }

    result = {
        'ot_sr': ot_sr,
        'compression_ratio': compute_compression_ratio(source, summary),
        'lengths': {
            'source': len(source),
            'summary': len(summary),
            'reference': len(reference) if reference else None,
        },
    }

    # ── АР-СР: авторский реферат vs реферат (если задан) ──
    if reference and reference.strip():
        result['ar_sr'] = {
            'rouge': compute_rouge(reference, summary),
            'bleu': compute_bleu(reference, summary),
            'chrf': compute_chrf(reference, summary),
            'meteor': compute_meteor(reference, summary),
            'bertscore': compute_bertscore(reference, summary, device=device),
            'bleurt': compute_bleurt(reference, summary),
        }
    else:
        result['ar_sr'] = None

    return result
