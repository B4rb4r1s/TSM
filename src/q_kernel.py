"""
q_kernel.py — Единое ядро мягких штрафов для Q-метрик.

Реализует семейство параметризованных ядер, использующихся при вычислении
компонент композитной метрики качества:

    Q = α·q_sem + β·q_lex + γ·q_align + δ·q_comp

Каждая q-компонента вычисляется через `q_kernel(z, ...)` и возвращает
значение в [0, 1], где 1 — «на эталоне», 0 — «сильное отклонение».

Семейство применяется как в оффлайн-оценке (TSM, reference-based Q),
так и в inference-оценке (KGS, reference-free Q_inf). Это гарантирует
единообразие штрафов между двумя компонентами методики.

Возможности ядра:
  1. Симметричная гауссиана exp(-z²/2) — baseline диссертации.
  2. Асимметричная гауссиана — штраф только с одной стороны отклонения:
       direction='penalize_low'  — штраф только при z < 0 (или z < τ_lo)
       direction='penalize_high' — штраф только при z > 0 (или z > τ_hi)
  3. Толерантные зоны [τ_lo, τ_hi] — внутри зоны q = 1.0, снаружи — гауссиана
     от расстояния до ближайшей границы.
  4. Сигмоидальное искажение z (g_funk) — смягчает положительную сторону
     отклонения; применяется к z_comp для поведения «длинный, но содержательный
     реферат штрафуется мягче слишком короткого».

Асимметричные шорткаты:
  q_sem_asymmetric(z)        — penalize_low  (высокий z_sem = хорошо)
  q_lex_symmetric(z)         — both          (и копирование, и парафраз плохо)
  q_comp_asymmetric(z, h, k) — both + g_funk (длинный ≠ плохой)
  q_align_asymmetric(z_lex, z_sem) — penalize_high  (штраф при lex > sem = копирование)
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Направление штрафа
# ---------------------------------------------------------------------------

class Direction(str, Enum):
    """Сторона, с которой ядро штрафует отклонения."""
    BOTH = "both"
    PENALIZE_LOW = "penalize_low"    # штраф только при z < τ_lo
    PENALIZE_HIGH = "penalize_high"  # штраф только при z > τ_hi


def _as_direction(value) -> Direction:
    """Приведение строки к Direction."""
    if isinstance(value, Direction):
        return value
    return Direction(value)


# ---------------------------------------------------------------------------
# Сигмоидальное искажение (для асимметрии компрессии)
# ---------------------------------------------------------------------------

def sigmoid(x, k: float = 2.0):
    """Численно устойчивая сигмоида 1 / (1 + exp(-k·x)).

    Работает и со скаляром (возвращает float), и с numpy/pandas массивом
    (возвращает numpy-массив). Для больших |x| защищает от переполнения
    в exp через ветвление по знаку x.
    """
    # Векторизованная ветка (pandas Series, numpy array, list)
    if hasattr(x, "__len__") or isinstance(x, np.ndarray):
        x_arr = np.asarray(x, dtype=float)
        pos = x_arr >= 0
        out = np.empty_like(x_arr)
        # x >= 0 : 1 / (1 + exp(-kx))
        out[pos] = 1.0 / (1.0 + np.exp(-k * x_arr[pos]))
        # x < 0  : exp(kx) / (1 + exp(kx))
        ex = np.exp(k * x_arr[~pos])
        out[~pos] = ex / (1.0 + ex)
        return out

    # Скалярная ветка
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-k * x))
    ex = math.exp(k * x)
    return ex / (1.0 + ex)


def g_funk(x, h: float = 2.0, k: float = 2.0):
    """Искажение g(x, h, k) = x / (1 + (h-1)·σ(x, k)).

    Работает со скаляром и с массивом (numpy/pandas Series).

    При h > 1 положительная сторона (x > 0) сжимается сильнее отрицательной:
      - z_comp = +2 → |g| < 2 (длинный реферат штрафуется мягче)
      - z_comp = -2 → |g| ≈ |z| (короткий реферат сохраняет полный штраф)
    При h = 1 возвращает x без искажения.
    """
    return x / (1.0 + (h - 1.0) * sigmoid(x, k))


# ---------------------------------------------------------------------------
# Основное ядро
# ---------------------------------------------------------------------------

def q_kernel(
    z: float,
    direction: Direction | str = Direction.BOTH,
    tau_lo: Optional[float] = None,
    tau_hi: Optional[float] = None,
    shape: str = "gaussian",
    h: float = 2.0,
    k: float = 2.0,
) -> float:
    """Мягкое ядро штрафа для нормализованного отклонения z.

    Args:
        z: z-отклонение (нормализованное).
        direction: сторона штрафа: BOTH, PENALIZE_LOW, PENALIZE_HIGH.
        tau_lo, tau_hi: границы целевой зоны. Если заданы обе —
            внутри зоны возвращается 1.0, снаружи — гауссиана
            от расстояния до ближайшей границы. Если не заданы —
            гауссиана от z (центр в 0).
        shape: 'gaussian' или 'gaussian_asymmetric' (применяет g_funk к z_excess
            перед гауссианой — для z_comp с асимметричной интерпретацией).
        h, k: параметры g_funk (учитываются только при shape='gaussian_asymmetric').

    Returns:
        Значение q ∈ [0, 1], где 1 — «на эталоне», 0 — «сильное отклонение».
    """
    direction = _as_direction(direction)

    # 1. Приводим z к «избыточному» значению относительно зоны
    if tau_lo is not None and tau_hi is not None:
        if tau_lo > tau_hi:
            raise ValueError(f"tau_lo={tau_lo} > tau_hi={tau_hi}")
        if tau_lo <= z <= tau_hi:
            return 1.0
        z_excess = z - tau_hi if z > tau_hi else z - tau_lo
    else:
        z_excess = z

    # 2. Проверка направления штрафа
    if direction is Direction.PENALIZE_LOW and z_excess > 0:
        return 1.0
    if direction is Direction.PENALIZE_HIGH and z_excess < 0:
        return 1.0

    # 3. Нелинейное искажение (для асимметричной компрессии)
    if shape == "gaussian_asymmetric":
        z_excess = g_funk(z_excess, h=h, k=k)
    elif shape != "gaussian":
        raise ValueError(f"Unknown shape: {shape!r}")

    # 4. Гауссиана
    return math.exp(-(z_excess ** 2) / 2.0)


# ---------------------------------------------------------------------------
# Высокоуровневые шорткаты для Q_inf
# ---------------------------------------------------------------------------

def q_sem_asymmetric(
    z_sem: float,
    tau_lo: Optional[float] = None,
    tau_hi: Optional[float] = None,
) -> float:
    """Семантическая типичность: штраф только при z_sem ниже зоны.

    Высокий z_sem (реферат семантически точнее АР-эталона) не штрафуется.
    """
    return q_kernel(z_sem, Direction.PENALIZE_LOW, tau_lo, tau_hi)


def q_lex_symmetric(
    z_lex: float,
    tau_lo: Optional[float] = None,
    tau_hi: Optional[float] = None,
) -> float:
    """Лексическая типичность: штраф с обеих сторон.

    Высокий z_lex — копирование, низкий — полный парафраз без опоры на термины.
    """
    return q_kernel(z_lex, Direction.BOTH, tau_lo, tau_hi)


def q_comp_asymmetric(
    z_comp: float,
    h: float = 2.0,
    k: float = 2.0,
    tau_lo: Optional[float] = None,
    tau_hi: Optional[float] = None,
) -> float:
    """Типичность сжатия: симметрия с мягкой положительной стороной.

    При h > 1 длинный реферат (z_comp > 0) штрафуется слабее,
    чем слишком короткий (z_comp < 0), где возможна потеря информации.
    """
    return q_kernel(
        z_comp,
        Direction.BOTH,
        tau_lo=tau_lo, tau_hi=tau_hi,
        shape="gaussian_asymmetric",
        h=h, k=k,
    )


def q_align_asymmetric(z_lex: float, z_sem: float) -> float:
    """Согласованность лекс./сем. профилей: штраф только при z_lex > z_sem.

    (z_lex - z_sem) > 0 — «слова без смысла» (копирование), штрафуем.
    (z_lex - z_sem) < 0 — «смысл без слов» (качественный парафраз), не штрафуем.
    """
    return q_kernel(z_lex - z_sem, Direction.PENALIZE_HIGH)


def q_align_symmetric(z_lex: float, z_sem: float) -> float:
    """Согласованность лекс./сем. профилей, симметричный вариант (baseline)."""
    return q_kernel(z_lex - z_sem, Direction.BOTH)


# ---------------------------------------------------------------------------
# Сборка Q_inf через режимы
# ---------------------------------------------------------------------------

Q_ALPHA = 0.45
Q_BETA = 0.25
Q_GAMMA = 0.15
Q_DELTA = 0.15


def compute_q_inf_symmetric(
    z_lex: float, z_sem: float, z_comp: float,
    alpha: float = Q_ALPHA, beta: float = Q_BETA,
    gamma: float = Q_GAMMA, delta: float = Q_DELTA,
) -> float:
    """Симметричная Q_inf (baseline из рукописи диссертации).

    Q_inf = α·exp(-z_sem²/2) + β·exp(-z_lex²/2)
          + γ·exp(-(z_lex−z_sem)²/2) + δ·exp(-z_comp²/2)
    """
    q_sem = q_kernel(z_sem, Direction.BOTH)
    q_lex = q_kernel(z_lex, Direction.BOTH)
    q_align = q_align_symmetric(z_lex, z_sem)
    q_comp = q_kernel(z_comp, Direction.BOTH)
    return alpha * q_sem + beta * q_lex + gamma * q_align + delta * q_comp


def compute_q_inf_asymmetric(
    z_lex: float, z_sem: float, z_comp: float,
    tau_lex_lo: Optional[float] = None, tau_lex_hi: Optional[float] = None,
    tau_sem_lo: Optional[float] = None, tau_sem_hi: Optional[float] = None,
    tau_comp_lo: Optional[float] = None, tau_comp_hi: Optional[float] = None,
    h_comp: float = 2.0, k_comp: float = 2.0,
    alpha: float = Q_ALPHA, beta: float = Q_BETA,
    gamma: float = Q_GAMMA, delta: float = Q_DELTA,
) -> float:
    """Асимметричная Q_inf (новый основной вариант).

    - q_sem:  штраф только при z_sem < τ_sem_lo (недостаток смысла)
    - q_lex:  штраф симметричный (копирование и полный парафраз — плохо)
    - q_align: штраф только при z_lex > z_sem (копирующий паттерн)
    - q_comp: симметрия с g_funk (длинный реферат штрафуется мягче короткого)

    Если границы τ не переданы — используются простые гауссианы вокруг 0.
    """
    q_sem = q_sem_asymmetric(z_sem, tau_sem_lo, tau_sem_hi)
    q_lex = q_lex_symmetric(z_lex, tau_lex_lo, tau_lex_hi)
    q_align = q_align_asymmetric(z_lex, z_sem)
    q_comp = q_comp_asymmetric(z_comp, h=h_comp, k=k_comp,
                               tau_lo=tau_comp_lo, tau_hi=tau_comp_hi)
    return alpha * q_sem + beta * q_lex + gamma * q_align + delta * q_comp


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Базовая симметричная гауссиана
    assert abs(q_kernel(0.0) - 1.0) < 1e-9
    assert abs(q_kernel(2.0) - math.exp(-2)) < 1e-9
    assert abs(q_kernel(-2.0) - math.exp(-2)) < 1e-9
    print("OK: базовая гауссиана")

    # Асимметричные направления
    assert q_kernel(2.0, Direction.PENALIZE_LOW) == 1.0
    assert abs(q_kernel(-2.0, Direction.PENALIZE_LOW) - math.exp(-2)) < 1e-9
    assert q_kernel(-2.0, Direction.PENALIZE_HIGH) == 1.0
    assert abs(q_kernel(2.0, Direction.PENALIZE_HIGH) - math.exp(-2)) < 1e-9
    print("OK: асимметричные направления")

    # Толерантная зона
    assert q_kernel(0.0, tau_lo=-1.0, tau_hi=1.0) == 1.0
    assert q_kernel(0.5, tau_lo=-1.0, tau_hi=1.0) == 1.0
    assert q_kernel(1.0, tau_lo=-1.0, tau_hi=1.0) == 1.0
    v = q_kernel(2.0, tau_lo=-1.0, tau_hi=1.0)  # z_excess = 1
    assert abs(v - math.exp(-0.5)) < 1e-9
    print("OK: толерантная зона")

    # Зона + направление (penalize_low не штрафует сверху)
    assert q_kernel(3.0, Direction.PENALIZE_LOW, tau_lo=-1.0, tau_hi=1.0) == 1.0
    v = q_kernel(-3.0, Direction.PENALIZE_LOW, tau_lo=-1.0, tau_hi=1.0)  # z_excess = -2
    assert abs(v - math.exp(-2)) < 1e-9
    print("OK: зона + направление")

    # g_funk: h=1 тождественен, h>1 сжимает положительную сторону
    assert abs(g_funk(2.0, h=1.0) - 2.0) < 1e-9
    assert abs(g_funk(-2.0, h=1.0) - (-2.0)) < 1e-9
    pos = g_funk(2.0, h=2.0, k=2.0)
    neg = g_funk(-2.0, h=2.0, k=2.0)
    assert abs(pos) < 2.0 and abs(neg) < 2.0
    assert abs(neg) > abs(pos), "отрицательная сторона должна остаться сильнее"
    print(f"OK: g_funk — g(+2)={pos:.4f}, g(-2)={neg:.4f}")

    # Векторизованные sigmoid/g_funk (для pandas/numpy в composite_metric.py)
    arr = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    sig_vec = sigmoid(arr, k=2.0)
    sig_scalar = np.array([sigmoid(float(v), k=2.0) for v in arr])
    assert np.allclose(sig_vec, sig_scalar), "векторная sigmoid != скалярной"
    g_vec = g_funk(arr, h=2.0, k=2.0)
    g_scalar = np.array([g_funk(float(v), h=2.0, k=2.0) for v in arr])
    assert np.allclose(g_vec, g_scalar), "векторная g_funk != скалярной"
    print(f"OK: vec sigmoid/g_funk — эквивалентны скалярным")

    # q_comp асимметрия
    v_pos = q_comp_asymmetric(2.0, h=2.0, k=2.0)
    v_neg = q_comp_asymmetric(-2.0, h=2.0, k=2.0)
    assert v_pos > v_neg, "длинный реферат должен штрафоваться мягче короткого"
    print(f"OK: q_comp_asymmetric — q(+2)={v_pos:.4f} > q(-2)={v_neg:.4f}")

    # q_align асимметрия
    v_copy = q_align_asymmetric(z_lex=2.0, z_sem=0.0)   # копирование
    v_para = q_align_asymmetric(z_lex=0.0, z_sem=2.0)   # парафраз
    assert v_para == 1.0, "хороший парафраз не должен штрафоваться"
    assert v_copy < 1.0, "копирование должно штрафоваться"
    print(f"OK: q_align — копирование={v_copy:.4f}, парафраз={v_para:.4f}")

    # Сравнение Q_inf симметричной и асимметричной на «семантически сильном» реферате
    z_lex, z_sem, z_comp = 0.2, 2.0, 0.5  # высокий z_sem, лексика в норме
    Q_sym = compute_q_inf_symmetric(z_lex, z_sem, z_comp)
    Q_asym = compute_q_inf_asymmetric(z_lex, z_sem, z_comp)
    assert Q_asym > Q_sym, f"Q_asym={Q_asym:.4f} должен быть выше Q_sym={Q_sym:.4f}"
    print(f"OK: Q_inf — sym={Q_sym:.4f}, asym={Q_asym:.4f}, Δ={Q_asym - Q_sym:+.4f}")

    # На «плохом» реферате (копирование)
    z_lex, z_sem, z_comp = 2.5, 0.5, 1.0
    Q_sym = compute_q_inf_symmetric(z_lex, z_sem, z_comp)
    Q_asym = compute_q_inf_asymmetric(z_lex, z_sem, z_comp)
    print(f"OK: Q_inf копирование — sym={Q_sym:.4f}, asym={Q_asym:.4f}")

    # На идеальном реферате
    Q_perfect_sym = compute_q_inf_symmetric(0.0, 0.0, 0.0)
    Q_perfect_asym = compute_q_inf_asymmetric(0.0, 0.0, 0.0)
    assert abs(Q_perfect_sym - 1.0) < 1e-9
    assert abs(Q_perfect_asym - 1.0) < 1e-9
    print("OK: Q_inf(0,0,0) = 1 для обеих версий")

    print("\n=== Все тесты q_kernel пройдены ===")
