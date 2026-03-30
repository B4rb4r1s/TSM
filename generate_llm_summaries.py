"""
generate_llm_summaries.py

Генерация рефератов научных текстов с помощью локальных LLM через Ollama.
Результаты записываются в data-tables/data-full.csv в колонки summary_<model>.

Требования:
  - Ollama запущен: ollama serve
  - Нужные модели загружены: ollama pull <model>
  - pip install pandas requests tqdm

Использование:
  python generate_llm_summaries.py
  python generate_llm_summaries.py --overwrite          # перезаписать все значения
  python generate_llm_summaries.py --models qwen2.5:7b mistral:7b
  python generate_llm_summaries.py --rows 5             # тест на 5 строках
  python generate_llm_summaries.py --csv path/to/other.csv

  python generate_llm_summaries.py --json data-tables/dataset-480.json --max-keywords 20 --min-score 0.3 --rows 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# ─────────────────────────── КОНФИГУРАЦИЯ ────────────────────────────────────

CSV_PATH = Path(__file__).parent / "data-tables" / "data-full.csv"
JSON_PATH = Path(__file__).parent / "data-tables" / "dataset-480.json"

OLLAMA_BASE_URL = "http://localhost:11434"

# Список моделей по умолчанию.
# Каждая модель → колонка summary_<model> (двоеточия/слэши заменяются на _).
# Закомментируйте ненужные или добавьте свои.
DEFAULT_MODELS = [
    # "forzer/GigaChat3-10B-A1.8B:latest",
    "qwen2.5:7b",
    # "qwen2.5:14b",
    # "saiga_llama3:8b",
    # "mistral:7b",
    # "llama3.1:8b",
]

# Системный промпт: инструкция для модели
SYSTEM_PROMPT = (
    "Ты — опытный научный редактор. Твоя задача — составить краткий реферат "
    "научной статьи на русском языке. Реферат должен:\n"
    "1) передавать цель исследования, методы, результаты и выводы;\n"
    "2) быть написан в безличной научной форме;\n"
    "3) содержать от 100 до 200 слов;\n"
    "4) не содержать ничего кроме самого текста реферата — "
    "никаких заголовков, нумерованных пунктов, пояснений;\n"
    "5) опираться на Ключевые слова из запроса пользователя:\n"
    "  - вес Ключевого слова показывает, насколько оно важно для текста реферата;\n"
    "  - допускаетя использовать НЕ ВСЕ Ключевые слова, а только необходимые."
)

# Шаблон пользовательского сообщения
USER_PROMPT_TEMPLATE = "Составь реферат следующей научной статьи.\n\nКлючевые слова: {keywords}\n\nПолный текс: {text}"

# Параметры генерации Ollama
OLLAMA_OPTIONS = {
    "temperature": 0.3,
    "num_predict": 1024,  # максимум токенов в ответе
}

# Параметры фильтрации ключевых слов
MAX_KEYWORDS = 20       # максимум ключевых слов в промпте
MIN_KEYWORD_SCORE = 0.1 # минимальный score для включения

# Пауза между запросами (секунды)
REQUEST_DELAY = 0.5

# Таймаут HTTP-запроса к Ollama (секунды)
REQUEST_TIMEOUT = 300

# ─────────────────────────── ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ─────────────────────────


def col_name(model: str) -> str:
    """Имя модели → имя колонки: 'qwen2.5:7b' → 'summary_qwen2.5_7b'."""
    # safe = model.replace(":", "_").replace("/", "_")
    # return f"summary_{safe}"
    return "KGSo_qwen2.5"


def check_ollama() -> bool:
    """Проверяет доступность Ollama."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def list_models() -> list[str]:
    """Список загруженных моделей в Ollama."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def _format_keywords(
    keywords: List[Dict[str, Any]],
    max_keywords: int,
    min_score: float,
) -> str:
    """Форматирует ключевые слова в строку для промпта.

    Сортирует по score (убывание), берёт top-N, фильтрует по min_score.
    """
    if not keywords:
        return ""

    filtered = [
        kw for kw in keywords
        if isinstance(kw, dict) and kw.get("score", 0) >= min_score
    ]
    filtered.sort(key=lambda x: x.get("score", 0), reverse=True)
    filtered = filtered[:max_keywords]

    parts = []
    for kw in filtered:
        surface = kw.get("surface_form", kw.get("keyword", ""))
        score = kw.get("score", 0)
        parts.append(f"{surface} ({score:.2f})")

    return ", ".join(parts)


def _normalize_text(text: str) -> str:
    """Нормализация текста для сопоставления: убираем пробелы, нижний регистр."""
    return " ".join(text.lower().split())[:500]


def load_keywords(json_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Загружает ключевые слова из JSON и строит словарь text_prefix → keywords.

    Сопоставление записей JSON со строками CSV выполняется по первым 500
    символам нормализованного текста (без лишних пробелов, в нижнем регистре).
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, List[Dict[str, Any]]] = {}
    for entry in data:
        text = entry.get("text", "")
        keywords = entry.get("keywords", [])
        if text and keywords:
            key = _normalize_text(text)
            mapping[key] = keywords

    return mapping


def get_keywords_for_row(
    source_text: str,
    keywords_map: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Находит ключевые слова для строки CSV по совпадению текста."""
    if not source_text:
        return []
    key = _normalize_text(source_text)
    return keywords_map.get(key, [])


def generate(model: str, keywords: str, text: str) -> str | None:
    """
    Запрашивает реферат у Ollama.
    Возвращает текст реферата или None при ошибке.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(keywords=keywords, text=text)},
        ],
        "stream": False,
        "options": OLLAMA_OPTIONS,
    }
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except requests.exceptions.Timeout:
        tqdm.write(f"    [WARN] Таймаут ({REQUEST_TIMEOUT}s) для модели {model}")
        return None
    except requests.exceptions.RequestException as e:
        tqdm.write(f"    [ERROR] Запрос завершился ошибкой: {e}")
        return None
    except (KeyError, ValueError) as e:
        tqdm.write(f"    [ERROR] Неожиданный ответ Ollama: {e}")
        return None


# ─────────────────────────── ОСНОВНАЯ ЛОГИКА ─────────────────────────────────


def ask_overwrite() -> bool:
    """Интерактивный вопрос о перезаписи."""
    while True:
        ans = input(
            "\nНекоторые строки уже содержат рефераты. "
            "Перезаписать их? [y/N]: "
        ).strip().lower()
        if ans in ("y", "yes", "д", "да"):
            return True
        if ans in ("n", "no", "н", "нет", ""):
            return False
        print("Пожалуйста, введите y или n.")


def run_model(
    df: pd.DataFrame,
    model: str,
    overwrite: bool,
    max_rows: int | None,
    csv_path: Path,
    keywords_map: Dict[str, List[Dict[str, Any]]] | None = None,
    max_keywords: int = MAX_KEYWORDS,
    min_keyword_score: float = MIN_KEYWORD_SCORE,
) -> int:
    """
    Генерирует рефераты для одной модели, сохраняя CSV после каждой строки.
    Возвращает количество успешно сгенерированных рефератов.
    """
    column = col_name(model)

    if column not in df.columns:
        df[column] = None

    # Строки, требующие генерации
    empty_mask = df[column].isna() | (df[column].astype(str).str.strip() == "")
    if overwrite:
        target_mask = pd.Series([True] * len(df), index=df.index)
    else:
        target_mask = empty_mask

    indices = df.index[target_mask].tolist()
    if max_rows is not None:
        indices = indices[:max_rows]

    skipped = len(df) - len(df.index[empty_mask])
    pending = len(indices)

    print(f"\n  {'─'*50}")
    print(f"  Модель  : {model}")
    print(f"  Колонка : {column}")
    print(f"  Уже есть: {skipped} строк")
    print(f"  К генерации: {pending} строк")

    if pending == 0:
        print("  [SKIP] Все строки уже заполнены.")
        return 0

    generated = 0
    with tqdm(indices, desc=f"  {model}", unit="стр", ncols=88) as pbar:
        for idx in pbar:
            row_id = df.at[idx, "id"]
            text = df.at[idx, "source_text"]

            if not isinstance(text, str) or not text.strip():
                tqdm.write(f"    [SKIP] id={row_id}: пустой source_text")
                continue

            # Получаем и форматируем ключевые слова
            kw_str = ""
            if keywords_map:
                kw_list = get_keywords_for_row(text, keywords_map)
                kw_str = _format_keywords(kw_list, max_keywords, min_keyword_score)

            t0 = time.time()
            summary = generate(model, kw_str, text)
            elapsed = time.time() - t0

            if summary is not None:
                df.at[idx, column] = summary
                generated += 1
                tqdm.write(
                    f"    [OK] id={row_id:>4} | {elapsed:5.1f}s | {len(summary)} chars"
                )
                df.to_csv(csv_path, index=False)
            else:
                tqdm.write(f"    [FAIL] id={row_id}: ошибка — строка пропущена")

            if REQUEST_DELAY > 0:
                time.sleep(REQUEST_DELAY)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Генерация рефератов локальными LLM через Ollama"
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS, metavar="MODEL",
        help="Список моделей Ollama (например: qwen2.5:7b mistral:7b)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Перезаписать существующие рефераты без вопроса",
    )
    parser.add_argument(
        "--rows", type=int, default=None, metavar="N",
        help="Обработать только первые N строк (для тестирования)",
    )
    parser.add_argument(
        "--csv", type=Path, default=CSV_PATH, metavar="PATH",
        help=f"Путь к CSV-файлу (по умолчанию: {CSV_PATH})",
    )
    parser.add_argument(
        "--json", type=Path, default=JSON_PATH, metavar="PATH",
        help=f"Путь к JSON с ключевыми словами (по умолчанию: {JSON_PATH})",
    )
    parser.add_argument(
        "--max-keywords", type=int, default=MAX_KEYWORDS, metavar="N",
        help=f"Максимум ключевых слов в промпте (по умолчанию: {MAX_KEYWORDS})",
    )
    parser.add_argument(
        "--min-score", type=float, default=MIN_KEYWORD_SCORE, metavar="F",
        help=f"Минимальный score ключевого слова (по умолчанию: {MIN_KEYWORD_SCORE})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Генератор рефератов  ·  Ollama LLM")
    print("=" * 60)

    # ── Проверка Ollama ───────────────────────────────────────────────────────

    if not check_ollama():
        print(
            "\n[ERROR] Ollama не запущен или недоступен.\n"
            "        Запустите в терминале: ollama serve"
        )
        sys.exit(1)

    available = list_models()
    print(f"\nЗагруженные модели: {available or '(нет)'}")

    missing = [m for m in args.models if m not in available]
    if missing:
        print(f"\n[WARN] Не найдены в Ollama: {missing}")
        print("       Загрузите: ollama pull <model>")
        ans = input("Продолжить с доступными моделями? [Y/n]: ").strip().lower()
        if ans in ("n", "no", "н", "нет"):
            sys.exit(0)
        args.models = [m for m in args.models if m in available]
        if not args.models:
            print("[ERROR] Нет доступных моделей.")
            sys.exit(1)

    # ── Загрузка данных ───────────────────────────────────────────────────────

    if not args.csv.exists():
        print(f"\n[ERROR] Файл не найден: {args.csv}")
        sys.exit(1)

    print(f"\nЧтение: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"Загружено: {len(df)} строк, {len(df.columns)} колонок")

    if args.rows:
        print(f"[TEST] Ограничение: первые {args.rows} строк")

    # ── Загрузка ключевых слов ───────────────────────────────────────────────

    keywords_map = None
    if args.json.exists():
        print(f"\nКлючевые слова: {args.json}")
        keywords_map = load_keywords(args.json)
        print(f"Загружено записей с ключевыми словами: {len(keywords_map)}")
    else:
        print(f"\n[WARN] JSON с ключевыми словами не найден: {args.json}")
        print("       Генерация будет выполнена без ключевых слов.")

    # ── Определение режима перезаписи ─────────────────────────────────────────

    overwrite = args.overwrite
    if not overwrite:
        has_data = any(
            col_name(m) in df.columns and df[col_name(m)].notna().any()
            for m in args.models
        )
        if has_data:
            overwrite = ask_overwrite()

    # ── Обработка ─────────────────────────────────────────────────────────────

    total = 0
    for model in args.models:
        count = run_model(
            df, model, overwrite, args.rows, args.csv,
            keywords_map, args.max_keywords, args.min_score,
        )
        total += count

    # ── Итог ──────────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print(f"  Готово. Сгенерировано рефератов: {total}")
    print(f"  Файл: {args.csv}")
    new_cols = [col_name(m) for m in args.models]
    for col in new_cols:
        if col in df.columns:
            filled = df[col].notna().sum()
            print(f"    {col}: {filled}/{len(df)} строк заполнено")
    print("=" * 60)


if __name__ == "__main__":
    main()
