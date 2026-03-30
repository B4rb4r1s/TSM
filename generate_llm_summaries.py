"""
generate_llm_summaries.py

Генерация рефератов научных текстов с помощью локальных LLM.
Поддерживаются два бэкенда: Ollama (по умолчанию) и HuggingFace Transformers.
Результаты записываются в data-tables/data-full.csv в колонки summary_<model>.

Требования:
  Ollama:       ollama serve + ollama pull <model> ; pip install pandas requests tqdm
  HuggingFace:  pip install pandas tqdm transformers torch accelerate

Использование:
  # Ollama (по умолчанию)
  python generate_llm_summaries.py
  python generate_llm_summaries.py --models qwen2.5:7b mistral:7b
  python generate_llm_summaries.py --rows 5

  # HuggingFace
  python generate_llm_summaries.py --backend huggingface --models Qwen/Qwen2.5-7B-Instruct
  python generate_llm_summaries.py --backend huggingface --models Qwen/Qwen2.5-7B-Instruct --hf-device cuda --hf-max-new-tokens 512

  # Ключевые слова
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

# ── HuggingFace ──────────────────────────────────────────────────────────────

HF_DEFAULT_DEVICE = "auto"          # "cpu", "cuda", "cuda:0", "auto"
HF_DEFAULT_MAX_NEW_TOKENS = 1024
HF_DEFAULT_TEMPERATURE = 0.3
HF_DEFAULT_DTYPE = "auto"           # "auto", "float16", "bfloat16", "float32"

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


# ─────────────────────────── HUGGINGFACE БЭКЕНД ─────────────────────────────


def load_hf_model(
    model_name: str,
    device: str = HF_DEFAULT_DEVICE,
    dtype: str = HF_DEFAULT_DTYPE,
) -> Tuple[Any, Any]:
    """Загружает модель и токенизатор HuggingFace.

    Возвращает кортеж (model, tokenizer).
    Требует установленных transformers и torch.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as e:
        print(
            f"\n[ERROR] Для бэкенда huggingface необходимы пакеты:\n"
            f"        pip install transformers torch accelerate\n"
            f"        Ошибка: {e}"
        )
        sys.exit(1)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    torch_dtype = dtype_map.get(dtype, "auto")

    print(f"\n  Загрузка модели HuggingFace: {model_name}")
    print(f"  Device: {device}, dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    print(f"  Модель загружена: {model_name}")
    return model, tokenizer


def generate_hf(
    hf_model: Any,
    hf_tokenizer: Any,
    keywords: str,
    text: str,
    max_new_tokens: int = HF_DEFAULT_MAX_NEW_TOKENS,
    temperature: float = HF_DEFAULT_TEMPERATURE,
) -> str | None:
    """Генерирует реферат с помощью HuggingFace модели.

    Использует chat template токенизатора (если поддерживается),
    иначе формирует промпт вручную.
    """
    try:
        import torch
    except ImportError:
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(keywords=keywords, text=text)},
    ]

    try:
        # Пробуем chat template (Qwen, Llama-chat, Mistral-Instruct и др.)
        input_text = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        # Fallback: простой формат без chat template
        input_text = (
            f"### System:\n{SYSTEM_PROMPT}\n\n"
            f"### User:\n{USER_PROMPT_TEMPLATE.format(keywords=keywords, text=text)}\n\n"
            f"### Assistant:\n"
        )

    try:
        inputs = hf_tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": hf_tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output = hf_model.generate(**inputs, **gen_kwargs)

        # Декодируем только новые токены (без промпта)
        new_tokens = output[0][input_len:]
        result = hf_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return result if result else None

    except Exception as e:
        tqdm.write(f"    [ERROR] HuggingFace генерация: {e}")
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
    backend: str = "ollama",
    hf_model: Any = None,
    hf_tokenizer: Any = None,
    hf_max_new_tokens: int = HF_DEFAULT_MAX_NEW_TOKENS,
    hf_temperature: float = HF_DEFAULT_TEMPERATURE,
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
            if backend == "huggingface":
                summary = generate_hf(
                    hf_model, hf_tokenizer, kw_str, text,
                    hf_max_new_tokens, hf_temperature,
                )
            else:
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
        description="Генерация рефератов локальными LLM (Ollama / HuggingFace)"
    )
    parser.add_argument(
        "--backend", choices=["ollama", "huggingface"], default="ollama",
        help="Бэкенд для генерации (по умолчанию: ollama)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="MODEL",
        help="Список моделей (Ollama: qwen2.5:7b; HF: Qwen/Qwen2.5-7B-Instruct)",
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
    # HuggingFace-специфичные аргументы
    parser.add_argument(
        "--hf-device", default=HF_DEFAULT_DEVICE, metavar="DEV",
        help=f"Устройство для HF модели: cpu, cuda, cuda:0, auto (по умолчанию: {HF_DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--hf-dtype", default=HF_DEFAULT_DTYPE,
        choices=["auto", "float16", "bfloat16", "float32"],
        help=f"Тип данных для HF модели (по умолчанию: {HF_DEFAULT_DTYPE})",
    )
    parser.add_argument(
        "--hf-max-new-tokens", type=int, default=HF_DEFAULT_MAX_NEW_TOKENS, metavar="N",
        help=f"Макс. новых токенов при HF генерации (по умолчанию: {HF_DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--hf-temperature", type=float, default=HF_DEFAULT_TEMPERATURE, metavar="F",
        help=f"Temperature для HF генерации (по умолчанию: {HF_DEFAULT_TEMPERATURE})",
    )
    args = parser.parse_args()

    # Модели по умолчанию зависят от бэкенда
    if args.models is None:
        args.models = DEFAULT_MODELS

    backend_label = "HuggingFace" if args.backend == "huggingface" else "Ollama"
    print("=" * 60)
    print(f"  Генератор рефератов  ·  {backend_label}")
    print("=" * 60)

    # ── Проверка бэкенда ─────────────────────────────────────────────────────

    hf_model_obj = None
    hf_tokenizer_obj = None

    if args.backend == "ollama":
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
    else:
        # HuggingFace: загрузка первой модели
        # (для HF обычно используется одна модель за запуск)
        if not args.models:
            print("\n[ERROR] Укажите модель: --models Qwen/Qwen2.5-7B-Instruct")
            sys.exit(1)
        hf_model_obj, hf_tokenizer_obj = load_hf_model(
            args.models[0], args.hf_device, args.hf_dtype,
        )

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
        # Для HF: если >1 модели, перезагружаем (первая уже загружена)
        if args.backend == "huggingface" and model != args.models[0]:
            hf_model_obj, hf_tokenizer_obj = load_hf_model(
                model, args.hf_device, args.hf_dtype,
            )

        count = run_model(
            df, model, overwrite, args.rows, args.csv,
            keywords_map, args.max_keywords, args.min_score,
            backend=args.backend,
            hf_model=hf_model_obj,
            hf_tokenizer=hf_tokenizer_obj,
            hf_max_new_tokens=args.hf_max_new_tokens,
            hf_temperature=args.hf_temperature,
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
