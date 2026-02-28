#!/usr/bin/env python3
"""
generate_summaries.py — Генерация рефератов для научных статей из cyberleninka.db

Использует 6 моделей:
  Экстрактивные: TextRank, LexRank (через sumy)
  Трансформерные: mT5, mbart, rut5, t5 (через HuggingFace transformers)

Выходной CSV совместим с analysis_summarization.ipynb
"""

import argparse
import csv
import json
import os
import sqlite3
import sys
import warnings
from pathlib import Path

import nltk
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Генерация рефератов из cyberleninka.db"
    )
    p.add_argument("--db", default="cyberleninka.db",
                   help="Путь к SQLite базе (по умолчанию: cyberleninka.db)")
    p.add_argument("--output", default="summaries.csv",
                   help="Путь к выходному CSV (по умолчанию: summaries.csv)")
    p.add_argument("--limit", type=int, default=0,
                   help="Ограничение количества статей (0 = все)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Размер батча для трансформеров (по умолчанию: 8)")
    p.add_argument("--models", default="all",
                   help="Модели через запятую: textrank,lexrank,mt5,mbart,rut5,t5 (по умолчанию: all)")
    p.add_argument("--resume", action="store_true",
                   help="Возобновить с прерванного места")
    p.add_argument("--max-source-tokens", type=int, default=1024,
                   help="Макс. длина исходного текста в токенах (по умолчанию: 1024)")
    p.add_argument("--save-every", type=int, default=100,
                   help="Промежуточное сохранение каждые N статей (по умолчанию: 100)")
    p.add_argument("--num-sentences", type=int, default=7,
                   help="Количество предложений для экстрактивных моделей (по умолчанию: 7)")
    return p.parse_args()


# ─── Константы ────────────────────────────────────────────────────────────────

ALL_MODELS = ["textrank", "lexrank", "mt5", "mbart", "rut5", "t5"]

CSV_COLUMNS = [
    "id", "source_text", "target_summary",
    "summary_TextRank", "summary_LexRank",
    "summary_mt5", "summary_mbart", "summary_rut5", "summary_t5",
]

MODEL_COL_MAP = {
    "textrank": "summary_TextRank",
    "lexrank":  "summary_LexRank",
    "mt5":      "summary_mt5",
    "mbart":    "summary_mbart",
    "rut5":     "summary_rut5",
    "t5":       "summary_t5",
}

HF_MODEL_IDS = {
    "mbart": "IlyaGusev/mbart_ru_sum_gazeta",
    "mt5":   "csebuetnlp/mT5_multilingual_XLSum",
    "rut5":  "IlyaGusev/rut5_base_sum_gazeta",
    "t5":    "utrobinmv/t5_summary_en_ru_zh_base_2048",
}


# ─── Загрузка данных ──────────────────────────────────────────────────────────

def load_articles(db_path, limit=0):
    """Загрузить статьи из SQLite."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT id, full_text, abstract
        FROM articles
        WHERE full_text IS NOT NULL AND full_text != ''
          AND abstract IS NOT NULL AND abstract != ''
        ORDER BY id
    """
    if limit > 0:
        query += f" LIMIT {limit}"
    cursor = conn.execute(query)
    articles = []
    for row in cursor:
        articles.append({
            "id": row[0],
            "source_text": row[1],
            "target_summary": row[2],
        })
    conn.close()
    return articles


# ─── Resume ───────────────────────────────────────────────────────────────────

def load_existing_results(csv_path):
    """Загрузить уже обработанные результаты из CSV для --resume."""
    results = {}
    if not os.path.exists(csv_path):
        return results
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            article_id = int(row["id"])
            results[article_id] = row
    print(f"[resume] Загружено {len(results)} уже обработанных статей из {csv_path}")
    return results


def determine_completed_models(existing_results):
    """Определить, какие модели уже полностью обработаны."""
    if not existing_results:
        return set()
    completed = set()
    for model_key, col_name in MODEL_COL_MAP.items():
        all_filled = all(
            row.get(col_name, "") != ""
            for row in existing_results.values()
        )
        if all_filled:
            completed.add(model_key)
    return completed


# ─── Экстрактивные модели (sumy) ─────────────────────────────────────────────

def ensure_nltk_data():
    """Скачать необходимые данные NLTK для sumy."""
    for resource in ["punkt_tab", "punkt"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            print(f"[nltk] Скачиваю {resource}...")
            nltk.download(resource, quiet=True)


def run_extractive(articles, model_name, num_sentences=7):
    """Запустить экстрактивную суммаризацию (TextRank или LexRank)."""
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.nlp.stemmers import Stemmer

    if model_name == "textrank":
        from sumy.summarizers.text_rank import TextRankSummarizer as Summarizer
    elif model_name == "lexrank":
        from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer
    else:
        raise ValueError(f"Неизвестная экстрактивная модель: {model_name}")

    LANGUAGE = "russian"
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)

    col = MODEL_COL_MAP[model_name]
    print(f"\n{'='*60}")
    print(f"  {model_name.upper()} (экстрактивная, {num_sentences} предложений)")
    print(f"{'='*60}")

    results = {}
    for art in tqdm(articles, desc=model_name):
        try:
            parser = PlaintextParser.from_string(art["source_text"], Tokenizer(LANGUAGE))
            summary_sentences = summarizer(parser.document, num_sentences)
            summary = " ".join(str(s) for s in summary_sentences)
            results[art["id"]] = summary
        except Exception as e:
            print(f"\n[{model_name}] Ошибка на id={art['id']}: {e}")
            results[art["id"]] = ""

    return results


# ─── Трансформерные модели ────────────────────────────────────────────────────

def run_transformer(articles, model_name, batch_size=8, max_source_tokens=1024):
    """Запустить трансформерную суммаризацию."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    hf_id = HF_MODEL_IDS[model_name]
    print(f"\n{'='*60}")
    print(f"  {model_name.upper()} ({hf_id})")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Устройство: {device}")
    print(f"  Загрузка модели...")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_id).to(device)
    model.eval()

    # Параметры генерации для каждой модели
    gen_kwargs = {
        "num_beams": 4,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }

    # Специфичные параметры
    if model_name == "mbart":
        gen_kwargs["max_new_tokens"] = 300
    elif model_name == "mt5":
        gen_kwargs["max_new_tokens"] = 300
    elif model_name == "rut5":
        gen_kwargs["max_new_tokens"] = 300
    elif model_name == "t5":
        gen_kwargs["max_new_tokens"] = 300
        # Эта модель ожидает префикс
        # utrobinmv/t5_summary_en_ru_zh_base_2048 использует "summarize: " префикс

    results = {}
    batches = [articles[i:i+batch_size] for i in range(0, len(articles), batch_size)]

    for batch in tqdm(batches, desc=f"{model_name} (batches)"):
        texts = []
        for art in batch:
            text = art["source_text"]
            # Добавить префикс для t5 модели
            if model_name == "t5":
                text = "summarize: " + text
            texts.append(text)

        try:
            inputs = tokenizer(
                texts,
                max_length=max_source_tokens,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    **gen_kwargs,
                )

            summaries = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for art, summary in zip(batch, summaries):
                results[art["id"]] = summary.strip()

        except Exception as e:
            print(f"\n[{model_name}] Ошибка на батче: {e}")
            for art in batch:
                if art["id"] not in results:
                    results[art["id"]] = ""

    # Освободить GPU память
    del model
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"  Модель {model_name} выгружена из памяти.")

    return results


# ─── Сохранение CSV ──────────────────────────────────────────────────────────

def save_csv(output_path, articles, all_results):
    """Сохранить результаты в CSV."""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for art in articles:
            row = {
                "id": art["id"],
                "source_text": art["source_text"],
                "target_summary": art["target_summary"],
            }
            for model_key, col_name in MODEL_COL_MAP.items():
                if model_key in all_results:
                    row[col_name] = all_results[model_key].get(art["id"], "")
                else:
                    row[col_name] = ""
            writer.writerow(row)
    print(f"\n[save] CSV сохранён: {output_path} ({len(articles)} строк)")


def save_csv_incremental(output_path, articles, all_results):
    """Промежуточное сохранение — перезаписывает файл."""
    save_csv(output_path, articles, all_results)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Определить список моделей
    if args.models == "all":
        models_to_run = ALL_MODELS[:]
    else:
        models_to_run = [m.strip().lower() for m in args.models.split(",")]
        for m in models_to_run:
            if m not in ALL_MODELS:
                print(f"Ошибка: неизвестная модель '{m}'. Допустимые: {ALL_MODELS}")
                sys.exit(1)

    extractive_models = [m for m in models_to_run if m in ("textrank", "lexrank")]
    transformer_models = [m for m in models_to_run if m in HF_MODEL_IDS]

    # Загрузка данных
    print(f"[db] Загрузка статей из {args.db}...")
    articles = load_articles(args.db, limit=args.limit)
    print(f"[db] Загружено {len(articles)} статей")

    if not articles:
        print("Нет статей для обработки.")
        sys.exit(0)

    # Resume
    all_results = {}
    if args.resume:
        existing = load_existing_results(args.output)
        if existing:
            # Восстановить результаты для каждой модели
            for model_key, col_name in MODEL_COL_MAP.items():
                model_results = {}
                for art_id, row in existing.items():
                    val = row.get(col_name, "")
                    if val:
                        model_results[art_id] = val
                if model_results:
                    all_results[model_key] = model_results

            # Определить какие модели уже полностью обработаны
            completed = determine_completed_models(existing)
            if completed:
                print(f"[resume] Полностью обработаны: {completed}")
                # Убрать завершённые модели из списка
                extractive_models = [m for m in extractive_models if m not in completed]
                transformer_models = [m for m in transformer_models if m not in completed]

    # NLTK данные для sumy
    if extractive_models:
        ensure_nltk_data()

    # ── Экстрактивные модели ──
    for model_name in extractive_models:
        results = run_extractive(articles, model_name, num_sentences=args.num_sentences)
        all_results[model_name] = results

        # Промежуточное сохранение
        save_csv_incremental(args.output, articles, all_results)

    # ── Трансформерные модели (по одной для экономии VRAM) ──
    for model_name in transformer_models:
        # Определить статьи для обработки (с учётом resume)
        if model_name in all_results:
            done_ids = set(
                aid for aid, val in all_results[model_name].items() if val
            )
            remaining = [a for a in articles if a["id"] not in done_ids]
            if not remaining:
                print(f"\n[{model_name}] Все статьи уже обработаны, пропускаю.")
                continue
            print(f"\n[{model_name}] Осталось обработать: {len(remaining)} из {len(articles)}")
        else:
            remaining = articles
            all_results[model_name] = {}

        # Обработка батчами с промежуточным сохранением
        total = len(remaining)
        processed = 0

        for chunk_start in range(0, total, args.save_every):
            chunk = remaining[chunk_start:chunk_start + args.save_every]

            chunk_results = run_transformer(
                chunk, model_name,
                batch_size=args.batch_size,
                max_source_tokens=args.max_source_tokens,
            )

            all_results[model_name].update(chunk_results)
            processed += len(chunk)

            # Промежуточное сохранение
            save_csv_incremental(args.output, articles, all_results)
            print(f"  [{model_name}] Прогресс: {processed}/{total}")

    # ── Финальное сохранение ──
    save_csv(args.output, articles, all_results)

    # Статистика
    print(f"\n{'='*60}")
    print(f"  ГОТОВО")
    print(f"{'='*60}")
    print(f"  Статей: {len(articles)}")
    print(f"  Моделей: {len(models_to_run)}")
    print(f"  Выходной файл: {args.output}")
    for model_key in models_to_run:
        col = MODEL_COL_MAP[model_key]
        if model_key in all_results:
            filled = sum(1 for v in all_results[model_key].values() if v)
            print(f"    {col}: {filled}/{len(articles)}")
    print()


if __name__ == "__main__":
    main()
