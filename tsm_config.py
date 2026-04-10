# =============================================================================
# ВЫБОР МЕТРИК ДЛЯ АНАЛИЗА
# =============================================================================


NAMES = {
    'compression_ratio_chars':'Сжатие (символы)',
    'compression_ratio_words':'Сжатие (слова)',

    'rouge1':'ROUGE-1',
    'rouge2':'ROUGE-2',
    'rougeL':'ROUGE-L',
    'bleu':'BLEU',
    'chrf':'chrF++',
    'meteor':'METEOR',
    'ter':'TER',

    'bertscore':'BERTScore',
    # 'bleurt':'BLEURT',
}

# Варианты: 'chars', 'words'
COMPRESSION_MODE = 'chars'

# Варианты: 'rouge1', 'rouge2', 'rougeL', 'bleu', 'chrf', 'meteor'
LEX_MODE = 'rougeL'
ROUGE_MEASURE = 'p'  # Варианты: 'f', 'p', 'r'
ROUGE_MEASURE_NAMES = {'f': 'F-мера', 'p': 'Precision', 'r': 'Recall'}


# Варианты: 'bertscore'
SEM_MODE = 'bertscore'
BERTSCORE_MEASURE = 'f'  # Варианты: 'f', 'p', 'r'
BERTSCORE_MEASURE_NAMES = {'f': 'F-мера', 'p': 'Precision', 'r': 'Recall'}

LEX_NAME = NAMES.get(LEX_MODE)
SEM_NAME = NAMES.get(SEM_MODE)

# -----------------------------------------------------------------------------
# ВЫБОР РЕЖИМА ПОРОГОВ
# -----------------------------------------------------------------------------
# 'centered'  - Центрированные пороги ±τσ вокруг (0,0) — рекомендуется для интерпретируемости
# 'reference' - Пороги на основе процентилей ОТ-АР — логически корректно
# 'adaptive'  - Пороги на основе процентилей машинных рефератов (исходный вариант)

# Варианты: 'Seq2Seq', 'LLMs', 'FULL'
MODE = 'TSM'
SOURCE = '480'
THRESHOLD_MODE = 'reference'  # <-- ИЗМЕНИТЕ ЗДЕСЬ: 'centered', 'reference', или 'adaptive'
CENTERED_TAU = 2.0           # Для режима 'centered': количество σ (1.5, 2.0, 2.5, 3.0)
PERCENTILE_LOW = 10          # Для режимов 'reference' и 'adaptive'
PERCENTILE_HIGH = 90         # Для режимов 'reference' и 'adaptive'

CALIBRATE_SIGMA_LIMIT = 5

MODEL_SHORT = {
    "IlyaGusev/mbart_ru_sum_gazeta":                "mBART",
    "IlyaGusev/rut5_base_sum_gazeta":               "ruT5",
    "IlyaGusev/saiga_llama3_8b":                    "Saiga (llama)",
    "IlyaGusev/saiga_nemo_12b":                     "Saiga (nemo)",
    "Qwen/Qwen2.5-7B-Instruct":                     "Qwen 2.5",
    "Qwen/Qwen3-8B":                                "Qwen 3",
    "ai-sage/GigaChat3-10B-A1.8B-bf16":             "GigaChat 3",
    "csebuetnlp/mT5_multilingual_XLSum":            "mT5",
    "lexrank":                                      "LexRank",
    "textrank":                                     "TextRank",
    "utrobinmv/t5_summary_en_ru_zh_base_2048":      "T5",
    "yandex/YandexGPT-5-Lite-8B-instruct":          "YandexGPT 5",
}


# -----------------------------------------------------------------------------
# Версии моделей
# -----------------------------------------------------------------------------

VERSION_SHORT = {
    'no_keywords':              '(без КС)',
    'extracted:KWE':            '(KWE)',
    'extracted:rutermextract':  '(ruterm)',
    'extracted:textrank':       '(TextRank)',
    'extracted:tfidf':          '(TF-IDF)',
    'abstractive':              '',
    'extractive':               '',
}

VERSION_GROUP_LABELS = {
    'no_keywords':              'Без ключевых слов',
    'extracted:KWE':            'С КС: KWE',
    'extracted:rutermextract':  'С КС: rutermextract',
    'extracted:textrank':       'С КС: TextRank',
    'extracted:tfidf':          'С КС: TF-IDF',
    'abstractive':              'Абстрактивные (Seq2Seq)',
    'extractive':               'Экстрактивные',
}

VERSION_ORDER = [
    'no_keywords',
    'extracted:KWE',
    'extracted:rutermextract',
    'extracted:textrank',
    'extracted:tfidf',
    'abstractive',
    'extractive',
]


def make_model_short_label(name: str, version: str) -> str:
    """Короткое имя для комбинации модель+версия."""
    base = MODEL_SHORT.get(name, name.split('/')[-1] if '/' in name else name)
    suffix = VERSION_SHORT.get(version, f'({version})' if version else '')
    if suffix:
        return f"{base} {suffix}"
    return base


# =============================================================================
# 8. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# =============================================================================

# Цветовая палитра для типов диагнозов (расширенная)
DIAGNOSIS_COLORS = {
    # 'good':             '#2ecc71', # зеленый — хорошие рефераты
    # 'copying':          '#e74c3c', # красный — копирование
    # 'incomplete':       '#9b59b6', # фиолетовый — неполнота
    # 'low_lexical':      '#3498db', # синий — низкая лексика
    # 'over_compressed':  '#e67e22', # оранжевый — слишком короткий
    # 'under_compressed': '#f1c40f', # жёлтый — слишком длинный
    # 'ambiguous':        '#95a5a6', # серый — неоднозначные

    'good':             '#2ecc71', # зеленый - хорошие рефераты
    'copying':          '#3498db', # красный - копирование
    'incomplete':       '#9b59b6', # фиолетовый - неполнота
    'low_lexical':      '#f39c12', # #3cd7e7 оранжевый - низкая лексика (новая категория)
    'ambiguous':        '#95a5a6', # серый - неоднозначные
    'under_compressed': "#e74d3c",
    'over_compressed':  '#008f8f', # #f39c12 #e67e22
}

DIAG_TRANSLATE = {
    'good':         'Целевая зона',
    'copying':      'Избыточное копирование',
    'incomplete':   'Семантическая неполнота',
    'low_lexical':  'Лексическая неполнота',
    'ambiguous':    'Неоднозначный паттерн',

    'under_compressed': 'Недостаточное сжатие',
    'over_compressed':  'Избыточное сжатие'
}

DIAGNOSIS_ZONES = [
    'Целевая зона',
    'Избыточное копирование',
    'Семантическая неполнота',
    'Низкая лекс. сходство',
    'Неоднозначный паттерн',
    'Недостаточное сжатие',
    'Избыточное сжатие',
]

# Цветовая палитра для моделей
MODEL_COLORS = {
    'reference':    '#f1c40f',
    'summary_lingvo':   '#2ecc71',
    'summary_TextRank': '#3498db',
    'summary_LexRank':  "#9b59b6",
    'summary_mt5':      '#f39c12',
    'summary_mbart':    '#e74d3c',
    'summary_rut5':     "#09a8a8",
    'summary_t5':       "#db7217",

    'summary_forzer_GigaChat3-10B-A1.8B_latest':                "#00be30",
    'summary_qwen2.5_7b':                                       "#085ee0",
    'summary_qwen3_8b':                                         "#8614b3",
    'summary_yandex_YandexGPT-5-Lite-8B-instruct-GGUF_latest':  "#eb0909",
}

MODEL_COLORS_2 = {
    'summary_lingvo':   'blue',
    'summary_TextRank': 'orange',
    'summary_LexRank':  'green',
    'summary_mt5':      'red',
    'summary_mbart':    'purple',
    'summary_rut5':     'brown',
    'summary_t5':       'pink',

    'summary_forzer_GigaChat3-10B-A1.8B_latest':                '#2ecc71',
    'summary_qwen2.5_7b':                                       '#3498db',
    'summary_qwen3_8b':                                         '#9b59b6',
    'summary_yandex_YandexGPT-5-Lite-8B-instruct-GGUF_latest':  '#e74c3c',
}

# ------
#  LLMs
# ------
SHORT_LABELS = [
    'GigaChat3',
    'Qwen2.5',
    'Qwen3',
    'YandexGPT-5'
]

INTER_LABELS = {
    "IlyaGusev/mbart_ru_sum_gazeta":                "mBART",
    "IlyaGusev/rut5_base_sum_gazeta":               "ruT5",
    "IlyaGusev/saiga_llama3_8b":                    "Saiga (llama)",
    "IlyaGusev/saiga_nemo_12b":                     "Saiga (nemo)",
    "Qwen/Qwen2.5-7B-Instruct":                     "Qwen 2.5",
    "Qwen/Qwen3-8B":                                "Qwen 3",
    "ai-sage/GigaChat3-10B-A1.8B-bf16":             "GigaChat 3",
    "csebuetnlp/mT5_multilingual_XLSum":            "mT5",
    "lexrank":                                      "LexRank",
    "textrank":                                     "TextRank",
    "utrobinmv/t5_summary_en_ru_zh_base_2048":      "T5",
    "yandex/YandexGPT-5-Lite-8B-instruct":          "YandexGPT 5",
}



# Исходные данные (df_diagnosed, df_raw) уже существуют
# Задаём фиксированные параметры (кроме h)
ALPHA = 0.45
BETA = 0.25
GAMMA = 0.15
DELTA = 0.15