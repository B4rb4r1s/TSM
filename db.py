"""
Модуль для работы с БД автоматического реферирования.

Использование:
    from db import Database

    db = Database()                         # подключение с параметрами по умолчанию
    db = Database("postgresql://u:p@host/dbname")  # или произвольный DSN

    # --- Публикации ---
    pub_id = db.add_publication(title="...", full_text="...", tag="ИТ")
    db.update_clean_text(pub_id, "очищенный текст")

    # --- Ключевые слова ---
    db.add_author_keywords(pub_id, ["нейросеть", "NLP"])
    method_id = db.add_extraction_method("TF-IDF")
    db.add_extracted_keywords(pub_id, method_id, [("нейросеть", 0.85), ("токен", 0.62)])

    # --- Рефераты ---
    model_id = db.add_model("mBART", version="large-cc25")
    abs_id   = db.add_abstract(pub_id, "machine", "Текст реферата...", model_id=model_id)

    # --- Метрики ---
    db.add_metric(pub_id, abs_id, "text_vs_machine", "ROUGE-1", 0.45)
    db.add_metric(pub_id, author_abs_id, "author_vs_machine", "BERTScore", 0.78,
                  compared_with_abstract_id=abs_id)
"""

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from contextlib import contextmanager
from datetime import date
from typing import Optional


REMOTE_DSN = "postgresql://user:1234@10.36.60.78:5533/articles"
DEFAULT_DSN = "postgresql://user:1234@localhost:5534/articles"


class Database:
    def __init__(self, dsn: str = DEFAULT_DSN):
        self.dsn = dsn
        self._conn = None

    # ------------------------------------------------------------------
    # Соединение
    # ------------------------------------------------------------------
    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            try:
                self._conn = psycopg.connect(REMOTE_DSN, row_factory=dict_row)
            except:
                self._conn = psycopg.connect(self.dsn, row_factory=dict_row)
        return self._conn

    @contextmanager
    def cursor(self, commit: bool = True):
        cur = self.conn.cursor()
        try:
            yield cur
            if commit:
                self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ------------------------------------------------------------------
    # Публикации
    # ------------------------------------------------------------------
    def add_publication(
        self,
        title: str,
        full_text: str,
        clean_text: Optional[str] = None,
        tag: Optional[str] = None,
        source: Optional[str] = None,
    ) -> int:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO publications
                    (title, full_text, clean_text, tag, source)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (title, full_text, clean_text, tag, source),
            )
            return cur.fetchone()["id"]

    def update_clean_text(self, publication_id: int, clean_text: str):
        with self.cursor() as cur:
            cur.execute(
                "UPDATE publications SET clean_text = %s WHERE id = %s",
                (clean_text, publication_id),
            )

    def get_publication(self, publication_id: int) -> Optional[dict]:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM publications WHERE id = %s", (publication_id,))
            return cur.fetchone()

    def iter_publications_with_abstracts(self, source: str, only_dirty: bool = True):
        """Итератор: публикации с авторскими рефератами для очистки.

        Параметры
        ----------
        source     : значение поля source ('17k', '480', ...).
        only_dirty : если True, выбирать только записи с clean_text IS NULL.

        Yields
        ------
        dict с ключами: id, full_text, author_abstract (str | None)
        """
        where = "p.source = %s"
        if only_dirty:
            where += " AND p.clean_text IS NULL"

        with self.cursor(commit=False) as cur:
            cur.execute(
                f"""
                SELECT p.id, p.full_text, a.text AS author_abstract
                FROM publications p
                LEFT JOIN abstracts a
                       ON a.publication_id = p.id AND a.abstract_type = 'author'
                WHERE {where}
                ORDER BY p.id
                """,
                (source,),
            )
            yield from cur

    def iter_publications_filtered(
        self,
        source: Optional[str] = None,
        tag: Optional[str] = None,
    ):
        """Итератор публикаций с фильтрацией по source и/или tag.

        Yields
        ------
        dict с ключами: id, title, full_text, clean_text, tag, source
        """
        conditions = []
        params: list = []
        if source is not None:
            conditions.append("source = %s")
            params.append(source)
        if tag is not None:
            conditions.append("tag = %s")
            params.append(tag)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        with self.cursor(commit=False) as cur:
            cur.execute(
                f"SELECT * FROM publications {where} ORDER BY id",
                params,
            )
            yield from cur

    def count_publications_filtered(
        self,
        source: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> int:
        """Количество публикаций, подходящих под фильтры."""
        conditions = []
        params: list = []
        if source is not None:
            conditions.append("source = %s")
            params.append(source)
        if tag is not None:
            conditions.append("tag = %s")
            params.append(tag)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        with self.cursor(commit=False) as cur:
            cur.execute(f"SELECT count(*) AS cnt FROM publications {where}", params)
            return cur.fetchone()["cnt"]

    def count_publications_with_extracted_keywords(
        self,
        extraction_method_id: int,
        source: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> int:
        """Количество публикаций, у которых уже есть extracted-KW для данного метода."""
        conditions = ["pk.source = 'extracted'", "pk.extraction_method_id = %s"]
        params: list = [extraction_method_id]
        if source is not None:
            conditions.append("p.source = %s")
            params.append(source)
        if tag is not None:
            conditions.append("p.tag = %s")
            params.append(tag)

        where = " AND ".join(conditions)

        with self.cursor(commit=False) as cur:
            cur.execute(
                f"""
                SELECT count(DISTINCT p.id) AS cnt
                FROM publications p
                JOIN publication_keywords pk ON pk.publication_id = p.id
                WHERE {where}
                """,
                params,
            )
            return cur.fetchone()["cnt"]

    def delete_extracted_keywords(
        self, publication_id: int, extraction_method_id: int
    ):
        """Удалить extracted-КС публикации для конкретного метода."""
        with self.cursor() as cur:
            cur.execute(
                """
                DELETE FROM publication_keywords
                WHERE publication_id = %s
                  AND source = 'extracted'
                  AND extraction_method_id = %s
                """,
                (publication_id, extraction_method_id),
            )

    # ------------------------------------------------------------------
    # Ключевые слова
    # ------------------------------------------------------------------
    def _get_or_create_keyword(self, cur, word: str) -> int:
        word = word.strip().lower()
        cur.execute(
            """
            INSERT INTO keywords (word) VALUES (%s)
            ON CONFLICT (word) DO UPDATE SET word = EXCLUDED.word
            RETURNING id
            """,
            (word,),
        )
        return cur.fetchone()["id"]

    def add_author_keywords(self, publication_id: int, words: list[str]):
        with self.cursor() as cur:
            for w in words:
                kw_id = self._get_or_create_keyword(cur, w)
                cur.execute(
                    """
                    INSERT INTO publication_keywords
                        (publication_id, keyword_id, source)
                    VALUES (%s, %s, 'author')
                    ON CONFLICT DO NOTHING
                    """,
                    (publication_id, kw_id),
                )

    def add_extraction_method(self, name: str, description: str = None) -> int:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO extraction_methods (name, description)
                VALUES (%s, %s)
                ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
                """,
                (name, description),
            )
            return cur.fetchone()["id"]

    def add_extracted_keywords(
        self,
        publication_id: int,
        extraction_method_id: int,
        keywords_with_weights: list[tuple[str, float]],
    ):
        with self.cursor() as cur:
            for word, weight in keywords_with_weights:
                kw_id = self._get_or_create_keyword(cur, word)
                cur.execute(
                    """
                    INSERT INTO publication_keywords
                        (publication_id, keyword_id, source, weight, extraction_method_id)
                    VALUES (%s, %s, 'extracted', %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (publication_id, kw_id, weight, extraction_method_id),
                )

    def get_keywords(self, publication_id: int, source: str = None) -> list[dict]:
        with self.cursor() as cur:
            if source:
                cur.execute(
                    """
                    SELECT k.word, pk.source, pk.weight, em.name AS method
                    FROM publication_keywords pk
                    JOIN keywords k ON k.id = pk.keyword_id
                    LEFT JOIN extraction_methods em ON em.id = pk.extraction_method_id
                    WHERE pk.publication_id = %s AND pk.source = %s
                    ORDER BY pk.weight DESC NULLS LAST
                    """,
                    (publication_id, source),
                )
            else:
                cur.execute(
                    """
                    SELECT k.word, pk.source, pk.weight, em.name AS method
                    FROM publication_keywords pk
                    JOIN keywords k ON k.id = pk.keyword_id
                    LEFT JOIN extraction_methods em ON em.id = pk.extraction_method_id
                    WHERE pk.publication_id = %s
                    ORDER BY pk.source, pk.weight DESC NULLS LAST
                    """,
                    (publication_id,),
                )
            return cur.fetchall()

    # ------------------------------------------------------------------
    # Модели
    # ------------------------------------------------------------------
    def add_model(
        self,
        name: str,
        version: str = None,
        description: str = None,
        parameters: dict = None,
    ) -> int:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO models (name, version, description, parameters)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (name, version) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
                """,
                (name, version, description, Jsonb(parameters) if parameters else None),
            )
            return cur.fetchone()["id"]

    # ------------------------------------------------------------------
    # Рефераты
    # ------------------------------------------------------------------
    def add_abstract(
        self,
        publication_id: int,
        abstract_type: str,
        text: str,
        model_id: int = None,
        keywords_config: dict = None,
    ) -> int:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO abstracts
                    (publication_id, abstract_type, text, model_id, keywords_config)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    publication_id,
                    abstract_type,
                    text,
                    model_id,
                    Jsonb(keywords_config) if keywords_config else None,
                ),
            )
            return cur.fetchone()["id"]

    def get_abstracts(
        self, publication_id: int, abstract_type: str = None
    ) -> list[dict]:
        with self.cursor() as cur:
            if abstract_type:
                cur.execute(
                    """
                    SELECT a.*, m.name AS model_name, m.version AS model_version
                    FROM abstracts a
                    LEFT JOIN models m ON m.id = a.model_id
                    WHERE a.publication_id = %s AND a.abstract_type = %s
                    ORDER BY a.created_at
                    """,
                    (publication_id, abstract_type),
                )
            else:
                cur.execute(
                    """
                    SELECT a.*, m.name AS model_name, m.version AS model_version
                    FROM abstracts a
                    LEFT JOIN models m ON m.id = a.model_id
                    WHERE a.publication_id = %s
                    ORDER BY a.abstract_type, a.created_at
                    """,
                    (publication_id,),
                )
            return cur.fetchall()

    # ------------------------------------------------------------------
    # Метрики сходства
    # ------------------------------------------------------------------
    def add_metric(
        self,
        publication_id: int,
        abstract_id: int,
        comparison_type: str,
        metric_name: str,
        metric_value: float,
        compared_with_abstract_id: int = None,
    ) -> int:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO similarity_metrics
                    (publication_id, abstract_id, comparison_type,
                     compared_with_abstract_id, metric_name, metric_value)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (publication_id, abstract_id, comparison_type,
                             compared_with_abstract_id, metric_name)
                DO UPDATE SET metric_value = EXCLUDED.metric_value,
                             calculated_at = CURRENT_TIMESTAMP
                RETURNING id
                """,
                (
                    publication_id,
                    abstract_id,
                    comparison_type,
                    compared_with_abstract_id,
                    metric_name,
                    metric_value,
                ),
            )
            return cur.fetchone()["id"]

    def get_metrics(
        self, publication_id: int, comparison_type: str = None
    ) -> list[dict]:
        with self.cursor() as cur:
            if comparison_type:
                cur.execute(
                    """
                    SELECT sm.*, a1.abstract_type AS abstract_type_a,
                           a2.abstract_type AS abstract_type_b
                    FROM similarity_metrics sm
                    JOIN abstracts a1 ON a1.id = sm.abstract_id
                    LEFT JOIN abstracts a2 ON a2.id = sm.compared_with_abstract_id
                    WHERE sm.publication_id = %s AND sm.comparison_type = %s
                    ORDER BY sm.metric_name
                    """,
                    (publication_id, comparison_type),
                )
            else:
                cur.execute(
                    """
                    SELECT sm.*, a1.abstract_type AS abstract_type_a,
                           a2.abstract_type AS abstract_type_b
                    FROM similarity_metrics sm
                    JOIN abstracts a1 ON a1.id = sm.abstract_id
                    LEFT JOIN abstracts a2 ON a2.id = sm.compared_with_abstract_id
                    WHERE sm.publication_id = %s
                    ORDER BY sm.comparison_type, sm.metric_name
                    """,
                    (publication_id,),
                )
            return cur.fetchall()
