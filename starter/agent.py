from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]

def _normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def _extract_constraints(message: str) -> list[str]:
    patterns = [
        "a key requirement is:",
        "for that, what matters is:",
        "what i need is:",
    ]

    lowered = message.lower()

    for pattern in patterns:
        position = lowered.find(pattern)

        if position != -1:
            value = message[position + len(pattern):]

            return [
                item.strip(" .")
                for item in value.split(";")
                if item.strip(" .")
            ]

    return []


class Agent:
    """Editable weak baseline: stateless BM25 retrieval with no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self.sessions = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions[session_id] = {"user_profile": user_profile, "history": [], "constraints": []}

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")

        # add memory to the session state to save user's message
        state = self._sessions[session_id]

        state["history"].append(user_message)

        new_constraints = _extract_constraints(user_message)

        for constraint in new_constraints:
            if constraint not in state["constraints"]:
                state["constraints"].append(constraint)

        query = " ".join(state["history"])

        unique_terms = list(dict.fromkeys(_terms(query)))[:40]

        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            recommendations: list[dict] = []
        else:
            candidate_k = max(100, top_k * 10)

            rows = self.connection.execute(
                "SELECT parent_asin, title, categories, features, "
                "details, store, description "
                "FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) "
                "LIMIT ?",
                (expression, candidate_k),
            ).fetchall()

            normalized_constraints = [
                _normalize(constraint)
                for constraint in state["constraints"]
            ]

            scored_rows = []

            for bm25_rank, row in enumerate(rows):

                product_text = _normalize(
                    " ".join(
                        str(value or "")
                        for value in row[1:]
                    )
                )

                matched_constraints = [
                    constraint
                    for constraint in normalized_constraints
                    if constraint in product_text
                ]

                # How many full constraints does this product satisfy?
                coverage = len(matched_constraints)

                # Prefer matching more specific / longer constraints
                specificity = sum(
                    len(constraint.split())
                    for constraint in matched_constraints
                )

                scored_rows.append(
                    (
                        coverage,
                        specificity,
                        bm25_rank,
                        row,
                    )
                )

                scored_rows.sort(
                    key=lambda item: (
                        -item[0],   # more constraints matched
                        -item[1],   # more specific matches
                        item[2],    # preserve BM25 when tied
                    )
)


            recommendations = [
                {"parent_asin": str(item[3][0])}
                for item in scored_rows[:top_k]
            ]

        # if turn <= 2:
        #     message = "What other requirements or preferences do you prefer?"
        #     ask_attribute = "other"
        # else:
        #     message = "Here are the closest matches I found."
        #     ask_attribute = None

        # if turn <= 1:
        #     message = "What other requirements or preferences do you prefer?"
        #     ask_attribute = "other"
        # else:
        #     message = "Here are the closest matches I found."
        #     ask_attribute = None

        if turn <= 3:
            message = "What other requirements or preferences do you prefer?"
            ask_attribute = "other"
        else:
            message = "Here are the closest matches I found."
            ask_attribute = None
        

            
        
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
