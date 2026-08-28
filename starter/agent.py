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


def _is_override(message: str) -> bool:
    text = message.lower()

    override_markers = [
        "actually",
        "instead",
        "changed my mind",
        "change of plan",
        "forget that",
        "forget about",
        "ignore my earlier",
        "no longer",
    ]

    return any(marker in text for marker in override_markers)

def _contains_term(text: str, term: str) -> bool:
    pattern = rf"\b{re.escape(term)}\b"
    return re.search(pattern, text, re.I) is not None

def _clean_vocab(values: list[str]) -> list[str]:
    cleaned = set()

    for value in values:
        value = _normalize(str(value))

        if not value:
            continue

        if value in STOPWORDS:
            continue

        if len(value) < 3:
            continue

        # Avoid giant accidental descriptions
        if len(value.split()) > 6:
            continue

        cleaned.add(value)

    # Longer terms first:
    # "navy blue" before "blue"
    return sorted(
        cleaned,
        key=lambda x: (-len(x.split()), -len(x))
    )

class Agent:
    """Editable weak baseline: stateless BM25 retrieval with no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self.sessions = {}
        with open("data/colors.json", encoding="utf-8") as f:
            self.colors = _clean_vocab(json.load(f))

        with open("data/materials.json", encoding="utf-8") as f:
            self.materials = _clean_vocab(json.load(f))
        self._build_index()

    def _extract_attributes(self, message: str) -> dict:
        text = _normalize(message)
        attributes = {}

        found_colors = [
            color
            for color in self.colors
            if _contains_term(text, color)
        ]

        found_materials = [
            material
            for material in self.materials
            if _contains_term(text, material)
        ]

        if found_colors:
            attributes["color"] = found_colors[0]

        if found_materials:
            attributes["material"] = found_materials[0]

        return attributes

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
        self._sessions[session_id] = {"user_profile": user_profile, "history": [], "constraints": [], "attributes": {}}

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
        override_message = _is_override(user_message)

        new_attributes = self._extract_attributes(user_message)

        if override_message:
            # User explicitly said to ignore the earlier preference
            state["attributes"].clear()

        for attribute, value in new_attributes.items():
            state["attributes"][attribute] = value

        retry_message = (
            "those options are not quite right yet"
            in user_message.lower()
        )

        new_constraints = []

        if not retry_message:
            state["history"].append(user_message)

            new_constraints = _extract_constraints(user_message)

            if override_message and new_constraints:
                state["constraints"] = new_constraints.copy()

            else:
                for constraint in new_constraints:
                    if constraint not in state["constraints"]:
                        state["constraints"].append(constraint)

        profile = state["user_profile"]
        preferences = profile.get("preference_tags", [])

        query = " ".join(state["history"])

        query_terms = _terms(query)

        preference_terms = []

        for preference in preferences:
            preference_terms.extend(
                _terms(preference)
            )

        all_terms = preference_terms + query_terms
        
        unique_terms = list(dict.fromkeys(all_terms))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            recommendations: list[dict] = []
        else:
            candidate_k = max(100, top_k * 10)

            rows = self.connection.execute(
                "SELECT parent_asin, title, categories, features, "
                "details, store, description, "
                "bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS bm25_score "
                "FROM products WHERE products MATCH ? "
                "ORDER BY bm25_score "
                "LIMIT ?",
                (expression, candidate_k),
            ).fetchall()

            normalized_constraints = [
                _normalize(constraint)
                for constraint in state["constraints"]
            ]

            scored_rows = []

            for row in rows:
                bm25_score = float(row[7])

                product_text = _normalize(
                    " ".join(
                        str(value or "")
                        for value in row[1:7]
                    )
                )

                matched_constraints = [
                    constraint
                    for constraint in normalized_constraints
                    if constraint in product_text
                ]

                coverage = len(matched_constraints)

                specificity = sum(
                    len(constraint.split())
                    for constraint in matched_constraints
                )

                attribute_matches = 0

                for attribute, value in state["attributes"].items():
                    normalized_value = _normalize(value)

                    if (
                        normalized_value
                        and _contains_term(
                            product_text,
                            normalized_value
                        )
                    ):
                        attribute_matches += 1

                score = -bm25_score

                # Strong reward for extracted attributes like material/color
                score += 8.0 * attribute_matches

                # Strong reward for satisfying explicit user constraints
                score += 8.0 * coverage

                # Smaller reward for matching more detailed constraints
                score += 1.5 * specificity

                scored_rows.append(
                    (
                        score,
                        row,
                    )
                )

            scored_rows.sort(
                key=lambda item: -item[0]
            )


            recommendations = [
                {
                    "parent_asin": str(item[1][0]),
                    "score": round(item[0], 6),
                }
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


        # if turn <= 3:
        #     message = "What other requirements or preferences do you prefer?"
        #     ask_attribute = "other"
        # else:
        #     message = "Here are the closest matches I found."
        #     ask_attribute = None
        
        # current_terms = _terms(user_message)
        # if retry_message:
        #     message = "Can you tell me one more specific product attribute you care about?"
        #     ask_attribute = "TODO"
        # elif len(current_terms) <= 1 and not state["constraints"]:
        #     message = "What will you use it for, and what features matter most?"
        #     ask_attribute = "TODO"
        # else:


        message = (
            "Here are the closest matches I found. "
            "What other requirement matters to you?"
        )

        ask_attribute = "other"
        
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
