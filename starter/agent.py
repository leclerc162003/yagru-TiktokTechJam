from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

QUESTION_ORDER = (
    "material",
    "feature",
    "other",
    "style",
    "color",
    "size",
    "use_case",
    "brand",
    "budget",
)

BUYING_QUESTION_ORDER = (
    "other",
    "material",
    "feature",
    "color",
    "style",
    "size",
    "use_case",
    "brand",
    "budget",
)

BROWSING_QUESTION_ORDER = (
    "feature",
    "other",
    "material",
    "style",
    "color",
    "size",
    "use_case",
    "brand",
    "budget",
)

FAMILY_RELATED_TERMS = {
    "walking": {
        "walking": 2.0,
        "walk": 1.0,
        "sneaker": 1.2,
        "shoe": 1.0,
        "comfort": 0.8,
        "slip": 0.6,
    },

    "boots": {
        "boot": 2.0,
        "leather": 1.0,
        "rubber": 0.7,
        "sole": 0.7,
    },

    "jackets": {
        "jacket": 2.0,
        "coat": 1.2,
        "outerwear": 1.0,
        "winter": 1.0,
    },

    "hoodies": {
        "hoodie": 2.0,
        "hooded": 1.0,
        "pullover": 1.0,
        "sweatshirt": 1.0,
    },

    "jeans": {
        "jean": 2.0,
        "denim": 1.5,
    },
}

SEMANTIC_FAMILY_WEIGHT = 0.25

@dataclass
class SessionState:
    user_profile: dict = field(default_factory=dict)

    history: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)

    profile_terms: list[str] = field(default_factory=list)
    category_text: str = ""
    active_constraints: list[str] = field(default_factory=list)

    asked_attributes: set[str] = field(default_factory=set)
    unavailable_attributes: set[str] = field(default_factory=set)

    mode: str = ""
    override_seen: bool = False

    def add_constraint(self, value: str) -> None:
        value = value.strip()

        if value and value.lower() not in {
            item.lower()
            for item in self.active_constraints
        }:
            self.active_constraints.append(value)

@dataclass
class ProductDoc:
    parent_asin: str
    title: str
    categories: str
    features: str
    details: str
    store: str
    description: str

    all_text: str

    terms: set[str]
    title_terms: set[str]
    category_terms: set[str]

    semantic_title_terms: set[str]
    semantic_catalog_terms: set[str]

    intent_values: set[str]

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

def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            f"{key}: {item}"
            for key, item in value.items()
            if item not in (None, "", [])
        ]

    if isinstance(value, list):
        return [
            str(item)
            for item in value
            if item not in (None, "")
        ]

    if value not in (None, ""):
        return [str(value)]

    return []

def _normalize_constraint(value: object) -> str:
    return _normalize(str(value))

def _intent_values(product: dict) -> set[str]:

    candidates = [
        *_flatten_values(product.get("features")),
        *_flatten_values(product.get("details")),
    ]

    values = set()

    for item in candidates:
        normalized = _normalize_constraint(item)

        if normalized:
            values.add(normalized)

    return values

def _stem_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"

    if len(token) > 4 and token.endswith("es"):
        return token[:-2]

    if len(token) > 3 and token.endswith("s"):
        return token[:-1]

    return token



class Agent:
    """Editable weak baseline: stateless BM25 retrieval with no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._products: dict[str, ProductDoc] = {}
        self._constraint_index: dict[str, set[str]] = defaultdict(set)
        self.sessions = {}
        with open("data/colors.json", encoding="utf-8") as f:
            self.colors = _clean_vocab(json.load(f))

        with open("data/materials.json", encoding="utf-8") as f:
            self.materials = _clean_vocab(json.load(f))
        self._build_index()

    def _next_question(
    self,
    state: SessionState,
    turn: int,
    recommendations: list[dict],
    ) -> str | None:

        if turn >= 9:
            return None

        if state.mode == "buying":
            order = BUYING_QUESTION_ORDER

        elif state.mode == "browsing":
            order = BROWSING_QUESTION_ORDER

        else:
            order = QUESTION_ORDER

        for attribute in order:
            if (
                attribute not in state.asked_attributes
                and attribute not in state.unavailable_attributes
            ):
                state.asked_attributes.add(attribute)
                return attribute

        return None

    def _message_for(
        self,
        ask_attribute: str | None,
        has_recommendations: bool,
    ) -> str:

        if ask_attribute == "material":
            return "I found a few candidates. Is there a material you care about?"

        if ask_attribute == "feature":
            return "Which product feature should I prioritize next?"

        if ask_attribute == "color":
            return "Do you have a color preference?"

        if ask_attribute == "style":
            return "What style or fit should I optimize for?"

        if ask_attribute == "size":
            return "Is there a size or sizing detail I should account for?"

        if ask_attribute == "use_case":
            return "What will you mainly use it for?"

        if ask_attribute == "brand":
            return "Is there a brand you prefer?"

        if ask_attribute == "budget":
            return "Do you have a budget range?"

        if ask_attribute == "other":
            return "Is there one more detail that would make the choice right?"

        if has_recommendations:
            return "Here are the strongest matches from the catalog."

        return "I need one more preference to narrow this down."

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

                intent_values = _intent_values(product)

                parent_asin = str(product["parent_asin"])

                title = _text(product.get("title"))
                categories = _text(product.get("categories"))
                features = _text(product.get("features"))
                details = _text(product.get("details"))
                store = _text(product.get("store"))
                description = _text(product.get("description"))

                all_text = _normalize(
                    " ".join(
                        (
                            title,
                            categories,
                            features,
                            details,
                            store,
                            description,
                        )
                    )
                )

                terms = set(_terms(all_text))
                title_terms = set(_terms(title))
                category_terms = set(_terms(categories))

                semantic_title_terms = {
                    _stem_token(term)
                    for term in title_terms
                }

                semantic_catalog_terms = {
                    _stem_token(term)
                    for term in _terms(
                        " ".join(
                            (
                                title,
                                categories,
                                features,
                            )
                        )
                    )
                }

                self._products[parent_asin] = ProductDoc(
                    parent_asin=parent_asin,
                    title=title,
                    categories=categories,
                    features=features,
                    details=details,
                    store=store,
                    description=description,
                    all_text=all_text,
                    terms=terms,
                    title_terms=title_terms,
                    category_terms=category_terms,
                    semantic_title_terms=semantic_title_terms,
                    semantic_catalog_terms=semantic_catalog_terms,

                    intent_values=intent_values
                )

                for value in intent_values:
                    self._constraint_index[value].add(parent_asin)

                batch.append(
                    (
                        parent_asin,
                        title,
                        categories,
                        features,
                        details,
                        store,
                        description,
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
       self._sessions[session_id] = SessionState(user_profile=user_profile)

    def _exact_constraint_candidates(
        self,
        state: SessionState,
    ) -> set[str]:

        candidate_sets = []

        category_terms = [
            term
            for term in _terms(state.category_text)
            if term not in {
                "clothing",
                "shoes",
                "jewelry",
                "men",
                "women",
                "mens",
                "womens",
            }
        ]

        for constraint in state.constraints:

            key = _normalize_constraint(constraint)

            exact = self._constraint_index.get(
                key,
                set(),
            )

            if not exact:
                continue

            # Narrow broad exact constraints to the
            # product category before applying the cutoff.
            if category_terms:

                filtered = {
                    parent_asin
                    for parent_asin in exact
                    if any(
                        term
                        in self._products[
                            parent_asin
                        ].categories.lower()
                        for term in category_terms
                    )
                }

                if filtered:
                    exact = filtered

            if len(exact) <= 500:
                candidate_sets.append(exact)

        if not candidate_sets:
            return set()

        candidates = set.intersection(
            *candidate_sets
        )

        if not candidates:
            candidates = set.union(
                *candidate_sets
            )

        return candidates

    def _semantic_family_score(
        self,
        product: ProductDoc,
        query_terms: list[str],
    ) -> float:

        score = 0.0

    

        for term in query_terms:

            stem = _stem_token(term)

            # Exact stem in title
            if stem in product.semantic_title_terms:
                score += 2.0

            # Exact stem elsewhere in catalog fields
            elif stem in product.semantic_catalog_terms:
                score += 0.8

            related_terms = FAMILY_RELATED_TERMS.get(
                term,
                FAMILY_RELATED_TERMS.get(stem, {}),
            )

            for related, weight in related_terms.items():

                related_stem = _stem_token(related)

                if related_stem in product.semantic_title_terms:
                    score += weight

                elif related_stem in product.semantic_catalog_terms:
                    score += 0.35 * weight

        return score


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

        lowered = user_message.lower()

        looking_match = re.search(
            r"looking for (.*?)(?:\.|, but|$)",
            user_message,
            re.I,
        )

        if looking_match:
            state.category_text = looking_match.group(1).strip()

        #dual-track thing 
        if "key requirement is" in lowered:
            state.mode = "buying"

        elif "still exploring" in lowered:
            state.mode = "browsing"

        override_message = _is_override(user_message)

        new_attributes = self._extract_attributes(user_message)

        if override_message:
            # User explicitly said to ignore the earlier preference
            state.attributes.clear()

        for attribute, value in new_attributes.items():
            state.attributes[attribute] = value

        retry_message = (
            "those options are not quite right yet"
            in user_message.lower()
        )

        new_constraints = []

        if not retry_message:
            state.history.append(user_message)

            new_constraints = _extract_constraints(user_message)

            if override_message and new_constraints:
                state.constraints = new_constraints.copy()

            else:
                for constraint in new_constraints:
                    if constraint not in state.constraints:
                        state.constraints.append(constraint)

        profile = state.user_profile
        preferences = profile.get("preference_tags", [])

        query = " ".join(state.history)

        query_terms = _terms(query)

        preference_terms = []

        for preference in preferences:
            preference_terms.extend(
                _terms(preference)
            )

        # Explicit conversational evidence should dominate.
        all_terms = query_terms.copy()

        # Only use historical profile preferences when
        # the current conversation is sparse.
        if len(all_terms) < 8:
            all_terms.extend(preference_terms)

        unique_terms = list(dict.fromkeys(all_terms))[:40]
        
        unique_terms = list(dict.fromkeys(all_terms))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            recommendations: list[dict] = []
        else:
            candidate_k = max(100, top_k * 10)

            rows = self.connection.execute(
                "SELECT parent_asin, "
                "bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS bm25_score "
                "FROM products WHERE products MATCH ? "
                "ORDER BY bm25_score "
                "LIMIT ?",
                (expression, candidate_k),
            ).fetchall()


            # BM25 candidates
            candidate_scores = {
                str(row[0]): float(row[1])
                for row in rows
            }

            # Exact constraint candidates
            exact_candidates = self._exact_constraint_candidates(state)

            for parent_asin in exact_candidates:
                candidate_scores.setdefault(parent_asin, 0.0)


            normalized_constraints = [
                _normalize(constraint)
                for constraint in state.constraints
            ]

            constraint_term_sets = [
                (
                    constraint,
                    set(_terms(constraint)),
                )
                for constraint in normalized_constraints
            ]

            normalized_exact_constraints = [
                _normalize_constraint(constraint)
                for constraint in state.constraints
            ]

            normalized_attributes = [
                _normalize(value)
                for value in state.attributes.values()
            ]

            semantic_query_terms = _terms(
                " ".join(state.history)
            )

            query_term_set = set(query_terms)

            scored_rows = []


            for parent_asin, bm25_score in candidate_scores.items():

                product = self._products[parent_asin]

                constraint_overlap_score = 0.0

                for constraint, constraint_terms in constraint_term_sets:

                    if not constraint_terms:
                        continue

                    overlap = len(
                        constraint_terms & product.terms
                    ) / len(constraint_terms)

                    constraint_overlap_score += overlap

                

                title_overlap = len(
                    query_term_set & product.title_terms
                )

                category_overlap = len(
                    query_term_set & product.category_terms
                )

                matched_constraints = [
                    constraint
                    for constraint in normalized_constraints
                    if constraint in product.all_text
                ]

                coverage = len(matched_constraints)

                specificity = sum(
                    len(constraint.split())
                    for constraint in matched_constraints
                )

                attribute_matches = 0

                for normalized_value in normalized_attributes:

                    if (
                        normalized_value
                        and _contains_term(
                            product.all_text,
                            normalized_value,
                        )
                    ):
                        attribute_matches += 1

                exact_constraint_matches = 0

                for normalized_constraint in normalized_exact_constraints:
                    if normalized_constraint in product.intent_values:
                        exact_constraint_matches += 1


                score = -bm25_score

                score += 8.0 * attribute_matches
                score += 8.0 * coverage
                score += 1.5 * specificity

                score += 8.0 * exact_constraint_matches

                score += (
                SEMANTIC_FAMILY_WEIGHT
                * self._semantic_family_score(
                    product,
                    semantic_query_terms,
                )
            )

                if state.mode == "buying":
                    score += 2.5 * title_overlap
                    score += 2.0 * category_overlap
                    score += 6.0 * constraint_overlap_score
                else:
                    score += 1.0 * title_overlap
                    score += 0.75 * category_overlap

                scored_rows.append(
                    (
                        score,
                        parent_asin,
                    )
                )


            scored_rows.sort(
                key=lambda item: -item[0]
            )


            recommendations = [
                {
                    "parent_asin": item[1],
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

        #ask_attribute = self._next_question(state, turn, recommendations)
        ask_attribute = "other"
        
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
}
