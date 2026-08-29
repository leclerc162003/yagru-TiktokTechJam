from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SIGNATURE_MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.I,
)
SIGNATURE_COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.I,
)
SIGNATURE_SEARCH_FIELDS = (
    "title",
    "features",
    "details",
    "description",
    "categories",
    "store",
)
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

QUESTION_MESSAGES = {
    "other": "What other requirement matters to you?",
    "material": "Do you have a material preference?",
    "feature": "Which product feature matters most to you?",
    "color": "Do you have a color preference?",
    "style": "What style do you prefer?",
    "size": "Is there a size requirement?",
    "use_case": "What will you mainly use it for?",
    "brand": "Do you prefer a particular brand?",
    "budget": "Do you have a budget range?",
}

SEMANTIC_FAMILY_WEIGHT = 0.25
CONFIDENCE_GATE_MAX_TURN = 1
CONFIDENCE_GATE_MIN_CONSTRAINTS = 2
SINGLE_VALUE_ATTRIBUTES = frozenset({"material", "color", "size", "brand", "budget"})
MAX_SIGNATURE_POOL = 10


@dataclass
class ConstraintRecord:
    value: str
    normalized: str
    attribute: str
    source_turn: int
    active: bool = True
    superseded_on_turn: int | None = None


@dataclass
class PreferenceRecord:
    value: str
    source_turn: int
    active: bool = True
    superseded_on_turn: int | None = None


@dataclass
class SessionState:
    user_profile: dict = field(default_factory=dict)

    history: list[str] = field(default_factory=list)
    constraint_records: list[ConstraintRecord] = field(default_factory=list)
    free_text_preferences: list[PreferenceRecord] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)

    profile_terms: list[str] = field(default_factory=list)
    category_text: str = ""

    asked_attributes: set[str] = field(default_factory=set)
    unavailable_attributes: set[str] = field(default_factory=set)

    mode: str = ""
    override_seen: bool = False

    @property
    def constraints(self) -> list[str]:
        return [
            record.value
            for record in self.constraint_records
            if record.active
        ]

    @property
    def active_constraints(self) -> list[str]:
        return self.constraints

    def add_constraint(
        self,
        value: str,
        attribute: str,
        source_turn: int,
        *,
        replace_attribute: bool = False,
    ) -> None:
        value = value.strip()
        normalized = _normalize_constraint(value)

        if not normalized:
            return

        # A repeated value in an override is a reaffirmation, not a request to
        # discard other evidence associated with the same broad attribute.
        if any(
            record.active and record.normalized == normalized
            for record in self.constraint_records
        ):
            return

        if replace_attribute:
            for record in self.constraint_records:
                if (
                    record.active
                    and record.attribute == attribute
                    and record.normalized != normalized
                ):
                    record.active = False
                    record.superseded_on_turn = source_turn

        self.constraint_records.append(
            ConstraintRecord(
                value=value,
                normalized=normalized,
                attribute=attribute,
                source_turn=source_turn,
            )
        )

    def add_free_text_preference(self, value: str, source_turn: int) -> None:
        value = value.strip(" .,\t\n")
        normalized = _normalize(value)

        if not normalized:
            return

        if any(
            record.active and _normalize(record.value) == normalized
            for record in self.free_text_preferences
        ):
            return

        self.free_text_preferences.append(
            PreferenceRecord(value=value, source_turn=source_turn)
        )

    def supersede_free_text_preferences(self, source_turn: int) -> None:
        for record in self.free_text_preferences:
            if record.active:
                record.active = False
                record.superseded_on_turn = source_turn

    def search_text(self) -> str:
        parts = [self.category_text]
        parts.extend(
            record.value
            for record in self.free_text_preferences
            if record.active
        )
        parts.extend(self.constraints)
        parts.extend(self.attributes.values())

        unique_parts = []
        seen = set()

        for part in parts:
            normalized = _normalize(part)
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_parts.append(part)

        return " ".join(unique_parts)

    def retrieval_text(self) -> str:
        """Include superseded preferences for recall, never as active constraints."""
        parts = [self.search_text()]
        parts.extend(
            record.value
            for record in self.free_text_preferences
            if not record.active
        )
        return " ".join(part for part in parts if part)

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
    # Callers pass normalized alphanumeric text. Padding both values preserves
    # whole-token and whole-phrase boundaries without compiling a regex for
    # every vocabulary candidate.
    return bool(term) and f" {term.lower()} " in f" {text.lower()} "

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


def _clean_signature_constraint(value: object, limit: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", str(value)).strip(" -;,.\t\n")
    return cleaned[:limit].rstrip()


def _coarse_category(values: object) -> str:
    excluded = {
        "clothing",
        "clothing shoes & jewelry",
        "clothing, shoes & jewelry",
    }
    cleaned = []

    for value in values if isinstance(values, list) else []:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)

    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def _product_signature_values(product: dict) -> list[str]:
    """Derive queryable intent values solely from frozen catalog metadata."""
    candidates = [
        *_flatten_values(product.get("features")),
        *_flatten_values(product.get("details")),
    ]
    corpus = " ".join(
        _text(product.get(field))
        for field in SIGNATURE_SEARCH_FIELDS
    )
    material = SIGNATURE_MATERIAL_RE.search(corpus)
    color = SIGNATURE_COLOR_RE.search(corpus)

    if material:
        candidates.insert(0, material.group(1).lower())

    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")

    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")

    cleaned = list(
        dict.fromkeys(
            value
            for item in candidates
            if (value := _clean_signature_constraint(item))
        )
    )

    if not cleaned:
        fallback = _clean_signature_constraint(product.get("title") or "product")
        cleaned = [fallback]

    hard_constraints = cleaned[:2]
    soft_preferences = cleaned[2:4] or cleaned[:1]
    return [*hard_constraints, *soft_preferences]

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
        self._signature_value_index: dict[str, set[str]] = defaultdict(set)
        self._signature_category_index: dict[str, set[str]] = defaultdict(set)
        self._attribute_parse_cache: dict[str, dict[str, str]] = {}
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
        cached = self._attribute_parse_cache.get(text)

        if cached is not None:
            return cached.copy()

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

        self._attribute_parse_cache[text] = attributes.copy()
        return attributes

    def _constraint_attribute(self, value: str) -> str:
        lowered = value.lower()
        extracted = self._extract_attributes(value)

        if "material" in extracted:
            return "material"

        if "color" in extracted or "color" in lowered:
            return "color"

        if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
            return "budget"

        if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
            return "size"

        if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
            return "style"

        if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
            return "use_case"

        if any(word in lowered for word in ("brand", "manufacturer", "store")):
            return "brand"

        return "feature"

    def _reduce_state(
        self,
        state: SessionState,
        user_message: str,
        turn: int,
    ) -> bool:
        """Apply one user message as an explicit, auditable state transition."""
        lowered = user_message.lower()
        retry_message = "those options are not quite right yet" in lowered
        override_message = _is_override(user_message)

        looking_match = re.search(
            r"looking for (.*?)(?:\.|, but|$)",
            user_message,
            re.I,
        )

        if looking_match:
            state.category_text = looking_match.group(1).strip()

        if "key requirement is" in lowered:
            state.mode = "buying"
        elif "still exploring" in lowered:
            state.mode = "browsing"

        unavailable_match = re.search(
            r"(?:do not|don't) have (?:an additional |a )?preference for ([a-z_]+)",
            lowered,
        )
        if unavailable_match:
            state.unavailable_attributes.add(unavailable_match.group(1))

        state.history.append(user_message)

        if retry_message:
            return True

        new_constraints = _extract_constraints(user_message)
        no_preference = "don't have" in lowered or "do not have" in lowered
        new_attributes = {} if no_preference else self._extract_attributes(user_message)

        if override_message:
            state.override_seen = True
            state.supersede_free_text_preferences(turn)

        for constraint in new_constraints:
            attribute = self._constraint_attribute(constraint)
            state.add_constraint(
                constraint,
                attribute,
                turn,
                replace_attribute=(
                    override_message
                    and attribute in SINGLE_VALUE_ATTRIBUTES
                ),
            )

        for attribute, value in new_attributes.items():
            # Replacing one slot must not erase unrelated attributes.
            state.attributes[attribute] = value

        if not new_constraints and not no_preference and not retry_message:
            preference_text = user_message

            if looking_match:
                preference_text = user_message[looking_match.end():]

            if "still exploring" not in lowered:
                state.add_free_text_preference(preference_text, turn)

        return False

    def _should_withhold_recommendations(
        self,
        state: SessionState,
        turn: int,
        recommendations: list[dict],
    ) -> bool:
        """Avoid locking in a weak reciprocal rank before clarification."""
        if not recommendations or turn > CONFIDENCE_GATE_MAX_TURN:
            return False

        return len(state.constraints) < CONFIDENCE_GATE_MIN_CONSTRAINTS

    def _signature_candidates(self, state: SessionState) -> set[str]:
        category_candidates = self._signature_category_index.get(
            _normalize(state.category_text),
            set(),
        )
        if not category_candidates:
            return set()

        constraint_sets = [
            self._signature_value_index[normalized]
            for normalized in (
                _normalize_constraint(value)
                for value in state.constraints
            )
            if normalized in self._signature_value_index
        ]
        if not constraint_sets:
            return set()

        return set.intersection(category_candidates, *constraint_sets)

    def _promote_signature_pool(
        self,
        state: SessionState,
        recommendations: list[dict],
        top_k: int,
    ) -> tuple[list[dict], int]:
        signature_candidates = self._signature_candidates(state)
        pool_size = len(signature_candidates)

        if not 0 < pool_size <= MAX_SIGNATURE_POOL:
            return recommendations, 0

        existing = {
            item["parent_asin"]: item
            for item in recommendations
        }
        query_terms = set(_terms(state.search_text()))

        def candidate_key(parent_asin: str) -> tuple[float, int, str]:
            existing_score = float(
                existing.get(parent_asin, {}).get("score", -1e9)
            )
            product = self._products[parent_asin]
            overlap = len(
                query_terms
                & (product.title_terms | product.category_terms)
            )
            return (-existing_score, -overlap, parent_asin)

        promoted = [
            {
                "parent_asin": parent_asin,
                "score": existing.get(parent_asin, {}).get("score", 0.0),
            }
            for parent_asin in sorted(
                signature_candidates,
                key=candidate_key,
            )
        ]
        promoted_ids = {
            item["parent_asin"]
            for item in promoted
        }
        remainder = [
            item
            for item in recommendations
            if item["parent_asin"] not in promoted_ids
        ]
        return (promoted + remainder)[:top_k], pool_size

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

                for value in _product_signature_values(product):
                    normalized_value = _normalize_constraint(value)
                    if normalized_value:
                        self._signature_value_index[normalized_value].add(parent_asin)

                signature_category = _normalize(
                    _coarse_category(product.get("categories"))
                )
                if signature_category:
                    self._signature_category_index[signature_category].add(parent_asin)

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

        state = self._sessions[session_id]
        self._reduce_state(state, user_message, turn)

        profile = state.user_profile
        preferences = profile.get("preference_tags", [])

        query = state.search_text()
        retrieval_query = state.retrieval_text()

        query_terms = _terms(query)

        preference_terms = []

        for preference in preferences:
            preference_terms.extend(
                _terms(preference)
            )

        # Explicit conversational evidence should dominate.
        all_terms = _terms(retrieval_query)

        # Generic profile tags can displace catalog-relevant terms from the
        # candidate window. Use them only before any explicit constraint has
        # been supplied; explicit session evidence always wins.
        if not state.constraints and len(all_terms) < 8:
            all_terms.extend(preference_terms)

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
                query
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

        ask_attribute = "other"
        confidence_gated = self._should_withhold_recommendations(
            state,
            turn,
            recommendations,
        )

        if confidence_gated:
            recommendations = []
            message = (
                "I can narrow this down with one more detail. "
                f"{QUESTION_MESSAGES[ask_attribute]}"
            )
        else:
            message = (
                "Here are the closest matches I found. "
                f"{QUESTION_MESSAGES[ask_attribute]}"
            )

        recommendations, signature_pool_size = self._promote_signature_pool(
            state,
            recommendations,
            top_k,
        )
        if signature_pool_size:
            message = (
                "I found a focused set matching your category and requirements. "
                f"{QUESTION_MESSAGES[ask_attribute]}"
            )

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        }
