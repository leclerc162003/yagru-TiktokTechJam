import json
from collections import Counter
import re

CATALOG = "data/catalog.jsonl"

colors = Counter()
materials = Counter()

COLOR_KEYS = {
    "color",
    "color name",
    "stone color",
    "band color",
    "lens color",
}

MATERIAL_KEYS = {
    "material",
    "material type",
    "fabric type",
    "metal type",
    "outer material",
    "inner material",
    "frame material",
    "handle material",
    "shaft material",
    "band material type",
    "sole material",
    "material composition",
    "material_composition",
}

def split_materials(value: str) -> list[str]:

    value = value.lower()

    # remove percentages
    value = re.sub(r"\d+(?:\.\d+)?%", "", value)

    # split on comma, slash, semicolon
    parts = re.split(r"[,;/]", value)

    result = []

    for part in parts:

        part = part.strip()

        if part:
            result.append(part)

    return result

def split_colors(value: str) -> list[str]:
    value = value.lower()

    # split only on separators
    parts = re.split(r"[,;/]", value)

    result = []

    for part in parts:
        part = part.strip()

        if part:
            result.append(part)

    return result


with open(CATALOG, encoding="utf-8") as f:
    for line in f:
        product = json.loads(line)

        details = product.get("details", {})

        if not isinstance(details, dict):
            continue

        for key, value in details.items():

            key_lower = key.lower().strip()

            # ---------- COLOR ----------
            if key_lower in COLOR_KEYS:
                for color in split_colors(str(value)):
                    colors[color] += 1

            # ---------- MATERIAL ----------
            if key_lower in MATERIAL_KEYS:
                for material in split_materials(str(value)):
                    materials[material] += 1


# print("COLORS")
# for value, count in colors.most_common(100):
#     print(count, value)

# print("\nMATERIALS")
# for value, count in materials.most_common(100):
#     print(count, value)


def product_text(product):

    parts = []

    parts.append(
        str(product.get("title", ""))
    )

    for feature in product.get("features", []):
        parts.append(str(feature))

    for description in product.get("description", []):
        parts.append(str(description))

    details = product.get("details", {})

    if isinstance(details, dict):
        for value in details.values():
            parts.append(str(value))

    return " ".join(parts).lower()

text = product_text(product)

found_materials = []

for material in materials:
    if material in text:
        found_materials.append(material)

found_colors = []

for color in colors:
    if color in text:
        found_colors.append(color)

import json

with open("data/colors.json", "w", encoding="utf-8") as f:
    json.dump(sorted(colors.keys()), f, indent=2)

with open("data/materials.json", "w", encoding="utf-8") as f:
    json.dump(sorted(materials.keys()), f, indent=2)