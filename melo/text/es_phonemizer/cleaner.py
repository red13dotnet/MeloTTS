"""Set of default text cleaners for Spanish phonemization"""

import re

_whitespace_re = re.compile(r"\s+")

rep_map = {
    "：": ",", "；": ",", "，": ",", "。": ".", "！": "!", "？": "?",
    "\n": ".", "·": ",", "、": ",", "...": ".", "…": ".", "$": ".",
    "“": "'", "”": "'", "‘": "'", "’": "'", "（": "'", "）": "'",
    "(": "'", ")": "'", "《": "'", "》": "'", "【": "'", "】": "'",
    "[": "'", "]": "'", "—": "", "～": "-", "~": "-", "「": "'", "」": "'",
}


def replace_punctuation(text: str) -> str:
    pattern = re.compile("|".join(re.escape(p) for p in rep_map))
    return pattern.sub(lambda x: rep_map[x.group()], text)


def lowercase(text: str) -> str:
    return text.lower()


def collapse_whitespace(text: str) -> str:
    return re.sub(_whitespace_re, " ", text).strip()


def remove_punctuation_at_begin(text: str) -> str:
    return re.sub(r'^[,.!?]+', '', text)


def remove_aux_symbols(text: str) -> str:
    return re.sub(r'[<>()\[\]"«»\']+', "", text)


def replace_symbols(text: str, lang: str = "en") -> str:
    text = text.replace(";", ",")
    text = text.replace("-", " ") if lang != "ca" else text.replace("-", "")
    text = text.replace(":", ",")
    if lang == "en":
        text = text.replace("&", " and ")
    elif lang == "fr":
        text = text.replace("&", " et ")
    elif lang == "pt":
        text = text.replace("&", " e ")
    elif lang == "ca":
        text = text.replace("&", " i ").replace("'", "")
    elif lang == "es":
        text = text.replace("&", "y").replace("'", "")
    return text


def spanish_cleaners(text: str) -> str:
    """Basic pipeline for Spanish text before phonemization."""
    text = lowercase(text)
    text = replace_symbols(text, lang="es")
    text = replace_punctuation(text)
    text = remove_aux_symbols(text)
    text = remove_punctuation_at_begin(text)
    text = collapse_whitespace(text)
    return re.sub(r'([^.,!?\-…])$', r'\1.', text)