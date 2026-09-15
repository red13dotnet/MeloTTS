"""Set of default text cleaners for French phonemization"""

import re

try:
    from .french_abbreviations import abbreviations_fr
except (ImportError, ValueError):
    from french_abbreviations import abbreviations_fr

_whitespace_re = re.compile(r"\s+")

rep_map = {
    "：": ",", "；": ",", "，": ",", "。": ".", "！": "!", "？": "?",
    "\n": ".", "·": ",", "、": ",", "...": ".", "…": ".", "$": ".",
    "“": "", "”": "", "‘": "", "’": "", "（": "", "）": "", "(": "",
    ")": "", "《": "", "》": "", "【": "", "】": "", "[": "", "]": "",
    "—": "", "～": "-", "~": "-", "「": "", "」": "", "¿": "", "¡": "",
}


def replace_punctuation(text: str) -> str:
    pattern = re.compile("|".join(re.escape(p) for p in rep_map))
    return pattern.sub(lambda x: rep_map[x.group()], text)


def expand_abbreviations(text: str, lang: str = "fr") -> str:
    if lang == "fr":
        _abbreviations = abbreviations_fr
    else:
        raise NotImplementedError(f"Language '{lang}' is not supported.")
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text: str) -> str:
    return text.lower()


def collapse_whitespace(text: str) -> str:
    return re.sub(_whitespace_re, " ", text).strip()


def remove_punctuation_at_begin(text: str) -> str:
    return re.sub(r'^[,.!?]+', '', text)


def remove_aux_symbols(text: str) -> str:
    return re.sub(r'[<>()\[\]"«»]+', "", text)


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


def french_cleaners(text: str) -> str:
    """Pipeline for French text before phonemization."""
    text = expand_abbreviations(text, lang="fr")
    text = replace_punctuation(text)
    text = replace_symbols(text, lang="fr")
    text = remove_aux_symbols(text)
    text = remove_punctuation_at_begin(text)
    text = collapse_whitespace(text)
    return re.sub(r'([^.,!?\-…])$', r'\1.', text)