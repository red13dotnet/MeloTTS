import re
from transformers import AutoTokenizer

from . import symbols
from .es_phonemizer import cleaner as es_cleaner
from .es_phonemizer import es_to_ipa


def distribute_phone(n_phone: int, n_word: int) -> list[int]:
    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


def text_normalize(text: str) -> str:
    return es_cleaner.spanish_cleaners(text)


def post_replace_ph(ph: str) -> str:
    rep_map = {
        "：": ",", "；": ",", "，": ",", "。": ".", "！": "!",
        "？": "?", "\n": ".", "·": ",", "、": ",", "...": "…",
    }
    if ph in rep_map:
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    return "UNK"


def refine_ph(phn: str):
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    return phn.lower(), tone


def refine_syllables(syllables):
    tones = []
    phonemes = []
    for phn_list in syllables:
        for phn in phn_list:
            refined_ph, tone = refine_ph(phn)
            phonemes.append(refined_ph)
            tones.append(tone)
    return phonemes, tones


model_id = 'dccuchile/bert-base-spanish-wwm-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_id)


def g2p(text: str, pad_start_end: bool = True, tokenized: list[str] | None = None):
    if tokenized is None:
        tokenized = tokenizer.tokenize(text)

    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))

    phones = []
    tones = []
    word2ph = []
    for group in ph_groups:
        w = "".join(group)
        phone_len = 0
        word_len = len(group)
        if w == '[UNK]':
            phone_list = ['UNK']
        else:
            phone_list = [p for p in es_to_ipa.es2ipa(w) if p != " "]

        for ph in phone_list:
            phones.append(ph)
            tones.append(0)
            phone_len += 1
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


def get_bert_feature(text: str, word2ph: list[int], device=None):
    from . import spanish_bert
    return spanish_bert.get_bert_feature(text, word2ph, device=device)