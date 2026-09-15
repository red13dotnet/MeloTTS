import re
import unicodedata
from transformers import AutoTokenizer
from anyascii import anyascii
from jamo import hangul_to_jamo
import torch

from . import punctuation, symbols
from melo.text.ko_dictionary import english_dictionary, etc_dictionary


def normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]", "", text)
    text = normalize_with_dictionary(text, etc_dictionary)
    text = normalize_english(text)
    return text.lower()


def normalize_with_dictionary(text: str, dic: dict[str, str]) -> str:
    if any(key in text for key in dic):
        pattern = re.compile("|".join(re.escape(key) for key in dic))
        return pattern.sub(lambda x: dic[x.group()], text)
    return text


def normalize_english(text: str) -> str:
    def fn(m):
        word = m.group()
        return english_dictionary.get(word, word)

    return re.sub(r"([A-Za-z]+)", fn, text)


g2p_kr = None


def korean_text_to_phonemes(text: str, character: str = "hangeul") -> str:
    global g2p_kr
    if g2p_kr is None:
        from g2pkk import G2p
        g2p_kr = G2p()

    if character == "english":
        text = normalize(text)
        text = g2p_kr(text)
        return anyascii(text)

    text = normalize(text)
    text = g2p_kr(text)
    return "".join(list(hangul_to_jamo(text)))


def text_normalize(text: str) -> str:
    return normalize(text)


def distribute_phone(n_phone: int, n_word: int) -> list[int]:
    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


model_id = 'kykim/bert-kor-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)


def g2p(norm_text: str):
    tokenized = tokenizer.tokenize(norm_text)
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))
    word2ph = []
    phs = []
    for group in ph_groups:
        text = "".join(group)
        if text == '[UNK]':
            phs += ['_']
            word2ph += [1]
            continue
        elif text in punctuation:
            phs += [text]
            word2ph += [1]
            continue

        phonemes = korean_text_to_phonemes(text)
        phone_len = len(phonemes)
        word_len = len(group)

        aaa = distribute_phone(phone_len, word_len)
        assert len(aaa) == word_len
        word2ph += aaa
        phs += list(phonemes)

    phones = ["_"] + phs + ["_"]
    tones = [0 for _ in phones]
    word2ph = [1] + word2ph + [1]
    assert len(word2ph) == len(tokenized) + 2
    return phones, tones, word2ph


def get_bert_feature(text: str, word2ph: list[int], device=None):
    from . import japanese_bert
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return japanese_bert.get_bert_feature(text, word2ph, device=device, model_id=model_id)