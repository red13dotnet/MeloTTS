import os
import re
import cn2an
from pypinyin import lazy_pinyin, Style
import jieba.posseg as psg
from transformers import AutoTokenizer

from .symbols import language_tone_start_map
from .tone_sandhi import ToneSandhi
from .english import g2p as g2p_en
from .chinese import _g2p as _chinese_g2p

punctuation = ["!", "?", "…", ",", ".", "'", "-"]
current_file_path = os.path.dirname(__file__)

with open(os.path.join(current_file_path, "opencpop-strict.txt"), "r", encoding="utf-8") as f:
    pinyin_to_symbol_map = {
        line.split("\t")[0]: line.strip().split("\t")[1]
        for line in f if line.strip()
    }

rep_map = {
    "：": ",", "；": ",", "，": ",", "。": ".", "！": "!", "？": "?",
    "\n": ".", "·": ",", "、": ",", "...": "…", "$": ".", "“": "'",
    "”": "'", "‘": "'", "’": "'", "（": "'", "）": "'", "(": "'",
    ")": "'", "《": "'", "》": "'", "【": "'", "】": "'", "[": "'",
    "]": "'", "—": "-", "～": "-", "~": "-", "「": "'", "」": "'",
}

tone_modifier = ToneSandhi()
model_id = 'bert-base-multilingual-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_id)


def replace_punctuation(text: str) -> str:
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    replaced_text = re.sub(r"[^\u4e00-\u9fa5_a-zA-Z\s" + "".join(punctuation) + r"]+", "", replaced_text)
    return re.sub(r"\s+", " ", replaced_text)


def g2p(text: str, impl: str = 'v2'):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    if impl == 'v1':
        _func = _g2p
    elif impl == 'v2':
        _func = _g2p_v2
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")
    phones, tones, word2ph = _func(sentences)
    assert sum(word2ph) == len(phones)
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


def _get_initials_finals(word):
    initials = []
    finals = []
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


def _g2p(segments):
    phones_list = []
    tones_list = []
    word2ph = []
    for seg in segments:
        seg_cut = psg.lcut(seg)
        initials = []
        finals = []
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        for word, pos in seg_cut:
            if pos == "eng":
                initials.append(['EN_WORD'])
                finals.append([word])
            else:
                sub_initials, sub_finals = _get_initials_finals(word)
                sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
                initials.append(sub_initials)
                finals.append(sub_finals)

        initials = sum(initials, [])
        finals = sum(finals, [])

        for c, v in zip(initials, finals):
            if c == 'EN_WORD':
                tokenized_en = tokenizer.tokenize(v)
                phones_en, tones_en, word2ph_en = g2p_en(text=None, pad_start_end=False, tokenized=tokenized_en)
                tones_en = [t + language_tone_start_map['EN'] for t in tones_en]
                phones_list += phones_en
                tones_list += tones_en
                word2ph += word2ph_en
            else:
                raw_pinyin = c + v
                if c == v:
                    assert c in punctuation
                    phone = [c]
                    tone = "0"
                    word2ph.append(1)
                else:
                    v_without_tone = v[:-1]
                    tone = v[-1]
                    pinyin = c + v_without_tone
                    assert tone in "12345"

                    if c:
                        v_rep_map = {"uei": "ui", "iou": "iu", "uen": "un"}
                        if v_without_tone in v_rep_map:
                            pinyin = c + v_rep_map[v_without_tone]
                    else:
                        pinyin_rep_map = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}
                        if pinyin in pinyin_rep_map:
                            pinyin = pinyin_rep_map[pinyin]
                        else:
                            single_rep_map = {"v": "yu", "e": "e", "i": "y", "u": "w"}
                            if pinyin[0] in single_rep_map:
                                pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                    assert pinyin in pinyin_to_symbol_map, (pinyin, seg, raw_pinyin)
                    phone = pinyin_to_symbol_map[pinyin].split(" ")
                    word2ph.append(len(phone))

                phones_list += phone
                tones_list += [int(tone)] * len(phone)
    return phones_list, tones_list, word2ph


def _g2p_v2(segments):
    spliter = '#$&^!@'
    phones_list = []
    tones_list = []
    word2ph = []

    for text in segments:
        assert spliter not in text
        text = re.sub(r'([a-zA-Z\s]+)', lambda x: f'{spliter}{x.group(1)}{spliter}', text)
        texts = [t for t in text.split(spliter) if len(t) > 0]

        for chunk in texts:
            if re.match(r'^[a-zA-Z\s]+$', chunk):
                tokenized_en = tokenizer.tokenize(chunk)
                phones_en, tones_en, word2ph_en = g2p_en(text=None, pad_start_end=False, tokenized=tokenized_en)
                tones_en = [t + language_tone_start_map['EN'] for t in tones_en]
                phones_list += phones_en
                tones_list += tones_en
                word2ph += word2ph_en
            else:
                phones_zh, tones_zh, word2ph_zh = _chinese_g2p([chunk])
                phones_list += phones_zh
                tones_list += tones_zh
                word2ph += word2ph_zh
    return phones_list, tones_list, word2ph


def text_normalize(text: str) -> str:
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    return replace_punctuation(text)


def get_bert_feature(text: str, word2ph: list[int], device=None):
    from . import chinese_bert
    return chinese_bert.get_bert_feature(
        text, word2ph, model_id='bert-base-multilingual-uncased', device=device
    )