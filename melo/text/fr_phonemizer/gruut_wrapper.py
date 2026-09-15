import importlib.util
import gruut
from gruut_ipa import IPA

try:
    from .base import BasePhonemizer
    from .punctuation import Punctuation
except (ImportError, ValueError):
    from base import BasePhonemizer
    from punctuation import Punctuation

# Table for str.translate to fix gruut/TTS phoneme mismatch
GRUUT_TRANS_TABLE = str.maketrans("g", "ɡ")


class Gruut(BasePhonemizer):
    def __init__(
        self,
        language: str,
        punctuations: str = Punctuation.default_puncs(),
        keep_puncs: bool = True,
        use_espeak_phonemes: bool = False,
        keep_stress: bool = False,
    ):
        super().__init__(language, punctuations=punctuations, keep_puncs=keep_puncs)
        self.use_espeak_phonemes = use_espeak_phonemes
        self.keep_stress = keep_stress

    @staticmethod
    def name() -> str:
        return "gruut"

    def phonemize_gruut(self, text: str, separator: str = "|", tie: bool = False) -> str:
        ph_list = []
        for sentence in gruut.sentences(text, lang=self.language, espeak=self.use_espeak_phonemes):
            for word in sentence:
                if word.is_break:
                    if ph_list:
                        ph_list[-1].append(word.text)
                    else:
                        ph_list.append([word.text])
                elif word.phonemes:
                    word_phonemes = []
                    for word_phoneme in word.phonemes:
                        if not self.keep_stress:
                            word_phoneme = IPA.without_stress(word_phoneme)

                        word_phoneme = word_phoneme.translate(GRUUT_TRANS_TABLE)
                        if word_phoneme:
                            word_phonemes.extend(word_phoneme)

                    if word_phonemes:
                        ph_list.append(word_phonemes)

        ph_words = [separator.join(word_phonemes) for word_phonemes in ph_list]
        return f"{separator} ".join(ph_words)

    def _phonemize(self, text: str, separator: str) -> str:
        return self.phonemize_gruut(text, separator, tie=False)

    def is_supported_language(self, language: str) -> bool:
        return gruut.is_language_supported(language)

    @staticmethod
    def supported_languages() -> list:
        return list(gruut.get_supported_languages())

    def version(self) -> str:
        return gruut.__version__

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("gruut") is not None