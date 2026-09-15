import abc

try:
    from .punctuation import Punctuation
except (ImportError, ValueError):
    from punctuation import Punctuation


class BasePhonemizer(abc.ABC):
    """Base phonemizer class."""

    def __init__(
        self,
        language: str,
        punctuations: str = Punctuation.default_puncs(),
        keep_puncs: bool = False,
    ):
        if not self.is_available():
            raise RuntimeError(f"{self.name()} not installed on your system")

        self._language = self._init_language(language)
        self._keep_puncs = keep_puncs
        self._punctuator = Punctuation(punctuations)

    def _init_language(self, language: str) -> str:
        if not self.is_supported_language(language):
            raise RuntimeError(
                f'language "{language}" is not supported by the {self.name()} backend'
            )
        return language

    @property
    def language(self) -> str:
        return self._language

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        ...

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        ...

    @classmethod
    @abc.abstractmethod
    def version(cls) -> str | tuple[int, ...]:
        ...

    @staticmethod
    @abc.abstractmethod
    def supported_languages() -> list[str] | dict:
        ...

    def is_supported_language(self, language: str) -> bool:
        return language in self.supported_languages()

    @abc.abstractmethod
    def _phonemize(self, text: str, separator: str) -> str:
        ...

    def _phonemize_preprocess(self, text: str) -> tuple[list[str], list]:
        text = text.strip()
        if self._keep_puncs:
            return self._punctuator.strip_to_restore(text)
        return [self._punctuator.strip(text)], []

    def _phonemize_postprocess(self, phonemized: list[str], punctuations: list) -> str:
        if self._keep_puncs:
            restored = self._punctuator.restore(phonemized, punctuations)
            return restored[0] if restored else ""
        return phonemized[0] if phonemized else ""

    def phonemize(
        self, text: str, separator: str = "|", language: str | None = None
    ) -> str:
        text_chunks, punctuations = self._phonemize_preprocess(text)
        phonemized = [self._phonemize(t, separator) for t in text_chunks]
        return self._phonemize_postprocess(phonemized, punctuations)

    def print_logs(self, level: int = 0):
        indent = "\t" * level
        print(f"{indent}| > phoneme language: {self.language}")
        print(f"{indent}| > phoneme backend: {self.name()}")