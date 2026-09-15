import abc

try:
    from .punctuation import Punctuation
except (ImportError, ValueError):
    from punctuation import Punctuation


class BasePhonemizer(abc.ABC):
    """Base phonemizer class

    Phonemization follows the following steps:
        1. Preprocessing:
            - remove empty lines
            - remove punctuation
            - keep track of punctuation marks

        2. Phonemization:
            - convert text to phonemes

        3. Postprocessing:
            - join phonemes
            - restore punctuation marks

    Args:
        language (str):
            Language used by the phonemizer.

        punctuations (list[str] | str):
            Punctuation marks to be preserved.

        keep_puncs (bool):
            Whether to preserve punctuation marks or not.
    """

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
        """Language initialization."""
        if not self.is_supported_language(language):
            raise RuntimeError(
                f'language "{language}" is not supported by the {self.name()} backend'
            )
        return language

    @property
    def language(self) -> str:
        """The language code configured to be used for phonemization."""
        return self._language

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        """The name of the backend."""
        ...

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """Returns True if the backend is installed, False otherwise."""
        ...

    @classmethod
    @abc.abstractmethod
    def version(cls) -> str | tuple[int, ...]:
        """Return the backend version as a tuple or string."""
        ...

    @staticmethod
    @abc.abstractmethod
    def supported_languages() -> list[str] | dict:
        """Return a list or dict of language codes supported by the backend."""
        ...

    def is_supported_language(self, language: str) -> bool:
        """Returns True if `language` is supported by the backend."""
        return language in self.supported_languages()

    @abc.abstractmethod
    def _phonemize(self, text: str, separator: str) -> str:
        """The main phonemization method."""

    def _phonemize_preprocess(self, text: str) -> tuple[list[str], list]:
        """Preprocess the text before phonemization."""
        text = text.strip()
        if self._keep_puncs:
            return self._punctuator.strip_to_restore(text)
        return [self._punctuator.strip(text)], []

    def _phonemize_postprocess(self, phonemized: list[str], punctuations: list) -> str:
        """Postprocess the raw phonemized output."""
        if self._keep_puncs:
            restored = self._punctuator.restore(phonemized, punctuations)
            return restored[0] if restored else ""
        return phonemized[0] if phonemized else ""

    def phonemize(
        self, text: str, separator: str = "|", language: str | None = None
    ) -> str:
        """Returns the `text` phonemized for the given language."""
        text_chunks, punctuations = self._phonemize_preprocess(text)
        phonemized = [self._phonemize(t, separator) for t in text_chunks]
        return self._phonemize_postprocess(phonemized, punctuations)

    def print_logs(self, level: int = 0):
        indent = "\t" * level
        print(f"{indent}| > phoneme language: {self.language}")
        print(f"{indent}| > phoneme backend: {self.name()}")