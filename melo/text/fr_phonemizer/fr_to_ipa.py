try:
    from .cleaner import french_cleaners
    from .gruut_wrapper import Gruut
except (ImportError, ValueError):
    from cleaner import french_cleaners
    from gruut_wrapper import Gruut


def remove_consecutive_t(input_str: str) -> str:
    result = []
    count = 0
    for char in input_str:
        if char == 't':
            count += 1
        else:
            if count < 3:
                result.extend(['t'] * count)
            count = 0
            result.append(char)
    if count < 3:
        result.extend(['t'] * count)
    return ''.join(result)


def fr2ipa(text: str) -> str:
    e = Gruut(language="fr-fr", keep_puncs=True, keep_stress=True, use_espeak_phonemes=True)
    phonemes = e.phonemize(text, separator="")
    return remove_consecutive_t(phonemes)


if __name__ == '__main__':
    sample = "Bonjour! Comment allez-vous aujourd'hui?"
    print(fr2ipa(sample))