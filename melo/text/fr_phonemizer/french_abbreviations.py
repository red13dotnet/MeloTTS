import re

# List of (regular expression, replacement) pairs for abbreviations in French:
abbreviations_fr: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\b{x[0]}\.", re.IGNORECASE), x[1])
    for x in [
        ("M", "monsieur"),
        ("Mlle", "mademoiselle"),
        ("Mlles", "mesdemoiselles"),
        ("Mme", "Madame"),
        ("Mmes", "Mesdames"),
        ("N.B", "nota bene"),
        ("p.c.q", "parce que"),
        ("Pr", "professeur"),
        ("qqch", "quelque chose"),
        ("rdv", "rendez-vous"),
        ("max", "maximum"),
        ("min", "minimum"),
        ("no", "numéro"),
        ("adr", "adresse"),
        ("dr", "docteur"),
        ("st", "saint"),
        ("co", "companie"),
        ("jr", "junior"),
        ("sgt", "sergent"),
        ("capt", "capitain"),
        ("col", "colonel"),
        ("av", "avenue"),
        ("av. J.-C", "avant Jésus-Christ"),
        ("apr. J.-C", "après Jésus-Christ"),
        ("art", "article"),
        ("boul", "boulevard"),
        ("c.-à-d", "c’est-à-dire"),
        ("etc", "et cetera"),
        ("ex", "exemple"),
        ("excl", "exclusivement"),
    ]
] + [
    (re.compile(rf"\b{x[0]}"), x[1])
    for x in [
        ("Mlle", "mademoiselle"),
        ("Mlles", "mesdemoiselles"),
        ("Mme", "Madame"),
        ("Mmes", "Mesdames"),
    ]
]