FIRST_NAMES = {
    "en": ["Tunde", "Ada", "Emeka", "Bola", "Ngozi", "Segun", "Ifeoma", "Musa"],
    "ig": ["Chidi", "Somtochukwu", "Uche", "Adaeze", "Obinna", "Chinelo"],
    "yo": ["Adewale", "Folake", "Babatunde", "Yetunde", "Ayodeji", "Simisola"],
    "ha": ["Aisha", "Ibrahim", "Fatima", "Sani", "Zainab", "Umar"],
    "pcm": ["Chi", "Bimbo", "Ekene", "Nkechi", "Tobi", "Ada"],
}
LAST_NAMES = {
    "en": ["Okafor", "Adeyemi", "Nwosu", "Bello", "Eze", "Balogun"],
    "ig": ["Okafor", "Nwosu", "Eze", "Okeke", "Obi", "Nnamdi"],
    "yo": ["Adeyemi", "Balogun", "Ogunlesi", "Afolabi", "Oyelaran"],
    "ha": ["Bello", "Sani", "Yusuf", "Abubakar", "Danjuma"],
    "pcm": ["Okafor", "Bello", "Eze", "Balogun"],
}


def pick_name(rng, lang="en"):
    firsts = FIRST_NAMES.get(lang, FIRST_NAMES["en"])
    lasts = LAST_NAMES.get(lang, LAST_NAMES["en"])
    return rng.choice(firsts), rng.choice(lasts)


def nickname(rng, first):
    if len(first) <= 3:
        return first
    return rng.choice([first[:3], first[:4], first[0] + first[1:3]])


def initials(name):
    return ".".join(p[0].upper() for p in name.split()) + "."
