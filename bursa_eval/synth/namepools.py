FIRST_NAMES = {
    "en": ["Tunde", "Ada", "Emeka", "Bola", "Ngozi", "Segun", "Ifeoma", "Musa",
           "Kelechi", "Damilola", "Efe", "Halima", "Chinedu", "Funke"],
    "ig": ["Chidi", "Somtochukwu", "Uche", "Adaeze", "Obinna", "Chinelo",
           "Ikenna", "Nkiru", "Chukwuma", "Ada", "Ebuka", "Ngozi"],
    "yo": ["Adewale", "Folake", "Babatunde", "Yetunde", "Ayodeji", "Simisola",
           "Olamide", "Bukola", "Tijani", "Morenike", "Kunle", "Aduke"],
    "ha": ["Aisha", "Ibrahim", "Fatima", "Sani", "Zainab", "Umar",
           "Hauwa", "Nasir", "Amina", "Bashir", "Halima", "Yakubu"],
    "pcm": ["Chi", "Bimbo", "Ekene", "Nkechi", "Tobi", "Ada",
            "Emeka", "Funmi", "Uche", "Bola"],
}
LAST_NAMES = {
    "en": ["Okafor", "Adeyemi", "Nwosu", "Bello", "Eze", "Balogun",
           "Danjuma", "Okonkwo", "Afolabi", "Ibrahim"],
    "ig": ["Okafor", "Nwosu", "Eze", "Okeke", "Obi", "Nnamdi", "Okonkwo", "Anyanwu"],
    "yo": ["Adeyemi", "Balogun", "Ogunlesi", "Afolabi", "Oyelaran", "Adebayo", "Ojo", "Bello"],
    "ha": ["Bello", "Sani", "Yusuf", "Abubakar", "Danjuma", "Musa", "Aliyu", "Garba"],
    "pcm": ["Okafor", "Bello", "Eze", "Balogun", "Nwosu", "Adeyemi"],
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
