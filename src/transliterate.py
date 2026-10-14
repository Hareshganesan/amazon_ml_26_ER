"""Transliterate Indian-script text to lowercase Latin so it can be matched against Latin S1 names/addresses.

Example: "राम मार्केटिंग प्राइवेट लिमिटेड" -> "ram marketing private limited".

Why a table instead of indic_transliteration.sanscript: sanscript targets Sanskrit schemes, so Tamil comes out
wrong ("பிரைவேட்" -> "bhiraiveDh"), Devanagari candra-o "ॉ" is left untouched, and every inherent vowel is kept
("rAma mArkeTiMga"). The nine Brahmic Unicode blocks (Devanagari 0900 ... Malayalam 0D00) share the ISCII layout,
so ONE table indexed by (codepoint - block start) handles all of them. The output aims at everyday English
spelling: long vowels collapse (aa -> a), nukta letters give z/f/r, anusvara gives n (m before p/b/m), and for the
northern scripts the inherent schwa is dropped word-finally and in V C_C V contexts (Hindi-style schwa deletion).
Frequent words are overridden by WORD_MAP, derived from true train pairs (see derive_word_map).

No third-party dependencies at runtime (rapidfuzz is only needed by derive_word_map). Text without Indic
characters is returned unchanged.
"""
import re
from functools import lru_cache

# block start -> script name; each block is 0x80 wide
BLOCKS = {0x0900: "Devanagari", 0x0980: "Bengali", 0x0A00: "Gurmukhi", 0x0A80: "Gujarati", 0x0B00: "Odia",
          0x0B80: "Tamil", 0x0C00: "Telugu", 0x0C80: "Kannada", 0x0D00: "Malayalam"}
NORTH = {"Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Odia"}   # scripts with schwa deletion
INDIC = re.compile(r"[\u0900-\u0D7F]")
ZW = {"\u200B", "\u200C", "\u200D", "\u2060", "\uFEFF"}

# offset -> Latin, shared by all nine scripts
CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh",
    0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d",
    0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "f", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y",
    0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh", 0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s",
    0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
    0x75: "y",                                                   # Gurmukhi yakash
}
VOWEL = {  # independent vowels
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li", 0x0D: "e",
    0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li",
    0x72: "", 0x73: "",                                          # Gurmukhi vowel carriers (matra follows)
}
MATRA = {  # dependent vowel signs
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e", 0x46: "e",
    0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "li", 0x63: "li",
}
NUKTA = {0x15: "q", 0x1C: "z", 0x21: "r", 0x22: "rh", 0x2B: "f", 0x2F: "y", 0x38: "sh"}   # consonant + nukta (ਸ਼)
CHILLU = {0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k"}   # Malayalam final consonants
LABIAL = {0x2A, 0x2B, 0x2C, 0x2D, 0x2E}
VIRAMA, NUKTA_SIGN, ANUSVARA, VISARGA = 0x4D, 0x3C, 0x02, 0x03
# Malayalam "റ്റ" is the tt sound, "ന്റ" is nt
PRE = [(re.compile("\u0D31\u0D4D\u0D31"), "\u0D1F"), (re.compile("\u0D28\u0D4D\u0D31"), "\u0D28\u0D4D\u0D1F")]

# Abbreviations seen in the data (normalise_rules word counts); not derivable by the similarity filter
ABBREV = {"प्रा.": "pvt", "लि.": "ltd", "પ્રા.": "pvt", "લિ.": "ltd", "ਪ੍ਰਾ.": "pvt", "ਲਿ.": "ltd",
          "एलएलपी": "llp", "ఎల్ఎల్పీ": "llp", "ಎಲ್ಎಲ್ಪಿ": "llp", "எல்எல்பி": "llp", "এলএলপি": "llp",
          "એલએલપી": "llp", "എൽഎൽപി": "llp"}
ABBREV = {"".join(c for c in k if c not in ZW): v for k, v in ABBREV.items()}
WORD_MAP = {}   # Indic token -> English word, filled from WORD_MAP_DATA below


def _script(ch):
    """Unicode block start of an Indic character, or None."""
    o = ord(ch)
    return (o & ~0x7F) if 0x0900 <= o <= 0x0D7F else None


def _rules(run, block):
    """Rule-based transliteration of one run of characters from a single Indic block."""
    script = BLOCKS[block]
    for pat, rep in PRE:
        run = pat.sub(rep, run)
    units = []   # [kind, latin, schwa, offset]; kind: C consonant, V vowel, N nasal
    aytham = False
    for ch in run:
        if ch in ZW:
            continue
        off = ord(ch) - block
        if script == "Tamil" and off == 0x03:                # aytham, must precede the visarga branch
            aytham = True
        elif off in CONS:
            lat = CONS[off]
            if script == "Tamil" and off == 0x1A:            # Tamil ச is mostly "s" ("சவுத்" = south)
                lat = "s"
            if aytham and off == 0x2A:                       # Tamil ஃப = f
                lat = "f"
            aytham = False
            units.append(["C", lat, True, off])
        elif off in MATRA:
            if units and units[-1][0] == "C":
                units[-1][2] = False
            units.append(["V", MATRA[off], False, off])
        elif off == VIRAMA:
            if units and units[-1][0] == "C":
                units[-1][2] = False
        elif off == NUKTA_SIGN:
            if units and units[-1][0] == "C" and units[-1][3] in NUKTA:
                units[-1][1] = NUKTA[units[-1][3]]
        elif off in VOWEL:
            units.append(["V", VOWEL[off], False, off])
        elif off in (0x00, 0x01, ANUSVARA) or (script == "Gurmukhi" and off == 0x70):
            units.append(["N", "m" if script == "Malayalam" else "n", False, off])
        elif off == VISARGA:
            units.append(["C", "h", False, off])
        elif script == "Malayalam" and off in CHILLU:
            units.append(["C", CHILLU[off], False, off])
        elif script == "Bengali" and off == 0x4E:            # khanda ta
            units.append(["C", "t", False, off])
        elif off == 0x71 and script in ("Odia", "Bengali"):  # wa "ୱ"/"ৱ" ("ପାୱାର୍" = power); Gurmukhi 0x71 is addak
            units.append(["C", "v", True, off])
        elif off == 0x50 and script == "Devanagari":        # ॐ
            units.append(["V", "om", False, off])
        elif 0x66 <= off <= 0x6F:
            units.append(["D", str(off - 0x66), False, off])
        # anything else (avagraha, addak, length marks, dandas) is dropped
    if script in NORTH:
        _delete_schwa(units)
    out = []
    for i, (kind, lat, schwa, off) in enumerate(units):
        if kind == "N" and lat == "n" and i + 1 < len(units) and units[i + 1][0] == "C" and units[i + 1][3] in LABIAL:
            lat = "m"
        out.append(lat + ("a" if schwa else ""))
    return "".join(out)


def _delete_schwa(units):
    """Hindi-style schwa deletion in place: word-final, then V C_C V right-to-left."""
    def vowel_after(i):
        """True if unit i carries a vowel sign or is followed by an independent vowel (used for schwa deletion)."""
        return units[i][2] or (i + 1 < len(units) and units[i + 1][0] == "V")

    n = len(units)
    if n > 1 and units[-1][0] == "C" and units[-1][2]:
        # keep the schwa after a conjunct only for r/y/v endings (mitra, aditya); English -ds/-cts endings drop it
        cluster = units[-2][0] == "C" and not vowel_after(n - 2) and units[-1][3] in (0x30, 0x2F, 0x35)
        other_vowel = any(u[0] == "V" or (u[0] == "C" and u[2]) for u in units[:-1])
        if other_vowel and not cluster:
            units[-1][2] = False
    for i in range(n - 2, 0, -1):
        u, left, right = units[i], units[i - 1], units[i + 1]
        if u[0] != "C" or not u[2]:
            continue
        left_vowel = left[0] == "V" or (left[0] == "C" and left[2])
        if left_vowel and (right[0] == "C" and vowel_after(i + 1) or right[0] == "V" and right[3] in VOWEL):
            u[2] = False   # V C_C V, or V C_V(independent) as in "एलएलपी" -> "elelpi"


@lru_cache(maxsize=300_000)
def _token(tok):
    """Transliterate one whitespace token (cached)."""
    key = _strip_zw(tok)
    if key in WORD_MAP:
        return WORD_MAP[key]
    if key in ABBREV:
        return ABBREV[key]
    out, i = [], 0
    while i < len(tok):
        block = _script(tok[i])
        if block is None:
            out.append(tok[i])
            i += 1
            continue
        j = i + 1
        while j < len(tok) and (_script(tok[j]) == block or tok[j] in ZW):
            j += 1
        run = tok[i:j]
        out.append(WORD_MAP.get(_strip_zw(run)) or _rules(run, block))   # "(इंडिया)" -> "(india)"
        i = j
    return "".join(out)


def _strip_zw(s):
    """Drop zero-width joiners/spaces (keys are stored without them)."""
    return "".join(c for c in s if c not in ZW)


def transliterate(text):
    """Indian-script runs -> Latin, then lowercase + collapse spaces. Text without Indic chars is returned as-is."""
    if not isinstance(text, str):
        return ""
    if not INDIC.search(text):
        return text
    try:
        return " ".join(filter(None, (_token(t) for t in text.split()))).lower()
    except Exception:   # never raise on odd input
        return text


def set_word_map(mapping):
    """Replace WORD_MAP (e.g. one derived from a training split) and clear the cache."""
    WORD_MAP.clear()
    WORD_MAP.update({_strip_zw(k): v for k, v in mapping.items()})
    _token.cache_clear()


def derive_word_map(cooc, min_n=20, tiers=((0.9, 20), (0.4, 50))):
    """Build Indic token -> English word from true-pair co-occurrences.

    cooc: iterable of (indic_token, n_pairs, [(latin_word, count), ...]) where the counts come from the Latin S1
    names of true pairs containing the token. A true translation co-occurs in ~100% of the token's pairs, but so
    can other words of the same fixed phrase ("प्राइवेट" always sits next to "limited"), so: take the first tier
    (min_share, min_sim) with any candidate and keep the candidate most similar (rapidfuzz ratio) to the rule
    transliteration. Tier 2 catches spelling variants that are only sometimes used ("lakshmi"/"laxmi").
    Entries equal to the rule output are skipped.
    """
    from rapidfuzz import fuzz
    out = {}
    for tok, n, cands in cooc:
        if n < min_n or not INDIC.search(tok):
            continue
        base = _rules_only(tok)
        for min_share, min_sim in tiers:
            best = [(fuzz.ratio(base, w), c, w) for w, c in cands if c / n >= min_share]
            best = [b for b in best if b[0] >= min_sim]
            if best:
                w = max(best)[2]
                if w != base:
                    out[tok] = w
                break
    return out


def _rules_only(tok):
    """Rule transliteration of a token ignoring WORD_MAP (used when deriving the map)."""
    out, i = [], 0
    while i < len(tok):
        block = _script(tok[i])
        if block is None:
            out.append(tok[i].lower())
            i += 1
            continue
        j = i + 1
        while j < len(tok) and (_script(tok[j]) == block or tok[j] in ZW):
            j += 1
        out.append(_rules(tok[i:j], block))
        i = j
    return "".join(out)


# ---- data-derived word map ------------------------------------------------------------------------------
# 1,051 entries. derive_word_map() looked at the 1,347 Indic name tokens that occur in >= 20 of the 551,240
# non-Latin true train pairs (Kaggle kernel amz-translit-eda, cooc.tsv) and kept the 1,051 whose translation
# differs from the rule output; the other 296 are already handled by the rules. Ordered by frequency.
# Zero-width chars stripped from keys. Plus 13 hand-listed abbreviations in ABBREV (प्रा. / लि. / LLP per script).
WORD_MAP_DATA = {
    "प्राइवेट": "private", "प्रा.": "pvt", "लि.": "ltd", "ప్రైవేట్": "private", "லிமிடெட்": "limited",
    "ಪ್ರೈವೇಟ್": "private", "பிரைவேட்": "private", "প্রাইভেট": "private", "પ્રાઇવેટ": "private",
    "ലിമിറ്റഡ്": "limited", "പ്രൈവറ്റ്": "private", "एलएलपी": "llp", "टेक": "tech", "इंटरनेशनल": "international",
    "श्री": "shree", "इंफ्रास्ट्रक्चर": "infrastructure", "आईटी": "it", "बिल्डर्स": "builders",
    "ପ୍ରାଇଭେଟ୍": "private", "ट्रेडिंग": "trading", "इंडस्ट्रीज": "industries", "प्रोड्यूसर": "producer",
    "एनर्जी": "energy", "कंसल्टिंग": "consulting", "मैनेजमेंट": "management", "इंजीनियरिंग": "engineering",
    "पावर": "power", "इंफोटेक": "infotech", "सॉफ्टवेयर": "software", "बिजनेस": "business",
    "कंसल्टेंट्स": "consultants", "इंफ्रा": "infra", "फाउंडेशन": "foundation", "फाइनेंस": "finance",
    "इम्पेक्स": "impex", "सॉल्यूशंस": "solutions", "प्रोडक्ट्स": "products", "हॉस्पिटैलिटी": "hospitality",
    "डेवलपर्स": "developers", "सर्विसेज": "services", "कंस्ट्रक्शन": "construction", "वेंचर्स": "ventures",
    "एग्रो": "agro", "एक्सपोर्ट्स": "exports", "प्रोजेक्ट्स": "projects", "कंस्ट्रक्शंस": "constructions",
    "एंटरप्राइजेज": "enterprises", "मीडिया": "media", "सिस्टम्स": "systems", "इन्वेस्टमेंट": "investment",
    "हेल्थकेयर": "healthcare", "कंसल्टेंसी": "consultancy", "प्रॉपर्टीज": "properties", "फूड्स": "foods",
    "टेक्नोलॉजीज": "technologies", "केयर": "care", "लॉजिस्टिक्स": "logistics", "टेक्नोलॉजी": "technology",
    "एस्टेट": "estate", "इन्वेस्टमेंट्स": "investments", "फूड": "food", "ਲਿਮਟਿਡ": "limited", "લિ.": "ltd",
    "પ્રા.": "pvt", "ਪ੍ਰਾਈਵੇਟ": "private", "क्लासिक": "classic", "होटल": "hotel", "ब्लू": "blue",
    "इनोवेटिव": "innovative", "विजन": "vision", "एस": "ace", "सुपर": "super", "डिजिटल": "digital", "ईस्ट": "east",
    "साउथ": "south", "हाई": "high", "फ्यूचर": "future", "लाइफ": "life", "सदर्न": "southern", "डायनामिक": "dynamic",
    "ईस्टर्न": "eastern", "सिटी": "city", "पायोनियर": "pioneer", "सिल्वर": "silver", "इंडियन": "indian",
    "लोटस": "lotus", "तिरुपति": "tirupati", "फॉर्च्यून": "fortune", "यूनिवर्सल": "universal", "बॉम्बे": "bombay",
    "गैलेक्सी": "galaxy", "प्रीमियर": "premier", "ड्रीम": "dream", "सेवन": "seven", "यूनिक": "unique",
    "एपेक्स": "apex", "अर्बन": "urban", "स्वस्तिक": "swastik", "गुजरात": "gujarat", "सन": "sun", "ग्रीन": "green",
    "रियल": "real", "सनराइज": "sunrise", "वन": "one", "गोल्डन": "golden", "मॉडर्न": "modern", "न्यू": "new",
    "परफेक्ट": "perfect", "सुप्रीम": "supreme", "ब्राइट": "bright", "ब्लैक": "black", "प्राइम": "prime",
    "अल्फा": "alpha", "व्हाइट": "white", "गुड": "good", "स्काई": "sky", "हाईटेक": "hitech", "ऑल": "all",
    "माय": "my", "ग्रेट": "great", "क्रिएटिव": "creative", "यूनाइटेड": "united", "एसएस": "ss", "फर्स्ट": "first",
    "ఎల్ఎల్పీ": "llp", "ಎಲ್ಎಲ್ಪಿ": "llp", "कृष्ण": "krishna", "माँ": "maa", "साईं": "sai", "मां": "maa",
    "எல்எல்பி": "llp", "এলএলপি": "llp", "એલએલપી": "llp", "టెక్": "tech", "ఇంటర్నేషనల్": "international",
    "ಟೆಕ್": "tech", "ಇಂಟರ್ನ್ಯಾಷನಲ್": "international", "ટેક": "tech", "டெக்": "tech", "குளோபல்": "global",
    "இன்டர்நேஷனல்": "international", "ইন্টারন্যাশনাল": "international", "টেক": "tech", "ਪ੍ਰਾ.": "pvt",
    "ਲਿ.": "ltd", "ઇન્ટરનેશનલ": "international", "సాఫ్ట్వేర్": "software", "కన్సల్టింగ్": "consulting",
    "కన్స్ట్రక్షన్": "construction", "ప్రొడ్యూసర్": "producer", "డెవలపర్స్": "developers", "ಶ್ರೀ": "shree",
    "ఫౌండేషన్": "foundation", "కన్సల్టెన్సీ": "consultancy", "కన్సల్టెంట్స్": "consultants", "శ్రీ": "shree",
    "ఎక్స్పోర్ట్స్": "exports", "హాస్పిటాలిటీ": "hospitality", "ఇండస్ట్రీస్": "industries",
    "ಇಂಜಿನಿಯರಿಂಗ್": "engineering", "టెక్నాలజీస్": "technologies", "కన్స్ట్రక్షన్స్": "constructions",
    "ಪವರ್": "power", "లాజిస్టిక్స్": "logistics", "എൽഎൽപി": "llp", "ಕನ್ಸಲ್ಟೆನ್ಸಿ": "consultancy", "కేర్": "care",
    "ಫುಡ್": "food", "ಲಾಜಿಸ್ಟಿಕ್ಸ್": "logistics", "టెక్నాలజీ": "technology", "ಕನ್ಸ್ಟ್ರಕ್ಷನ್": "construction",
    "ప్రొడక్ట్స్": "products", "మేనేజ్మెంట్": "management", "ఫుడ్": "food", "ఇన్ఫోటెక్": "infotech", "ಐಟಿ": "it",
    "ఎస్టేట్": "estate", "ಎನರ್ಜಿ": "energy", "ಇಂಡಸ್ಟ್ರೀಸ್": "industries", "ಇನ್ಫೋಟೆಕ್": "infotech",
    "ಕನ್ಸಲ್ಟೆಂಟ್ಸ್": "consultants", "సర్వీసెస్": "services", "ಎಸ್ಟೇಟ್": "estate", "పవర్": "power",
    "వెంచర్స్": "ventures", "ఇంపెక్స్": "impex", "బిల్డర్స్": "builders", "ఫైనాన్స్": "finance",
    "ಸರ್ವಿಸಸ್": "services", "ஜெய்": "jay", "ఫుడ్స్": "foods", "ಸಾಫ್ಟ್ವೇರ್": "software",
    "ఇంజనీరింగ్": "engineering", "ಇಂಪೆಕ್ಸ್": "impex", "எண்டர்பிரைசஸ்": "enterprises", "ಪ್ರಾಪರ್ಟೀಸ್": "properties",
    "ಫೈನಾನ್ಸ್": "finance", "హెల్త్కేర్": "healthcare", "ಕೇರ್": "care", "మీడియా": "media",
    "ప్రాపర్టీస్": "properties", "సిస్టమ్స్": "systems", "ஸ்ரீ": "sree", "ಮ್ಯಾನೇಜ್ಮೆಂಟ್": "management",
    "ಫೌಂಡೇಶನ್": "foundation", "ಪ್ರೊಡಕ್ಟ್ಸ್": "products", "ಕನ್ಸಲ್ಟಿಂಗ್": "consulting", "ఐటీ": "it",
    "ఇన్ఫ్రాస్ట్రక్చర్": "infrastructure", "సొల్యూషన్స్": "solutions", "ಸಿಸ್ಟಮ್ಸ್": "systems",
    "ஃபவுண்டேஷன்": "foundation", "ఎనర్జీ": "energy", "ಕನ್ಸ್ಟ್ರಕ್ಷನ್ಸ್": "constructions", "ಫುಡ್ಸ್": "foods",
    "ಹೆಲ್ತ್ಕೇರ್": "healthcare", "ಎಂಟರ್ಪ್ರೈಸಸ್": "enterprises", "ಮೀಡಿಯಾ": "media", "ಟೆಕ್ನಾಲಜಿ": "technology",
    "ప్రాజెక్ట్స్": "projects", "ஃபுட்ஸ்": "foods", "એસ્ટેટ": "estate", "ಎಕ್ಸ್ಪೋರ್ಟ್ಸ್": "exports",
    "ಪ್ರೊಡ್ಯೂಸರ್": "producer", "బిజినెస్": "business", "ஃபைனான்ஸ்": "finance", "ಬಿಲ್ಡರ್ಸ್": "builders",
    "ಟ್ರೇಡಿಂಗ್": "trading", "ಡೆವಲಪರ್ಸ್": "developers", "ಬಿಸಿನೆಸ್": "business", "கன்ஸ்ட்ரக்ஷன்ஸ்": "constructions",
    "அக்ரோ": "agro", "ಟೆಕ್ನಾಲಜೀಸ್": "technologies", "ವೆಂಚರ್ಸ್": "ventures", "ஹாஸ்பிடாலிட்டி": "hospitality",
    "எக்ஸ்போர்ட்ஸ்": "exports", "கன்சல்டிங்": "consulting", "ఎంటర్ప్రైజెస్": "enterprises",
    "ಸೊಲ್ಯೂಷನ್ಸ್": "solutions", "கன்சல்டன்சி": "consultancy", "શ્રી": "shree", "பிசினஸ்": "business",
    "லாஜிஸ்டிக்ஸ்": "logistics", "বিজনেস": "business", "புராடக்ட்ஸ்": "products",
    "இன்ஃப்ராஸ்ட்ரக்சர்": "infrastructure", "இன்ஜினியரிங்": "engineering", "இன்ஃபோடெக்": "infotech",
    "ట్రేడింగ్": "trading", "હેલ્થકેર": "healthcare", "சிஸ்டம்ஸ்": "systems", "ஃபுட்": "food", "ஐடி": "it",
    "கன்சல்டன்ட்ஸ்": "consultants", "সিস্টেমস": "systems", "வென்ச்சர்ஸ்": "ventures",
    "புராப்பர்ட்டீஸ்": "properties", "பில்டர்ஸ்": "builders", "கேர்": "care", "કન્સલ્ટન્ટ્સ": "consultants",
    "ടെക്": "tech", "ইম্পেক্স": "impex", "சர்வீசஸ்": "services", "மேனேஜ்மெண்ட்": "management",
    "ಇನ್ಫ್ರಾಸ್ಟ್ರಕ್ಚರ್": "infrastructure", "সফটওয়্যার": "software", "কনস্ট্রাকশনস": "constructions",
    "இம்பெக்ஸ்": "impex", "ಹಾಸ್ಪಿಟಾಲಿಟಿ": "hospitality", "এনার্জি": "energy", "டெவலப்பர்ஸ்": "developers",
    "પ્રોડ્યુસર": "producer", "ટેક્નોલોજી": "technology", "কনসালট্যান্টস": "consultants",
    "மார்க்கெட்டிங்": "marketing", "புரொட்யூசர்": "producer", "ટ્રેડિંગ": "trading", "பவர்": "power",
    "சொல்யூஷன்ஸ்": "solutions", "ಪ್ರಾಜೆಕ್ಟ್ಸ್": "projects", "এক্সপোর্টস": "exports", "শিব": "shiv", "আইটি": "it",
    "டிரேடிங்": "trading", "কনসালটেন্সি": "consultancy", "இண்டஸ்ட்ரீஸ்": "industries", "સિસ્ટમ્સ": "systems",
    "ઇન્ફ્રાસ્ટ્રક્ચર": "infrastructure", "மீடியா": "media", "લોજિસ્ટિક્સ": "logistics", "எனர்ஜி": "energy",
    "புராஜெக்ட்ஸ்": "projects", "ડેવલપર્સ": "developers", "হসপিটালিটি": "hospitality", "ডেভেলপারস": "developers",
    "শ্রী": "shree", "ট্রেডিং": "trading", "எஸ்டேட்": "estate", "કન્સ્ટ્રક્શન": "construction",
    "কনস্ট্রাকশন": "construction", "ஹெல்த்கேர்": "healthcare", "પ્રોપર્ટીઝ": "properties",
    "டெக்னாலஜீஸ்": "technologies", "মার্কেটিং": "marketing", "এন্টারপ্রাইজেস": "enterprises",
    "મેનેજમેન્ટ": "management", "প্রজেক্টস": "projects", "ফাউন্ডেশন": "foundation", "ফুডস": "foods", "ফুড": "food",
    "કન્સલ્ટિંગ": "consulting", "બિલ્ડર્સ": "builders", "டெக்னாலஜி": "technology", "ફૂડ્સ": "foods",
    "ઇન્વેસ્ટમેન્ટ": "investment", "সলিউশনস": "solutions", "પ્રોજેક્ટ્સ": "projects", "ফাইন্যান্স": "finance",
    "சாஃப்ட்வேர்": "software", "கன்ஸ்ட்ரக்ஷன்": "construction", "એન્ટરપ્રાઇઝિસ": "enterprises",
    "টেকনোলজিস": "technologies", "কনসাল্টিং": "consulting", "સર્વિસીસ": "services", "এস্টেট": "estate",
    "એક્સપોર્ટ્સ": "exports", "সার্ভিসেস": "services", "ફાઇનાન્સ": "finance", "প্রডিউসার": "producer",
    "ઇમ્પેક્સ": "impex", "પ્રોડક્ટ્સ": "products", "ইনভেস্টমেন্টস": "investments", "প্রপার্টিজ": "properties",
    "ઇન્ફોટેક": "infotech", "હોસ્પિટાલિટી": "hospitality", "টেকনোলজি": "technology", "આઈટી": "it",
    "એનર્જી": "energy", "অ্যাগ্রো": "agro", "ફાઉન્ડેશન": "foundation", "ઇન્ડસ્ટ્રીઝ": "industries",
    "প্রোডাক্টস": "products", "પાવર": "power", "পাওয়ার": "power", "બિઝનેસ": "business",
    "হেলথকেয়ার": "healthcare", "কেয়ার": "care", "ટેક્નોલોજીસ": "technologies", "એગ્રો": "agro",
    "વેન્ચર્સ": "ventures", "ইনফ্রাস্ট্রাকচার": "infrastructure", "ইন্ডাস্ট্রিজ": "industries",
    "સોફ્ટવેર": "software", "ম্যানেজমেন্ট": "management", "ઇન્વેસ્ટમેન્ટ્સ": "investments",
    "કન્સલ્ટન્સી": "consultancy", "ફૂડ": "food", "કેર": "care", "ভেঞ্চারস": "ventures",
    "કન્સ્ટ્રક્શન્સ": "constructions", "ইনভেস্টমেন্ট": "investment", "ইনফ্রা": "infra", "સોલ્યુશન્સ": "solutions",
    "ইঞ্জিনিয়ারিং": "engineering", "ഇന്റർനാഷണൽ": "international", "એન્જિનિયરિંગ": "engineering",
    "லக்ஷ்மி": "laxmi", "বিল্ডার্স": "builders", "ইনফোটেক": "infotech", "లైఫ్": "life", "লজিস্টিকস": "logistics",
    "મીડિયા": "media", "মিডিয়া": "media", "వన్": "one", "డ్రీమ్": "dream", "ಗ್ಯಾಲಕ್ಸಿ": "galaxy", "హై": "high",
    "ಸದರ್ನ್": "southern", "ప్రీమియర్": "premier", "సన్": "sun", "బ్రైట్": "bright", "యూనిక్": "unique",
    "ಕ್ಲಾಸಿಕ್": "classic", "స్కై": "sky", "ಸನ್ರೈಸ್": "sunrise", "ఫార్చూన్": "fortune", "వైట్": "white", "మై": "my",
    "గ్రేట్": "great", "సదర్న్": "southern", "రాయల్": "royal", "గెలాక్సీ": "galaxy", "டிரீம்": "dream",
    "రియల్": "real", "సిటీ": "city", "ಸನ್": "sun", "ஈஸ்ட்": "east", "పర్ఫెక్ట్": "perfect", "సాయి": "sai",
    "హైటెక్": "hitech", "ಅಪೆಕ್ಸ್": "apex", "మోడర్న్": "modern", "అర్బన్": "urban", "ಕ್ರಿಯೇಟಿವ್": "creative",
    "నార్త్": "north", "ಈಸ್ಟರ್ನ್": "eastern", "డైనమిక్": "dynamic", "సూపర్": "super", "శివం": "shivam",
    "பிரீமியர்": "premier", "ಡ್ರೀಮ್": "dream", "ರಿಯಲ್": "real", "న్యూ": "new", "ಮೈ": "my", "ಸೂಪರ್": "super",
    "ఫ్యూచర్": "future", "சிவா": "shiva", "ಪರ್ಫೆಕ್ಟ್": "perfect", "ശ്രീ": "shree", "పయనీర్": "pioneer",
    "ನಾರ್ತ್": "north", "ఆల్": "all", "சில்வர்": "silver", "ಯುನೀಕ್": "unique", "பயனியர்": "pioneer",
    "എന്റർപ്രൈസസ്": "enterprises", "ಸೌತ್": "south", "గ్రీన్": "green", "ಮಾ": "maa", "యునైటెడ్": "united",
    "సుప్రీమ్": "supreme", "ಓಂ": "om", "స్వస్తిక్": "swastik", "ಪ್ರೈಮ್": "prime", "ಡೈನಾಮಿಕ್": "dynamic",
    "ఏపెక్స్": "apex", "ಸಿಟಿ": "city", "గుడ్": "good", "கிருஷ்ணா": "krishna", "ಗೋಲ್ಡನ್": "golden", "ಈಸ್ಟ್": "east",
    "విజన్": "vision", "ಇನೋವೇಟಿವ್": "innovative", "యూనివర్సల్": "universal", "ఏస్": "ace", "డిజిటల్": "digital",
    "ஹை": "high", "ಡಿಜಿಟಲ್": "digital", "சக்தி": "shakti", "ಒನ್": "one", "വെഞ്ചേഴ്സ്": "ventures", "ఈస్ట్": "east",
    "யுனிக்": "unique", "ఇండియన్": "indian", "సిల్వర్": "silver", "கோல்டு": "gold", "ಮಾಡರ್ನ್": "modern",
    "ನ್ಯೂ": "new", "ரியல்": "real", "హోటల్": "hotel", "ರಾಯಲ್": "royal", "ಹೈಟೆಕ್": "hitech", "ஏஸ்": "ace",
    "ഇൻഡസ്ട്രീസ്": "industries", "క్లాసిక్": "classic", "లోటస్": "lotus", "ಫ್ಯೂಚರ್": "future", "ಸ್ಕೈ": "sky",
    "ఫస్ట్": "first", "సౌత్": "south", "మా": "maa", "సన్రైజ్": "sunrise", "બ્લૂ": "blue", "லோட்டஸ்": "lotus",
    "ಇಂಡಿಯನ್": "indian", "ஒன்": "one", "வொயிட்": "white", "ಪ್ರೀಮಿಯರ್": "premier", "ವೈಟ್": "white", "బ్లూ": "blue",
    "ಫಾರ್ಚೂನ್": "fortune", "ಗುಡ್": "good", "சன்ரைஸ்": "sunrise", "സർവീസസ്": "services", "ടെക്നോളജി": "technology",
    "കൺസ്ട്രക്ഷൻ": "construction", "சூப்பர்": "super", "শিবম": "shivam", "ಸಾಯಿ": "sai", "হাই": "high",
    "ஃபர்ஸ்ட்": "first", "ಬ್ಲ್ಯಾಕ್": "black", "ఆల్ఫా": "alpha", "ಬ್ಲೂ": "blue", "പ്രൊഡക്ട്സ്": "products",
    "ಅರ್ಬನ್": "urban", "ಬಾಂಬೆ": "bombay", "കൺസൾട്ടന്റ്സ്": "consultants", "কৃষ্ণ": "krishna", "ಪಯೊನೀರ್": "pioneer",
    "டைனமிக்": "dynamic", "ಏಸ್": "ace", "ప్రైమ్": "prime", "கேலக்ஸி": "galaxy", "ಗ್ರೇಟ್": "great",
    "ક્લાસિક": "classic", "ಹೈ": "high", "പവർ": "power", "ಸ್ವಸ್ತಿಕ್": "swastik", "బాంబే": "bombay",
    "ஃபியூச்சர்": "future", "ಬ್ರೈಟ್": "bright", "ഇൻഫ്രാസ്ട്രക്ചർ": "infrastructure", "ഹെൽത്ത്കെയർ": "healthcare",
    "ಲೋಟಸ್": "lotus", "ಸಿಲ್ವರ್": "silver", "ఎస్ఎస్": "ss", "சாய்": "sai", "ഫിനാൻസ്": "finance",
    "ક્રિએટિવ": "creative", "விஷன்": "vision", "ಆಲ್ಫಾ": "alpha", "பிரைம்": "prime", "நியூ": "new", "இண்டோ": "indo",
    "ಗ್ರೀನ್": "green", "ബിൽഡേഴ്സ്": "builders", "బ్లాక్": "black", "லைஃப்": "life", "క్రియేటివ్": "creative",
    "ഇംപെക്സ്": "impex", "எஸ்எஸ்": "ss", "யுனிவர்சல்": "universal", "മാർക്കറ്റിംഗ്": "marketing", "செவன்": "seven",
    "ಸುಪ್ರೀಮ್": "supreme", "புளூ": "blue", "பம்பாய்": "bombay", "ফিউচার": "future", "ஆல்": "all",
    "ગેલેક્સી": "galaxy", "பாலாஜி": "balaji", "ഫൗണ്ടേഷൻ": "foundation", "சன்": "sun", "সানরাইজ": "sunrise",
    "মডার্ন": "modern", "குஜராத்": "gujarat", "ગુજરાત": "gujarat", "சிவம்": "shivam", "ಆಲ್": "all",
    "ಫಸ್ಟ್": "first", "കൺസൾട്ടിംഗ്": "consulting", "ఇన్నోవేటివ్": "innovative", "ഹോസ്പിറ്റാലിറ്റി": "hospitality",
    "ஆதித்யா": "aditya", "વ્હાઇટ": "white", "ஆல்ஃபா": "alpha", "ಲೈಫ್": "life", "લાઇફ": "life",
    "കൺസൾട്ടൻസി": "consultancy", "ವಿಷನ್": "vision", "പ്രോപ്പർട്ടീസ്": "properties", "சுப்ரீம்": "supreme",
    "ട്രേഡിംഗ്": "trading", "ক্লাসিক": "classic", "ગોલ્ડન": "golden", "சதர்ன்": "southern",
    "ഡെവലപ്പേഴ്സ്": "developers", "સ્કાય": "sky", "বোম্বে": "bombay", "સધર્ન": "southern",
    "എക്സ്പോർട്സ്": "exports", "ఈస్టర్న్": "eastern", "பிரைட்": "bright", "പ്രോജക്ട്സ്": "projects",
    "ஃபார்ச்சூன்": "fortune", "இந்தியன்": "indian", "എൻജിനീയറിംഗ്": "engineering", "પ્રાઇમ": "prime",
    "ഫുഡ്സ്": "foods", "ബിസിനസ്": "business", "மை": "my", "ગ્રેટ": "great", "பர்ஃபெக்ட்": "perfect", "ஸ்கை": "sky",
    "ಯುನೈಟೆಡ್": "united", "ઇન્ડિયન": "indian", "மாடர்ன்": "modern", "சிட்டி": "city", "திருப்பதி": "tirupati",
    "പ്രൊഡ്യൂസർ": "producer", "એસ": "ace", "સ્વસ્તિક": "swastik", "ডায়নামিক": "dynamic", "প্রাইম": "prime",
    "ઈસ્ટ": "east", "হাইটেক": "hitech", "கிளாசிக்": "classic", "পারফেক্ট": "perfect", "વિઝન": "vision",
    "ഫുഡ്": "food", "ইনোভেটিভ": "innovative", "கிரேட்": "great", "યુનિક": "unique", "সেভেন": "seven",
    "ડાયનેમિક": "dynamic", "সান": "sun", "ઇનોવેટિવ": "innovative", "সিটি": "city", "സോഫ്റ്റ്വെയർ": "software",
    "சிவ்": "shiv", "লোটাস": "lotus", "ஈஸ்டர்ன்": "eastern", "நார்த்": "north", "മാനേജ്മെന്റ്": "management",
    "சவுத்": "south", "பாபா": "baba", "அர்பன்": "urban", "প্রিমিয়ার": "premier", "পায়োনিয়ার": "pioneer",
    "অ্যাপেক্স": "apex", "ಎಸ್ಎಸ್": "ss", "കെയർ": "care", "গুজরাট": "gujarat", "લોટસ": "lotus", "સિટી": "city",
    "সিলভার": "silver", "ரெட்": "red", "ಶಿವಂ": "shivam", "ಯುನಿವರ್ಸಲ್": "universal", "অল": "all", "மா": "maa",
    "ലോജിസ്റ്റിക്സ്": "logistics", "ડ્રીમ": "dream", "ফার্স্ট": "first", "ফরচুন": "fortune", "பிக்": "big",
    "આલ્ફા": "alpha", "டிஜிட்டல்": "digital", "குரு": "guru", "লাইফ": "life", "স্কাই": "sky", "હાઇ": "high",
    "ব্লু": "blue", "ഇൻഫോടെക്": "infotech", "ઈસ્ટર્ન": "eastern", "ઓલ": "all", "கிரீன்": "green", "સેવન": "seven",
    "મોડર્ન": "modern", "ടെക്നോളജീസ്": "technologies", "সুপার": "super", "গুড": "good", "கோல்டன்": "golden",
    "കൺസ്ട്രക്ഷൻസ്": "constructions", "ইস্ট": "east", "ફ્યુચર": "future", "সুপ্রিম": "supreme",
    "યુનાઇટેડ": "united", "તિરુપતિ": "tirupati", "হোয়াইট": "white", "ক্রিয়েটিভ": "creative",
    "സിസ്റ്റംസ്": "systems", "ராயல்": "royal", "அபெக்ஸ்": "apex", "સિલ્વર": "silver", "ஹைடெக்": "hitech",
    "এসএস": "ss", "ইস্টার্ন": "eastern", "গ্যালাক্সি": "galaxy", "સાઉથ": "south", "ഐടി": "it", "ஹோட்டல்": "hotel",
    "ஆனந்த்": "anand", "யுனைடெட்": "united", "સન": "sun", "ગુડ": "good", "ইন্ডিয়ান": "indian", "রিয়েল": "real",
    "સુપર": "super", "યુનિવર્સલ": "universal", "હાઇટેક": "hitech", "મા": "maa", "બોમ્બે": "bombay",
    "എസ്റ്റേറ്റ്": "estate", "স্বস্তিক": "swastik", "பிளாக்": "black", "ইউনিক": "unique", "பாரத்": "bharat",
    "ડિજિટલ": "digital", "গ্রেট": "great", "আরবান": "urban", "ওয়ান": "one", "മീഡിയ": "media", "আলফা": "alpha",
    "ஸ்வஸ்திக்": "swastik", "ব্রাইট": "bright", "সাউথ": "south", "ગ્રીન": "green", "પરફેક્ટ": "perfect",
    "ଏଲ୍ଏଲ୍ପି": "llp", "ન્યૂ": "new", "પાયોનિયર": "pioneer", "બ્રાઇટ": "bright", "মাই": "my", "নর্থ": "north",
    "એસએસ": "ss", "સનરાઇઝ": "sunrise", "নিউ": "new", "എനർജി": "energy", "બ્લેક": "black", "ફોર્ચ્યુન": "fortune",
    "અર્બન": "urban", "கிரியேட்டிவ்": "creative", "માય": "my", "তিরুপতি": "tirupati", "பெஸ்ட்": "best",
    "রয়্যাল": "royal", "গ্রীন": "green", "ইউনাইটেড": "united", "ইউনিভার্সাল": "universal", "এইস": "ace",
    "ব্ল্যাক": "black", "પ્રીમિયર": "premier", "സൊല്യൂഷൻസ്": "solutions", "বিজয়": "vijay", "રિયલ": "real",
    "સુપ્રીમ": "supreme", "ভিশন": "vision", "ਐਲਐਲਪੀ": "llp", "সাউদার্ন": "southern", "વન": "one", "মা": "maa",
    "એપેક્સ": "apex", "ড্রিম": "dream", "இன்னோவேட்டிவ்": "innovative", "ডিজিটাল": "digital", "ફર્સ્ટ": "first",
    "സെവൻ": "seven", "സിറ്റി": "city", "സ്മാർട്ട്": "smart", "ബ്ലാക്ക്": "black", "വിഷൻ": "vision",
    "പയനിയർ": "pioneer", "എസ്എസ്": "ss", "ഈസ്റ്റേൺ": "eastern", "ന്യൂ": "new", "റിയൽ": "real", "ਗਲੋਬਲ": "global",
    "യൂണിവേഴ്സൽ": "universal", "ഗുഡ്": "good", "ഇന്ത്യൻ": "indian", "ഫസ്റ്റ്": "first", "ഹോട്ടൽ": "hotel",
    "മൈ": "my", "ബോംബെ": "bombay", "ഡ്രീം": "dream", "സൺ": "sun", "ഓൾ": "all", "ഡൈനാമിക്": "dynamic",
    "ആൽഫ": "alpha", "വൈറ്റ്": "white", "ലോട്ടസ്": "lotus", "ਟੈਕ": "tech", "സ്കൈ": "sky", "ഹൈ": "high",
    "സതേൺ": "southern", "നോർത്ത്": "north", "ಜೇ": "jay", "ലൈഫ്": "life", "ഗാലക്സി": "galaxy",
    "ଇଣ୍ଟରନ୍ୟାସନାଲ୍": "international", "ഡിജിറ്റൽ": "digital", "അപെക്സ്": "apex", "ਇੰਟਰਨੈਸ਼ਨਲ": "international",
    "ക്ലാസിക്": "classic", "സൺറൈസ്": "sunrise", "വൺ": "one", "ബ്രൈറ്റ്": "bright", "യുണൈറ്റഡ്": "united",
    "മോഡേൺ": "modern", "മാ": "maa", "ଟେକ୍": "tech", "ഹൈടെക്": "hitech", "പെർഫെക്റ്റ്": "perfect",
    "തിരുപ്പതി": "tirupati", "സ്വസ്തിക്": "swastik", "സൗത്ത്": "south", "ക്രിയേറ്റീവ്": "creative",
    "ഗുജറാത്ത്": "gujarat", "ഈസ്റ്റ്": "east", "പ്രീമിയർ": "premier", "ഗ്രീൻ": "green", "ബ്ലൂ": "blue",
    "സുപ്രീം": "supreme", "ഇന്നൊവേറ്റീവ്": "innovative", "ପ୍ରୋଜେକ୍ଟସ୍": "projects", "ഫ്യൂച്ചർ": "future",
    "ਪ੍ਰੋਜੈਕਟਸ": "projects", "லட்சுமி": "lakshmi", "എയ്സ്": "ace", "ഗോൾഡൻ": "golden", "യുണീക്ക്": "unique",
    "ଇନଫ୍ରାଷ୍ଟ୍ରକଚର୍": "infrastructure", "പ്രൈം": "prime", "അർബൻ": "urban", "કૃષ્ણ": "krishna", "ఓం": "om",
    "ਕੇਅਰ": "care", "ਇਸਟੇਟ": "estate", "ഇന്തോ": "indo", "സിൽവർ": "silver", "ഫോർച്യൂൺ": "fortune",
    "ଇଣ୍ଡଷ୍ଟ୍ରିଜ୍": "industries", "ହସ୍ପିଟାଲିଟି": "hospitality", "ଏଷ୍ଟେଟ୍": "estate", "ଫୁଡ୍": "food",
    "ਐਗਰੋ": "agro", "ਐਂਟਰਪ੍ਰਾਈਜ਼ਿਜ਼": "enterprises", "ମ୍ୟାନେଜମେଣ୍ଟ୍": "management", "ଫୁଡ୍ସ୍": "foods",
    "ଏନର୍ଜି": "energy", "ഗ്രേറ്റ്": "great", "ଇନଫୋଟେକ୍": "infotech", "સાંઈ": "sai", "ଫାଇନାନ୍ସ୍": "finance",
    "ਸਿਸਟਮਜ਼": "systems", "ଫାଉଣ୍ଡେସନ୍": "foundation", "ପାୱାର୍": "power", "ସଫ୍ଟୱେୟାର୍": "software",
    "ହେଲ୍ଥକେୟାର୍": "healthcare", "ସଲ୍ୟୁସନ୍ସ୍": "solutions", "ଏକ୍ସପୋର୍ଟସ୍": "exports",
    "କନଷ୍ଟ୍ରକସନ୍": "construction", "ଡେଭଲପର୍ସ୍": "developers", "ଶ୍ରୀ": "shree", "କନସଲଟାନ୍ସି": "consultancy",
    "ਹੈਲਥਕੇਅਰ": "healthcare", "ਪ੍ਰੋਡਕਟਸ": "products", "ପ୍ରୋପର୍ଟିଜ୍": "properties", "ଟେକ୍ନୋଲୋଜି": "technology",
    "ਸਰਵਿਸਿਜ਼": "services", "ଶିବ": "shiv", "സൂപ്പർ": "super", "ਸੋਲਿਊਸ਼ਨਜ਼": "solutions",
    "ଟେକ୍ନୋଲୋଜିସ୍": "technologies", "କେୟାର୍": "care", "ਇਨਵੈਸਟਮੈਂਟਸ": "investments", "ਕੰਸਲਟੈਂਟਸ": "consultants",
    "ਕੰਸਲਟੈਂਸੀ": "consultancy", "ਪਾਵਰ": "power", "କନଷ୍ଟ୍ରକସନ୍ସ୍": "constructions", "ଇମ୍ପେକ୍ସ୍": "impex",
    "ਫੂਡਜ਼": "foods", "ଇନଭେଷ୍ଟମେଣ୍ଟସ୍": "investments", "ବିଜନେସ୍": "business", "କନସଲଟିଂ": "consulting",
    "ଇନଫ୍ରା": "infra", "ਹਾਸਪਿਟੈਲਿਟੀ": "hospitality", "ମିଡିଆ": "media", "ଭେଞ୍ଚର୍ସ୍": "ventures",
    "ਫਾਊਂਡੇਸ਼ਨ": "foundation", "ਲਕਸ਼ਮੀ": "lakshmi", "କନସଲଟାଣ୍ଟସ୍": "consultants", "ଇଞ୍ଜିନିୟରିଂ": "engineering",
    "ਪ੍ਰੋਡਿਊਸਰ": "producer", "ଏଣ୍ଟରପ୍ରାଇଜେସ୍": "enterprises", "ଆଇଟି": "it", "ਸਾਫਟਵੇਅਰ": "software",
    "ବିଲଡର୍ସ୍": "builders", "സായി": "sai", "ଇନଭେଷ୍ଟମେଣ୍ଟ୍": "investment", "ਇੰਡਸਟ੍ਰੀਜ਼": "industries",
    "ਬਿਲਡਰਜ਼": "builders", "ପ୍ରୋଡକ୍ଟସ୍": "products", "ਟੈਕਨਾਲੋਜੀਜ਼": "technologies", "ସିଷ୍ଟମସ୍": "systems",
    "ਇੰਜੀਨੀਅਰਿੰਗ": "engineering", "ਕੰਸਟ੍ਰਕਸ਼ਨ": "construction", "ପ୍ରୋଡ୍ୟୁସର୍": "producer",
    "ମାର୍କେଟିଂ": "marketing", "ਡਿਵੈਲਪਰਜ਼": "developers", "ਇੰਪੈਕਸ": "impex", "ਇਨਵੈਸਟਮੈਂਟ": "investment",
    "ਕੰਸਲਟਿੰਗ": "consulting", "ਮਾਰਕੀਟਿੰਗ": "marketing", "ਟ੍ਰੇਡਿੰਗ": "trading", "ਲੌਜਿਸਟਿਕਸ": "logistics",
    "ସର୍ଭିସେସ୍": "services", "ਇਨਫੋਟੈਕ": "infotech", "ਇਨਫ੍ਰਾ": "infra", "ਇਨਫ੍ਰਾਸਟ੍ਰਕਚਰ": "infrastructure",
    "ଲଜିଷ୍ଟିକ୍ସ୍": "logistics", "ਵੈਂਚਰਜ਼": "ventures", "ਐਨਰਜੀ": "energy", "സായ്": "sai", "ਐਕਸਪੋਰਟਸ": "exports",
    "ਟੈਕਨਾਲੋਜੀ": "technology", "ਮੈਨੇਜਮੈਂਟ": "management", "କ୍ରିଏଟିଭ୍": "creative", "ଟ୍ରେଡିଂ": "trading",
    "ਮੀਡੀਆ": "media", "ਫਾਈਨੈਂਸ": "finance", "ਪ੍ਰੋਪਰਟੀਜ਼": "properties", "ਆਈਟੀ": "it", "ਸ੍ਰੀ": "sree",
    "ਫੂਡ": "food", "ਕੰਸਟ੍ਰਕਸ਼ਨਜ਼": "constructions", "ମାଇ": "my", "ପ୍ରାଇମ୍": "prime", "ਸਨ": "sun",
    "ମଡର୍ନ୍": "modern", "ଫର୍ଚୁନ୍": "fortune", "ହ୍ୱାଇଟ୍": "white", "ଗ୍ରୀନ୍": "green", "ਸ਼ਿਆਮ": "shyam",
    "ਬਿਜ਼ਨਸ": "business", "ਡਿਜੀਟਲ": "digital", "ଲାଇଫ୍": "life", "ਫਾਰਚੂਨ": "fortune", "ଏପେକ୍ସ୍": "apex",
    "ସ୍ୱସ୍ତିକ": "swastik", "ଇନୋଭେଟିଭ୍": "innovative", "ସିଲଭର୍": "silver", "ਗੋਲਡ": "gold", "ତିରୁପତି": "tirupati",
    "ਹੋਟਲ": "hotel", "ਕ੍ਰਿਏਟਿਵ": "creative", "କୃଷ୍ଣ": "krishna", "ପର୍ଫେକ୍ଟ୍": "perfect", "ସ୍କାଇ": "sky",
    "ଫ୍ୟୁଚର୍": "future", "ଲୋଟସ୍": "lotus", "ଭିଜନ୍": "vision", "ଶିବମ୍": "shivam", "ਡ੍ਰੀਮ": "dream",
    "ଗ୍ରେଟ୍": "great", "ਈਸਟ": "east", "ਫਿਊਚਰ": "future", "ବେଷ୍ଟ୍": "best", "ਏਸ": "ace", "ବ୍ଲାକ୍": "black",
    "ਮਾਂ": "maa", "ਸੁਪਰੀਮ": "supreme", "ਅਲਫ਼ਾ": "alpha", "ୟୁନିକ୍": "unique", "ବିଜୟ": "vijay", "ହାଇ": "high",
    "ସେଭେନ୍": "seven", "ଇଷ୍ଟର୍ନ୍": "eastern", "ପାୟୋନିୟର୍": "pioneer", "ਪ੍ਰੀਮੀਅਰ": "premier", "ਸਵਾਸਤਿਕ": "swastik",
    "ବ୍ଲୁ": "blue", "ଷ୍ଟାର୍": "star", "ਬ੍ਰਾਈਟ": "bright", "ਰੈੱਡ": "red", "ସୁପର୍": "super", "ଡିଜିଟାଲ୍": "digital",
    "ਯੂਨੀਕ": "unique", "ୟୁନିଭର୍ସାଲ୍": "universal", "ୟୁନାଇଟେଡ୍": "united", "ਵਿਜ਼ਨ": "vision", "ਗੋਲਡਨ": "golden",
    "ନର୍ଥ": "north", "ਯੂਨੀਵਰਸਲ": "universal", "ਸਨਰਾਈਜ਼": "sunrise", "ਗਲੈਕਸੀ": "galaxy", "ଏସ୍": "ace", "ସନ୍": "sun",
    "ਸੈਵਨ": "seven", "ਸਿਲਵਰ": "silver", "ଡାଇନାମିକ୍": "dynamic", "ସୁପ୍ରିମ୍": "supreme", "ਡਾਇਨਾਮਿਕ": "dynamic",
    "ସନ୍ରାଇଜ୍": "sunrise", "ਸਕਾਈ": "sky", "ଗୁଡ୍": "good", "କ୍ଲାସିକ୍": "classic", "ଫାର୍ଷ୍ଟ୍": "first",
    "ਸਦਰਨ": "southern", "ਮਾਈ": "my", "ਲੋਟਸ": "lotus", "ਸਟਾਰ": "star", "ਰੀਅਲ": "real", "ਸਾਊਥ": "south",
    "ਐੱਸਐੱਸ": "ss", "ଆଲଫା": "alpha", "ਵਿਜੈ": "vijay", "ଏସ୍ଏସ୍": "ss", "ਗ੍ਰੇਟ": "great", "ବ୍ରାଇଟ୍": "bright",
    "ସାଉଥର୍ନ୍": "southern", "ਆਦਿਤਿਆ": "aditya", "ਹਾਈਟੈਕ": "hitech", "ਇਨੋਵੇਟਿਵ": "innovative", "ସୂର୍ଯ୍ୟ": "surya",
    "ରୟାଲ୍": "royal", "ਆਲ": "all", "ରିଅଲ୍": "real", "ୱାନ୍": "one", "ਨਿਊ": "new", "ਲਾਈਫ਼": "life", "ਬੈਸਟ": "best",
    "ਪ੍ਰਾਈਮ": "prime", "ਸੁਪਰ": "super", "ਮਾਡਰਨ": "modern", "ନ୍ୟୁ": "new", "ହାଇଟେକ୍": "hitech", "ਗੁੱਡ": "good",
    "ବମ୍ବେ": "bombay", "ଆଦିତ୍ୟ": "aditya", "ਹਾਈ": "high", "ସାଉଥ୍": "south", "ପ୍ରିମିୟର୍": "premier",
    "ਈਸਟਰਨ": "eastern", "ਗ੍ਰੀਨ": "green", "ସିଟି": "city", "ଗାଲାକ୍ସି": "galaxy", "ਬਲੈਕ": "black", "ਵਨ": "one",
    "ਗੁਜਰਾਤ": "gujarat", "ਸਮਾਰਟ": "smart", "ਕਲਾਸਿਕ": "classic", "ଇଷ୍ଟ୍": "east", "ਯੂਨਾਈਟਿਡ": "united",
    "ਪਾਇਨੀਅਰ": "pioneer", "ଅର୍ବାନ୍": "urban", "ਤਿਰੂਪਤੀ": "tirupati", "ਬਲੂ": "blue", "ଗୁଜରାଟ": "gujarat",
    "ਸਿਟੀ": "city", "ਪਰਫੈਕਟ": "perfect", "ਅਰਬਨ": "urban", "ମା": "maa", "ଡ୍ରିମ୍": "dream", "ਰੌਇਲ": "royal",
    "ଓଁ": "om", "ਰਾਇਲ": "royal", "ਨਾਰਥ": "north", "ਸਾਈਂ": "sai", "ਵਾਈਟ": "white", "ਬੰਬੇ": "bombay",
    "ਫਰਸਟ": "first", "ਸੂਰਜ": "surya", "ਸ਼੍ਰੀ": "shree", "ਵ੍ਹਾਈਟ": "white", "ਨੌਰਥ": "north", "ਫਸਟ": "first",
    "ਬੌਂਬੇ": "bombay", "ਸੂਰਿਆ": "surya",
}
set_word_map(WORD_MAP_DATA)
