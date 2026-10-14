"""French normalisation rules for the Amazon ML Challenge 2026 entity-resolution task.

France is ~15% of the test set and absent from training, so the model never learned French legal
forms or address abbreviations; these dictionaries are the main lever.

The dictionaries below are a hand-written *candidate pool* (FR_LEGAL 52, FR_ADDR 89, FR_ADDR_DROP 11), and
this pool is what the pipeline imports. ``analysis/fr_analysis.py`` checks it against the French rows of
``test_source{1,2,3}.tsv`` and writes ``analysis/fr_rules_observed.py``: the subset occurring at least five
times (34 / 59 / 9). That file documents which rules are backed by observed tokens; the remaining pool
entries occur fewer than five times in the French rows, so they touch at most a handful of records.

Token-wise rules mirror ``LEGAL`` and ``ADDR`` in ``features.py``; values may contain spaces
(``zi`` -> ``zone industrielle``) because normalisation re-joins tokens with a space.
"""
import re

import unicodedata


def ascii_fold(s):
    """Lowercase and strip accents; non-Latin scripts are kept as-is (same as features.ascii_fold)."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


_non_word = re.compile(r"[^\w]+")
_postcode = re.compile(r"\b(\d{5})\b")
# "S.A.R.L." / "S A S" survive punctuation splitting as single letters; glue them back together.
_initials = re.compile(r"\b(?:[a-z][^\w]+){2,}[a-z]\b")

# --- names ---------------------------------------------------------------------------------------

# True legal forms and grammatical fillers, always dropped from the "core" name.
FR_LEGAL = {
    # limited companies / partnerships
    "sarl", "sarlu", "eurl", "sas", "sasu", "sa", "snc", "scs", "sca", "sasp",
    # civil / property companies
    "sci", "scm", "scp", "sccv", "scpi", "scia",
    # regulated professions
    "selarl", "selas", "selasu", "selca", "selafa", "aarpi",
    # sole traders / farming / co-ops
    "ei", "eirl", "earl", "gaec", "scea", "scop", "scic", "sem", "spl", "gie", "sep",
    # company-form abbreviations, the French equivalents of "Co." / "Company", which the team's
    # LEGAL set already drops for US names
    "ets", "etablissements", "etablissement", "cie", "compagnie", "sc",
    # grammatical fillers (features.LEGAL drops "the" / "and" / "of" the same way)
    "et", "and", "de", "du", "des", "la", "le", "les", "l", "d", "au", "aux", "en",
    # "ste" is deliberately absent: in French names it is Sainte (Ecole Ste-Therese)
    # far more often than an abbreviation of societe.
}

# Descriptive business words. These LOOK like fillers but can be the only thing separating two
# businesses: the France audit found decoys built as "<true name> + <a real word>", and "Groupe"
# is one of the words used. Dropping them turns "X Groupe" and "X" into the same core name, so
# they are kept by default and dropping is opt-in via normalise_fr_name(drop_filler=True).
# Verified on the 119 audited pairs: dropping or keeping them flips no verdict (0 of 119).
FR_FILLER = {
    "groupe", "association", "asso", "societe", "entreprise", "cabinet",
}

# Token rewrites inside names. Saint/Sainte appear abbreviated in one source and spelled out in
# another ("Ecole Ste-Therese" vs "Ecole Sainte Therese"), so they are collapsed, not dropped.
FR_NAME_MAP = {
    "st": "saint", "ste": "saint", "sts": "saint", "stes": "saint",
    "sainte": "saint", "saints": "saint", "saintes": "saint",
}

# Junk prefixes/suffixes seen wrapped around French names.
FR_NAME_JUNK = re.compile(r"^[\s<>\-*_.,;:/\|\"']+|[\s<>\-*_.,;:/\|\"']+$")

# --- addresses -----------------------------------------------------------------------------------

# Abbreviation -> canonical form, applied token-wise after ascii folding.
FR_ADDR = {
    # ways
    "r": "rue", "ru": "rue", "rue": "rue",
    "av": "avenue", "ave": "avenue", "avn": "avenue", "aven": "avenue",
    "bd": "boulevard", "bld": "boulevard", "blvd": "boulevard", "boul": "boulevard",
    "ch": "chemin", "chem": "chemin", "che": "chemin",
    "all": "allee", "allee": "allee", "allees": "allee",
    "imp": "impasse", "impa": "impasse",
    "pl": "place", "plc": "place",
    "rte": "route", "rt": "route",
    "fg": "faubourg", "fbg": "faubourg",
    "sq": "square",
    "qu": "quai", "qai": "quai",
    "crs": "cours",
    "sen": "sentier", "sent": "sentier",
    "pass": "passage",
    "prom": "promenade",
    "pte": "porte",
    "rpt": "rond point", "rdpt": "rond point",
    "mte": "montee",
    "gale": "galerie", "galerie": "galerie",
    # honorifics, very common inside French street names
    "gal": "general", "gle": "general", "gnl": "general", "gen": "general",
    # NOT expanded, checked against the data: "pas" is the departement Pas-de-Calais (14.3k S2
    # rows, all Calais) far more often than an abbreviation of passage; "cap" is Lege-Cap-Ferret
    # (commune); "pres" is
    # pre/pres (meadow, as in "allee des Pres"), "mar" is the Basse Mar in Nantes, "car" is
    # "car ferry" in Calais, "dom" is as often the title Dom as it is domaine.
    "mal": "marechal",
    "pdt": "president", "pst": "president",
    "dr": "docteur", "prof": "professeur", "me": "maitre", "mgr": "monseigneur",
    "cdt": "commandant", "col": "colonel", "lt": "lieutenant",
    "vla": "villa",
    "ham": "hameau",
    "ld": "lieu dit", "lieudit": "lieu dit",
    "quart": "quartier", "qrt": "quartier",
    "espl": "esplanade",
    "vc": "voie communale", "voie": "voie",
    # buildings / sub-addresses
    "bat": "batiment", "bt": "batiment", "bati": "batiment",
    "res": "residence", "resid": "residence", "rsd": "residence",
    "esc": "escalier",
    "appt": "appartement", "apt": "appartement", "app": "appartement",
    "etg": "etage",
    "imm": "immeuble",
    "ctre": "centre", "ctr": "centre", "cc": "centre commercial",
    # zones
    "zi": "zone industrielle", "za": "zone artisanale", "zac": "zone amenagement concerte",
    "zae": "zone activite economique", "zone": "zone",
    # saints (gender and abbreviation collapsed)
    "st": "saint", "ste": "saint", "sts": "saint", "stes": "saint",
    "sainte": "saint", "saints": "saint", "saintes": "saint",
}

# Postal-routing and house-number noise dropped from addresses (inconsistent across sources).
#
# "france" / "fr" are deliberately NOT here. Dropping them token-wise mangles the region name
# "Hauts-de-France" -> "hauts de", which is worse than leaving a country label in place. The
# country/region/departement component is an admin-area problem, not a token problem: drop it at
# component level with FR_ADMIN (see analysis/fr_admin.py) if it is dropped at all.
#
# Note also that "bis"/"ter" only fire when separated ("5 bis"); "5bis" is a single alphanumeric
# token and survives intact. Given that house numbers are unreliable across sources anyway
# (the same business appears at 5bis, 5B and 16bis), this is left as-is rather than split.
FR_ADDR_DROP = {
    "bis", "ter", "quater", "quinquies",
    "cedex", "cdx", "bp", "cs", "tsa", "boite", "postale",
}

# Website records: ~11% of matched French S2/S3 rows are a domain, not a name
# ("adeptestransportssarl.com" for "Adeptes Transports SARL"), which tokenises to one
# unsplittable token and scores 0 against the real name.
FR_TLD = re.compile(r"\.(?:com|fr|net|org|eu|info|biz)$", re.I)
_alnum = re.compile(r"[^a-z0-9]+")

# NOTE on postcodes: the French TEST rows carry NO 5-digit postcode at all (0 of 238 audited
# addresses). The format is "<number> <street>, <City>, <Region>", components in any order.
# fr_postcode()/fr_department() are kept for completeness but return "" on this data.

# --- helpers -------------------------------------------------------------------------------------


def glue_initials(s):
    """Collapse dotted/spaced initialisms so "S.A.R.L." and "S A S" reach the legal-form set."""
    return _initials.sub(lambda m: re.sub(r"[^\w]", "", m.group(0)), s)


def normalise_fr_name(s, drop_legal=True, drop_filler=False):
    """Normalise a French business name: fold accents, strip junk, drop legal forms.

    drop_filler additionally removes FR_FILLER (groupe, societe, association, ...). It defaults to
    False because those words can be the only difference between two businesses; turn it on only if
    a measurement on the full French names shows it helps.
    """
    s = glue_initials(FR_NAME_JUNK.sub("", ascii_fold(s).replace("&", " and ")))
    tokens = [FR_NAME_MAP.get(t, t) for t in _non_word.split(s) if t]
    drop = (FR_LEGAL | FR_FILLER) if drop_filler else FR_LEGAL
    if drop_legal:
        tokens = [t for t in tokens if t not in drop]
    return " ".join(tokens)


def fr_core_name(s):
    """French core name with tokens sorted, so word swaps compare equal."""
    return " ".join(sorted(normalise_fr_name(s).split()))


def normalise_fr_addr(s, drop_noise=True):
    """Normalise a French address: fold accents, expand abbreviations, drop routing noise."""
    tokens = []
    for t in _non_word.split(ascii_fold(s)):
        if not t or (drop_noise and t in FR_ADDR_DROP):
            continue
        tokens.append(FR_ADDR.get(t, t))
    return " ".join(tokens)


def squash_name(s):
    """Name reduced to bare alphanumerics with any TLD removed, for domain-vs-name comparison.

    Deliberately keeps legal forms and does not collapse saint/sainte: the domain spells the
    name out as one run ("etsterreseurl.com" -> "etsterreseurl"), so dropping tokens on one
    side only would break the very comparison this enables.
    """
    return _alnum.sub("", FR_TLD.sub("", ascii_fold(s).strip()))


def is_domain_name(s):
    """True when a business_name is really a website."""
    return bool(FR_TLD.search(s.strip()))


def fr_postcode(s):
    """Return the 5-digit French postcode found in an address, or ''."""
    m = _postcode.search(s)
    return m.group(1) if m else ""


def fr_department(s):
    """Return the 2-digit departement prefix of the address postcode, or ''."""
    return fr_postcode(s)[:2]


def add_fr_columns(df):
    """Add French normalised columns to a dataframe of test rows (in place)."""
    df["fr_name"] = df["business_name"].map(normalise_fr_name)
    df["fr_core"] = df["business_name"].map(fr_core_name)
    df["fr_addr"] = df["business_address"].map(normalise_fr_addr)
    df["fr_postcode"] = df["business_address"].map(fr_postcode)
    df["fr_dept"] = df["fr_postcode"].str[:2]
    return df
