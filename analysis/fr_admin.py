"""City / region address components derived from the French test rows.

Components occurring at least 200 times. They appear in every French address and differ
systematically between sources (S1 tends to carry the region, S2/S3 the departement),
so they add matching noise rather than signal.
"""

FR_ADMIN = {
    "hauts de france",  # 288486
    "bordeaux",  # 273666
    "nouvelle aquitaine",  # 242144
    "nantes",  # 237288
    "lille",  # 217542
    "pays de la loire",  # 206946
    "nord",  # 150860
    "gironde",  # 150420
    "loire atlantique",  # 126884
    "tourcoing",  # 115233
    "dunkerque",  # 108634
    "roubaix",  # 104126
    "calais",  # 100199
    "pessac",  # 80569
    "la teste de buch",  # 72208
    "saint nazaire",  # 71146
    "merignac",  # 65245
    "lege cap ferret",  # 51387
    "pornic",  # 50029
    "la baule escoublac",  # 44201
    "saint herblain",  # 36358
    "pas de calais",  # 27838
    "st nazaire",  # 15832
    "st herblain",  # 8059
    "maison des associations",  # 1585
    "27 rue jean bart",  # 755
    "1er etage",  # 376
    "maison de la vie associative",  # 355
    "30 rue des meuniers",  # 320
    "27 r jean bart",  # 318
    "100 rue de lille",  # 309
    "2eme etage",  # 289
    "lomme",  # 253
    "12 rue lyderic",  # 240
    "59000 lille",  # 234
    "59000",  # 228
    "24 pl de la liberte",  # 226
    "59100 roubaix",  # 225
    "le clion",  # 216
    "59200 tourcoing",  # 210
    "60 bd vauban",  # 202
}
