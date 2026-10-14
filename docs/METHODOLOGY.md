# ML Challenge 2026: Business Entity Resolution

**Team Name:** Thadaladi

**Team Members:** Ganesan P V (Team Leader), Mithilesh Kumar U S, Venkatkaushal A, N Sankkarshana

**Submission Date:** 27 September 2026 (final submission; final public LB **0.981722**)

> A PDF version of this document with eight figures (pipeline, blocking funnel, candidate-set sizes, feature
> importance, cross-encoder gains, France analysis, leaderboard history) is included as `METHODOLOGY.pdf`.

| Requested item | Where in this document |
|---|---|
| Methodology used | §1 Executive Summary, §2 Methodology, §6 Validation |
| Candidate generation / blocking strategy | §3 Candidate Generation (final candidate set: 5.91 per S1 entity, 10.25M pairs, ~98.2% recall) |
| Model architecture and feature engineering | §4 Matching Model (LightGBM, ~73 features), §5 Cross-Encoder, Final Ensemble and France |
| Any other relevant information | §6 Validation, §7 Results & Error Analysis (incl. what did not work), §8 Data Sources & Compliance, §9 Conclusion, Appendix (code) |

**Final solution at a glance**

| Stage | What | Output |
|---|---|---|
| 0. Normalise | case/accent folding, Indic → Latin transliteration, legal-form and abbreviation rules, French rules | clean text views |
| 1. Retrieval | per country: TF-IDF char n-grams (name+address, core name, address) → SVD → GPU top-k, ∪ multilingual-e5-small | ~150 candidates per S1, recall 0.988 |
| 2. Pruner | LightGBM on retrieval scores, keep top 5 with p ≥ 0.01 | **5.91 candidates per S1**, recall 0.982 |
| 3. Matcher | LightGBM on ~73 pair features, trained on test-like (orphan + dense decoy) data | P(match) per pair |
| 4. Cross-encoder | multilingual-e5-base fine-tuned on raw pair text, score stacked into the LightGBM | P(match) per pair |
| 5. Ensemble | US/India: 6 cross-encoders + stage 3. France: French-noise-augmented cross-encoders + stage 3 | blended P(match) |
| 6. Decision | one S1 per record; entity's first record p > 0.75, extra records p > 0.35 (France 0.70 / 0.40) | `matching_results.tsv` |

---

## 1. Executive Summary

We treat the task as **record-to-reference linkage**. Source 1 is deduplicated and, in the training
ground truth, every Source 2/3 record matches **at most one** Source 1 entity — so instead of scoring
all pairs we search Source 1 for the single best match of each Source 2/3 record, then decide
"match / no match" with a precision-first threshold tuned directly on macro F0.5.

The pipeline has four stages. **Blocking** retrieves candidates per country from three TF-IDF views
plus multilingual-e5 embeddings; a **learned pruner** cuts the set to **5.91 candidates per S1 entity at
0.9816 recall** — 10.25M pairs out of 6.73 trillion same-country pairs (17.27 trillion overall); a **~73-feature LightGBM** ranks what
survives under a one-to-one assignment constraint; and a **fine-tuned cross-encoder** reads each
surviving pair as raw text, its score stacked back in as one more feature.

Public leaderboard: 0.9732 (v5) → 0.975339 (v9) → 0.980065 (one cross-encoder) → **0.981722 (final)**.
The final model is an ensemble: six raw-text e5-base cross-encoders plus the stage-3 LightGBM for US/India,
and two French-noise-augmented cross-encoders for France.

Three country-specific problems shaped the design and are treated explicitly: Indian records written in
Indic scripts, French records absent from training entirely, and a decoy population (~26% of S2/S3)
that the generator built from real entities rather than at random.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the full training set (streamed, no sampling):

| Fact | Value | Consequence |
|---|---|---|
| Train size | S1 2.21M, S2 5.03M, S3 5.29M | Blocking must be scalable |
| Test size | S1 1.73M, S2 4.89M, S3 5.08M | ~10M queries at inference |
| Test countries (S1) | India 810k, US 663k, **France 259k (15%)** | France is unseen in train |
| Singleton S1 entities | 5.6% | An empty submission scores ≈0.056; recall matters too |
| Matches per S1 entity | mostly 2–6 (max 11) | Multi-match output per entity |
| S2/S3 ids linked to >1 S1 entity | **0** | One-to-one assignment constraint |
| S2/S3 records with no match | ~26% | Decoys must be rejected (threshold) |

General noise: legal-suffix movement (`LLC Moncada Learning Center`), junk prefixes (`--`, `<<`), case
changes, typos, abbreviations (St/Street, Pvt/Private), component reordering in addresses
(`OH, Columbus, 5559 Orville Avenue`), and missing addresses.

#### Core-name collisions are dataset-wide

Share of S1 entities whose normalised core name is shared with at least one other S1 entity of the same
country:

| | train | test |
|---|---|---|
| India | 54.19% | 53.43% |
| US | 46.42% | 38.67% |
| France | — (not in train) | 54.58% |

The shapes differ: US collisions are single generic words (`meridian` ×556, `cedar` ×345), India and
France are templated phrases (`india mumbai`, `delhi new services`; `bordeaux club` ×530). More
important is the effect on real matches — on sampled true train pairs the S2/S3 core name equals its S1
core name in 62.2% (India) / 55.7% (US) of cases, and **in roughly half of those at least one other S1
entity carries the identical name** (unambiguous: 41.9% India, 52.3% US; median 1 rival in India,
tail to 555 in the US). A correct name match is frequently not a sufficient one, which is what the
one-to-one assignment constraint absorbs.

#### India: non-Latin scripts

About **36% of India S2/S3 records** carry Indic text while the S1 record is in Latin, so the two share
no characters and character-n-gram TF-IDF cannot retrieve them. We transliterate Indic script to Latin
inside `ascii_fold()`, before the blocking index is built. Measured on 50k India S1 entities with
173,733 true matches:

| Queries | | R@1 | R@5 | R@10 | R@20 | R@50 |
|---|---|---|---|---|---|---|
| Indic-script (62,992) | before | 0.908 | 0.941 | 0.953 | 0.962 | 0.974 |
| | **after** | **0.978** | **0.993** | **0.995** | **0.997** | **0.999** |
| Latin (110,741) | before = after | 0.961 | 0.981 | 0.986 | 0.990 | 0.993 |
| All India | before | 0.942 | 0.967 | 0.974 | 0.980 | 0.986 |
| | **after** | **0.967** | **0.985** | **0.990** | **0.993** | **0.995** |

At top-20, missed matches fall from 3.8% to 0.27% — **93% fewer**. Latin records are identical to four
decimals, so there is no collateral damage. On 276,000 held-out true pairs, mean `core_name` similarity
between the Indic record and its Latin counterpart rises from **8.6 to 99.7** (`fuzz.ratio`).

`transliterate.py` is our own ~200 lines of pure-stdlib Python with **zero runtime dependencies**. All
nine Indic Unicode blocks share an internal layout, so one table indexed by `codepoint − block_start`
covers every script, plus Hindi-style schwa deletion (`राम` → `ram`), nukta letters, anusvara → `n`/`m`,
Tamil/Malayalam special cases, and a 1,051-entry word map (plus 13 abbreviations) learned from our own
training pairs. We evaluated `indic-transliteration` (sanscript, MIT) as a baseline and rejected it:
Sanskrit-oriented output and broken Tamil (`பிரைவேட்` → `bhiraiveDh`), `fuzz.ratio` 48 against 99.7.

Residual: **mixed-script names** where one token is Indic and the rest Latin (`జై Solutions Private
Limited` → `jai` against S1's `Jay`), and Indic **state names** inside otherwise-Latin addresses. These
account for 8–12% of the remaining errors (section 7).

#### France: 15% of test, absent from training

France is in no training row, so rules are the main lever. Four findings change how France must be
handled.

**1. French addresses carry effectively no postcode.** The format is `<number> <street>, <City>,
<Région>` with components in arbitrary order. Zero of 238 audited French addresses contain a 5-digit
code. A handful of postcode-bearing components do exist in the wider French text (`59000 lille` 234
occurrences, `59100 roubaix` 225, `59200 tourcoing` 210), but they are a fraction of a percent, so
`postcode_eq` is −1 on essentially every French pair. The model agrees: its gain importance is **0.00013**, the
second-lowest of the v5 model's 58 features.

**2. Address agreement is not evidence of a match in France.** In a hand-audit of 30 French entities
(119 pairs, 111 correct / 8 wrong, **93.3% precision**), six of the eight false positives sat at the
*exact same street address* as the true entity, while genuine matches frequently disagreed on the house
number (one business appearing at `5bis`, `5B` and `16bis` of one street).

**3. The sources describe location at different administrative depths.** S1 tends to carry the région,
S2/S3 the département (`Nantes, Pays de la Loire` vs `Calais, Pas-de-Calais`), so the trailing address
component of a cross-source French pair is systematically mismatched. The same holds in India, where
S1 carries a district that S2/S3 drop (`Kolkata, Kolkata, Howrah` → `KOLKATA, KOLKATA`) — 44.7% of
sampled Indian true pairs. Street types are also spelled out in S1 and abbreviated in S2/S3 (`rue`
170,819 in S1 vs `r` 178,584 in S2).

**4. ~11% of matched French S2/S3 records store a website instead of a name**
(`adeptestransportssarl.com`), which tokenises to one unsplittable token and scores `name_ratio = 0.0`.

**Does France actually behave?** With no French labels we cannot score it directly, so we compared its
*output distribution* against the labelled countries on the test set:

| | France | India | US |
|---|---|---|---|
| share of S1 entities receiving ≥1 match at t = 0.75 | 0.9410 | 0.9425 | 0.9426 |
| records per *matched* S1 entity (v5 model; §5.6 reports the final model per *all* entities) | 3.42 | 3.54 | 3.57 |

France tracks the labelled countries closely, which is the best available evidence that the rules
transfer rather than merely firing. The one visible difference is calibration: **9% of French records
land in the uncertain 0.05–0.9 band against 6% elsewhere**, consistent with a country the model has
never trained on.

`french_rules.py` holds a hand-written **candidate pool** (`FR_LEGAL` 52, `FR_ADDR` 89,
`FR_ADDR_DROP` 11), and the pool is what the pipeline imports. `analysis/fr_analysis.py` checks it against
the French test rows and writes `analysis/fr_rules_observed.py`, the subset occurring at least five times:
`FR_LEGAL` **34**, `FR_ADDR` **59**, `FR_ADDR_DROP` **9**. That file documents which rules are backed by
observed tokens; the remaining pool entries occur fewer than five times in the French rows, so they touch
at most a handful of records. Legal forms are dropped from the sorted *core* name (`fr_core_name`) but
deliberately **kept** in the full name view (`drop_legal=False` in `features.add_text_columns`), because
the legal form is itself a decoy signal (section 4). Six plausible rules were **rejected on the
evidence** — `cap`→capitaine is `Lège-Cap-Ferret` (61,600 occurrences), `pas`→passage is the département
`Pas-de-Calais`, `pres`→président is `pré` (meadow), and likewise `mar`, `car`, `dom`. Collapsing
`st`/`ste`/`sainte` → `saint` reconciles ~24k address rows (`saint nazaire` 71,146 vs `st nazaire`
15,832).

### 2.2 Solution Strategy

**Approach type:** two-stage cascade blocking + gradient-boosted pair classifier, precision-tuned.

**Core innovation:** the query direction is flipped to S2/S3 → S1 with a one-to-one assignment
constraint, so each record contributes at most one (possibly false) link; the decision threshold is
tuned directly on macro F0.5 including singletons; and the candidate set is *pruned by a model* rather
than by a fixed K, which is what makes a 5.91-candidate set affordable.

```
normalise + transliterate → stage-1 retrieval (lexical + semantic)
  → stage-2 pruner → stage-3 pair features + LightGBM
  → cross-encoder score stacked as a feature (ensemble of fine-tuned e5-base models)
  → best S1 per record, first/extra thresholds → group by S1
  (a stage-4 sibling re-scoring stage was built and rejected, see §6–7)
```

**Engineering constraints.** Inference is 10M S2/S3 records against a 1.73M-entity S1 index on a single
Kaggle T4 with 31 GB of RAM, so the pipeline is chunked at 500k records: blocking on GPU (fp16 SVD plus
`torch.topk`), features on CPU. The sibling stage is the memory-hungry one, because it needs the text of
every test record rather than just the candidate pairs — the first attempt exhausted memory during the
test phase. Five consecutive runs — v10, v12, v13, v14 and v14b — were killed at 100% RAM during the test phase
before the cause was found: five text caches each allowed up to 2M entries, the full 5M-row test tables
were held in memory, and every sibling text was retained alongside them. The fix was bounded caches
(400k entries), chunked reading, per-record texts written to disk, the blocking indexes freed before
stage 4 begins, and memory logged per chunk so a regression is visible immediately. Every subsequent run
passed the point where the earlier ones died.

One deliberate consequence: **test pass-1 features are written to disk**, so stage 4, threshold sweeps
and ensembles can be recomputed on CPU in about 30 minutes without re-running blocking. Full run:
**≈6 hours on one T4**.

---

## 3. Candidate Generation (Blocking)

Blocking is the binding constraint on this task: a true match that never becomes a candidate cannot be
recovered downstream, and at ~10M queries against a 1.73M-entity index the search must stay cheap. v5
replaces the earlier fixed-K search with a **cascade**, which is what allows a small final candidate set
without giving up recall. The organisers noted that smaller candidate sets rank higher, and this is the
section where that is won.

### Stage 1 — retrieval (lexical + semantic, per country)

Two retrievers, unioned, both restricted to the same country (country is used as an open set of labels,
so France needs no special casing):

**Lexical.** TF-IDF over name + normalised address, char_wb n-grams. Vectors are projected to a dense
subspace with TruncatedSVD and L2-normalised; a GPU inner-product search takes the **top-30**, and the
**exact sparse TF-IDF cosine** is then recomputed for those 30 and used to keep the **top-10**. The
two-step fetch-then-rescore matters: SVD alone loses real matches, and exact cosine over the whole
index costs ~25 CPU-hours per 1M queries, but exact cosine over 30 pre-filtered candidates is free.

**Semantic.** `intfloat/multilingual-e5-small` (118M parameters, MIT), **top-10** by cosine. This is
what recovers records whose surface forms diverge — trade names, heavy corruption, and the residual
script cases transliteration does not fully normalise.

Transliteration is applied inside `ascii_fold()` **before** the blocking text columns are built, so
Indic-script records are indexed in the same Latin space as everything else.

**Stage-1 result: recall 0.9825 at 18.35 candidates per query** — equivalently **~105 per S1 entity**,
since there are ~5.9 S2/S3 records per S1 entity. (Both denominators appear in team discussion; they
describe the same number.)

### Stage 2 — learned pruner

A small model scores every stage-1 candidate on cheap features and keeps the **top-5 with p ≥ 0.01**.

**Stage-2 result: recall 0.9789 at 1.12 candidates per query — 6.67 per S1 entity.**

So the pruner discards **94%** of the candidate set for **0.36 points** of recall (0.9825 → 0.9789).
That trade is the core of v5: the full feature set and the ranking model then run on 6.67 candidates
per entity instead of 105, which is what makes a ~73-feature model affordable at test scale.

The pruner's own output is also reused downstream — `p_prune` and `margin_p_prune` (its score, and its
margin over the next candidate) are the **2nd and 3rd most important features** in the final model,
worth 0.186 of total gain between them. The pruner is therefore not only a filter but one of the
strongest signals available.

| stage | recall | candidates / S1 |
|---|---|---|
| stage 1, v5 (TF-IDF name+addr top-10 ∪ e5 top-10) | 0.9825 | ~105 |
| stage 2, v5 (pruner, top-5 at p ≥ 0.01) | 0.9789 | 6.67 |
| **stage 1, v7 (+ address-only and core-name views)** | **0.9872** | 147 |
| **stage 2, v7** | **0.9830** | **6.43** |

(Per-query figures for v5 are 18.35 and 1.12; there are ~5.9 S2/S3 records per S1 entity, so the two
denominators describe the same number.)

### v7 — two more retrieval views

The v5 retriever searched a single combined name+address view, which fails in two situations the error
review made concrete: a record with an **empty address** has nothing to match on in a combined view, and
a **trade name** shares no name tokens with its S1 entity. v7 adds two narrower views to the union:

- **address-only** (`n_addr`), which retrieves trade-name records through the address alone
- **core-name-only** (`c_name`), which retrieves empty-address records through the name alone

The cost is a larger stage-1 set (105 → 147 candidates per S1); the pruner absorbs it and still returns
a *smaller* final set than v5 (6.43 vs 6.67) at **0.9830 recall, up 0.41 points**. That is the single
largest recall gain since the cascade itself, and it came directly from reading errors rather than from
tuning.

### Wider blocking

With the cascade in place, retrieval width becomes a tunable trade rather than a fixed cost, because the
pruner absorbs whatever stage 1 returns. Widening it — per-view `k` 10 → 15, `fetch` 30 → 40, name and
address views 5 → 8, pruner keeping the top 8 at p ≥ 0.005:

| | before | after |
|---|---|---|
| stage-1 recall | 0.9872 | **0.9891** |
| stage-2 recall (the ceiling) | 0.9830 | **0.9856** |
| candidates per record | 1.08 | 1.16 |
| stage-4 F0.5 | 0.98602 | **0.98647** |

**+0.26 recall points for about 7% more candidates.** A deliberate trade, and the right direction given
the gain map in section 7: blocking is the largest remaining prize, and every point of stage-2 recall is
a point the downstream stages can no longer lose. Wider blocking was used in one earlier ensemble member; the
final v18 run kept the standard width (k 10, fetch 30, name/address views 5, pruner top 5 at p ≥ 0.01).
Final candidates per S1 entity: **5.91** (measured below).

### The final candidate set

The numbers above trace how blocking developed. This is the candidate set behind the final submission
(`candidate_pairs.tsv` from the v18 run). Every record in the final `matching_results.tsv` comes from it.

| Measure | Value | What it means |
|---|---|---|
| Recall after stage 1 (retrieval) | **0.9884** | 98.84% of true matches are retrieved |
| Recall after stage 2 (pruner) | **0.9816** | 98.16% of true matches remain candidates; the other ~1.8% can never be matched |
| Candidates per S1 entity: mean | **5.91** | the matcher chooses among about 6 records per entity |
| Candidates per S1 entity: median / p90 / max | 6 / 9 / 314 | 90% of entities have 9 or fewer |
| S1 entities with no candidate | 0.050% | 870 of 1,732,544 |

Candidate counts are measured on the final candidate file. Recall needs labels, so it is measured on the
dense validation sample (v18b run, same recipe).

**1,732,544 S1 rows, 10,246,717 candidate pairs in total.**

Per country:

| Country | S1 rows | Pairs | Mean | Median | P90 | Max | Zero-candidate | Reduction vs per-country cross product |
|---|---|---|---|---|---|---|---|---|
| US | 663,106 | 3,823,585 | 5.77 | 6 | 8 | 69 | 0.007% | 99.99985% |
| India | 809,986 | 4,755,562 | 5.87 | 6 | 9 | 298 | 0.084% | 99.99988% |
| France | 259,452 | 1,667,570 | 6.43 | 6 | 10 | 314 | 0.054% | 99.99955% |
| **Overall** | **1,732,544** | **10,246,717** | **5.91** | 6 | 9 | 314 | 0.050% | **99.99985%** (99.99994% vs the global product) |

Blocking runs per country, so the fair baseline is the **per-country** cross product — US ≈2.53T pairs,
India ≈3.82T, France ≈0.37T, **≈6.73 trillion. We score 10.25 million**, a reduction of **99.99985%**.
Against the global product (1.73M S1 × 9.97M records = 17.27 trillion) the reduction is **99.99994%**.
Either way, 98.16% of true matches are retained.

Two country-level observations worth recording:

- **France has the highest mean (6.43) and max (314) candidates per S1, and the weakest reduction
  ratio** of the three. Blocking is measurably less selective on the country it never saw labelled —
  small in absolute terms, but consistent and in the expected direction.
- **India has the highest zero-candidate share (0.084%)**, roughly 12× the US rate, which matches the
  gain map's finding that blocking loss is disproportionately an India problem. All three are under 0.1%.

The shipped stage-2 recall (0.9816) is **below** the 0.9856 measured during the wider-blocking
experiment above. The two come from different runs and different measurement sets, and we did not
re-run the comparison to isolate the cause, so the figure to quote is the shipped one: **0.9816**.
The development numbers are kept only to show the direction each change moved.

Full pipeline runtime for this configuration: **20,457.9s (5.68 h)** on one T4 — roughly 1.2 h training,
16 min building the test index, and 4.2 h of test inference.

**Blocking is still the largest single loss.** `FN_blocking` — the true S1 never became a candidate —
accounts for 11,776 of the v5 validation errors, second only to threshold hesitancy. The stage-2 recall
is the ceiling the ranking model works under — 0.9789 in v5, 0.9830 in v7 — and no amount of feature
work recovers a match that was never retrieved.

---

## 4. Matching Model

**Model:** LightGBM binary classifier over **~73 pair features**, 2-fold cross-validation grouped by S1
entity so no entity appears in both folds, final model retrained on all pairs. (The gain figures below
are measured on the v5 58-feature model, which is the last version with a published importance list;
the v14 additions are described at the end of this section.)

**Decision rule (v5):** each S2/S3 record keeps its single best S1 candidate if p > t; results are grouped
by S1 entity. Threshold tuned on out-of-fold probabilities by grid search on macro F0.5 (the final
submission uses the first/extra rule of §5.3):

| t | 0.3 | 0.5 | 0.6 | 0.7 | **0.75** | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| OOF macro F0.5 | 0.9723 | 0.9770 | 0.9782 | 0.9790 | **0.97898** | 0.9788 | 0.9764 |

The curve is flat between 0.7 and 0.8, so the choice is not delicate — but it is decidedly
precision-side of centre, as F0.5 demands.

### The feature groups

| group | features | top gain |
|---|---|---|
| **Candidate-set / cascade** (19) | `p_prune`, `margin_p_prune`, `cos_e5`, `cos_name_addr`, `cos_c_name`, their `rank_*`, `gap_*` and `margin_*` variants, `n_cand`, `n_cand_pruned`, `cands_same_name`, `s1_name_cnt`, `q_name_cnt`, `addr_rank_in_name`, `addr_gap_in_name` | `p_prune` 0.102, `margin_p_prune` 0.084 |
| **Address** (9) | `nums_jacc`, `addr_tset`, `addr_ratio`, `addr_tsort`, `street_jacc`, `addr_q_in_s1`, `addr_s1_in_q`, `postcode_eq`, `addr_empty` | **`nums_jacc` 0.307** |
| **Decoy / name-difference** (9) | `n_decoy_words`, `n_safe_words`, `n_typo_extra`, `n_other_extra`, `n_missing`, `n_diff_tok`, `diff_tok_minfreq`, `diff_tok_maxfreq`, `legal_form_conflict` | `n_decoy_words` 0.079 |
| **House number** (6) | `house_shift`, `house_same`, `house_small_shift`, `house_large_shift`, `house_other_street`, `house_missing` | `house_shift` 0.062 |
| **Name similarity** (14) | `name_ratio`, `name_tset`, `core_ratio`, `core_tsort`, `core_partial`, `core_jw`, `core_eq`, `len_diff`, `sq_ratio`, `sq_partial`, `name_q_in_s1`, `name_s1_in_q`, `is_domain`, `q_non_latin` | `name_ratio` 0.021 |
| **Source** (1) | `q_src3` | 0.0007 |

**The ten strongest by gain:** `nums_jacc` 0.307, `p_prune` 0.102, `margin_p_prune` 0.084,
`n_decoy_words` 0.079, `house_shift` 0.062, `house_same` 0.051, `house_small_shift` 0.046,
`legal_form_conflict` 0.036, `name_ratio` 0.021, `diff_tok_minfreq` 0.019.

Two observations worth recording. First, **the address number features dominate** — `nums_jacc` alone
carries 31% of total gain, and with the six house-number features the numeric side of the address is
over 47%. Name similarity, the intuitive signal, is under 5%. Second, **three features are dead**:
`core_eq` (0.0), `addr_empty` (0.00002) and `postcode_eq` (0.00013). `postcode_eq` is structurally
inert for France (effectively no postcodes, §2.1) and near-duplicated elsewhere by `nums_jacc`; `core_eq` is
subsumed by `core_ratio`. They cost nothing but should be dropped for clarity.

### v14 additions — features from the error review

The v5 feature set was built from intuition about how records get corrupted. The v14 additions were
built from a hand-read of 180 actual errors (section 7), which found that address stubs account for 58%
of `FN_threshold`, and that generic-word swaps, spelled ordinals and duplicated words each recur:

- **Learned address abbreviations** (`addr_learned.py`) — 50 abbreviation → full-form pairs **learned
  from training pairs**, kept only where the pair co-occurs at least 50 times with a share ≥ 0.9, e.g.
  `(India, gj)` → `gujarat`, `(US, tx)` → `texas`. Data-derived; no external gazetteer or postal table.
- **Ordinals normalised to digits** (`fifth` → `5th`), which also fixes the `1St Floor` family, plus an
  ordinal- and abbreviation-normalised `token_set_ratio`.
- **Address coverage in both directions** on the normalised address — the asymmetric containment score
  that a stub needs, since a symmetric score punishes a short address for what it omits rather than
  scoring what it actually contains.
- **Rarity-weighted (IDF) Jaccard of core-name tokens** — losing a distinctive word costs more than
  losing a generic one, which is what separates `Gurgaon Academy` → `Gurgaon Services` from ordinary
  legal-form noise.
- **Duplicated-word count difference** — token *sets* cannot see `Agnello Microelectronics
  Microelectronics`, so a bag-of-words count is needed.

Effect: stage 3 alone 0.98126 → **0.98151**; with the sibling stage 0.98605 → **0.98633**. Small but
consistent across folds, so kept.

### Credit

- **Transliteration, decoy / house / legal-form / typo features** (`transliterate.py`,
  `decoy_rules.py`) — Venkatkaushal A
- **French normalisation rules, containment features, admin-level findings** (`french_rules.py`) —
  N Sankkarshana
- **multilingual-e5 blocking and the cascade design/test** — Mithilesh Kumar U S
- **Pipeline, training, validation harness and submissions** — Ganesan P V
- **Final day (27 Sep):** cross-encoder models C (Venkatkaushal A), D (Mithilesh Kumar U S) and
  F (N Sankkarshana); models A, B, E, the French models H, H2 and G, the ensembling, the France recipe and
  the final submission (Ganesan P V); code package and reproducibility check (Venkatkaushal A); blocking
  statistics (Mithilesh Kumar U S); documentation (N Sankkarshana)

### Decoy features in detail

About 26% of S2/S3 records match nothing, and the generator builds many of them from a real S1 entity
rather than from scratch: the parent's name plus one extra word, the same name with a different legal
form, or the same street a few doors down. Word lists were mined from 200k sampled decoys and 200k
sampled true pairs, keeping words with P(decoy | extra word) ≥ 0.95 seen in ≥20 decoys — 74 words for
US, 139 for India. They appear in **43% (US) / 54% (India)** of decoys that have an S1 parent, against
**0.01% / 0.03%** of true-match extra tokens. `legal_form_conflict` adds a further **21% / 13.5%**;
either signal covers **63% / 62%**.

`n_typo_extra` separates a *corrupted* word from a *substituted* one. On labelled train data, when
exactly one word differs:

| the differing word is… | P(decoy) US | P(decoy) India |
|---|---|---|
| a close misspelling of the original | **0.086** | **0.047** |
| a non-word, not a misspelling | 0.393 | 0.621 |
| another real word | **0.657** | **0.672** |

House numbers behave the same way, in the other direction: true matches share a house number 71% (US) /
55% (India) of the time against 1% / 16% for decoys, and decoys sit on the same street with a *shifted*
number 65% / 27% of the time.

**France caveat.** France has no labels, so its 7-word decoy list is *inferred* from the generator's
US/India fingerprint: decoy words occur in a tight equal-count cluster and carry an unusually low
legal-form conflict rate (US: 9.0% against 27.5% for other words). The seven French words fit both
(n between 6,400 and 7,000, conflict 0.059–0.082); `fils` (n=1,009, 0.221) and `associes` (n=552,
0.289) were considered and rejected because they fail it. One entry deserves watching: `france` has a
conflict rate of 0.165, roughly double the rest of its cluster, and appears on a confirmed true match in
the French hand audit.

---

## 5. Cross-Encoder, Final Ensemble and France (27 Sep, final day)

The pair features in section 4 compare strings with fixed rules: a ratio, a token overlap, a
house-number distance. A cross-encoder reads both records **together** and learns which differences
matter. Corrupted text is exactly where the hand-written rules keep misjudging.

`intfloat/multilingual-e5-base` (278M parameters, MIT) is fine-tuned as a pair classifier, and **its
score becomes one more feature of the stage-3 LightGBM**, stacked out-of-fold. It is not a separate model
whose output we average: the gradient-boosted matcher decides, per pair, how much to trust it. Every
model in this section re-scores the **saved v18 test pairs**, so the candidate set is fixed. Nothing on
this day changed blocking.

### 5.1 Architecture and training

| | |
|---|---|
| Model | `intfloat/multilingual-e5-base` + a 1-logit classification head (`AutoModelForSequenceClassification`) |
| Input | `"<record name> \| <record address>"` and `"<S1 name> \| <S1 address>"` as a sentence pair, **raw strings**, max 96 tokens (128 for model D) |
| Training pairs | 400k dense training pairs (the v18 candidate pairs after pruning, with orphans and full decoy density), labels from the ground truth |
| Optimiser | AdamW, lr 3e-5 (2e-5 for E/F), weight decay 0.01, 6% linear warm-up, fp16, batch 48, 1–2 epochs |
| Stacking | `--single_fold`: the CE trains on one half of the training pairs and scores the other half. The fused LightGBM (73 features + CE score) trains on that second half, so the CE feature it sees is out-of-sample |
| Test | The trained CE scores all ~10M saved test pairs, and the fused LightGBM re-scores them → best S1 per record + probability (`test_best_ce.npz`) |
| Cost | ~1.5 h training + ~2–3 h test scoring: ~3–4.5 h per model (model F: 3.1 h on a P100) |

Re-scoring saved test pairs, rather than re-running the 6-hour blocking pipeline, made it possible to
train and apply eight cross-encoders on one day across the four team accounts.

### 5.2 Raw text beats normalised text

The single most useful finding of the final day. Feeding the cross-encoder our *normalised* text
(`core name | address`) is worse than feeding it the **original strings**. All numbers are dense
validation (section 6) on the same v18 training pairs:

| Model | CE alone | Stage 3 + CE feature | Gain over stage 3 |
|---|---|---|---|
| v18 stage 3, no cross-encoder | – | 0.98097 | – |
| CE e5-small, normalised text | 0.96031 | 0.98310 | +0.0021 |
| CE e5-base, normalised text | 0.96648 | 0.98383 | +0.0029 |
| CE e5-small, **raw text** | 0.97752 | 0.98508 | +0.0041 |
| CE **e5-base, raw text** | 0.98146 | **0.98639** | **+0.0054** |
| … + first/extra decision rule (5.3) | | **0.98741** | +0.0064 |

Two findings:

- **Raw text roughly doubles the gain.** Normalisation exists to help the *rule-based* features: it
  strips legal forms, expands abbreviations, sorts tokens. Every one of those operations destroys
  information the cross-encoder can use. The legal form is a decoy signal (section 4), the house number
  is the strongest discriminator, and the extra word is often the whole reason a pair is a decoy. The
  pre-processing that makes hand-written features work makes a learned model weaker.
- **Stacking beats replacing.** The cross-encoder alone (0.98146) is barely better than the LightGBM it
  would replace (0.98097). Stacked as a feature it adds +0.0054, because the two are wrong on different
  pairs.

The e5-small result was reproduced in an independent second run (+0.00214 and +0.00212), so differences
of this size are real.

### 5.3 Decision rule: first record vs extra records

Macro F0.5 is averaged per S1 entity. An entity's first accepted record moves it from 0 to a positive
score; a second one only improves an already-positive score. A single global threshold prices both the
same. We therefore use two thresholds: an entity's **first** (highest-probability) record needs
**p > 0.75**, and its **extra** records need only **p > 0.35**.

Measured on the two entity halves of dense validation (gain over the best single threshold):

| Rule | Half 1 | Half 2 |
|---|---|---|
| first > 0.75, extra > 0.35 | **+0.00084** | **+0.00073** |
| first > 0.70, extra > 0.40 | +0.00067 | +0.00072 |
| first > 0.75, extra > 0.40 | +0.00085 | +0.00067 |
| first > 0.75, extra > 0.30 | +0.00078 | +0.00065 |

The rule needs no retraining and only changes which candidates are accepted, so every predicted pair is
still a candidate. A separate third-record threshold added at most +0.00007 on one half and nothing on
the other, so it was not used.

### 5.4 Ensemble for US and India

Six raw-text e5-base cross-encoders were trained, differing only in training half (fold), epochs,
learning rate and sequence length, so each is about equally strong. Each was run on a different team
account (Kaggle or Colab):

| Model | Fold | Epochs | LR | max_len | Run by |
|---|---|---|---|---|---|
| A (LB 0.980065 alone) | 0 | 1 | 3e-5 | 96 | Ganesan |
| B | 1 | 2 | 3e-5 | 96 | Ganesan |
| C | 0 | 2 | 3e-5 | 96 | Venkatkaushal |
| D | 1 | 1 | 3e-5 | 128 | Mithilesh |
| E | 0 | 2 | 2e-5 | 96 | Ganesan |
| F | 1 | 1 | 2e-5 | 96 | Sankkarshana |

**Averaging** (`country_mix.py`): per record, each model votes its best S1 with its probability (0 for
any other S1), and the S1 with the highest weighted mean wins.

**Weights:** A–F get 2/13 each, and the v18 stage-3 LightGBM *without* the cross-encoder gets 1/13
(≈ 8%). A small share of the no-CE model was measured on dense validation as a mixing weight w:

| w (share of the no-CE stage-3 probability) | Half 1 | Half 2 |
|---|---|---|
| 0.0 | 0.98754 | 0.98735 |
| **0.1** | **+0.00005** | **+0.00006** |
| 0.2 | +0.00001 | +0.00008 |
| 0.3 | −0.00025 | −0.00026 |
| 0.5 | −0.00174 | −0.00171 |

A little of the independent model helps; more than ~20% dilutes the stronger one.

**Equal-strength members only.** Averaging a *weaker* model into a stronger one makes it worse:
e5-base + e5-small scored 0.98708 against 0.98741 for e5-base alone. Diversity is not free insurance.

**Agreement between members** (test decisions at p > 0.6): two CE folds (A vs B) disagree on 0.45% of US
and 0.46% of India records. A CE and the no-CE LightGBM disagree on 1.21% / 1.57%. The members are close,
so averaging mainly smooths borderline cases.

### 5.5 France: no labels, so a separate recipe

France is 15% of the test and appears in no training row. The cross-encoders are trained on US/India pairs
only, and the test predictions show that they are measurably less sure of themselves on France:

| Model (test, best candidate per record) | US: unsure (0.1 < p < 0.9) | India | France |
|---|---|---|---|
| v18 stage 3 (LightGBM, French rules) | 4.79% | 5.02% | 7.16% |
| CE A (raw e5-base) | 2.34% | 2.03% | **5.64%** |
| CE B | 2.18% | 1.81% | **5.99%** |
| CE H (French-noise augmented) | – | – | 5.27% |
| CE H2 (French-noise augmented) | – | – | **5.00%** |
| CE G (CamemBERT) | – | – | 6.10% |

| Decision disagreement at p > 0.6 | US | India | France |
|---|---|---|---|
| CE A vs CE B (two training halves) | 0.45% | 0.46% | **1.79%** |
| CE A vs v18 stage 3 | 1.21% | 1.57% | **2.77%** |
| CE H vs CE H2 (augmented) | – | – | 1.32% |

French uncertainty is 2.5–3× higher, and two CE folds disagree 4× as often as in US/India. Reading the
disagreements (115 sampled French records) shows the reason: the French
generator uses noise the models never saw in training:

- **Family/partner suffixes:** `Communale SARL & Fils` ↔ `Communale Compagnie SARL`,
  `CLINIQUE PIERRE & (FILS)` ↔ `Clinique Pierre`, `Sapeurs EI & Associés` ↔ `Sapeurs Parents EI`.
- **Generic-word swaps inside the name:** `Cnl Maternelle SAS` ↔ `Cnl Club SAS`, `Lexcuse Groupement SAS`
  ↔ `Lexcuse Club SAS` (same street and number, a true match).
- **Administrative depth:** S1 ends with the région (`Nouvelle-Aquitaine`), the record with the
  département (`Gironde`) or nothing, and address components appear in any order.
- **Websites and acronyms as names:** `timaisoneurl.com` ↔ `Ti Maison EURL`, `YF` ↔ `YOT Federation SAS`.

**What we tried for France** (all label-free):

1. **Blend the CEs with the rule-based stage 3**, which has the French normalisation rules (section 2.1).
   Half/half was used in the intermediate day-3 submissions.
2. **CamemBERT** (`almanach/camembert-base`, MIT): a French-pretrained encoder, trained the same way on
   US/India pairs and applied to French test pairs (model G). It was *more* uncertain on France (6.10%)
   than the multilingual models, so it got **weight 0** in the final blend.
3. **French-style noise augmentation** (`--fr_aug`, models H and H2). The record side of every
   US/India training pair is perturbed with the noise patterns observed in the unlabelled French test text
   (`fr_noise()` in `ce_stack.py`): drop a non-first name word (p = 0.2); append `& Fils` / `& Associés` /
   `Et Fils` / `(France)` / `& Cie` (p = 0.2); move the legal form to the front (p = 0.1); drop the last
   address component, i.e. the région (p = 0.3); shuffle address components (p = 0.3). This teaches the
   model that these differences are harmless without any French label. This is consistent with section
   4, where `fils` and `associes` were rejected as decoy words: both suffixes appear on true matches. H/H2 are the most confident and
   most self-consistent models on France (5.0–5.3% unsure, 1.32% disagreement between the two folds).

**Yardstick without labels.** The data generator appears country-agnostic: US and India have almost the
same matches-per-entity distribution (3.39 vs 3.35 predicted matches per S1, near-identical histograms),
and France has the same no-match share (section 6). So France's predicted matches-per-S1 histogram should
look like US/India's. We compared each candidate France recipe against the US/India prediction histogram
(L1 distance; 0–8+ matches per entity):

| France recipe (rule 0.75 / 0.35) | Mean matches per S1 | L1 vs US/India |
|---|---|---|
| v18 stage 3 only | 3.255 | 0.0591 |
| CE A only | 3.271 | 0.0522 |
| CE A, B, E, F averaged | 3.295 | 0.0399 |
| ½ CE ensemble + ½ stage 3 | 3.289 | 0.0412 |
| CamemBERT G only | 3.288 | 0.0435 |
| H + H2 (augmented) | 3.301 | 0.0374 |
| **Final blend: 70% H/H2 + 15% CE ensemble + 15% stage 3** (scored at 0.75 / 0.35 for comparability; 3.28 at the shipped 0.70 / 0.40) | **3.300** | **0.0374** |
| US / India (predicted, for reference) | 3.394 / 3.352 | – |

The augmented models move France closest to the labelled countries. The final France blend is
**70% avg(H, H2) + 15% avg(A–F) + 15% v18 stage 3**. It keeps the augmented models' behaviour, with some
weight on the independent rule-based model.

**France thresholds: first > 0.70, extra > 0.40.** These came from dense validation of *blended*
probabilities. When ~30% of an independent model is mixed into the CE-stacked probability, the best rule
shifts from 0.75/0.35 to **0.70/0.40** (+0.00005 / +0.00021 on the two halves). A 50/50 blend shifts it
further, to 0.65/0.45 (+0.00053 / +0.00055). Blending flattens the probabilities, so the best "first"
threshold drops and the "extra" threshold rises. France's final matches per entity is 3.28 (US 3.39,
India 3.35), slightly more cautious on the country without labels.

### 5.6 Final submission

| | US + India | France |
|---|---|---|
| Members | CE A–F (2/13 each) + v18 stage 3 (1/13) | 70% avg(H, H2) + 15% avg(A–F) + 15% v18 stage 3 |
| Decision | first > 0.75, extra > 0.35 | first > 0.70, extra > 0.40 |
| Matched records per S1 entity (all entities, including empty ones) | 3.39 (US), 3.35 (India) | 3.28 |
| S1 left empty | 5.82% (US), 5.95% (India) | 6.04% |

Totals: **5,817,893 matched records** for 1,732,544 S1 entities, inside **10,246,717 candidate pairs**.
Every match is a candidate, no record is matched twice, and the file was rebuilt from the saved
predictions and checked to reproduce the uploaded `matching_results.tsv` exactly.

**Public leaderboard, step by step on the final day** (each step is a real LB submission):

| Step | LB | Change vs best so far |
|---|---|---|
| CE A alone | 0.980065 | – |
| + CE B; France ½ CE + ½ stage 3 | 0.98107 | +0.0010 |
| CE A, B, E, F; France on cross-encoders only | 0.980814 | −0.00026 vs the France blend: the rule-based stage 3 helps France |
| CE A, B, C, E, F + 1/11 stage 3; France ½ CE + ½ stage 3 | 0.981163 | +0.0001 |
| + French-augmented CE H/H2 for France, France thresholds 0.70/0.40 | 0.981721 | **+0.0006**: the largest single step of the evening |
| **+ CE D; France 70% H/H2 + 15% CE + 15% stage 3 (final)** | **0.981722** | +0.000001 |

The France-only changes (17:49 and 22:35) moved the overall score by −0.0003 and +0.0006. France is 15% of
the entities, so this is roughly −0.002 / +0.004 on France itself. This is the only direct evidence about
France, since it has no labels.

---

## 6. Validation

Validation is the instrument every decision in this document depends on, and it was wrong twice in ways
that cost us real leaderboard points. Both failures and their fixes are documented here because they
are the most transferable thing we learned.

### Dense validation is the only trusted instrument

`--dense_bg` builds the validation set with **orphans plus every decoy or orphan record that lands on a
sampled entity** — the same crowding per entity the test has. The earlier "realistic" validation kept
only a proportional sample of non-matching records (~7% of what the test gives each entity).

For **pairwise** features the thin version was fine: its gap to the leaderboard was stable (0.0057 and
0.0058 on two consecutive submissions). For anything that reads a **neighbourhood** it was catastrophically optimistic —
see the sibling stage in section 7.

**Calibration:** LB ≈ dense validation − ~0.007 for single cross-encoder models (model A: dense 0.98741 →
LB 0.980065). Ensembles cannot be scored on dense validation as a whole, because each member holds out a
different half; they were judged by their members' scores, the test gate and the LB.

### Orphan records

Test has 21% fewer S1 entities than train but the same number of records to link:

| | S1 entities | S2/S3 records | records with no match |
|---|---|---|---|
| train | 2.21M | ~10.3M | ~25% |
| test | 1.73M (**21% fewer**) | 9.97M | **~40%** |

The simplest explanation that fits: the test was generated like train, then ~21% of its S1 entities
were deleted, leaving their records behind. We call these **orphans** — and they are not decoys. A
decoy is a fabricated record with no real counterpart; an orphan is a genuine record whose counterpart
was removed, so it may be a *perfect* match for an entity that is not there, and the correct answer is
still "no match".

Predicted unmatched share under that explanation: **41%**, against ~40% observed. All countries behave
identically, France included, which rules out a France-specific cause. Simulated in training with
`--orphan_frac 0.21`.

### The submission gate

Because validation is the thing that failed, a model may not be cleared by validation alone.
`src/gate_check.py` compares a candidate submission's **test** decisions against its own stage 3:

- accepted records may change by only about **±1–2%**
- it must **not cover fewer S1 entities**
- both checks, overall **and per country**

v14c — the submission that scored 0.964 — accepted 4.2% more records at t=0.8 while covering *fewer*
entities. It fails this gate, which is what the gate is for.

---

## 7. Results & Error Analysis

### Version history

| When | Version | Change | Validation | Public LB |
|---|---|---|---|---|
| 25 Sep | v01 | exact normalised-name baseline | 0.384 | — |
| 25 Sep | v1–v4b | TF-IDF blocking, LightGBM, e5, transliteration, French rules | 0.949 → 0.960 | 0.878 → 0.944 |
| 26 Sep 09:11 | **v5** | cascade pruner, decoy/house/typo features | 0.97898 | **0.9732** |
| 26 Sep 12:53 | v6 | + French pseudo-labels | 0.97833 | 0.969 → **removed** |
| 26 Sep 15:21 | **v9** | + address/core-name blocking views, house-as-text, de-leet | 0.98100 | **0.975339** |
| 26 Sep 19:11 | v14c | + siblings + rescue rule (validated on thin data) | 0.98645 | **0.964** ✗ |
| 26 Sep 22:56 | — | 6-run stage-3 ensemble | — | 0.976451 |
| 27 Sep 04:05 | v18 + CE | e5-small cross-encoder, normalised text, + rule | 0.98310 | 0.97709 |
| 27 Sep 11:33 | **v18 + CE** | **e5-base cross-encoder, raw text, + rule** | **0.98741** | **0.980065** |
| 27 Sep 16:19 | ens. | CE A+B; France ½ CE + ½ stage 3 | – | 0.98107 |
| 27 Sep 17:49 | ens. | CE A+B+E+F; France on cross-encoders only | – | 0.980814 |
| 27 Sep 20:10 | ens. | CE A+B+C+E+F + 1/11 stage 3; France ½ CE + ½ stage 3 | – | 0.981163 |
| 27 Sep 22:35 | ens. | + French-augmented CE H/H2 for France (50% H/H2, 25% CE, 25% stage 3), France thresholds 0.70/0.40 | – | 0.981721 |
| 27 Sep 23:02 | **final** | + CE D; 6 raw e5-base CEs + stage 3 (US/India); France 70% H/H2, 15% CE, 15% stage 3; thresholds 0.70/0.40 | — (ensemble; best member 0.98741) | **0.981722** |

The organisers extended the final day by two extra submissions, which is why 27 Sep has seven uploads.


### Why validation had to be made realistic

Early experiments were badly optimistic. A v1 experiment scored **0.978** out-of-fold when candidates
were searched against a 100k-entity index; the same pipeline against the **full** 2.2M index scored
**0.949**. The gap is entirely blocking: a small index makes the true match easy to retrieve and there
are far fewer near-duplicate distractors.

Validation therefore now mirrors test conditions in three respects:

1. **Full-size index.** Candidates are searched against the complete S1 index, not a sample.
2. **Decoys included.** The negative population contains the same proportion of no-match S2/S3 records
   as the test set (~26%), so the threshold is tuned against realistic rejection pressure.
3. **Grouped folds.** Cross-validation is grouped by S1 entity, so no entity's records appear in both
   folds — without this, near-duplicate records of the same entity leak across the split.

The OOF figure quoted above (0.9790) is produced under all three conditions. It nevertheless remained
~0.6 points above the public LB, and the next subsection explains why.

### Test-like validation: orphan records

Validation tracked the leaderboard consistently but always sat above it, by a gap too stable to be noise.
The cause is the orphan records described in section 6.

**Simulation.** We drop 21% of sampled entities from the candidate lists, recompute the candidate-set
features (so the removed entity no longer appears as a rival), and keep their records as no-match
queries:

| Model | Normal validation | Test-like (orphans) | Trained with orphans |
|---|---|---|---|
| Stage 3 | 0.98126 | 0.97889 (−0.0024) | 0.97980 |
| Stage 4 (siblings) | 0.98640 | 0.98488 (−0.0015) | 0.98547 |

Three conclusions:

1. **Test-like validation tracks the leaderboard much more closely.** For v9: test-like validation
   0.97889 against **0.97534** on the LB, a gap of 0.0036, where normal validation gave 0.98100 and a
   gap of 0.0057. Most of the residual gap was orphans, not France.
2. **The sibling stage is more robust to orphans** (−0.0015) than stage 3 (−0.0024) — sensible, since an
   orphan has no siblings to corroborate it, and stage 4 notices.
3. **Training with orphans halves the damage.** The false-match rate on orphan records drops from 11.5%
   to 6.3% at t = 0.4, for a small cost on ordinary records.

Implemented as `--orphan_frac 0.21` in the training sampler (v16, v17 and the final v18).

### What did not work

Recorded because an ablation that fails is as informative as one that succeeds, and because two of
these look obviously promising:

| Tried | Result | Kept? |
|---|---|---|
| **Siblings / group consistency** | +0.005 on thin validation, **−0.0002 on dense**; as submitted (v14c) LB 0.964 | no |
| **Empty-entity rescue rule** | 60–80% of rescued entities are true singletons | no |
| **F0.5-optimal per-entity decisions** | −0.0017 | no |
| **Averaging a weaker model into a stronger one** | e5-base + e5-small 0.98708 < e5-base alone 0.98741 | no |
| Mithilesh's 3 extra features | +0.00001 | no |
| Abstain rule on name-tied candidates | +0.00005 | no |
| **French pseudo-labels** — confident test predictions reused as training data | LB 0.9732 → **0.969** on the France-only swap test | no |
| Bigger LightGBM / XGBoost matchers | +0.0003 (within noise) | no |
| Per-country thresholds | no gain over a single global threshold | no |
| Group-size cap on matches per S1 entity | affects 333 entities; no measurable effect | no |
| A second sibling round (stage 5) | 0.98629 → **0.98447** | no |
| **Stage 4 siblings + rescue, as submitted (v14c)** | validation ≈0.985 → LB **0.964** | withdrawn, see section 6 |
| Cross-encoder as the *only* matcher (no LightGBM) | dense 0.98146 vs 0.98639 stacked | no, stacked instead |
| Mixing > 20% of the no-CE model into the CE model | w = 0.3: −0.00025; w = 0.5: −0.0017 (dense) | no, 1/13 used |
| Separate threshold for an entity's third+ record | ≤ +0.00007 on one half, 0 on the other | no |
| Abstain when the top-2 candidates are close (CE probabilities) | −0.00001 to −0.0003 on both halves | no |
| "Clear winner" rule (accept p1 > 0.2–0.5 when p2 is tiny) | −0.00015 to −0.0011 on both halves | no |
| Empty-address segment rule (accept no-address records with a unique exact core name at low p) | 7,457 validation records, only 87% truly matched; adds false positives | no |
| CamemBERT (`almanach/camembert-base`) for France | more uncertain on French pairs (6.10% vs 5.0–5.6%); weight 0 in the final blend | no |

The pseudo-label result is the instructive one. France has no training labels, so reusing confident
French test predictions as training data is the natural move — and it measurably *hurt*, on both v6 and
v8. The model reinforces its own systematic errors on exactly the population where it has no ground
truth to correct them. The same logic explains the failed second sibling round: once sibling groups are
built from sibling-adjusted probabilities, the corroboration stops being independent evidence.

### Error breakdown (v5 validation)

| error type | meaning | count | share |
|---|---|---|---|
| `FN_threshold` | right S1 was top candidate but p ≤ 0.75 | **14,463** | 40.7% |
| `FN_blocking` | right S1 never became a candidate | **11,776** | 33.1% |
| `FN_ranked_below` | another candidate scored higher | 3,753 | 10.6% |
| `FP_decoy` | a no-match record was matched | 3,607 | 10.1% |
| `FN_pruned` | lost in the cascade pruner | 2,304 | 6.5% |
| `FP_wrong_entity` | matched to the wrong S1 | 841 | 2.4% |

**Nearly three-quarters of all error is recall, not precision** — the model failing to commit
(`FN_threshold`) or never seeing the answer (`FN_blocking`). False positives together are 12.5%. Under
F0.5 this is the right side to err on, but the balance suggests the threshold and the blocking ceiling,
not the classifier, are where the remaining points are.

A hand-read of 180 errors (60 each from `FN_threshold`, `FP_decoy`, `FN_ranked_below`, all with
non-empty addresses) is summarised below:

- **`FN_ranked_below` is largely already solved** — 83% of it is the trade-name pattern (different name,
  same address) currently being fixed.
- **The largest new pattern is address stubs** — one side keeps only city + state while the other has
  the full street (58% of `FN_threshold`). Symmetric address similarity punishes a stub for what it
  omits rather than scoring what it contains.
- **The house-number signal is neutral *inside* the error set** — it fires on 57% of true matches and
  60% of decoys there. It is strongly discriminative overall, but it cannot separate the rows v5 is
  already getting wrong.
- **Estimated recoverable share:** `FN_threshold` 85%, `FN_ranked_below` 67%, `FP_decoy` **58%**. About
  42% of `FP_decoy` (≈1,440 of 3,607) are pairs a human would confidently call a match — one character
  of name difference at an identical address — and should be treated as a floor rather than a backlog.

### Where the remaining points actually are

The error counts above say how *often* each failure happens, not what fixing it is worth. This is the
maximum gain if one error type were repaired perfectly, measured at stage 4, t = 0.4, base 0.98629.
**The stage-4 basis means the absolute numbers inherit the low-density problem of section 6**; the
*ranking* of error types is still informative, since it is driven by record counts and by how many
entities each failure empties, neither of which depends on the sibling stage:

| Error type | Records | Max gain | |
|---|---|---|---|
| True S1 never a candidate (blocking) | 8,169 | **+0.0053** | India +0.0075, US +0.0038 |
| Missed at threshold | 1,376 | +0.0043 | |
| Decoy matched | 1,345 | +0.0021 | |
| Wrong entity | 942 | +0.0016 | |
| Ranked below | 92 | +0.0004 | |

Blocking is both the largest bucket and the largest prize, and it is disproportionately an **India**
problem (+0.0075 against +0.0038 for the US). "Missed at threshold" is the notable entry: only 1,376
records, but nearly as valuable as blocking's 8,169, because a missed record often means an entity with
*no* record at all — a zero rather than a reduced score. That asymmetry is what the shipped
first/extra decision rule (§5.3) exploits — and what the rejected rescue rule (§7, "What did not work")
tried and failed to exploit, by accepting low-probability records for entities that were correctly empty.

**More training data does not help.** Stage 4 trained on 50% / 75% / 100% of training entities scores
0.98615 / 0.98628 / 0.98633. The curve is flat, so the constraint is the signal available per pair, not
the number of pairs.

### Error review — Mithilesh (task 5)

An independent hand-read of the v5 error set, complementing the 180-row review above:

- **An empty record address dominates every failure mode** — 62% of blocking misses, 84% of ranking
  misses and 66% of wrong-entity errors. A record with no address is close to unmatchable by the
  address-weighted feature set, and the address features carry most of the model's gain.
- **House-number padding** appears in 45% of decoy false matches, which is consistent with the mined
  distribution: decoys sit on the parent's street with an altered number.
- **About a third of wrong-entity errors are unresolvable** — the same name, no address, and two real
  businesses behind them. No feature separates those.

Three features followed: numeric house-number match, filler-stripped name Jaccard, and cleaned exact
name. Measured on dense validation they are worth **+0.00001** — within noise, so they are not used.
The error tags above stand on their own: they are counts over the v5 error set.

Proposed features from that review, in priority order: asymmetric `addr_coverage` (containment of the
shorter address in the longer), `street_match_without_number`, character-level `addr_fuzz` plus ordinal
normalisation (`FIFTH ST` / `5th Street`), `translit_fuzzy_name`, and multiset (bag) token features to
catch duplicated words.

---

## 8. Data Sources & Compliance

All data used is from the provided competition dataset. **No external data, APIs, web lookups,
geocoding services, business registries or public gazetteers were used at any point**, including for
building dictionaries.

| Component | Licence | Notes |
|---|---|---|
| `intfloat/multilingual-e5-small` | MIT | 118M parameters — blocking embeddings |
| `intfloat/multilingual-e5-base` | MIT | 278M parameters — cross-encoder (final models A–F, H, H2); both well under the 8B limit |
| `almanach/camembert-base` | MIT | 110M parameters — French cross-encoder G, trained and evaluated, **weight 0 in the final submission** |
| LightGBM | MIT | pruner, matcher and fused models |
| RapidFuzz | MIT | string similarity; also offline for the transliteration word map |
| scikit-learn | BSD-3-Clause | TF-IDF, TruncatedSVD |
| PyTorch | BSD-3-Clause | GPU search and cross-encoder training |
| transformers / sentence-transformers | Apache-2.0 | model loading and inference |
| pandas / NumPy / pyarrow | BSD-3-Clause / Apache-2.0 | data handling |

**No hosted LLM or external API is used anywhere in the pipeline.**

**Explicitly avoided:** `unidecode` (GPL) — our Indic transliteration is a hand-written stdlib module
with no third-party dependency; and `indic-transliteration` (MIT), evaluated and rejected on quality
rather than licence.

**Derived resources, all mined from the provided training data or the provided test text:**

- Decoy and safe word lists (74 US / 139 India) — mined from 200k sampled train decoys and 200k sampled
  train true pairs.
- Transliteration word map (1,051 entries + 13 abbreviations) — derived from Indic tokens occurring in
  at least 20 of the 551,240 non-Latin true *training* pairs.
- Learned address abbreviation map — derived from train pairs.
- French rule dictionaries (`FR_LEGAL` 34, `FR_ADDR` 59, `FR_ADDR_DROP` 9) — small hand-written
  candidate lists, pruned to entries that actually occur in the French **test text**. Reading token
  frequencies of the provided test text is permitted; no French gazetteer, postal file or external word
  list was consulted, and six plausible rules were discarded because the data contradicted them.
- The French noise augmentation (`fr_noise()`: `& Fils`, `& Associés`, `(France)`, `& Cie` suffixes, legal form
  moved to the front, dropped région, shuffled components) — patterns read off the provided French **test text**;
  it uses no French labels and no external data.
- The French `FR_ADMIN` city/région vocabulary — derived by frequency from the test addresses
  themselves, not from any geographic reference.

All code and data are kept private: private Kaggle notebooks and a private dataset, with no public
repository, gist or dataset at any point.

---

## 9. Conclusion

A four-stage cascade — blocking, a learned pruner, a feature-based matcher and a cross-encoder — reaches
**0.981722** on the public leaderboard (final ensemble; 0.980065 with a single cross-encoder). The decisions that mattered were structural rather than
incremental: flipping the query direction to S2/S3 → S1, tuning the threshold directly on macro F0.5,
pruning the candidate set with a model rather than a fixed K, and giving a learned model the **raw**
text rather than our carefully normalised version.

That last one is the finding we would carry furthest. Normalisation exists to make hand-written features
work: it strips legal forms, expands abbreviations, sorts tokens. Every one of those operations deletes
something a cross-encoder can use, and feeding it raw strings roughly doubled its contribution
(+0.0029 → +0.0054). The pre-processing that helps rules hurts learning.

The final gain came from ensembling and from treating France as its own problem. Six equal-strength
cross-encoders plus a small share of the independent LightGBM lifted US/India. For France, which has no
labels, we injected the French generator's noise into US/India training text, judged the result by how
closely France's matches-per-entity distribution matched the labelled countries, and used thresholds
re-tuned for blended probabilities (0.70 / 0.40). Together: 0.980065 → **0.981722**.

Two failures taught us more than most of the successes. **Orphan records** — the ~21% of test entities
that appear to have been deleted after generation, leaving their records behind — explained a
validation-to-leaderboard gap we had been blaming on France. And a **density blind spot** in that same
validation set made a group-level feature look like our biggest win when on the test it cost 0.0113
(v9 0.975339 → v14c 0.964); sampling entities preserves the marginal distribution of pairs while destroying the joint
distribution inside a group, so a validation set can be simultaneously unbiased for pairwise decisions
and badly biased for anything that reads a neighbourhood. Both now have countermeasures in the pipeline:
`--orphan_frac`, `--dense_bg`, and a submission gate measured on test behaviour rather than validation
score.

What remains is mostly retrieval. Blocking sets a 98.2% recall ceiling that nothing downstream can lift, and
it is disproportionately an India problem (+0.0075 available against +0.0038 for the US). Roughly 42% of
decoy false positives appear unrecoverable from the two records alone — one character of name difference
at an identical address — which puts a real floor under the score.

---

## Appendix

### A. Code Artefacts

`src/`:

- `metric.py` — local macro F0.5 scorer (same definition as the official metric)
- `features.py` — text normalisation and pair features; `ascii_fold()` transliterates Indic script
- `transliterate.py` — Indic → Latin, own implementation, stdlib only, no runtime dependencies
- `decoy_rules.py` — decoy word / legal-form / typo / house-number features
- `french_rules.py` — French normalisation: `FR_LEGAL`, `FR_ADDR`, `FR_ADDR_DROP`,
  `normalise_fr_name()`, `normalise_fr_addr()`, `squash_name()`, `glue_initials()`
- `gpu_blocking.py` — TF-IDF → SVD → GPU top-K candidate generation and multilingual-e5 retrieval
- `v1_full.py` — training sampler, realistic validation and the one-to-one decision (imported by `v5_full.py`)
- `siblings.py` — stage-4 sibling features (built, **not used** in the final submission)
- `addr_learned.py` — address abbreviations learned from training pairs (50 pairs)
- `v5_full.py` — **entry point**: blocking, stage-2 pruner, stage-3 matcher, test inference, output files
- `ce_stack.py` — cross-encoder fine-tuning (e5-base, raw text, optional `--fr_aug`) stacked into LightGBM;
  re-scores the saved test pairs
- `fopt.py`, `fopt_probe.py`, `rescue_apply.py` — decision-rule helpers imported by the pipeline
- `export_ids.py` — exports S1 and S2/S3 id order for the CPU-side submission build
- `country_mix.py` — per-country weighted averaging of model predictions
- `local_ensemble.py` — first/extra decision rule, writes and checks `matching_results.tsv` and
  `candidate_pairs.tsv`
- `gate_check.py` — test-behaviour gate run before every submission
- `ce_rule_local.py`, `mix_rule_local.py`, `mix_grid2.py`, `blend_thresh.py`, `margin_rule.py`, `empty_rule.py`,
  `ens_oof_check.py` — final-day experiments on saved dense-validation predictions (§5, §7)
- `blocking_probe.py` — recall@K measurement
- `eda.py` — dataset statistics

Analysis scripts (`analysis/`; not part of the inference pipeline):

- `fr_analysis.py` → `notes_french.md`, `fr_rules_observed.py` — French token frequencies, rule
  pruning against the test rows, the 30-entity match audit and the missed-match check
- `fr_admin.py` — the `FR_ADMIN` city/région vocabulary, derived by frequency from the test addresses
- `task2_analysis.py` → `notes_task2.md` — name-sharing study and corruption taxonomy
- `error_tag.py` → `errors_review.md` — the v5 error tagger behind the 180-row hand review
The hand-labelled audit rows and the analysis notes are not published here because they contain
records from the competition dataset.

### B. Additional Results

Blocking recall@K, name+address view, 200k-entity index: R@1 0.965, R@5 0.981, R@10 0.985, R@50 0.993.

**Compute used for the final submission** (free Kaggle / Colab GPUs across the four team accounts):

| Step | Hardware | Time |
|---|---|---|
| v18 base pipeline (blocking, pruner, stage 3, saved pairs) | 1× T4, 31 GB RAM | 5.68 h |
| Each cross-encoder A–F (train + score ~10M test pairs) | 1× T4 / P100 | ~3–4.5 h each |
| French cross-encoders H, H2 (France test pairs only) | 1× T4 / P100 | ~1.2 h each |
| Blend, decision rule, file checks | laptop CPU | minutes |

Threshold sweep and stage recalls are given in sections 3 and 4.

Final submission statistics: 5,817,893 matched records (3.36 per S1 entity), 5.91% of S1 entities left
empty; per country US 3.39 / India 3.35 / France 3.28 matched records per entity.
