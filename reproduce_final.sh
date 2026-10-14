#!/usr/bin/env bash
# Reproduces the final submission (public LB 0.981722) end to end: data -> blocking -> matching -> output/.
#
# Usage:  bash reproduce_final.sh <dataset dir containing train/ and test/>
#
# Hardware: steps 1-2 need a CUDA GPU with 16 GB (we used free Kaggle/Colab T4 and P100 GPUs, 31 GB RAM).
#           Step 3 runs on CPU in minutes.
# Time:     step 1 ~5.7 h; step 2 ~3-4.5 h per model A-F and ~1.2 h per French model H/H2 (~26 GPU hours total).
#           The models in step 2 are independent, so we ran them in parallel on the four team accounts.
# Models are downloaded from the Hugging Face hub at run time: intfloat/multilingual-e5-small and
# intfloat/multilingual-e5-base (both MIT). No other external data or service is used.
set -euo pipefail
DATA=${1:?usage: bash reproduce_final.sh <dataset dir>}
cd "$(dirname "$0")"
mkdir -p models

# ---- 1. base pipeline (v18): blocking, pruner, stage-3 LightGBM; saves training pairs and test pairs ----------------
#  -> output/candidate_pairs.tsv (the final candidate set), output/test_best_stage3.npz, output/train_pairs.parquet,
#     output/test_parts/ (saved test pairs + features for step 2)
#  (--siblings/--sib_v2 also compute stage 4, which the final submission does NOT use.)
python src/v5_full.py --data "$DATA" --out output --n_train 120000 --seed 0 --chunk 300000 \
  --orphan_frac 0.21 --dense_bg --save_test --save_train --dump_errors --extra_views n_addr \
  --block_fields name_addr c_name:5 n_addr:5 --siblings --sib_v2 --v14 --extra_t 0.3 0.4 0.5 0.6 0.7 0.8 0.9
cp output/test_best_stage3.npz models/S.npz

# ---- 2. cross-encoders: fine-tune multilingual-e5-base on raw pair text, stack into LightGBM, re-score test pairs ---
#  ce_stack.py always writes output/test_best_ce.npz; each result is copied to models/<tag>.npz.
CE="python src/ce_stack.py --run output --data $DATA --model intfloat/multilingual-e5-base --text_mode raw \
    --max_train 400000 --single_fold --test"
#  tag  fold epochs lr    max_len bs pbs   extra flags
while read -r TAG FOLD EP LR LEN BS PBS EXTRA; do
  $CE --sf_fold "$FOLD" --epochs "$EP" --lr "$LR" --max_len "$LEN" --bs "$BS" --pbs "$PBS" $EXTRA
  cp output/test_best_ce.npz "models/$TAG.npz"
done <<'MODELS'
A  0 1 3e-5 96  48 384
B  1 2 3e-5 96  48 384
C  0 2 3e-5 96  48 384
D  1 1 3e-5 128 32 256
E  0 2 2e-5 96  48 384
F  1 1 2e-5 96  48 384
H  1 1 3e-5 96  48 384 --test_country France --fr_aug
H2 0 1 3e-5 96  48 384 --test_country France --fr_aug
MODELS

# ---- 3. blend per country, decision rule, write and check both output files (CPU) ----------------------------------
python src/export_ids.py "$DATA"                       # -> output/test_ids.npz (S1 / S2+S3 id order)
M=models
# France helpers: avg(H, H2) and avg(A..F), each restricted to its best S1 per record
python src/country_mix.py --ids output/test_ids.npz --usin $M/H.npz $M/H2.npz --fr $M/H.npz $M/H2.npz --out $M/avg_H.npz
python src/country_mix.py --ids output/test_ids.npz --usin $M/A.npz $M/B.npz $M/C.npz $M/D.npz $M/E.npz $M/F.npz \
  --fr $M/A.npz $M/B.npz $M/C.npz $M/D.npz $M/E.npz $M/F.npz --out $M/avg_CE6.npz
# US/India: A-F at 2/13 each + stage 3 at 1/13.  France: 70% avg(H,H2) + 15% avg(A-F) + 15% stage 3 (14:3:3).
H14=$(printf "$M/avg_H.npz %.0s" $(seq 14)); C3=$(printf "$M/avg_CE6.npz %.0s" 1 2 3); S3=$(printf "$M/S.npz %.0s" 1 2 3)
python src/country_mix.py --ids output/test_ids.npz \
  --usin $M/A.npz $M/B.npz $M/C.npz $M/D.npz $M/E.npz $M/F.npz $M/A.npz $M/B.npz $M/C.npz $M/D.npz $M/E.npz $M/F.npz $M/S.npz \
  --fr $H14 $C3 $S3 --out $M/final_mix.npz
# Decision: one S1 per record; an entity's first record p > 0.75, extra records p > 0.35 (France 0.70 / 0.40).
# Writes final/matching_results.tsv and final/candidate_pairs.tsv and checks headers, row counts, one S1 per record,
# and that every match is a candidate.
python src/local_ensemble.py --ids output/test_ids.npz --npz $M/final_mix.npz --cands output/candidate_pairs.tsv \
  --t 0.6 --t_first 0.75 --t_extra 0.35 --fr_first 0.7 --fr_extra 0.4 --out final
echo "done: final/matching_results.tsv, final/candidate_pairs.tsv"
