#!/usr/bin/env bash
set -euo pipefail

cd /root/asd-cag
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

mkdir -p outputs/logs outputs/repro_abide1

SITES=(Trinity KKI SBL OHSU Caltech)

echo "=== Paper alignment check ===" | tee outputs/logs/repro_abide1_master.log
python scripts/check_paper_alignment.py \
  --config configs/cag_abide1_paper.yaml \
  --subjects_csv data/abide1/subjects.csv \
  --fc_dir data/abide1/fc \
  2>&1 | tee -a outputs/logs/repro_abide1_master.log

echo "=== Start ABIDE-I paper default full CAG: lambda 1/1/10 ===" | tee -a outputs/logs/repro_abide1_master.log

for site in "${SITES[@]}"; do
  echo "=== Running site ${site}, paper default 1/1/10 ===" | tee -a outputs/logs/repro_abide1_master.log

  python -u scripts/train_abide1.py \
    --config configs/cag_abide1_paper.yaml \
    --subjects_csv data/abide1/subjects.csv \
    --fc_dir data/abide1/fc \
    --test_site "${site}" \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0001 \
    --n_env 2 \
    --n_layers 2 \
    --lambda_e1 1 \
    --lambda_e2 1 \
    --lambda_s 10 \
    --dropout 0.2 \
    --mask_type node \
    --use_le1 1 \
    --use_le2 1 \
    --use_ls 1 \
    --device cuda \
    2>&1 | tee "outputs/logs/train_${site}_paper_1_1_10_ep50.log"

  rm -rf "outputs/abide1/${site}_paper_1_1_10" "checkpoints/abide1/${site}_paper_1_1_10"
  cp -r "outputs/abide1/${site}" "outputs/abide1/${site}_paper_1_1_10"
  cp -r "checkpoints/abide1/${site}" "checkpoints/abide1/${site}_paper_1_1_10"

  echo "=== Summary ${site} paper 1/1/10 ===" | tee -a outputs/logs/repro_abide1_master.log
  cat "outputs/abide1/${site}_paper_1_1_10/summary.json" | tee -a outputs/logs/repro_abide1_master.log
done

echo "=== Caltech Fig.4 external-best 10/100/10 sanity run ===" | tee -a outputs/logs/repro_abide1_master.log

python -u scripts/train_abide1.py \
  --config configs/cag_abide1_external_best.yaml \
  --subjects_csv data/abide1/subjects.csv \
  --fc_dir data/abide1/fc \
  --test_site Caltech \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.0001 \
  --n_env 2 \
  --n_layers 2 \
  --lambda_e1 10 \
  --lambda_e2 100 \
  --lambda_s 10 \
  --dropout 0.2 \
  --mask_type node \
  --use_le1 1 \
  --use_le2 1 \
  --use_ls 1 \
  --device cuda \
  2>&1 | tee outputs/logs/train_Caltech_external_best_10_100_10_ep50.log

rm -rf outputs/abide1/Caltech_external_best_10_100_10 checkpoints/abide1/Caltech_external_best_10_100_10
cp -r outputs/abide1/Caltech outputs/abide1/Caltech_external_best_10_100_10
cp -r checkpoints/abide1/Caltech checkpoints/abide1/Caltech_external_best_10_100_10

echo "=== Final summaries ===" | tee -a outputs/logs/repro_abide1_master.log
for d in outputs/abide1/*_paper_1_1_10 outputs/abide1/Caltech_external_best_10_100_10; do
  echo "=== $d ===" | tee -a outputs/logs/repro_abide1_master.log
  cat "$d/summary.json" | tee -a outputs/logs/repro_abide1_master.log
done

python - <<'PY'
from pathlib import Path
import json
import pandas as pd

rows = []
for p in sorted(Path("outputs/abide1").glob("*_paper_1_1_10/summary.json")):
    site = p.parent.name.replace("_paper_1_1_10", "")
    s = json.loads(p.read_text())
    row = {"site": site, "profile": "paper_1_1_10"}
    row.update(s)
    rows.append(row)

p = Path("outputs/abide1/Caltech_external_best_10_100_10/summary.json")
if p.exists():
    s = json.loads(p.read_text())
    row = {"site": "Caltech", "profile": "external_best_10_100_10"}
    row.update(s)
    rows.append(row)

df = pd.DataFrame(rows)
Path("outputs/repro_abide1").mkdir(parents=True, exist_ok=True)
df.to_csv("outputs/repro_abide1/summary_table.csv", index=False)
print(df.to_string(index=False))
PY

echo "=== Done ===" | tee -a outputs/logs/repro_abide1_master.log
