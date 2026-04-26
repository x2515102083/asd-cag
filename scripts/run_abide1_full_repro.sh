#!/usr/bin/env bash
set -euo pipefail

cd /root/asd-cag
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

mkdir -p outputs/logs outputs/repro_abide1

SITES=(Trinity KKI SBL OHSU Caltech)
RUNTIME_CONFIG_DIR="outputs/repro_abide1/runtime_configs"
RUNTIME_OUTPUT_ROOT="outputs/repro_abide1/runtime_outputs/abide1"
RUNTIME_CHECKPOINT_ROOT="outputs/repro_abide1/runtime_checkpoints/abide1"
PAPER_CONFIG="${RUNTIME_CONFIG_DIR}/cag_abide1_paper.yaml"
EXTERNAL_CONFIG="${RUNTIME_CONFIG_DIR}/cag_abide1_external_best.yaml"

mkdir -p "${RUNTIME_CONFIG_DIR}" "${RUNTIME_OUTPUT_ROOT}" "${RUNTIME_CHECKPOINT_ROOT}"

python - <<'PY'
from pathlib import Path
import yaml

runtime_config_dir = Path("outputs/repro_abide1/runtime_configs")
runtime_config_dir.mkdir(parents=True, exist_ok=True)

for src_name, dst_name in [
    ("configs/cag_abide1_paper.yaml", "cag_abide1_paper.yaml"),
    ("configs/cag_abide1_external_best.yaml", "cag_abide1_external_best.yaml"),
]:
    src = Path(src_name)
    cfg = yaml.safe_load(src.read_text()) or {}
    cfg.setdefault("training", {})
    cfg["training"]["output_dir"] = "outputs/repro_abide1/runtime_outputs/abide1"
    cfg["training"]["checkpoint_dir"] = "outputs/repro_abide1/runtime_checkpoints/abide1"
    (runtime_config_dir / dst_name).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

echo "=== Paper alignment check ===" | tee outputs/logs/repro_abide1_master.log
python scripts/check_paper_alignment.py \
  --config "${PAPER_CONFIG}" \
  --subjects_csv data/abide1/subjects.csv \
  --fc_dir data/abide1/fc \
  2>&1 | tee -a outputs/logs/repro_abide1_master.log

echo "=== Start ABIDE-I paper default full CAG: lambda 1/1/10 ===" | tee -a outputs/logs/repro_abide1_master.log

for site in "${SITES[@]}"; do
  echo "=== Running site ${site}, paper default 1/1/10 ===" | tee -a outputs/logs/repro_abide1_master.log

  python -u scripts/train_abide1.py \
    --config "${PAPER_CONFIG}" \
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
  cp -r "${RUNTIME_OUTPUT_ROOT}/${site}" "outputs/abide1/${site}_paper_1_1_10"
  cp -r "${RUNTIME_CHECKPOINT_ROOT}/${site}" "checkpoints/abide1/${site}_paper_1_1_10"

  echo "=== Summary ${site} paper 1/1/10 ===" | tee -a outputs/logs/repro_abide1_master.log
  cat "outputs/abide1/${site}_paper_1_1_10/summary.json" | tee -a outputs/logs/repro_abide1_master.log
done

echo "=== Caltech Fig.4 external-best 10/100/10 sanity run ===" | tee -a outputs/logs/repro_abide1_master.log

python -u scripts/train_abide1.py \
  --config "${EXTERNAL_CONFIG}" \
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
cp -r "${RUNTIME_OUTPUT_ROOT}/Caltech" outputs/abide1/Caltech_external_best_10_100_10
cp -r "${RUNTIME_CHECKPOINT_ROOT}/Caltech" checkpoints/abide1/Caltech_external_best_10_100_10

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
