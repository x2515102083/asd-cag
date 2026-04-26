# ASD-CAG Reproduction

This repository contains a clean PyTorch Geometric code framework for reproducing the method described in "A causal adversarial graph neural network for multi-center autism spectrum disorder identification".

The target benchmark is ABIDE-I CPAC `filt_noglobal` `rois_cc400` ASD vs TC classification with external-site testing. The code keeps data handling, graph construction, CAG model components, adversarial losses, pseudo-environment clustering, evaluation, and biomarker extraction separate.

## Data Layout

The code does not alter data. Expected local paths are:

```text
data/abide1/fc/*.npy
data/abide1_raw/cpac/filt_noglobal/rois_cc400/*.1D
data/synthetic_abide/fc/*.npy
data/synthetic_abide/subjects.csv
```

`subjects.csv` should contain at least `subject_id`, `label`, and `site`. Compatible aliases are supported: `id` or `sub_id`, `dx` or `diagnosis` or `y`, and `site_id` or `center`. Labels are converted to `ASD=1` and control/TC/TDC=`0`.

## Installation

Install PyTorch and PyG according to the CUDA version on your server. Do not rely on `requirements.txt` to choose CUDA wheels.

```powershell
pip install -r requirements.txt
```

For GPU training, install PyTorch/PyG from the official CUDA-specific instructions first, then run the command above for the remaining packages.

## Create FC From .1D

```powershell
python scripts/make_fc_from_1d.py --raw_dir data/abide1_raw/cpac/filt_noglobal/rois_cc400 --out_dir data/abide1/fc
```

The script loads each `.1D` time-series file, computes `np.corrcoef(time_series.T)`, applies `nan_to_num`, and writes an `.npy` FC matrix.

## Check Data

```powershell
python scripts/check_dataset.py --subjects_csv data/synthetic_abide/subjects.csv --fc_dir data/synthetic_abide/fc
```

It reports subject count, label counts, site counts, missing FC count, and the first FC shape.

## Smoke Test

```powershell
python scripts/train_synthetic.py
```

The synthetic command defaults to 2 epochs and exists only as a quick pipeline check.

## ABIDE-I Training

```powershell
python scripts/train_abide1.py --subjects_csv data/abide1/subjects.csv --fc_dir data/abide1/fc --test_site Trinity
```

If `data/abide1/subjects.csv` is missing, the script fails with a message asking for ABIDE-I metadata containing at least `subject_id,label,site`. It does not invent labels.

External-site repeats in PowerShell:

```powershell
foreach ($s in "Trinity","KKI","SBL","OHSU","Caltech") { python scripts/train_abide1.py --subjects_csv data/abide1/subjects.csv --fc_dir data/abide1/fc --test_site $s }
```

## Biomarker Extraction

```powershell
python scripts/extract_biomarkers.py --checkpoint checkpoints/best.pt --subjects_csv data/abide1/subjects.csv --fc_dir data/abide1/fc --out_dir outputs/biomarkers
```

The extractor loads a checkpoint, accumulates ASD subject node-mask probabilities, writes `causal_nodes.csv`, and writes `mask_probabilities.npy`.

## Implementation Notes

- This is a faithful implementation of the method described in the paper where implementation details are specified.
- Exact numerical reproduction may require the authors' original random external-site selections, random seeds, and unpublished preprocessing/metadata details.
- The default graph uses full `n x n` adjacency as in the paper. With CC400 data this is about 153k directed edges per subject, so reduce `batch_size` on memory-limited GPUs if needed.
- The paper's default CAG path masks nodes with Gumbel-Sigmoid. `edge` and `node_edge` mask types are available as ablations, but the default config uses `node`.
- Pseudo environment labels are refreshed at the beginning of each epoch by k-means on causal graph embeddings from the training loader.
- Loss terms are optimized as positive losses while Gradient Reversal Layers apply the adversarial lambdas internally.
- The code does not alter, delete, rename, or move data.

## Paper Alignment Notes

- The English paper default CAG lambdas are `lambda_e1=1`, `lambda_e2=1`, and `lambda_s=10`; this is the default in `configs/cag_abide1.yaml` and `configs/cag_abide1_paper.yaml`.
- Fig. 4 external best uses `lambda_e1=10`, `lambda_e2=100`, and `lambda_s=10`; use `configs/cag_abide1_external_best.yaml` for that sanity run.
- `configs/cag_abide1_legacy_50_20_5.yaml` preserves the old `50/20/5` setting only for reproducing previous local experiments.
- The paper ABIDE-I count is `ASD=486`, `TDC=531`, `total=1017`. If the local `subjects.csv` reports 1035 subjects, that dataset does not exactly match the paper ABIDE-I cohort.
- `Ls` uses `BCEWithLogitsLoss` semantics with a single spurious logit. The causal classifier still uses two-class softmax probabilities for evaluation.
- ABIDE-II reproduction and complete biomarker statistics remain full-reproduction TODOs.
