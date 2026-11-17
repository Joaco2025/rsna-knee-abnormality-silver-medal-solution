# RSNA Knee Abnormality Detection
### Silver Medal Kaggle Competition Solution

> **Competition:** [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)  
> **Achievement:** **Silver Medal** (exact final private leaderboard rank/score: *[to be confirmed]*; last documented public leaderboard milestone in repository logs as of 2026-10-21: **0.959** macro ROC-AUC, rank **178**)  
> **Third-Party Code & Attribution:** See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [§16 Contributors and Acknowledgments](#16-contributors-and-acknowledgments).

---

## 1. Project Overview

This repository contains the collaborative codebase, experimental pipeline, preprocessing utilities, NLP pseudo-labeling scripts, SLURM HPC job definitions, and Kaggle inference notebooks developed for the **RSNA Knee Abnormality Detection** competition.

The project was developed on the **Yuca** HPC cluster (ACARUS, Universidad de Sonora, SLURM project `p082`) and evolved through two main phases:
1. **Foundation & Baseline Pipeline:** Exploratory data analysis (EDA), DICOM geometric canonicalization, LLM-based pseudo-labeling from radiology reports, the `KneeNet` 2.5D multi-slot attention architecture, and multi-fold training (`0.875` → `0.935` public LB).
2. **Ensembling, Public-Model Integration & Structural Corrections:** Isolated experiment tracking (`mlflow-skinny` + SQLite), paired-bootstrap validation, offline fusion, integration of strong public competition models (Raptor CoAtNet, R6/R7 ConvNeXt-S, S6 ConvNeXt-S), and structural post-processing corrections (`Dual-S6` + `dsj` + `C2`), reaching **0.959** macro ROC-AUC on the public leaderboard.

---

## 2. Kaggle Competition Context

- **Competition:** [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) (RSNA × Kaggle)
- **Evaluation Metric:** Macro-averaged ROC-AUC across **12** musculoskeletal knee MRI findings.
- **Submission Constraints:** Offline Kaggle Notebooks (no internet access) with a 9-hour runtime budget.
- **Competition Close:** October 22, 2026 (23:59 UTC).

---

## 3. Silver Medal Achievement

This team solution earned a **Silver Medal** in the RSNA Knee Abnormality Detection Kaggle competition.

### Documented Public Leaderboard Progression

The following public leaderboard milestones are recorded in the repository's experiment logs (`experiments/journal.md`, `docs/trio_base_d.md`, `docs/documentacion_integral_rsna_knee.md`) up to the 2026-10-21 cutoff:

| Milestone | Composition | Public Macro ROC-AUC | Notes |
|---|---|---:|---|
| Baseline | `KneeNet` ResNet-34 (epoch 7) | 0.875 | Internal baseline |
| Base Ensemble | 50% ResNet-34 + 50% ConvNeXt-Tiny | 0.897 | Paired bootstrap Δ +0.0139 vs. R34 |
| Trio Base+D v2 | 25% R34 + 25% R34-D (slice MIL attention) + 50% ConvNeXt-Tiny | 0.904 | `docs/trio_base_d.md` |
| Owen v3 / v5 | 15 / 20 models (5 folds × 3–4 model families) | 0.912 / 0.918 | Multi-fold ensembles |
| v9b / v9c2 | Out-of-fold (OOF) model packages | 0.935 | Internal multi-family OOF |
| Public One-Click Base | Raptor 384 + R7 ConvNeXt-S (50/50) | 0.956–0.957 | Public Kaggle baseline |
| **Dual-S6 + dsj + C2** | Public blend + Dual-S6 + structural corrections (`λ`, `γ`) | **0.959** | Rank ~210 → 197 → **178** (2026-10-21) |

> **Note:** Final private leaderboard score and exact final rank after competition close are marked as *[to be confirmed]*.

---

## 4. Problem Statement

Given multi-planar knee MRI DICOM studies (sagittal, coronal, and axial series), predict the probability of **12 musculoskeletal findings** per study (fixed `LABELS` order in `rsna/modelo.py`):

1. `ACL`
2. `MCL`
3. `Medial Meniscus`
4. `Lateral Meniscus`
5. `Medial OA`
6. `Lateral OA`
7. `PF OA` (Patellofemoral Osteoarthritis)
8. `Effusion`
9. `Synovitis`
10. `Baker's` (Baker's cyst)
11. `Contusion`
12. `Fracture`

### Key Challenges Addressed
- **Scarce & Noisy Ground Truth:** Only **58 studies ("Gold")** have complete official labels; the remaining **4,349 studies** rely on noisy pseudo-labels extracted from free-text radiology reports, where unmentioned findings (`NaN`) cannot be assumed negative.
- **Gold Calibration Leakage:** Because the 58 Gold studies were used to calibrate Bayesian label mappings, model selection on Gold alone overfits; decisions were therefore anchored on internal validation (`val_int`, 415 studies via `md5(uid) % 10 == 0`) and 5-fold OOF (`md5 % 5`).
- **Focal Pathology Dilution:** Findings like meniscal tears or small fractures may appear on only 1–3 slices, requiring Multiple-Instance Learning (MIL) slice attention rather than naive global pooling.
- **Anatomical Laterality:** Naive horizontal flip TTA swaps medial and lateral compartments, corrupting meniscal and OA predictions; laterality normalization and anatomical-mirror handling are required.

---

## 5. Technical Architecture and Workflow

```
Raw DICOM Studies (24,371 series / 819,078 slices) + 4,407 Radiology Reports
   │
   ├──► [NLP Pipeline: etiquetas/extraer.py] (Qwen2.5-72B via Ollama)
   │       └─► Structured status + severity → Masked pseudo-labels (NaN => w=0)
   │
   ├──► [DICOM Pipeline: rsna/preproc.py & rsna/raptor_preproc.py]
   │       └─► LPS canonical reorientation + slice ordering by ImagePositionPatient
   │           └─► Fixed uint8 tensor (4 slots × 32 slices × 320 × 320)
   │
   ├──► [Training & Evaluation: rsna/entrenar*.py, rsna/evaluar_ensamble.py]
   │       └─► 2.5D KneeNet / KneeNetAtt + CoAtNet / ConvNeXt / DINOv2 arms
   │       └─► Paired bootstrap evaluation (val_int / 5-fold OOF, MLflow SQLite)
   │
   └──► [Kaggle Inference & Post-Processing: kaggle/]
           └─► Rank blending + Dual-S6 + spurious-shortcut de-biasing (λ, γ / C2)
           └─► submission.csv (stride 1, ~7h 15m Kaggle runtime)
```

---

## 6. Data Preprocessing Pipeline

- **Slot Selection (`rsna/preproc.py`):** Each study is mapped to **4 canonical slots**:
  - Slot 0: Sagittal Fluid-Sensitive (`sag_fs`)
  - Slot 1: Sagittal Non-Fluid-Sensitive (`sag_nofs`)
  - Slot 2: Coronal Fluid-Sensitive (`cor_fs`)
  - Slot 3: Axial Fluid-Sensitive (`ax_fs`)
  - Within each plane, candidate series with `n_slices >= 5` are ranked by proximity to `DEPTH = 32` to avoid selecting 3D acquisitions with hundreds of slices when an equivalent 2D series exists. Missing slots are zero-padded and masked via a binary `present` flag.
- **Precomputed Corpus (`rsna/preprocesar.py`, `rsna/raptor_preproc.py`, `rsna/dd44_corpus.py`):** Studies are preprocessed into fixed `uint8` tensors for fast GPU data loading without touching raw DICOMs during training. `rsna/raptor_preproc.py` provides a bit-for-bit validated port of the public Raptor preprocessing pipeline (384×384 / 224×224 windows).

---

## 7. DICOM Processing

Implemented in `rsna/preproc.py` and audited across `eda/01_eda_tabular.py`–`eda/08_verificar_series.py`:
- **LPS Canonical Orientation:** Slices are reoriented to standard patient LPS coordinates (`+x` left, `+y` posterior, `+z` cranial) using `ImageOrientationPatient` rather than trusting scanner-specific axis conventions.
- **Spatial Slice Ordering:** Slices are ordered physically along the slice normal via `ImagePositionPatient`, avoiding unreliable `InstanceNumber` or filename sorting.
- **Corrupted/Mixed-Bit-Depth Resilience:** `_pixels()` falls back to raw `uint8` buffer reshaping when DICOM headers declare 16-bit allocation for 8-bit pixel payloads, and unreadable slices are skipped gracefully.

---

## 8. NLP and Radiology Report Processing

Implemented in `etiquetas/extraer.py`, `etiquetas/consolidar.py`, and `etiquetas/evaluar_extraccion.py`:
- **Local LLM Extraction:** Free-text radiology reports (in any language) are processed locally using **Qwen2.5-72B** served via **Ollama** with deterministic decoding (`temperature=0`, `seed=0`) and JSON-schema-constrained output (`status` ∈ `{present, possible, absent, not_mentioned}`, `severity` ∈ `{mild, moderate, severe, unspecified, na}`).
- **Silence Masking (`w = 0`):** Findings marked `not_mentioned` (`NaN`) are masked out of the training loss (`w = 0`) instead of being imputed as negative—preventing severe false-negative bias (e.g., Baker's cyst prevalence in Gold vs. report mentions).
- **Pilot Validation:** Evaluated against 58 Gold studies in `etiquetas/comparar_piloto.py` and `etiquetas/piloto_58/comparacion.txt` (84% overall agreement across the 12 findings).

---

## 9. Machine Learning Methodology

- **Validation Strategy:**
  - Deterministic 5-fold split (`md5(StudyInstanceUID) % 5`) over 4,349 pseudo-labeled studies; fold 0 (`val_int`, 415 studies via `md5 % 10 == 0`) serves as the primary internal decision set.
  - **Paired Bootstrap (`rsna/evaluar_ensamble.py`):** Compares candidate ensembles against baselines using identical study resamples (500–2,000 replicates), reporting Δ macro AUC, 95% confidence intervals, and `P(Δ > 0)`.
- **Pre-registered Experimentation:** Experiments were logged in `experiments/journal.md` and tracked in local SQLite MLflow (`rsna/registrar_mlflow.py`) before execution, with lessons recorded in `experiments/lessons.md`.
- **Negative Results Documented:** PatchCore/Anomalib anomaly scoring, naive 12×12 stacking on Gold, and stride-3 inference were systematically tested and rejected.

---

## 10. Model Architecture and Inference Pipeline

### `KneeNet` (`rsna/modelo.py`)
- **2.5D Input Preparation (`prepare()`):** Extracts `T` equispaced slices per slot (with random offset during training, center offset at inference) and stacks each slice with its two immediate neighbors `(idx-1, idx, idx+1)` into 3-channel 2.5D inputs normalized with ImageNet statistics.
- **Backbone & MIL Slice Attention:** A 2D pretrained encoder (`resnet34`, `convnext_*`, `efficientnet_*` from `torchvision`) extracts per-slice features, adds learned `slot_emb`, aggregates slices per slot via a 2-layer `Tanh` attention mechanism (`self.attn`), masks absent slots (`present`), concatenates all 4 slot representations, and outputs 12 logits via a `Dropout → Linear(512) → GELU → Dropout → Linear(12)` head.

### Final Competition Inference & Post-Processing
- **Multi-Family Rank Blending (`kaggle/`):** Combines public ConvNeXt-S (`R6`/`R7`), Raptor CoAtNet (`384`/`224`), `Dual-S6` (distilled S6 within the Raptor half + JEV-supervised S6 on top), and internal `KneeNet` / `DINOv2` arms.
- **Structural De-biasing (`dsj` & `C2`):**
  - `λ` correction: subtracts spurious center/DICOM-metadata correlation on `Synovitis`.
  - `γ` (`C2`) correction: rank-based adjustment reducing spurious `Medial OA` correlation on `Lateral OA` and `PF OA`.
- **Execution:** Runs offline at `stride=1` in Kaggle Notebooks (~7h 15m total runtime).

---

## 11. Technologies and Libraries Used

- **Deep Learning:** PyTorch (`2.8.0+rocm6.4` on Yuca MI210 GPUs), `torchvision`, `timm`
- **Medical Imaging:** `pydicom`, `pylibjpeg`, `pylibjpeg-libjpeg`, `pylibjpeg-openjpeg`, `opencv-python-headless`, `Pillow`
- **Data & Scientific Computing:** `numpy`, `pandas`, `pyarrow`, `scipy`, `scikit-learn`
- **NLP / LLM Pseudo-labeling:** Ollama + `qwen2.5:72b`
- **Experiment Tracking & Visualization:** `mlflow-skinny==3.8.1`, `sqlalchemy==2.0.45`, `alembic==1.17.2`, `matplotlib`, `tqdm`
- **HPC Orchestration:** SLURM (`jobs/*.slurm`)

---

## 12. Repository Structure

```text
.
├── README.md                     # Project overview & documentation (this file)
├── THIRD_PARTY_NOTICES.md        # Attribution & Apache 2.0 notices for Kaggle notebook ports
├── requirements.txt              # Python dependencies (ROCm PyTorch + medical imaging stack)
├── requirements.lock.txt         # Locked dependency versions
├── docs/                         # Technical documentation, strategy plans, and reconstructed Spanish docs
├── eda/                          # Exploratory data analysis & DICOM header/series verification scripts
├── etiquetas/                    # LLM-based radiology report extraction & label consolidation
├── experiments/                  # Experiment journal, lessons learned, and analysis reports
├── imagenes/                     # Experimental VLM/MedGemma image probing script
├── jobs/                         # SLURM batch scripts for HPC training, preprocessing, and evaluation
├── kaggle/                       # Kaggle offline inference notebooks and model arms (v5, DINOv2)
├── research/                     # Hypothesis testing & post-0.958 analysis scripts (`score_0958`, `ensemble_hipotesis_2026-10-19`)
└── rsna/                         # Core package: preprocessing, KneeNet, training, CoAtNet ports, evaluation, MLflow
```

---

## 13. Installation and Environment Setup

> **Note:** The training environment in `requirements.txt` targets AMD Instinct MI210 GPUs (`ROCm 6.4`). On NVIDIA CUDA machines, install the matching CUDA build of `torch` and `torchvision` before installing the remaining packages.

```bash
git clone https://github.com/Joaco2025/rsna-knee-abnormality-silver-medal-solution.git
cd rsna-knee-abnormality-silver-medal-solution

python -m venv .venv
source .venv/bin/activate

pip install --no-cache-dir -r requirements.txt
```

Set `KNEE_DATA` to your local competition dataset directory if not running on the original Yuca cluster path:

```bash
export KNEE_DATA=/path/to/rsna-knee-abnormality-detection
```

---

## 14. Instructions for Reproducing the Workflow

1. **Acquire Competition Data:** Download the dataset from Kaggle (see [§15](#15-dataset-acquisition-instructions)).
2. **EDA & Header Verification:**
   ```bash
   python eda/03_extraer_cabeceras.py
   python eda/08_verificar_series.py
   ```
3. **NLP Pseudo-Label Extraction (requires local Ollama server with `qwen2.5:72b`):**
   ```bash
   python etiquetas/extraer.py --host 127.0.0.1:11434 --model qwen2.5:72b --subset labeled --out ./out_labels
   python etiquetas/consolidar.py
   ```
4. **DICOM Preprocessing:**
   ```bash
   python rsna/preprocesar.py
   ```
5. **Model Training & Paired Bootstrap Evaluation:**
   - Local/CLI: `python rsna/entrenar.py` (see `jobs/entrenar.slurm` for full arguments).
   - Ensemble evaluation: `python rsna/evaluar_ensamble.py`.
6. **Kaggle Inference:**
   - Upload trained checkpoints and `rsna/` / `kaggle/` modules as private Kaggle datasets and run `kaggle/inferencia.ipynb` or `kaggle/notebooks/0957_mas_v5_diag.ipynb` offline.

---

## 15. Dataset Acquisition Instructions

**No competition data, DICOM images, study IDs, or model weights are included in this repository** (in compliance with Kaggle competition rules and the MIRA data license).

To obtain the dataset:
1. Visit the official competition page: [RSNA Knee Abnormality Detection on Kaggle](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).
2. Accept the competition rules and data license.
3. Download via the Kaggle CLI:
   ```bash
   kaggle competitions download -c rsna-knee-abnormality-detection
   ```

---

## 16. Contributors and Acknowledgments

### Team Contributors
- **Owen Solis** (`owen` branch) — EDA, DICOM geometric verification, LLM pseudo-labeling pipeline, `KneeNet` architecture, baseline ResNet-34 / ConvNeXt-Tiny models, and multi-fold OOF packages (`v3`–`v9`).
- **Joaco / jcastro (`Joaco2025`)** (`main` / `joako` branch) — Isolated HPC infrastructure & MLflow tracking, paired-bootstrap evaluation, offline ensemble fusion, public-model integration (`Raptor`, `R7`, `Dual-S6`), and structural corrections (`dsj`, `C2`).
- **Fernando Borquez (`FernyBoy`)** — Collaborative repository host (`FernyBoy/RSNA-Knee`) and model family contributions (`cnx140`, `eff140`, `swin_lados`, `cnxs_lados`).

### Infrastructure & Third-Party Acknowledgments
- **ACARUS / Universidad de Sonora** — Yuca HPC cluster (`p082` project, PI: Victor Manuel Minjares Neriz).
- **Kaggle Community & Open-Source Authors:** This solution builds upon public Kaggle notebooks, weights, and label datasets by `heliosli`, `mattiaangeli`, `dreaddevelopment`, `nartaa`, `varshithakolupuri`, `goodpjw2008`, `jqnx25`, `OmerZalman`, `riadmohamed42`, `pilkwang`, `stevenleehans`, and others. Full file-level attribution and Apache 2.0 notices are documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

---

## 17. Limitations and Reproducibility Notes

- **Dataset & Weights Excluded:** Running training or inference requires downloading the official RSNA Knee dataset from Kaggle and obtaining any external public checkpoints separately.
- **Hardware & Path Defaults:** SLURM scripts in `jobs/` and default paths in some scripts reference `/lustre/proyectos/p082/...` and AMD ROCm (`MI210`) configurations from the Yuca HPC cluster; override `KNEE_DATA` and output paths when running elsewhere.
- **External Checkpoint Licenses:** Third-party labels and weights referenced in `research/score_0958/eligibility.md` and `docs/documentacion_integral_rsna_knee.md` carry their own licenses (e.g., Apache 2.0, CC0, CC BY-NC 4.0); verify each resource's license before commercial or derivative use.
- **Historical Spanish Documentation:** Detailed experiment journals (`experiments/journal.md`), lessons (`experiments/lessons.md`), and the earlier Spanish reconstruction (`docs/documentacion_integral_rsna_knee.md`) are preserved in their original language for historical fidelity.
