# fmriBERT

BERT-style self-supervised pretraining applied to fMRI voxel timeseries for music perception and auditory imagery decoding. Developed as part of my PhD thesis at [PLACEHOLDER: University Name] ([PLACEHOLDER: link to thesis]).

## Overview

This project adapts masked language modeling and next-sequence prediction — the core ideas behind BERT — to fMRI brain imaging data. Instead of word tokens, the model operates on **voxel activation patterns** extracted from the Superior Temporal Gyrus (STG), a brain region involved in auditory processing.

The pipeline has two stages:

1. **Self-supervised pretraining** on unlabeled fMRI timeseries, learning general neural representations
2. **Supervised finetuning** on downstream classification tasks (timbre discrimination, genre classification, pitch decoding)

### Pretraining Tasks

- **Next-Sequence Prediction (CLS)**: Given two consecutive 5-TR windows of voxel activations, predict whether the second follows the first or is from a different timepoint (binary, via CLS token)
- **Masked Region Prediction (MSK)**: Mask a subset of voxels in the input and predict their values (regression, via masked tokens)
- **Time Direction (TimeDIR)**: Predict whether a sequence of voxel activations is in forward or reversed temporal order (binary, self-supervised)

### Downstream Tasks

- **Same-Timbre**: Given two fMRI responses to musical stimuli, predict whether both correspond to the same instrument timbre (e.g., both Clarinet, or Clarinet vs. Trumpet)
- **Same-Genre**: Predict whether two samples come from the same music genre (10 genres from the OpenGenre dataset)
- **Timbre Decoding**: Classify instrument timbre directly from a single fMRI sample
- **Pitch Class Decoding**: Decode the musical pitch class from fMRI responses

## Architecture

The model (`voxel_transformer.py`) is a custom encoder-decoder transformer adapted for fMRI data:

- **Input**: Flattened voxel activations from STG ROI (~420 dimensions after thresholding), with special token dimensions prepended (CLS, MSK, SEP)
- **Tokenization**: Each TR (repetition time = 1 fMRI volume) is one token in the sequence. A typical input is a pair of 5-TR windows → 10 tokens + CLS
- **Encoder**: Multi-head self-attention blocks with learned positional embeddings
- **Decoder**: Dual-head output for CLS (binary) and MSK (multi-class/regression) tasks
- **Transfer**: For finetuning, pretrained encoder weights are loaded into `transfer_transformer.py`, which adds a task-specific classification head

## Datasets

### OpenGenre
- 17 subjects listening to 10 music genres
- 12 training runs + 6 test runs per subject
- 40 clips of 15 seconds per run, TR = 1.5s → 400 TRs per run
- Source: [Nakai et al.](https://openneuro.org/datasets/ds003720)

### PitchClass (Auditory Imagery)
- Subjects hearing and imagining musical stimuli (Clarinet and Trumpet timbres)
- 8 runs per subject
- Includes both "heard" and "imagined" conditions for cross-modal decoding

### TimeDIR
- Derived from OpenGenre data by reversing temporal sequences
- Used for unsupervised pretraining without behavioral labels

## Project Structure

```
fmriBERT/
├── Constants.py                 # Configuration, paths, subject/genre metadata
├── helpers.py                   # Data processing, masking, label creation, metrics
│
├── voxel_transformer.py         # Main transformer model (pretraining)
├── transfer_transformer.py      # Transfer model (finetuning)
├── talking_transformer.py       # Extended model for analysis/interpretability
├── transformer.py               # Vanilla transformer baseline
│
├── pretrain.py                  # OpenGenre pretraining (CLS + MSK tasks)
├── pretrain_timedir.py          # TimeDIR pretraining
├── unpairedpretrain.py          # Unpaired/unsupervised pretraining
├── pairedpretrain.py            # Paired pretraining with behavioral labels
│
├── pairedfinetune.py            # Same-timbre finetuning (main downstream task)
├── sametimbre.py                # Same-timbre finetuning (alternative implementation)
├── samegenre.py                 # Same-genre finetuning
├── timbre_decoding.py           # Single-sample timbre classification
├── timedir_pitchclass.py        # Pitch class decoding from TimeDIR pretrained models
├── unpaireddecode.py            # Downstream decoding from unpaired pretraining
│
├── make_pretraining_data.py     # Data preparation: fMRI → voxel sequences (OpenGenre)
├── make_timedir_data.py         # Data preparation for TimeDIR task
├── makepcdata.py                # Data preparation for PitchClass tasks
├── make_pitchclass_dicts.py     # Stimulus-to-pitch-class metadata
├── make_audimgpaired_data.py    # Paired heard/imagined data preparation
├── make_ptdcrossval_data.py     # Cross-validation split creation
├── make_test_data.py            # Test set preparation
│
├── pitchclass_data.py           # PyTorch Dataset for pitch class data
├── pitchclass_datasets.py       # Extended dataset handling
├── toy_datasets.py              # Synthetic data for testing/debugging
│
├── inference.py                 # Run inference with trained models
├── pretrain_inference.py        # Inference for pretraining evaluation
├── check_pretrained.py          # Inspect and evaluate pretrained models
├── pretrain_results.py          # Aggregate and plot pretraining results
├── pretrain_attention.py        # Attention weight analysis
│
├── classi_baseline.py           # Non-transformer classification baseline
├── parameter_distributions.py   # Hyperparameter search grids
├── canaries.py                  # Data integrity validation
├── audimg.py                    # MVPA integration (exploratory)
├── heat2binary.py               # ROI mask thresholding
├── getROIs.py                   # ROI mask loading
├── merge_and_lateralize.py      # Hemisphere ROI combination
├── plotdetrending.py            # Preprocessing visualization
├── scrape_ofiles.py             # SLURM log parsing
├── scrape_samegenre.py          # Same-genre results aggregation
│
├── targets/                     # Subject metadata, stimulus logs, condition labels
│   ├── A00XXXX_run-0X.txt       # Per-subject per-run stimulus codes
│   ├── condition_labels.txt     # Condition label mappings
│   ├── subj-id-accession-key.csv
│   └── ...
│
├── pretrain_timedir_logs/       # Example pretraining logs
│
├── scripts/                     # SLURM job scripts and launchers
│   ├── pretrain.sh / .script    # OpenGenre pretraining
│   ├── pretrain_timedir.sh / .script  # TimeDIR pretraining
│   ├── paired_pretrain.sh / .script   # Paired pretraining
│   ├── unpaired_pretrain.sh / .script # Unpaired pretraining
│   ├── finetune.sh / .script    # Finetuning (sametimbre, samegenre)
│   └── decode.sh / .script      # Decoding (timbre, unpaired, pitchclass)
│
└── deprecated/                  # Old/unused files preserved for reference
```

## Usage

### 1. Data Preparation

Prepare voxel timeseries from preprocessed fMRI data (requires fmriprep output and STG ROI masks):

```bash
python make_pretraining_data.py
```

For TimeDIR pretraining data:
```bash
python make_timedir_data.py
```

### 2. Pretraining

Pretrain on OpenGenre with cross-validation (all 12 folds):
```bash
bash scripts/pretrain.sh "opengenre_run" 0-11 0.5 0.0001
```

Pretrain on TimeDIR task:
```bash
bash scripts/pretrain_timedir.sh "timedir_run" 0-7 0.00001 CLS_only
```

Paired pretraining on auditory imagery data:
```bash
bash scripts/paired_pretrain.sh "paired_run" 0-7 0.00001 CLS_only audimg
```

Or call the Python scripts directly:
```bash
python pretrain.py -count 0 -LR 1e-4 -binweight 0.5 -gpu 0 -m "opengenre_pretrain"
python pretrain_timedir.py -heldout_run 1 -task both -CLS_task_weight 0.5 -save_model True
```

### 3. Finetuning

Fine-tune on same-timbre or same-genre using the unified launcher:
```bash
bash scripts/finetune.sh "exp1" sametimbre both 8 0-4
bash scripts/finetune.sh "exp1" samegenre both 10 0-4
```

Or call the Python scripts directly:
```bash
python pairedfinetune.py -heldout_run 1 -pretrain_task timedir -pretrain_idx 0 -LR 1e-4
python samegenre.py -pretrain_task both -pretrain_idx 10 -count 0
```

### 4. Decoding

Run timbre or pitch class decoding:
```bash
bash scripts/decode.sh "exp1" timbre 0-7 0.0000001 fresh fresh
bash scripts/decode.sh "exp1" pitchclass 0-7 0.00001 CLS_only 0
```

### 5. Evaluation

Check pretrained model performance:
```bash
python check_pretrained.py
```

Run inference:
```bash
python inference.py
```

## Configuration

Edit `Constants.py` to set:
- `env`: `"local"` or `"discovery"` (HPC cluster)
- Paths to fMRI data, ROI masks, and output directories
- `ATTENTION_HEADS`, `EPOCHS`, and other hyperparameters
- `threshold`: Probability threshold for ROI voxel inclusion (default: 23)

Key model parameters:
- Voxel space dimension: 420 (STG after thresholding + token dimensions)
- Input sequence: Pairs of 5-TR windows
- OpenGenre: 400 TRs per run, 10 genres, 17 subjects

## Requirements

- Python 3.9+
- PyTorch
- nibabel (for NIfTI fMRI data)
- nilearn
- numpy, scipy
- scikit-learn

## Citation

If you use this code, please cite:

```
[PLACEHOLDER: thesis citation]
```

## License

[PLACEHOLDER: license]
