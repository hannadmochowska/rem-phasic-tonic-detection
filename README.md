# rem-phasic-tonic-detection

Signal processing pipeline for automated detection and classification of phasic and tonic REM sleep substates from polysomnography (PSG) recordings, using EOG-based event detection, unsupervised Hidden Markov Modelling, and downstream EEG spectral analysis.

---

## Overview

REM sleep comprises two neurophysiologically distinct microstates — phasic REM (characterised by bursts of rapid eye movements and sawtooth waves) and tonic REM (quiescent intervals between bursts). This repository implements a two-stage unsupervised classification pipeline:

1. **Stage 1 — Rule-based classification**: Hilbert envelope burst detection on the horizontal EOG signal with subject-adaptive percentile thresholds (P90/P20), labelling each 4 s window as phasic, tonic, or transition.
2. **Stage 2 — HMM classification**: Per-subject Gaussian Hidden Markov Model (K=2 or K=3) fit on four z-scored EEG/EOG features (EOG envelope, sawtooth power, spindle power, theta power), validated against rule-based labels using ARI and NMI.

Downstream EEG analysis includes Welch PSD, canonical bandpower (delta, theta, alpha, beta, gamma), and FOOOF-based aperiodic exponent fitting over low (2–30 Hz) and high (30–48 Hz) frequency bands.

---

## Repository structure

```
rem-phasic-tonic-detection/
├── scripts/
│   ├── detect_phasic_tonic.py              # Main rule-based classification pipeline (Sleep-EDF)
│   ├── edf_structure.py                    # Utility: inspect EDF channel names and metadata
│   ├── rem_hmm_unsupervised.py             # Unsupervised HMM on EEG features (EEG-only input)
│   ├── rem_phasic90_tonic20_ieeg.py        # Adaptation for intracranial EEG dataset
│   ├── rem_phasic90_tonic20_theta_psd.py   # Theta-band PSD analysis per substate
│   ├── rem_phasic_tonic_theta_percentiles.py  # Threshold sweep and percentile optimisation
│   └── rem_phasic_tonic_theta_pzoz_combined.py  # Combined Fpz-Cz / Pz-Oz theta analysis
├── results/
│   ├── percentiles/                        # Output from threshold sweep (P80–P95 × P10–P20)
│   ├── hmm_unsupervised_results/           # HMM results (EOG envelope + EEG features)
│   ├── hmm_results_eeg_first/              # HMM ablation: EEG features only (no EOG input)
│   ├── quinn_method_results/               # Replication of Quinn et al. (2019) on raw EOG
│   ├── iEEG_90:20_percentiles/             # Rule-based results for intracranial EEG dataset
│   └── theta-detection/                    # Theta bandpower and PSD per substate
└── README.md
```

---

## Scripts

### `detect_phasic_tonic.py`
Main classification pipeline for the Sleep-EDF Cassette dataset. Loads `.edf` PSG recordings and expert hypnograms, applies bandpass filtering (0.1–60 Hz), detects EOG bursts via Hilbert envelope thresholding (90th percentile within REM), classifies 4 s windows as phasic/tonic/transition, and computes per-subject summary statistics and EEG spectral analysis (PSD, bandpower, FOOOF). Outputs per-subject CSV tables and group-level figures.

### `edf_structure.py`
Utility script for inspecting EDF file structure: prints channel names, sampling rates, signal durations, and header metadata. Useful for verifying data layout before running the main pipeline.

### `rem_hmm_unsupervised.py`
Unsupervised HMM classification. Detects EOG burst events using the same Hilbert envelope approach as the rule-based pipeline, then fits a per-subject Gaussian HMM (K=2 and K=3) on three EEG features: sawtooth power (2–6 Hz, Fpz-Cz), spindle power (12–15 Hz, Pz-Oz), and theta power (4–8 Hz, Pz-Oz), all z-scored within subject. EOG envelope is used post hoc to label states (highest mean EOG = phasic) but is not included in the HMM feature matrix. Computes ARI and NMI against rule-based P90/P20 labels and outputs feature profile plots and PCA scatter.

### `rem_phasic90_tonic20_ieeg.py`
Adaptation of the main pipeline for an intracranial EEG (iEEG) dataset. Handles fragmented EDF files per night, NumPy hypnogram arrays from U-Sleep (10 s epochs, codes 0–4), and different channel names (EOG1, C3-Cz, Oz-Cz) at 250 Hz. All classification logic (P90/P20 windows, PSD, bandpower, FOOOF) is identical to the Sleep-EDF version.

### `rem_phasic90_tonic20_theta_psd.py`
Dedicated theta-band analysis. Computes Welch PSD and integrated theta bandpower (4–8 Hz) per REM substate at both Fpz-Cz and Pz-Oz, and plots per-subject and group-level comparisons. Run after the main pipeline to produce theta-specific figures.

### `rem_phasic_tonic_theta_percentiles.py`
Threshold sweep analysis. Evaluates 12 combinations of phasic (P80, P85, P90, P95) and tonic (P10, P15, P20) percentile thresholds across all 25 recordings, reporting mean and SD of resulting amplitude thresholds and the three-way window balance for each combination.

### `rem_phasic_tonic_theta_pzoz_combined.py`
Combined frontal/parietal theta analysis. Computes and plots theta bandpower and PSD at Fpz-Cz and Pz-Oz side by side for phasic, tonic, and transition substates, and runs paired t-tests between substates at each channel.

---

## Dependencies

```
python >= 3.9
numpy
pandas
scipy
matplotlib
mne
pyedflib
hmmlearn
scikit-learn
fooof        # or specparam (newer package name)
```

Install with:

```bash
pip install numpy pandas scipy matplotlib mne pyedflib hmmlearn scikit-learn fooof
```

---

## Dataset

This project uses the **Sleep-EDF Expanded Dataset (Sleep Cassette subset)**.

- **Source**: https://physionet.org/content/sleep-edfx/1.0.0/
- **Recordings used**: 25 Sleep Cassette (SC) recordings from healthy adults
- **Channels**: Fpz-Cz, Pz-Oz (EEG), horizontal EOG, expert hypnogram
- **Sampling rate**: 100 Hz

To reproduce the analysis:

1. Download the **Sleep Cassette (SC)** recordings from PhysioNet.
2. Place the `.edf` PSG files and corresponding hypnogram files in:

```
data/sleep-edf/
```

Expected structure:

```
data/sleep-edf/
    SC4001E0-PSG.edf
    SC4001EC-Hypnogram.edf
    SC4002E0-PSG.edf
    SC4002EC-Hypnogram.edf
    ...
```

---

## Running the pipeline

**Step 1 — Rule-based classification and EEG analysis:**

```bash
python scripts/detect_phasic_tonic.py
```

Outputs per-subject CSVs and figures to `results/percentiles/`.

**Step 2 — HMM classification:**

```bash
python scripts/rem_hmm_unsupervised.py
```

Outputs ARI/NMI summary, feature profile plots, and PCA scatter to `results/hmm_unsupervised_results/`.

**Step 3 — Theta analysis (optional):**

```bash
python scripts/rem_phasic90_tonic20_theta_psd.py
python scripts/rem_phasic_tonic_theta_pzoz_combined.py
```

**Threshold sweep (optional):**

```bash
python scripts/rem_phasic_tonic_theta_percentiles.py
```

---

## Key parameters

| Parameter | Value | Description |
|---|---|---|
| `BURST_PCT` | 90 | Phasic threshold (percentile of within-REM EOG envelope) |
| `TONIC_PCT` | 20 | Tonic threshold (percentile of within-REM EOG envelope) |
| `MIN_BURST_DUR` | 0.5 s | Minimum burst duration |
| `MERGE_BURST_GAP` | 0.5 s | Merge gap between adjacent bursts |
| `MIN_TONIC_GAP_S` | 5.0 s | Minimum burst-free interval for tonic label |
| `WINDOW_S` | 4.0 s | Fixed window length for classification |
| HMM states | K=3 | Three-state model (phasic, transition, tonic) |
| HMM features | 4 | EOG envelope, sawtooth (2–6 Hz), spindle (12–15 Hz), theta (4–8 Hz) |

---

## Citation

If you use this code, please cite:

> Dmochowska, H. (2026). *Automated detection and characterisation of phasic and tonic REM sleep substates using EOG-based classification and EEG analysis*. Donders Institute for Brain, Cognition and Behaviour, Radboud University Medical Centre.

And the dataset:

> Kemp, B., et al. (2000). Analysis of a sleep-dependent neuronal feedback loop. *IEEE Transactions on Biomedical Engineering*, 47(9), 1185–1194.

> Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220.
