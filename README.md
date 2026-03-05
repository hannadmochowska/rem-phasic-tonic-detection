# rem-phasic-tonic-detection
Signal processing pipeline for detecting and classifying phasic and tonic REM sleep episodes from polysomnography recordings using EOG-based event detection and spectral analysis.

## Dataset

This project uses the **Sleep-EDF Expanded Dataset (Sleep Cassette subset)**.

Dataset source:
https://physionet.org/content/sleep-edfx/1.0.0/

To reproduce the analysis:

1. Download the **Sleep Cassette (SC)** recordings from PhysioNet.
2. Place the `.edf` PSG files and corresponding hypnogram files in:

data/sleep-edf/

Example structure:

data/
    sleep-edf/
        SC4001E0-PSG.edf
        SC4001EC-Hypnogram.edf
        SC4002E0-PSG.edf
        ...
