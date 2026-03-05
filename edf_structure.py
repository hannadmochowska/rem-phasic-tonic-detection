import mne
import pyedflib
from pathlib import Path


# CONFIG

edf_path = Path("/Users/hanna/Documents/UCD/classes/semester 2/Internship/dataset/sleep-cassette/SC4001E0-PSG.edf")



# 1️ LOAD WITH MNE (high-level structure)

print("Loading with MNE...")

raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

print("General Info:")
print(raw.info)

print("\nChannel Names:")
for ch in raw.ch_names:
    print(" -", ch)


# 2️ LOAD WITH PYEDFLIB (low-level EDF header)

print("EDF HEADER DETAILS")

f = pyedflib.EdfReader(str(edf_path))

n_signals = f.signals_in_file
signal_labels = f.getSignalLabels()

print(f"Number of signals: {n_signals}\n")

for i in range(n_signals):
    print(f"Signal {i+1}:")
    print("  Label:", signal_labels[i])
    print("  Sample Frequency:", f.getSampleFrequency(i))
    print("  Physical Dimension:", f.getPhysicalDimension(i))
    print("  Physical Min/Max:", f.getPhysicalMinimum(i), "/", f.getPhysicalMaximum(i))
    print()


f.close()


# 3️ SIMPLE ANATOMICAL CLASSIFICATION

print("CHANNEL CLASSIFICATION")

classification = {
    "Frontal": [],
    "Occipital": [],
    "Posterior": [],
    "Central": [],
    "Parietal": [],
    "EOG": [],
    "EMG": [],
    "Other": []
}

for ch in raw.ch_names:
    ch_upper = ch.upper()

    if "FP" in ch_upper or "F" in ch_upper:
        classification["Frontal"].append(ch)

    elif "O" in ch_upper:
        classification["Occipital"].append(ch)

    elif "PZ" in ch_upper or "PO" in ch_upper:
        classification["Posterior"].append(ch)

    elif "CZ" in ch_upper or "C" in ch_upper:
        classification["Central"].append(ch)

    elif "P" in ch_upper:
        classification["Parietal"].append(ch)

    elif "EOG" in ch_upper:
        classification["EOG"].append(ch)

    elif "EMG" in ch_upper:
        classification["EMG"].append(ch)

    else:
        classification["Other"].append(ch)


for region, channels in classification.items():
    print(f"{region}:")
    for ch in channels:
        print("  -", ch)
    print()
