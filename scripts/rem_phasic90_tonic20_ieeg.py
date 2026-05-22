"""
REM Phasic/Tonic Detection and Spectral Analysis Pipeline
Adapted for the human intracranial EEG (iEEG) dataset.

Key differences from the Sleep-EDF version:
  - Data: fragmented EDF files per night (e.g. 15_night1_01.edf, _02.edf, ...)
  - Hypnogram: .npy arrays (U-Sleep output, 10-s epochs, codes 0-4)
  - Channels: EOG1 for classification; C3-Cz and Oz-Cz for downstream EEG analysis
  - Sampling rate: 250 Hz

Everything else — classification logic (90/20 percentile windows), PSD, bandpower,
FOOOF, merging, and all plotting — is unchanged from the original script.

Data layout expected:
  data/
    iEEG/        (subject 15)
    iEEG-2/      (subject 31)
    iEEG-3/      (subject 28)
    iEEG-4/      (subject 7)
    iEEG-5/      (subject 87)
    iEEG-6/      (subject 2)
  Each folder contains:
    converted_ec/edf_files/{subj}_night{N}_{frag}.edf
    U_sleep_API_10s/{subj}_night{N}_full_hypnogram.npy
                   (or fragment files: {subj}_night{N}_01_hypnogram.npy, ...)

Output:
  results/
    {subj}_night{N}.rem_states_onset_aligned.csv
    REM_summary.csv
    REM_period_counts_all_subjects.csv
    fig1_EEG_PSD_per_subject_and_combined.png
    fig2_EEG_bandpower.png
    fig3_EEG_theta_C3Cz_vs_OzCz.png
    fig4_EEG_examples_all_subjects.png
    fig5_FOOOF_aperiodic_slopes.png   (requires fooof/specparam)
    fig6_FOOOF_power_spectrum_with_slope_fit.png
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import glob
import os

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import welch
from scipy.stats import ttest_rel

try:
    from fooof import FOOOF
    HAS_FOOOF = True
except Exception:
    try:
        from specparam import FOOOF
        HAS_FOOOF = True
    except Exception:
        FOOOF = None
        HAS_FOOOF = False


# ── Dataset configuration ────────────────────────────────────────────────────

DATA_ROOT = Path(__file__).parent / "data"   # adjust if needed

SUBJECT_FOLDERS = {
    "15": "iEEG",
    "31": "iEEG-2",
    "28": "iEEG-3",
    "7":  "iEEG-4",
    "87": "iEEG-5",
    "2":  "iEEG-6",
}

# Intracranial EEG is stored in BrainVision format in these two subfolders.
INTRA_FOLDERS = ["converted_intra_upscale", "converted_intra_upscale_strips"]

# Channel name fragments to exclude when falling back to auto-detection
INTRA_EXCLUDE_PATTERNS = re.compile(r"EOG|EMG|EDF Annotations", re.IGNORECASE)

# Per-subject designated analysis channels from Tim's thesis (Supplementary Table 2).
# Hippocampus contacts: TL03 (S7), TL05 (S2, S15, S31, S87), TL07 (S28)
# Cortex contacts:      TLR01 (S15, S87), TLR06 (S2, S7, S28), TLR12 (S31)
SUBJECT_CHANNEL_MAP: Dict[str, List[str]] = {
    "2":  ["TL05", "TLR06"],
    "7":  ["TL03", "TLR06"],
    "15": ["TL05", "TLR01"],
    "28": ["TL07", "TLR06"],
    "31": ["TL05", "TLR12"],
    "87": ["TL05", "TLR01"],
}

# Not used for intracranial analysis (kept for backwards compatibility)
EEG_ANALYSIS_CHANNELS = ()

# Limit recordings for a quick test run (set to None to process all)
MAX_RECORDINGS = None


# ── Config dataclass (unchanged from original) ───────────────────────────────

@dataclass
class Cfg:
    data_root: Path

    # EOG channel used for ALL phasic/tonic/transition detection
    eog_channels: List[str] = None

    # EEG channels for downstream spectral analysis
    eeg_channels: Tuple[str, ...] = None

    # Preprocessing
    filter_lfreq_hz: float = 0.1
    filter_hfreq_hz: float = 60.0

    # Windowing
    window_s: float = 4.0

    # Per-recording threshold computation (from individual EOG sample distribution)
    # 90th percentile of |EOG samples| in REM → phasic peak threshold
    # 20th percentile of |EOG samples| in REM → tonic max amplitude threshold
    phasic_window_percentile: float = 90.0
    tonic_window_percentile:  float = 20.0
    eog_window_metric: str = "max_abs"

    # Event detection parameters (used to detect rapid eye movements)
    phasic_peak_threshold_uV: float = 100.0   # overridden per recording at runtime
    max_event_duration_s:     float = 0.5
    min_event_separation_s:   float = 0.25
    edge_guard_first_s:       float = 2.0
    bin_s:                    float = 1.0

    # Tonic detection: max |EOG| below this threshold → tonic window
    tonic_max_abs_uV: float = 60.0            # overridden per recording at runtime

    # Buffer: relabel tonic windows within buffer_s of a phasic window as transition
    buffer_s: float = 0.0

    # Merge adjacent same-state windows separated by < merge_gap_s
    merge_gap_s: float = 2.0

    # Plot settings
    plot_span_s:     float = 70.0
    plot_pad_left_s: float = 20.0
    plot_downsample: int   = 5
    phasic_alpha:    float = 0.30
    tonic_alpha:     float = 0.22
    transition_alpha: float = 0.14

    # PSD / Welch
    psd_epoch_s:       float = 4.0
    psd_epoch_overlap: float = 0.0
    welch_nperseg_s:   float = 2.0
    welch_overlap:     float = 0.5
    psd_fmax_hz:       float = 50.0
    psd_to_db:         bool  = True
    combined_psd_df_hz: float = 0.5

    # Bandpower bands
    bandpower_bands: Dict[str, Tuple[float, float]] = None

    # Output filenames
    per_csv_suffix:             str = ".rem_states_onset_aligned.csv"
    summary_csv:                str = "REM_summary.csv"
    rem_period_counts_all_csv:  str = "REM_period_counts_all_subjects.csv"
    fig_eeg_psd_png:            str = "fig1_EEG_PSD_per_subject_and_combined.png"
    fig_eeg_bandpower_png:      str = "fig2_EEG_bandpower.png"
    theta_combined_png:         str = "fig3_EEG_theta_C3Cz_vs_OzCz.png"
    eeg_examples_all_png:       str = "fig4_EEG_examples_all_subjects.png"
    aperiodic_subject_csv:      str = "aperiodic_slopes_by_subject.csv"
    aperiodic_stats_csv:        str = "aperiodic_slopes_group_stats.csv"
    aperiodic_fig_png:          str = "fig5_FOOOF_aperiodic_slopes.png"
    psd_fit_fig_png:            str = "fig6_FOOOF_power_spectrum_with_slope_fit.png"

    # Which channel is the "primary" for combined plots and FOOOF
    main_eeg_channel:   str = "C3-Cz"
    frontal_channel:    str = "C3-Cz"
    parietal_channel:   str = "Oz-Cz"

    # FOOOF settings
    fooof_peak_width_limits: Tuple[float, float] = (1.0, 8.0)
    fooof_max_n_peaks:       int   = 6
    fooof_min_peak_height:   float = 0.05
    fooof_aperiodic_mode:    str   = "fixed"
    aperiodic_lowband:       Tuple[float, float] = (2.0, 30.0)
    aperiodic_highband:      Tuple[float, float] = (30.0, 48.0)

    def __post_init__(self):
        if self.eog_channels is None:
            self.eog_channels = ["EOG1"]
        if self.eeg_channels is None:
            self.eeg_channels = list(EEG_ANALYSIS_CHANNELS)
        if self.bandpower_bands is None:
            self.bandpower_bands = {
                "delta (1-4)":  (1.0,  4.0),
                "theta (4-8)":  (4.0,  8.0),
                "alpha (8-12)": (8.0,  12.0),
                "beta (12-30)": (12.0, 30.0),
                "gamma (30-50)":(30.0, 50.0),
            }


# ── iEEG-specific I/O ────────────────────────────────────────────────────────

def discover_recordings(data_root: Path) -> List[Tuple[str, str, Path, Path]]:
    """
    Scan the data folder and return a list of (subject, night, edf_dir, hyp_dir)
    for every subject/night that has both EDF fragments and a hypnogram.
    """
    recordings = []
    for subject, folder_name in SUBJECT_FOLDERS.items():
        edf_dir = data_root / folder_name / "converted_ec" / "edf_files"
        hyp_dir = data_root / folder_name / "U_sleep_API_10s"

        if not edf_dir.is_dir():
            continue

        # Find night numbers from EDF filenames
        nights = set()
        for f in edf_dir.glob(f"{subject}_night*_*.edf"):
            m = re.match(rf"^{subject}_night(\d+)_\d+\.edf$", f.name)
            if m:
                nights.add(m.group(1))

        for night in sorted(nights):
            # Require at least one EDF fragment
            frags = sorted(edf_dir.glob(f"{subject}_night{night}_*.edf"))
            if not frags:
                continue

            # Require a hypnogram (full file or numbered fragments)
            hyp_full  = hyp_dir / f"{subject}_night{night}_full_hypnogram.npy"
            hyp_frags = sorted(hyp_dir.glob(f"{subject}_night{night}_0*_hypnogram.npy"))
            if not hyp_full.exists() and not hyp_frags:
                print(f"  [SKIP] S{subject}N{night}: no hypnogram found")
                continue

            recordings.append((subject, night, edf_dir, hyp_dir))

    return recordings


def load_night_raw(edf_dir: Path, subject: str, night: str) -> mne.io.BaseRaw:
    """
    Load and concatenate all EDF fragments for one subject/night.
    Returns a preloaded MNE Raw object.
    """
    frags = sorted(edf_dir.glob(f"{subject}_night{night}_*.edf"))
    if not frags:
        raise FileNotFoundError(f"No EDF fragments for S{subject}N{night} in {edf_dir}")

    raws = []
    for f in frags:
        r = mne.io.read_raw_edf(str(f), preload=False, verbose="ERROR")
        r.rename_channels(lambda c: c.strip())
        raws.append(r)

    if len(raws) == 1:
        raws[0].load_data(verbose="ERROR")
        return raws[0]

    raw = mne.concatenate_raws(raws, verbose="ERROR")
    raw.load_data(verbose="ERROR")
    return raw


def load_intra_night_raw(intra_dir: Path, subject: str, night: str):
    """
    Load and concatenate all BrainVision fragments for one subject/night
    from a single intracranial folder.  Returns None if no files found.
    """
    frags = sorted(intra_dir.glob(f"{subject}_night{night}_*.vhdr"))
    if not frags:
        return None
    raws = []
    for vhdr in frags:
        try:
            r = mne.io.read_raw_brainvision(str(vhdr), preload=False, verbose="ERROR")
            r.rename_channels(lambda c: c.strip())
            raws.append(r)
        except Exception as e:
            print(f"    [WARN] Could not load {vhdr.name}: {e}")
    if not raws:
        return None
    if len(raws) == 1:
        raws[0].load_data(verbose="ERROR")
        return raws[0]
    raw = mne.concatenate_raws(raws, verbose="ERROR")
    raw.load_data(verbose="ERROR")
    return raw


def get_intracranial_channels(raw) -> List[str]:
    """Return all channel names that are not EOG, EMG or annotations."""
    return [c for c in raw.ch_names if not INTRA_EXCLUDE_PATTERNS.search(c)]


def get_rem_intervals_from_npy(
    hyp_dir: Path, subject: str, night: str, t_end_s: float
) -> List[Tuple[float, float]]:
    """
    Load a U-Sleep .npy hypnogram and return a list of (start_s, end_s) REM intervals.
    Stage codes: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM.
    Epochs are 10 s, non-overlapping.
    """
    EPOCH_S = 10.0
    REM_CODE = 4

    # Try full hypnogram first; fall back to concatenating numbered fragments
    hyp_full = hyp_dir / f"{subject}_night{night}_full_hypnogram.npy"
    if hyp_full.exists():
        hyp = np.load(str(hyp_full))
    else:
        frags = sorted(hyp_dir.glob(f"{subject}_night{night}_0*_hypnogram.npy"))
        if not frags:
            raise FileNotFoundError(f"No hypnogram for S{subject}N{night}")
        hyp = np.concatenate([np.load(str(f)) for f in frags])

    rem_ints: List[Tuple[float, float]] = []
    i = 0
    while i < len(hyp):
        if hyp[i] == REM_CODE:
            start = i * EPOCH_S
            while i < len(hyp) and hyp[i] == REM_CODE:
                i += 1
            end = min(i * EPOCH_S, t_end_s)
            if end > start:
                rem_ints.append((start, end))
        else:
            i += 1

    return merge_intervals(rem_ints, gap_s=0.0)


# ── Channel resolution (unchanged) ───────────────────────────────────────────

def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _find_channel(raw_ch_names: List[str], wanted: str) -> str | None:
    w = wanted.strip()
    for ch in raw_ch_names:
        if ch.strip() == w:
            return ch
    wlow = w.lower()
    subs = [ch for ch in raw_ch_names if wlow in ch.lower()]
    if len(subs) == 1:
        return subs[0]
    if len(subs) > 1:
        return sorted(subs, key=len)[0]
    wn = _norm_name(w)
    eq = [ch for ch in raw_ch_names if _norm_name(ch) == wn]
    if len(eq) == 1:
        return eq[0]
    subs2 = [ch for ch in raw_ch_names if wn in _norm_name(ch)]
    if len(subs2) == 1:
        return subs2[0]
    if len(subs2) > 1:
        return sorted(subs2, key=len)[0]
    return None


def pick_channels_robust(raw: mne.io.BaseRaw, wanted: List[str]) -> List[str]:
    resolved = []
    for w in wanted:
        hit = _find_channel(list(raw.ch_names), w)
        if hit is None:
            raise ValueError(
                f"Missing channel '{w}'.\nAvailable: {list(raw.ch_names)}"
            )
        resolved.append(hit)
    return resolved


# ── Signal loading from a Raw object ─────────────────────────────────────────

def _safe_hfreq(raw: mne.io.BaseRaw, hfreq_hz: float) -> float | None:
    sf = float(raw.info["sfreq"])
    nyq = sf / 2.0
    if hfreq_hz is None:
        return None
    return float(min(hfreq_hz, nyq - 0.5)) if nyq > 1.0 else None


def _apply_filter(raw: mne.io.BaseRaw, cfg: Cfg) -> mne.io.BaseRaw:
    l = cfg.filter_lfreq_hz
    h = _safe_hfreq(raw, cfg.filter_hfreq_hz)
    if h is not None and h <= l:
        return raw
    raw.filter(l_freq=l, h_freq=h, fir_design="firwin", verbose="ERROR")
    return raw


def load_eog_from_raw(
    raw: mne.io.BaseRaw, cfg: Cfg
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Extract and filter the EOG channel from an already-loaded Raw object."""
    raw_copy = raw.copy()
    _apply_filter(raw_copy, cfg)
    resolved = pick_channels_robust(raw_copy, list(cfg.eog_channels))
    picks = mne.pick_channels(raw_copy.ch_names, include=resolved)
    x = raw_copy.get_data(picks=picks).mean(axis=0) * 1e6   # → µV
    sf = float(raw_copy.info["sfreq"])
    t = np.arange(x.size) / sf
    return t, x, sf


def load_eeg_channel_from_raw(
    raw: mne.io.BaseRaw, cfg: Cfg, ch: str
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Extract and filter a single EEG channel from an already-loaded Raw object."""
    raw_copy = raw.copy()
    _apply_filter(raw_copy, cfg)
    resolved = pick_channels_robust(raw_copy, [ch])
    picks = mne.pick_channels(raw_copy.ch_names, include=resolved)
    if len(picks) != 1:
        raise RuntimeError(f"Expected 1 pick for {ch}, got {len(picks)}")
    x = raw_copy.get_data(picks=picks)[0] * 1e6   # → µV
    sf = float(raw_copy.info["sfreq"])
    t = np.arange(x.size) / sf
    return t, x, sf


# ── Utilities (unchanged) ─────────────────────────────────────────────────────

def runs_true(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = mask.astype(bool)
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends   = np.where(d == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]
    return list(zip(starts.tolist(), ends.tolist()))


def merge_intervals(
    intervals: List[Tuple[float, float]], gap_s: float
) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    ints = sorted(intervals)
    out = [ints[0]]
    for s, e in ints[1:]:
        ps, pe = out[-1]
        if s <= pe + gap_s:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] < b[1] and a[1] > b[0]


def subtract_intervals(
    base: List[Tuple[float, float]], cuts: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    if not base:
        return []
    if not cuts:
        return base[:]
    cuts = sorted(cuts)
    out = []
    for bs, be in base:
        cur = bs
        for cs, ce in cuts:
            if ce <= cur:
                continue
            if cs >= be:
                break
            if cs > cur:
                out.append((cur, min(cs, be)))
            cur = max(cur, ce)
            if cur >= be:
                break
        if cur < be:
            out.append((cur, be))
    return [(s, e) for s, e in out if e > s]


def tile_from_gap_starts(
    intervals: List[Tuple[float, float]], win_s: float
) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for s, e in intervals:
        cur = float(s)
        while cur + win_s <= e:
            out.append((cur, cur + win_s))
            cur += win_s
    return out


def chunk(
    intervals: List[Tuple[float, float]], epoch_s: float, overlap_frac: float
) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    step = epoch_s * (1.0 - overlap_frac)
    if step <= 0:
        raise ValueError("psd_epoch_overlap must be < 1.0")
    out = []
    for s, e in intervals:
        cur = s
        while cur + epoch_s <= e:
            out.append((cur, cur + epoch_s))
            cur += step
    return out


def round3(x):
    return np.nan if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 3)


def total_duration(intervals: List[Tuple[float, float]]) -> float:
    return float(np.sum([(e - s) for s, e in intervals])) if intervals else 0.0


def rem_period_index(t_s: float, rem_periods: List[Tuple[float, float]]) -> int:
    for i, (a, b) in enumerate(rem_periods):
        if (t_s >= a) and (t_s < b):
            return i
    return -1


def rem_period_counts_table(
    rem_periods: List[Tuple[float, float]],
    phasic_bouts: List[Tuple[float, float]],
    tonic_bouts: List[Tuple[float, float]],
    transition_bouts: List[Tuple[float, float]],
) -> pd.DataFrame:
    rows = []
    for i, (rs, re_) in enumerate(rem_periods):
        def _count(bouts):
            return int(np.sum([(s >= rs) and (e <= re_) for s, e in bouts]))
        rows.append({
            "rem_period_idx":     int(i),
            "rem_start_s":        round3(rs),
            "rem_end_s":          round3(re_),
            "rem_duration_s":     round3(re_ - rs),
            "n_phasic_bouts":     _count(phasic_bouts),
            "n_tonic_bouts":      _count(tonic_bouts),
            "n_transition_bouts": _count(transition_bouts),
            "n_total_bouts":      _count(phasic_bouts) + _count(tonic_bouts) + _count(transition_bouts),
        })
    return pd.DataFrame(rows)


# ── Classification (event-based, matching original script behavior) ───────────

def compute_rem_windows(
    rem: List[Tuple[float, float]], win_s: float
) -> List[Tuple[float, float]]:
    windows: List[Tuple[float, float]] = []
    for a, b in rem:
        cur = float(a)
        while cur + win_s <= float(b):
            windows.append((cur, cur + win_s))
            cur += win_s
    return windows


def eog_metric_for_window(
    x_uV: np.ndarray, sf: float, s: float, e: float, metric: str = "max_abs"
) -> float:
    i0 = max(0, int(round(s * sf)))
    i1 = min(x_uV.size, int(round(e * sf)))
    if i1 <= i0:
        return np.nan
    seg = x_uV[i0:i1]
    if seg.size == 0:
        return np.nan
    metric = str(metric).lower()
    if metric == "rms":
        return float(np.sqrt(np.mean(np.square(seg))))
    if metric == "ptp":
        return float(np.ptp(seg))
    return float(np.max(np.abs(seg)))


def phasic_threshold_from_percentile(
    x_uV: np.ndarray,
    sf: float,
    rem_intervals: List[Tuple[float, float]],
    percentile: float,
    fallback_uV: float,
) -> float:
    """Compute per-recording amplitude threshold from the absolute EOG sample
    distribution within REM.  Returns the chosen percentile of |EOG| samples."""
    if x_uV.size == 0 or sf <= 0:
        return float(fallback_uV)
    mask = np.zeros(x_uV.size, dtype=bool)
    for a, b in rem_intervals:
        s = max(0, int(round(a * sf)))
        e = min(x_uV.size, int(round(b * sf)))
        if e > s:
            mask[s:e] = True
    vals = np.abs(x_uV[mask]) if mask.any() else np.abs(x_uV)
    if vals.size == 0:
        return float(fallback_uV)
    thr = float(np.percentile(vals, percentile))
    if not np.isfinite(thr) or thr <= 0:
        return float(fallback_uV)
    return thr


def edge_guard_ok(ev_rel: np.ndarray, cfg: Cfg) -> bool:
    if ev_rel.size == 0:
        return False
    all_in_first = np.all(ev_rel < cfg.edge_guard_first_s)
    bins = np.floor(ev_rel / cfg.bin_s).astype(int)
    in_two_bins = np.unique(bins).size >= 2
    return (not all_in_first) or in_two_bins


def detect_em_events(x_uV: np.ndarray, sf: float, cfg: Cfg) -> np.ndarray:
    """Detect rapid eye-movement events: peaks above threshold, within max duration,
    separated by at least min_event_separation_s."""
    x = np.abs(x_uV)
    above = x >= cfg.phasic_peak_threshold_uV
    max_len = int(round(cfg.max_event_duration_s * sf))

    peaks = []
    for s, e in runs_true(above):
        if (e - s) < max_len:
            peaks.append(s + int(np.argmax(x[s:e])))

    if not peaks:
        return np.array([], dtype=int)

    peaks = np.asarray(sorted(peaks), dtype=int)
    min_sep = int(round(cfg.min_event_separation_s * sf))
    kept = [int(peaks[0])]
    for p in peaks[1:]:
        if int(p) - kept[-1] >= min_sep:
            kept.append(int(p))
    return np.asarray(kept, dtype=int)


def detect_phasic(
    event_times_s: np.ndarray,
    rem: List[Tuple[float, float]],
    cfg: Cfg,
) -> List[Tuple[float, float]]:
    """Phasic windows: 4-s REM windows containing ≥2 EM events passing timing rules."""
    if event_times_s.size < 2:
        return []
    out = []
    i = 0
    while i < event_times_s.size - 1:
        s = float(event_times_s[i])
        e = s + cfg.window_s
        if not any((s >= a) and (e <= b) for a, b in rem):
            i += 1
            continue
        ev = event_times_s[(event_times_s >= s) & (event_times_s < e)]
        if ev.size >= 2 and edge_guard_ok(ev - s, cfg):
            out.append((s, e))
            i = int(np.searchsorted(event_times_s, e, side="left"))
        else:
            i += 1
    return out


def detect_tonic_maxabs(
    x_uV: np.ndarray,
    sf: float,
    rem: List[Tuple[float, float]],
    cfg: Cfg,
) -> List[Tuple[float, float]]:
    """Tonic windows: 4-s REM windows where max |EOG| is below the tonic threshold."""
    x_abs = np.abs(x_uV)
    win_n = int(round(cfg.window_s * sf))
    if win_n <= 1:
        return []
    tonic_windows: List[Tuple[float, float]] = []
    for a, b in rem:
        cur = float(a)
        while cur + cfg.window_s <= float(b):
            i0 = int(round(cur * sf))
            i1 = i0 + win_n
            if i1 > x_abs.size:
                break
            if float(np.max(x_abs[i0:i1])) < cfg.tonic_max_abs_uV:
                tonic_windows.append((cur, cur + cfg.window_s))
            cur += cfg.window_s
    return tonic_windows


def apply_buffer_rule(
    tonic_windows: List[Tuple[float, float]],
    phasic_windows: List[Tuple[float, float]],
    buffer_s: float,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Relabel tonic windows that fall within buffer_s of a phasic window as transition."""
    if buffer_s <= 0 or not tonic_windows or not phasic_windows:
        return tonic_windows, []
    expanded = [(ps - buffer_s, pe + buffer_s) for ps, pe in phasic_windows]
    kept, relabeled = [], []
    for tw in tonic_windows:
        (relabeled if any(overlap(tw, ex) for ex in expanded) else kept).append(tw)
    return kept, relabeled


def classify_rem_substates(
    x_uV: np.ndarray,
    sf: float,
    rem: List[Tuple[float, float]],
    cfg: Cfg,
) -> Tuple[List, List, List, pd.DataFrame, float, float]:
    """Event-based REM microstate classification (matches original script).

    Uses fixed amplitude thresholds from cfg (same as original script):
      phasic_peak_threshold_uV  (default 100 µV) — EM event detection threshold
      tonic_max_abs_uV          (default  60 µV) — tonic window amplitude ceiling

    Steps:
    1. Detect rapid eye-movement events: peaks above phasic_peak_threshold_uV.
    2. Phasic windows: 4-s windows with ≥2 events passing timing/spacing rules.
    3. Tonic windows: 4-s windows where max |EOG| < tonic_max_abs_uV.
    4. Apply optional buffer (relabels tonic windows near phasic as transition).
    5. Transition: remaining REM intervals tiled into 4-s windows.
    """
    ph_thr = cfg.phasic_peak_threshold_uV
    to_thr = cfg.tonic_max_abs_uV
    cfg_ev = cfg   # thresholds already in cfg; no override needed

    # 2. Detect EM events
    event_samples = detect_em_events(x_uV, sf, cfg_ev)
    event_times_s = event_samples.astype(float) / sf

    # 3. Phasic windows (event-based)
    phasic = detect_phasic(event_times_s, rem, cfg_ev)

    # 4. Tonic windows (amplitude-based)
    tonic = detect_tonic_maxabs(x_uV, sf, rem, cfg_ev)

    # 5. Buffer rule
    tonic, relabeled = apply_buffer_rule(tonic, phasic, cfg_ev.buffer_s)

    # 6. Transition = remaining REM gaps + relabeled tonic-near-phasic
    occupied = sorted(phasic + tonic)
    rem_gaps = subtract_intervals(rem, occupied)
    transition_tiled = tile_from_gap_starts(rem_gaps, cfg.window_s)
    transition = transition_tiled + relabeled

    # Build window-level DataFrame for diagnostics
    all_wins = (
        [(s, e, "phasic") for s, e in phasic] +
        [(s, e, "tonic") for s, e in tonic] +
        [(s, e, "transition") for s, e in transition]
    )
    if all_wins:
        all_wins.sort()
        df = pd.DataFrame(all_wins, columns=["start_s", "end_s", "label"])
        df["start_s"] = df["start_s"].map(round3)
        df["end_s"] = df["end_s"].map(round3)
        df["duration_s"] = (df["end_s"] - df["start_s"]).map(round3)
        df["eog_metric_uV"] = [
            round3(eog_metric_for_window(x_uV, sf, s, e, cfg.eog_window_metric))
            for s, e in zip(df["start_s"], df["end_s"])
        ]
    else:
        df = pd.DataFrame(columns=["start_s", "end_s", "label", "duration_s", "eog_metric_uV"])

    return phasic, tonic, transition, df, ph_thr, to_thr


def classify_rem_windows_percentile(
    x_uV: np.ndarray,
    sf: float,
    rem: List[Tuple[float, float]],
    cfg: Cfg,
) -> Tuple[List, List, List, pd.DataFrame, float, float]:
    """Percentile-based REM window classification (matches original Sleep-EDF script).

    For each 4-s REM window computes max |EOG| (eog_window_metric).
    Phasic  = windows >= phasic_window_percentile  (default 90th percentile)
    Tonic   = windows <= tonic_window_percentile   (default 20th percentile)
    Transition = all remaining windows
    """
    windows = compute_rem_windows(rem, cfg.window_s)
    if not windows:
        return [], [], [], pd.DataFrame(), np.nan, np.nan

    metrics = np.asarray(
        [eog_metric_for_window(x_uV, sf, s, e, cfg.eog_window_metric) for s, e in windows],
        dtype=float,
    )
    finite = np.isfinite(metrics)
    if not np.any(finite):
        return [], [], [], pd.DataFrame(), np.nan, np.nan

    ph_thr = float(np.percentile(metrics[finite], cfg.phasic_window_percentile))
    to_thr = float(np.percentile(metrics[finite], cfg.tonic_window_percentile))

    labels, phasic, tonic, transition = [], [], [], []
    for (s, e), m in zip(windows, metrics):
        if not np.isfinite(m):
            label = "transition"; transition.append((s, e))
        elif m >= ph_thr:
            label = "phasic";    phasic.append((s, e))
        elif m <= to_thr:
            label = "tonic";     tonic.append((s, e))
        else:
            label = "transition"; transition.append((s, e))
        labels.append(label)

    df = pd.DataFrame({
        "start_s":        [round3(s) for s, _ in windows],
        "end_s":          [round3(e) for _, e in windows],
        "duration_s":     [round3(e - s) for s, e in windows],
        "eog_metric_uV":  [round3(v) for v in metrics],
        "label":          labels,
    })
    return phasic, tonic, transition, df, ph_thr, to_thr


# ── FOOOF / aperiodic (unchanged) ─────────────────────────────────────────────

def fit_fooof_exponent(
    freqs: np.ndarray, psd: np.ndarray,
    fit_range: Tuple[float, float], cfg: Cfg
) -> float:
    if freqs.size == 0 or psd.size == 0:
        return np.nan
    lo, hi = fit_range
    mask = (freqs >= lo) & (freqs <= hi) & np.isfinite(psd) & (psd > 0)
    f = freqs[mask]; p = psd[mask]
    if f.size < 6:
        return np.nan
    if not HAS_FOOOF:
        raise ImportError("fooof / specparam not installed.")
    fm = FOOOF(
        peak_width_limits=list(cfg.fooof_peak_width_limits),
        max_n_peaks=int(cfg.fooof_max_n_peaks),
        min_peak_height=float(cfg.fooof_min_peak_height),
        aperiodic_mode=str(cfg.fooof_aperiodic_mode),
        verbose=False,
    )
    fm.fit(f, p, [float(lo), float(hi)])
    ap = getattr(fm, "aperiodic_params_", None)
    if ap is None or len(ap) < 2:
        return np.nan
    return float(ap[1])


def fit_fooof_model(
    freqs: np.ndarray, psd: np.ndarray,
    fit_range: Tuple[float, float], cfg: Cfg
):
    if freqs.size == 0 or psd.size == 0:
        return None, np.array([]), np.array([])
    lo, hi = fit_range
    mask = (freqs >= lo) & (freqs <= hi) & np.isfinite(psd) & (psd > 0)
    f = freqs[mask]; p = psd[mask]
    if f.size < 6:
        return None, f, p
    if not HAS_FOOOF:
        raise ImportError("fooof / specparam not installed.")
    fm = FOOOF(
        peak_width_limits=list(cfg.fooof_peak_width_limits),
        max_n_peaks=int(cfg.fooof_max_n_peaks),
        min_peak_height=float(cfg.fooof_min_peak_height),
        aperiodic_mode=str(cfg.fooof_aperiodic_mode),
        verbose=False,
    )
    fm.fit(f, p, [float(lo), float(hi)])
    return fm, f, p


def fooof_aperiodic_fit_db(freqs: np.ndarray, fm) -> np.ndarray:
    if fm is None or freqs.size == 0:
        return np.array([])
    ap = getattr(fm, "aperiodic_params_", None)
    if ap is None or len(ap) < 2:
        return np.full(freqs.shape, np.nan, dtype=float)
    offset   = float(ap[0])
    exponent = float(ap[1])
    return 10.0 * (offset - exponent * np.log10(freqs))


def compute_aperiodic_from_psd_dict(psd_dict: Dict, cfg: Cfg) -> Dict[str, float]:
    out = {}
    for state in ["phasic", "tonic", "transition"]:
        f = psd_dict[state]["freqs"]
        m = psd_dict[state]["mean"]
        out[f"{state}_low_exp"]  = fit_fooof_exponent(f, m, cfg.aperiodic_lowband, cfg)
        out[f"{state}_high_exp"] = fit_fooof_exponent(f, m, cfg.aperiodic_highband, cfg)
        out[f"{state}_n_epochs"] = int(psd_dict[state]["n"])
    return out


def make_aperiodic_subject_table(payloads: List[Dict], cfg: Cfg, channel: str) -> pd.DataFrame:
    rows = []
    for p in payloads:
        if "eeg_psd_channels" not in p or channel not in p["eeg_psd_channels"]:
            continue
        aps = p.get("aperiodic_channels", {}).get(channel, {})
        row = {"subject": p["name"], "channel": channel}
        row.update(aps)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_stats_table(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    comparisons = [("phasic", "tonic"), ("phasic", "transition"), ("tonic", "transition")]
    for band in ["low", "high"]:
        for a, b in comparisons:
            ca, cb = f"{a}_{band}_exp", f"{b}_{band}_exp"
            sub = df[[ca, cb]].dropna()
            if sub.empty:
                continue
            t_res = ttest_rel(sub[ca].values, sub[cb].values, nan_policy="omit")
            diff  = sub[ca].values - sub[cb].values
            rows.append({
                "channel":    channel,
                "band":       band,
                "comparison": f"{a}_vs_{b}",
                "n_subjects": int(len(sub)),
                "mean_diff":  round3(np.mean(diff)),
                "t_stat":     round3(t_res.statistic),
                "p_value":    float(t_res.pvalue) if np.isfinite(t_res.pvalue) else np.nan,
            })
    return pd.DataFrame(rows)


# ── PSD helpers (unchanged) ───────────────────────────────────────────────────

def welch_mean_sem(
    x_uV: np.ndarray, sf: float,
    epochs: List[Tuple[float, float]], cfg: Cfg
):
    if not epochs:
        return np.array([]), np.array([]), np.array([]), 0

    nperseg = int(round(cfg.welch_nperseg_s * sf))
    if nperseg < 8:
        raise ValueError("welch_nperseg_s too small for sampling rate.")
    noverlap = min(int(round(cfg.welch_overlap * nperseg)), nperseg - 1)

    psds, freqs_ref = [], None
    for s, e in epochs:
        i0, i1 = int(round(s * sf)), int(round(e * sf))
        seg = x_uV[i0:i1]
        if seg.size < nperseg:
            continue
        f, p = welch(seg, fs=sf, nperseg=nperseg, noverlap=noverlap,
                     detrend="linear", scaling="density",
                     window="hann", average="mean")
        keep = f <= cfg.psd_fmax_hz
        f, p = f[keep], p[keep]
        if freqs_ref is None:
            freqs_ref = f
        elif f.shape != freqs_ref.shape or not np.allclose(f, freqs_ref):
            raise RuntimeError("Welch frequency bins mismatch.")
        psds.append(p)

    if not psds:
        return np.array([]), np.array([]), np.array([]), 0

    psds = np.asarray(psds)
    mean = psds.mean(axis=0)
    sem  = (psds.std(axis=0, ddof=1) / np.sqrt(psds.shape[0])
            if psds.shape[0] > 1 else np.zeros_like(mean))
    return freqs_ref, mean, sem, int(psds.shape[0])


def interp_to(freqs, y, target):
    if freqs.size == 0 or y.size == 0:
        return np.full_like(target, np.nan, dtype=float)
    return np.interp(target, freqs, y, left=np.nan, right=np.nan)


def bandpower_from_psd(
    freqs: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float
) -> float:
    if freqs.size == 0 or psd.size == 0:
        return np.nan
    m = (freqs >= f_lo) & (freqs < f_hi)
    if not np.any(m):
        return np.nan
    return float(np.trapz(psd[m], freqs[m]))


def compute_bandpower_table_channel(
    payloads: List[Dict], cfg: Cfg, channel: str
) -> pd.DataFrame:
    rows = []
    for p in payloads:
        if "eeg_psd_channels" not in p or channel not in p["eeg_psd_channels"]:
            continue
        for state in ["phasic", "tonic", "transition"]:
            f = p["eeg_psd_channels"][channel][state]["freqs"]
            m = p["eeg_psd_channels"][channel][state]["mean"]
            for band_name, (lo, hi) in cfg.bandpower_bands.items():
                rows.append({
                    "subject":        p["name"],
                    "state":          state,
                    "band":           band_name,
                    "band_lo_hz":     lo,
                    "band_hi_hz":     hi,
                    "bandpower_uV2":  bandpower_from_psd(f, m, lo, hi),
                })
    return pd.DataFrame(rows)


# ── Plotting (unchanged) ──────────────────────────────────────────────────────

STATE_COLORS  = {"phasic": "tab:red", "tonic": "tab:blue", "transition": "tab:gray"}
STATE_PRETTY  = {"phasic": "Phasic REM", "tonic": "Tonic REM", "transition": "Transition REM"}


def _add_state_shading(ax, bouts, cfg: Cfg, tmin: float, tmax: float):
    for label in ["transition", "tonic", "phasic"]:
        alpha = (cfg.transition_alpha if label == "transition"
                 else cfg.tonic_alpha if label == "tonic" else cfg.phasic_alpha)
        color = STATE_COLORS[label]
        for s, e in bouts[label]:
            if e > tmin and s < tmax:
                ax.axvspan(max(s, tmin), min(e, tmax), alpha=alpha, color=color, lw=0)


def _state_legend(cfg: Cfg):
    return [
        Patch(facecolor=STATE_COLORS["phasic"],     alpha=cfg.phasic_alpha,     label="Phasic REM"),
        Patch(facecolor=STATE_COLORS["tonic"],      alpha=cfg.tonic_alpha,      label="Tonic REM"),
        Patch(facecolor=STATE_COLORS["transition"], alpha=cfg.transition_alpha, label="Transition REM"),
    ]


def subject_excerpt(cfg: Cfg, t: np.ndarray, phasic_bouts):
    if phasic_bouts:
        anchor = phasic_bouts[0][0]
        tmin = max(0.0, anchor - cfg.plot_pad_left_s)
        tmax = min(float(t[-1]), tmin + cfg.plot_span_s)
    else:
        tmin = 0.0
        tmax = min(float(t[-1]), cfg.plot_span_s)
    return tmin, tmax


def _plot_psd_curves(ax, psd_dict: Dict, cfg: Cfg, title: str, ylabel_prefix: str):
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"{ylabel_prefix} PSD (dB, µV²/Hz)" if cfg.psd_to_db else f"{ylabel_prefix} PSD (µV²/Hz)")
    ax.set_xlim(0, cfg.psd_fmax_hz)
    eps = 1e-20
    for label in ["phasic", "tonic", "transition"]:
        f    = psd_dict[label]["freqs"]
        mean = psd_dict[label]["mean"]
        sem  = psd_dict[label]["sem"]
        n    = psd_dict[label]["n"]
        if f.size == 0:
            continue
        if cfg.psd_to_db:
            md = 10 * np.log10(mean + eps)
            lo = 10 * np.log10(np.maximum(mean - sem, 0) + eps)
            hi = 10 * np.log10(mean + sem + eps)
        else:
            md = mean; lo = np.maximum(mean - sem, 0); hi = mean + sem
        ax.plot(f, md, label=f"{STATE_PRETTY[label]} (n={n})")
        ax.fill_between(f, lo, hi, alpha=0.18)
    ax.legend(loc="upper right")


def plot_subject_eog_preview(ax, payload: Dict, cfg: Cfg, show_legend: bool = False):
    t, x = payload["t"], payload["x_uV"]
    bouts = {k: payload[f"{k}_bouts"] for k in ("phasic", "tonic", "transition")}
    tmin, tmax = subject_excerpt(cfg, t, bouts["phasic"])
    m  = (t >= tmin) & (t <= tmax)
    tt = t[m][::cfg.plot_downsample]
    xx = x[m][::cfg.plot_downsample]
    ax.plot(tt, xx, lw=1)
    _add_state_shading(ax, bouts, cfg, tmin, tmax)
    ax.set_title(f"{payload['name']} — EOG (classification)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("EOG1 (µV)")
    if show_legend:
        ax.legend(handles=_state_legend(cfg), loc="upper right", frameon=True)


def plot_subject_eeg_psd(ax, payload: Dict, cfg: Cfg):
    ch = cfg.main_eeg_channel
    if "eeg_psd_channels" not in payload or ch not in payload["eeg_psd_channels"]:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"Missing EEG channel: {ch}", ha="center", va="center")
        return
    _plot_psd_curves(ax, payload["eeg_psd_channels"][ch], cfg,
                     title=f"{payload['name']} — EEG PSD ({ch})",
                     ylabel_prefix=f"EEG ({ch})")


def plot_combined_psd_from_channel(ax, payloads: List[Dict], cfg: Cfg, channel: str):
    ax.set_title(f"Combined EEG PSD across subjects (mean ± SEM) [{channel}]")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB, µV²/Hz)" if cfg.psd_to_db else "PSD (µV²/Hz)")
    ax.set_xlim(0, cfg.psd_fmax_hz)
    target = np.arange(0.0, cfg.psd_fmax_hz + 1e-9, cfg.combined_psd_df_hz)
    eps = 1e-20
    for label, pretty in [("phasic", "Phasic REM"), ("tonic", "Tonic REM"), ("transition", "Transition REM")]:
        curves = []
        for p in payloads:
            if "eeg_psd_channels" not in p or channel not in p["eeg_psd_channels"]:
                continue
            f = p["eeg_psd_channels"][channel][label]["freqs"]
            m = p["eeg_psd_channels"][channel][label]["mean"]
            curves.append(interp_to(f, m, target))
        if not curves:
            continue
        curves  = np.vstack(curves)
        mean    = np.nanmean(curves, axis=0)
        n_eff   = np.sum(np.isfinite(curves), axis=0)
        sem     = np.where(n_eff > 0, np.nanstd(curves, axis=0, ddof=1) / np.sqrt(n_eff), np.nan)
        if cfg.psd_to_db:
            md = 10 * np.log10(mean + eps)
            lo = 10 * np.log10(np.maximum(mean - sem, 0) + eps)
            hi = 10 * np.log10(mean + sem + eps)
        else:
            md = mean; lo = np.maximum(mean - sem, 0); hi = mean + sem
        ax.plot(target, md, label=pretty)
        ax.fill_between(target, lo, hi, alpha=0.18)
    ax.legend(loc="upper right")


def plot_bandpower_bar_single_channel(ax, bp_df: pd.DataFrame, channel: str):
    states = ["phasic", "tonic", "transition"]
    bands  = list(bp_df["band"].dropna().unique())
    stats  = []
    for band in bands:
        for state in states:
            vals = bp_df[(bp_df["band"] == band) & (bp_df["state"] == state)]["bandpower_uV2"].astype(float).values
            vals = vals[np.isfinite(vals)]
            stats.append((band, state,
                          float(np.mean(vals)) if vals.size else np.nan,
                          float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0))
    stats_df = pd.DataFrame(stats, columns=["band", "state", "mean", "sem"])
    x = np.arange(len(bands)); width = 0.25
    offsets = {"phasic": -width, "tonic": 0.0, "transition": width}
    for state in states:
        means = [stats_df[(stats_df.band == b) & (stats_df.state == state)]["mean"].values[0] for b in bands]
        sems  = [stats_df[(stats_df.band == b) & (stats_df.state == state)]["sem"].values[0]  for b in bands]
        ax.bar(x + offsets[state], means, width=width,
               label=STATE_PRETTY[state], yerr=sems, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(bands, rotation=0)
    ax.set_ylabel("Bandpower (µV²)")
    ax.set_title(f"EEG bandpower by REM state (mean ± SEM) [{channel}]")
    ax.legend(loc="upper right")


def plot_theta_only_channel(ax, payloads: List[Dict], cfg: Cfg, channel: str):
    ax.set_title(f"Theta-range EEG PSD (mean ± SEM) [{channel}]")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (dB, µV²/Hz)" if cfg.psd_to_db else "PSD (µV²/Hz)")
    f_lo, f_hi = 3.0, 10.0; ax.set_xlim(f_lo, f_hi)
    target = np.arange(f_lo, f_hi + 1e-9, cfg.combined_psd_df_hz); eps = 1e-20
    for label, pretty in [("phasic", "Phasic REM"), ("tonic", "Tonic REM"), ("transition", "Transition REM")]:
        curves = []
        for p in payloads:
            if "eeg_psd_channels" not in p or channel not in p["eeg_psd_channels"]:
                continue
            f = p["eeg_psd_channels"][channel][label]["freqs"]
            m = p["eeg_psd_channels"][channel][label]["mean"]
            curves.append(interp_to(f, m, target))
        if not curves:
            continue
        curves = np.vstack(curves)
        mean   = np.nanmean(curves, axis=0)
        n_eff  = np.sum(np.isfinite(curves), axis=0)
        sem    = np.where(n_eff > 0, np.nanstd(curves, axis=0, ddof=1) / np.sqrt(n_eff), np.nan)
        if cfg.psd_to_db:
            md = 10 * np.log10(mean + eps)
            lo = 10 * np.log10(np.maximum(mean - sem, 0) + eps)
            hi = 10 * np.log10(mean + sem + eps)
        else:
            md = mean; lo = np.maximum(mean - sem, 0); hi = mean + sem
        ax.plot(target, md, label=pretty)
        ax.fill_between(target, lo, hi, alpha=0.18)
    ax.legend(loc="best")


def plot_theta_bandpower_bar(ax, bp_df: pd.DataFrame, title: str):
    states = ["phasic", "tonic", "transition"]
    pretty = STATE_PRETTY
    theta_rows = bp_df[bp_df["band"].astype(str).str.lower().str.contains("theta")].copy()
    if theta_rows.empty:
        ax.set_axis_off(); ax.text(0.5, 0.5, "No theta data.", ha="center", va="center"); return
    means, sems = [], []
    for s in states:
        vals = theta_rows[theta_rows["state"] == s]["bandpower_uV2"].astype(float).values
        vals = vals[np.isfinite(vals)]
        means.append(float(np.mean(vals)) if vals.size else np.nan)
        sems.append(float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
    x = np.arange(len(states))
    ax.bar(x, means, yerr=sems, capsize=5)
    ax.set_xticks(x); ax.set_xticklabels([pretty[s] for s in states])
    ax.set_ylabel("Theta bandpower (µV²)"); ax.set_title(title)


def plot_aperiodic_boxplots(ax_low, ax_high, df: pd.DataFrame, channel: str):
    state_order = ["phasic", "tonic", "transition"]
    pretty = ["Phasic", "Tonic", "Transition"]
    if df.empty:
        for ax in [ax_low, ax_high]:
            ax.set_axis_off(); ax.text(0.5, 0.5, "No aperiodic data", ha="center", va="center")
        return
    ax_low.boxplot([df[f"{s}_low_exp"].dropna().values  for s in state_order], tick_labels=pretty)
    ax_low.set_title(f"FOOOF aperiodic exponent 2–30 Hz [{channel}]"); ax_low.set_ylabel("Exponent")
    ax_high.boxplot([df[f"{s}_high_exp"].dropna().values for s in state_order], tick_labels=pretty)
    ax_high.set_title(f"FOOOF aperiodic exponent 30–48 Hz [{channel}]"); ax_high.set_ylabel("Exponent")


def plot_group_psd_with_fooof_fit(ax, payloads, cfg, channel, fit_range):
    lo, hi = fit_range
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB, µV²/Hz)" if cfg.psd_to_db else "PSD (µV²/Hz)")
    ax.set_xlim(max(0, lo - 1.0), min(cfg.psd_fmax_hz, hi + 2.0))
    target = np.arange(0.0, cfg.psd_fmax_hz + 1e-9, cfg.combined_psd_df_hz); eps = 1e-20
    for label, pretty, color in [("phasic", "Phasic REM", "tab:red"), ("tonic", "Tonic REM", "tab:blue")]:
        curves = []
        for p in payloads:
            if "eeg_psd_channels" not in p or channel not in p["eeg_psd_channels"]:
                continue
            f = p["eeg_psd_channels"][channel][label]["freqs"]
            m = p["eeg_psd_channels"][channel][label]["mean"]
            curves.append(interp_to(f, m, target))
        if not curves:
            continue
        mean = np.nanmean(np.vstack(curves), axis=0)
        ax.plot(target, 10 * np.log10(mean + eps), color=color, lw=2, label=f"{pretty} PSD")
        fm, f_fit, _ = fit_fooof_model(target, mean, fit_range, cfg)
        if f_fit.size:
            ax.plot(f_fit, fooof_aperiodic_fit_db(f_fit, fm),
                    color=color, lw=2, ls="--", label=f"{pretty} slope fit")
    ax.legend(loc="best"); ax.set_title(f"{channel} | {lo:.0f}–{hi:.0f} Hz")


def _pick_example_window(bouts, dur_s: float = 10.0):
    if not bouts:
        return None
    for s, e in bouts:
        if (e - s) >= dur_s:
            return (s, s + dur_s)
    s, e = bouts[0]
    return (s, min(e, s + dur_s))


def _slice(t, x, s, e):
    m = (t >= s) & (t <= e)
    if not np.any(m):
        return np.array([]), np.array([])
    return (t[m] - s), x[m]


def save_all_eeg_examples_grid(payloads, cfg, out_path: Path, dur_s: float = 10.0):
    states  = [("phasic_bouts", "Phasic REM", "tab:red"),
               ("tonic_bouts",  "Tonic REM",  "tab:blue"),
               ("transition_bouts", "Transition REM", "tab:gray")]
    n_subj  = len(payloads)
    n_cols  = len(states)
    fig, axes = plt.subplots(n_subj, n_cols,
                             figsize=(5.6 * n_cols, 2.8 * n_subj),
                             sharex=False, sharey=False)
    if n_subj == 1:
        axes = np.array([axes])
    for r, p in enumerate(payloads):
        name   = p["name"]
        t_eog, eog = p["t"], p["x_uV"]
        t_eeg, eeg = p["t_eeg"], p["eeg_uV"]
        for c, (key, pretty, color) in enumerate(states):
            ax    = axes[r, c]
            bouts = p.get(key, [])
            win   = _pick_example_window(bouts, dur_s=dur_s)
            if win is None:
                ax.set_axis_off(); ax.text(0.5, 0.5, f"{name}\nNo {pretty}", ha="center", va="center"); continue
            s, e = win
            tt_eeg, xx_eeg = _slice(t_eeg, eeg, s, e)
            tt_eog, xx_eog = _slice(t_eog, eog, s, e)
            ax.axvspan(0, e - s, alpha=0.08, color=color, lw=0)
            ax.plot(tt_eeg, xx_eeg, lw=0.9, label="EEG" if (r == 0 and c == 0) else None)
            ax.plot(tt_eog, xx_eog, lw=0.9, alpha=0.65, label="EOG" if (r == 0 and c == 0) else None)
            if r == 0: ax.set_title(pretty)
            if c == 0: ax.set_ylabel(f"{name}\nµV")
            ax.set_xlabel("Time (s)"); ax.grid(False)
    fig.suptitle(f"EEG+EOG example windows — EEG={cfg.main_eeg_channel}, EOG used for labeling", y=0.995)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ── Core analysis per recording ───────────────────────────────────────────────

def analyze_recording(
    subject: str, night: str,
    edf_dir: Path, hyp_dir: Path,
    cfg: Cfg,
):
    """
    Full analysis for one subject/night.
    Loads fragmented EDF files, reads .npy hypnogram, classifies REM windows,
    computes PSD/bandpower/FOOOF on EEG channels.
    Returns (bout_df, summary_dict, plot_payload, rem_period_df).
    """
    name = f"S{subject}N{night}"
    print(f"  Loading EDF fragments for {name} ...")
    raw = load_night_raw(edf_dir, subject, night)

    # EOG (for classification)
    t, x_uV, sf = load_eog_from_raw(raw, cfg)
    t_end = float(t[-1])

    # EEG for examples (use first analysis channel)
    try:
        t_eeg, eeg_uV, _ = load_eeg_channel_from_raw(raw, cfg, cfg.main_eeg_channel)
    except Exception:
        t_eeg = t
        eeg_uV = np.zeros_like(x_uV)

    # Hypnogram → REM intervals
    rem = get_rem_intervals_from_npy(hyp_dir, subject, night, t_end)
    if not rem:
        raise RuntimeError(f"No REM intervals found for {name}")

    total_rem_true = total_duration(rem)

    # Classify REM substates (percentile-based, matching original Sleep-EDF script)
    ph, to_kept, transition, win_df, ph_thr_uV, to_thr_uV = \
        classify_rem_windows_percentile(x_uV, sf, rem, cfg)

    ph_b  = merge_intervals(ph,          cfg.merge_gap_s)
    to_b  = merge_intervals(to_kept,     cfg.merge_gap_s)
    tr_b  = merge_intervals(transition,  cfg.merge_gap_s)

    rem_period_df = rem_period_counts_table(rem, ph_b, to_b, tr_b)

    total_ph = total_duration(ph_b)
    total_to = total_duration(to_b)
    total_tr = total_duration(tr_b)
    pct_ph   = 100.0 * total_ph / total_rem_true if total_rem_true > 0 else np.nan
    pct_to   = 100.0 * total_to / total_rem_true if total_rem_true > 0 else np.nan
    pct_tr   = 100.0 * total_tr / total_rem_true if total_rem_true > 0 else np.nan

    n_rem_periods     = len(rem)
    n_phasic_bouts    = len(ph_b)
    n_tonic_bouts     = len(to_b)
    n_transition_bouts = len(tr_b)

    # Build bout-level CSV rows
    rows = []
    for label, bouts in [("phasic", ph_b), ("tonic", to_b), ("transition", tr_b)]:
        for s, e in bouts:
            rp = rem_period_index(s, rem)
            rem_start, rem_end = (rem[rp] if rp >= 0 else (np.nan, np.nan))
            mx = eog_metric_for_window(x_uV, sf, s, e, cfg.eog_window_metric)
            rows.append({
                "subject":                          subject,
                "night":                            night,
                "start_s":                          round3(s),
                "end_s":                            round3(e),
                "duration_s":                       round3(e - s),
                "label":                            label,
                "eog_metric_uV":                    round3(mx),
                "classification_metric":            cfg.eog_window_metric,
                "phasic_percentile_threshold_uV":   round3(ph_thr_uV),
                "tonic_percentile_threshold_uV":    round3(to_thr_uV),
                "phasic_percentile":                cfg.phasic_window_percentile,
                "tonic_percentile":                 cfg.tonic_window_percentile,
                "rem_period_idx":                   int(rp) if rp >= 0 else np.nan,
                "rem_period_start_s":               round3(rem_start) if rp >= 0 else np.nan,
                "rem_period_end_s":                 round3(rem_end)   if rp >= 0 else np.nan,
                "total_rem_s":                      round3(total_rem_true),
                "total_phasic_s":                   round3(total_ph),
                "total_tonic_s":                    round3(total_to),
                "total_transition_s":               round3(total_tr),
                "pct_phasic_of_rem":                round3(pct_ph),
                "pct_tonic_of_rem":                 round3(pct_to),
                "pct_transition_of_rem":            round3(pct_tr),
            })

    df = pd.DataFrame(rows).sort_values(["start_s", "end_s"]).reset_index(drop=True)

    summary = {
        "recording":                        name,
        "subject":                          subject,
        "night":                            night,
        "total_rem_s":                      round3(total_rem_true),
        "total_phasic_s":                   round3(total_ph),
        "total_tonic_s":                    round3(total_to),
        "total_transition_s":               round3(total_tr),
        "pct_phasic_of_rem":                round3(pct_ph),
        "pct_tonic_of_rem":                 round3(pct_to),
        "pct_transition_of_rem":            round3(pct_tr),
        "n_rem_periods":                    int(n_rem_periods),
        "n_phasic_bouts":                   int(n_phasic_bouts),
        "n_tonic_bouts":                    int(n_tonic_bouts),
        "n_transition_bouts":               int(n_transition_bouts),
        "mean_phasic_bouts_per_rem":        round3(n_phasic_bouts / n_rem_periods) if n_rem_periods else np.nan,
        "mean_tonic_bouts_per_rem":         round3(n_tonic_bouts  / n_rem_periods) if n_rem_periods else np.nan,
        "mean_transition_bouts_per_rem":    round3(n_transition_bouts / n_rem_periods) if n_rem_periods else np.nan,
        "phasic_percentile_threshold_uV":   round3(ph_thr_uV),
        "tonic_percentile_threshold_uV":    round3(to_thr_uV),
        "classification_metric":            cfg.eog_window_metric,
        "phasic_percentile":                cfg.phasic_window_percentile,
        "tonic_percentile":                 cfg.tonic_window_percentile,
    }

    # PSD epochs (same windows used for all EEG channels)
    ph_ep = chunk(ph_b, cfg.psd_epoch_s, cfg.psd_epoch_overlap)
    to_ep = chunk(to_b, cfg.psd_epoch_s, cfg.psd_epoch_overlap)
    tr_ep = chunk(tr_b, cfg.psd_epoch_s, cfg.psd_epoch_overlap)

    # Load intracranial channels from both BrainVision folders
    eeg_psd_channels   = {}
    aperiodic_channels = {}
    subject_data_dir   = cfg.data_root / SUBJECT_FOLDERS[subject]

    for folder_name in INTRA_FOLDERS:
        intra_dir = subject_data_dir / folder_name
        if not intra_dir.is_dir():
            continue
        intra_raw = load_intra_night_raw(intra_dir, subject, night)
        if intra_raw is None:
            print(f"    [WARN] {name}: no BrainVision files in {folder_name}")
            continue
        # Use per-subject designated channels from thesis (Supplementary Table 2).
        # Fall back to auto-detection only if the subject has no entry in the map.
        designated = SUBJECT_CHANNEL_MAP.get(subject)
        if designated is not None:
            intra_chs = [c for c in designated if c in intra_raw.ch_names]
            missing   = [c for c in designated if c not in intra_raw.ch_names]
            if missing:
                print(f"    [WARN] {name}/{folder_name}: designated channels not found: {missing}")
        else:
            intra_chs = get_intracranial_channels(intra_raw)
        print(f"    {name}/{folder_name}: using channels: {intra_chs}")

        for ch_name in intra_chs:
            if ch_name in eeg_psd_channels:
                continue  # already loaded from a previous folder
            try:
                _, x_ch, sf_ch = load_eeg_channel_from_raw(intra_raw, cfg, ch_name)
            except Exception as err:
                print(f"    [WARN] {name}: channel '{ch_name}' — {err}")
                continue

            f_ph, m_ph, se_ph, n_ph = welch_mean_sem(x_ch, sf_ch, ph_ep, cfg)
            f_to, m_to, se_to, n_to = welch_mean_sem(x_ch, sf_ch, to_ep, cfg)
            f_tr, m_tr, se_tr, n_tr = welch_mean_sem(x_ch, sf_ch, tr_ep, cfg)

            eeg_psd_channels[ch_name] = {
                "phasic":     {"freqs": f_ph, "mean": m_ph, "sem": se_ph, "n": n_ph},
                "tonic":      {"freqs": f_to, "mean": m_to, "sem": se_to, "n": n_to},
                "transition": {"freqs": f_tr, "mean": m_tr, "sem": se_tr, "n": n_tr},
            }
            aperiodic_channels[ch_name] = compute_aperiodic_from_psd_dict(
                eeg_psd_channels[ch_name], cfg
            )

    print(f"    {name}: total intracranial channels analysed: {list(eeg_psd_channels)}")

    plot_payload = {
        "name":             name,
        "t":                t,
        "x_uV":             x_uV,
        "t_eeg":            t_eeg,
        "eeg_uV":           eeg_uV,
        "phasic_bouts":     ph_b,
        "tonic_bouts":      to_b,
        "transition_bouts": tr_b,
        "eeg_psd_channels": eeg_psd_channels,
        "aperiodic_channels": aperiodic_channels,
        "window_classification": win_df,
    }

    return df, summary, plot_payload, rem_period_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = Cfg(
        data_root=DATA_ROOT,
        eog_channels=["EOG1"],
        eeg_channels=list(EEG_ANALYSIS_CHANNELS),   # empty = auto-detect all non-EOG channels
        filter_lfreq_hz=0.1,
        filter_hfreq_hz=60.0,
        phasic_window_percentile=90.0,
        tonic_window_percentile=20.0,
        eog_window_metric="max_abs",
        merge_gap_s=2.0,
        main_eeg_channel="C3-Cz",
        frontal_channel="C3-Cz",
        parietal_channel="Oz-Cz",
    )

    results_dir = DATA_ROOT.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    recordings = discover_recordings(DATA_ROOT)
    print(f"\nFound {len(recordings)} recordings:\n")
    for s, n, _, _ in recordings:
        print(f"  S{s}  Night {n}")
    print()

    if MAX_RECORDINGS is not None:
        recordings = recordings[:MAX_RECORDINGS]

    summaries:          List[Dict] = []
    payloads:           List[Dict] = []
    rem_period_tables:  List[pd.DataFrame] = []
    failed:             List[Dict] = []

    for subject, night, edf_dir, hyp_dir in recordings:
        name = f"S{subject}N{night}"
        print(f"\n── {name} ──")
        try:
            df, summary, payload, rem_period_df = analyze_recording(
                subject, night, edf_dir, hyp_dir, cfg
            )
        except Exception as e:
            import traceback
            print(f"  SKIP ({type(e).__name__}): {e}")
            traceback.print_exc()
            failed.append({"recording": name, "error_type": type(e).__name__, "error": str(e)})
            continue

        out_csv = results_dir / f"{name}{cfg.per_csv_suffix}"
        df.to_csv(out_csv, index=False)

        summaries.append(summary)
        payloads.append(payload)

        rem_period_df = rem_period_df.copy()
        rem_period_df.insert(0, "recording", name)
        rem_period_df.insert(1, "subject",   subject)
        rem_period_df.insert(2, "night",     night)
        rem_period_tables.append(rem_period_df)

        print(f"  REM={summary['total_rem_s']:.0f}s  "
              f"Phasic={summary['pct_phasic_of_rem']:.1f}%  "
              f"Tonic={summary['pct_tonic_of_rem']:.1f}%  "
              f"Transition={summary['pct_transition_of_rem']:.1f}%")
        print(f"  Saved: {out_csv.name}")

    if failed:
        pd.DataFrame(failed).to_csv(results_dir / "failed_recordings.csv", index=False)
        print(f"\nFailed: {len(failed)} recording(s) — see failed_recordings.csv")

    if not summaries:
        print("\nNo recordings processed successfully.")
        return

    # Summary CSV
    df_sum = pd.DataFrame(summaries)
    df_sum.to_csv(results_dir / cfg.summary_csv, index=False)
    print(f"\nSaved: {cfg.summary_csv}")

    # REM period table
    rem_all = pd.concat(rem_period_tables, ignore_index=True) if rem_period_tables else pd.DataFrame()
    rem_all.to_csv(results_dir / cfg.rem_period_counts_all_csv, index=False)
    print(f"Saved: {cfg.rem_period_counts_all_csv}")

    if not payloads:
        return

    # Collect all EEG channels present across all recordings
    all_channels = sorted({ch for p in payloads for ch in p.get("eeg_psd_channels", {})})
    print(f"\nChannels found across all recordings: {all_channels}")

    n      = len(payloads)
    n_ch   = len(all_channels)

    # ── FIG 1: per-subject EOG preview + PSD for every channel ───────────────
    # Layout: rows = subjects + 1 combined row; cols = EOG | ch1 | ch2 | ...
    n_cols1 = 1 + n_ch
    col_ratios = [1.0] + [1.0] * n_ch
    fig1 = plt.figure(figsize=(7 * n_cols1, 3.0 * n + 4.5))
    gs1  = fig1.add_gridspec(n + 1, n_cols1,
                             height_ratios=[1.0] * n + [1.1],
                             hspace=0.6, wspace=0.28)
    for i, p in enumerate(payloads):
        plot_subject_eog_preview(fig1.add_subplot(gs1[i, 0]), p, cfg, show_legend=(i == 0))
        for j, ch in enumerate(all_channels):
            cfg_ch = replace(cfg, main_eeg_channel=ch)
            plot_subject_eeg_psd(fig1.add_subplot(gs1[i, 1 + j]), p, cfg_ch)
    # Combined row: one combined PSD panel per channel
    for j, ch in enumerate(all_channels):
        plot_combined_psd_from_channel(fig1.add_subplot(gs1[n, 1 + j]), payloads, cfg, ch)
    fig1.suptitle(
        f"EEG PSD by REM state — all channels | "
        f"EOG1 classification, {cfg.filter_lfreq_hz}–{cfg.filter_hfreq_hz} Hz filter",
        y=0.995
    )
    fig1.tight_layout()
    out1 = results_dir / cfg.fig_eeg_psd_png
    fig1.savefig(out1, dpi=150); plt.close(fig1)
    print(f"Saved: {out1.name}")

    # ── FIG 2: bandpower bar charts — one panel per channel ──────────────────
    fig2, axes2 = plt.subplots(1, n_ch, figsize=(14 * n_ch, 5), squeeze=False)
    for j, ch in enumerate(all_channels):
        bp_df = compute_bandpower_table_channel(payloads, cfg, ch)
        plot_bandpower_bar_single_channel(axes2[0, j], bp_df, ch)
    fig2.suptitle(
        f"EEG bandpower across REM states — all channels | Labels from EOG1",
        y=0.995
    )
    fig2.tight_layout()
    out2 = results_dir / cfg.fig_eeg_bandpower_png
    fig2.savefig(out2, dpi=150); plt.close(fig2)
    print(f"Saved: {out2.name}")

    # ── FIG 3: theta — 2 rows (PSD, bar) × n_ch columns ─────────────────────
    fig3, axes3 = plt.subplots(2, n_ch, figsize=(8 * n_ch, 10), squeeze=False)
    for j, ch in enumerate(all_channels):
        bp_ch = compute_bandpower_table_channel(payloads, cfg, ch)
        plot_theta_only_channel(axes3[0, j], payloads, cfg, ch)
        plot_theta_bandpower_bar(axes3[1, j], bp_ch,
                                 f"Theta bandpower [{ch}]")
    fig3.suptitle(
        f"Theta-range EEG — all channels | "
        f"Same EOG1-labeled epochs | {cfg.filter_lfreq_hz}–{cfg.filter_hfreq_hz} Hz filter",
        y=0.995
    )
    fig3.tight_layout()
    out3 = results_dir / cfg.theta_combined_png
    fig3.savefig(out3, dpi=150); plt.close(fig3)
    print(f"Saved: {out3.name}")

    # ── FIG 4: EEG example traces ─────────────────────────────────────────────
    out4 = results_dir / cfg.eeg_examples_all_png
    save_all_eeg_examples_grid(payloads, cfg, out4, dur_s=10.0)
    print(f"Saved: {out4.name}")

    # ── FIG 5 & 6: FOOOF aperiodic — all channels in one figure each ─────────
    try:
        # Collect aperiodic data for all channels; concatenate into one CSV each
        all_ap_dfs    = []
        all_stats_dfs = []
        for ch in all_channels:
            ap_df    = make_aperiodic_subject_table(payloads, cfg, ch)
            ap_stats = paired_stats_table(ap_df, ch)
            all_ap_dfs.append(ap_df)
            all_stats_dfs.append(ap_stats)

        pd.concat(all_ap_dfs,    ignore_index=True).to_csv(
            results_dir / cfg.aperiodic_subject_csv, index=False)
        pd.concat(all_stats_dfs, ignore_index=True).to_csv(
            results_dir / cfg.aperiodic_stats_csv,   index=False)
        print(f"Saved: {cfg.aperiodic_subject_csv}, {cfg.aperiodic_stats_csv}")

        # FIG 5: n_ch rows × 2 cols (low band | high band)
        fig5, axes5 = plt.subplots(n_ch, 2, figsize=(12, 5 * n_ch), squeeze=False)
        for j, (ch, ap_df) in enumerate(zip(all_channels, all_ap_dfs)):
            plot_aperiodic_boxplots(axes5[j, 0], axes5[j, 1], ap_df, ch)
            axes5[j, 0].set_title(f"Aperiodic exponent 2–30 Hz [{ch}]")
            axes5[j, 1].set_title(f"Aperiodic exponent 30–48 Hz [{ch}]")
        fig5.suptitle(
            f"FOOOF aperiodic exponents — all channels\n"
            f"Phasic = top {cfg.phasic_window_percentile:.0f}%, "
            f"Tonic = bottom {cfg.tonic_window_percentile:.0f}% EOG windows",
            y=0.995,
        )
        fig5.tight_layout()
        fig5.savefig(results_dir / cfg.aperiodic_fig_png, dpi=150); plt.close(fig5)
        print(f"Saved: {cfg.aperiodic_fig_png}")

        # FIG 6: n_ch rows × 2 cols (low band | high band)
        fig6, axes6 = plt.subplots(n_ch, 2, figsize=(16, 5 * n_ch), squeeze=False)
        for j, ch in enumerate(all_channels):
            plot_group_psd_with_fooof_fit(axes6[j, 0], payloads, cfg, ch, cfg.aperiodic_lowband)
            plot_group_psd_with_fooof_fit(axes6[j, 1], payloads, cfg, ch, cfg.aperiodic_highband)
        fig6.suptitle(
            f"Power spectrum + FOOOF slope fit — all channels — phasic vs tonic REM\n"
            f"Phasic = top {cfg.phasic_window_percentile:.0f}%, "
            f"Tonic = bottom {cfg.tonic_window_percentile:.0f}% EOG windows",
            y=0.995,
        )
        fig6.tight_layout()
        fig6.savefig(results_dir / cfg.psd_fit_fig_png, dpi=150); plt.close(fig6)
        print(f"Saved: {cfg.psd_fit_fig_png}")

    except ImportError as e:
        print(f"\nSkipping FOOOF analysis (not installed): {e}")

    print(f"\nAll outputs saved to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
