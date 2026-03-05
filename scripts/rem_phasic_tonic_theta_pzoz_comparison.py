"""
REM Phasic/Tonic Detection and Spectral Analysis Pipeline

This program processes polysomnography (PSG) recordings to detect and classify REM sleep microstates (phasic, tonic, and transition) and analyze their EEG spectral properties.

Pipeline overview:

1. Load data
   • Load PSG recordings (.edf) and corresponding hypnogram annotation files
   • Extract EOG signal (for event detection) and EEG signal (for spectral analysis)

2. Preprocess signals
   • Apply bandpass filtering (default 0.1–60 Hz)
   • Convert signals to microvolts

3. Extract REM sleep periods
   • Read hypnogram annotations
   • Identify all REM sleep intervals in the recording

4. Detect eye movement (EM) events
   • Identify peaks in the EOG signal above a defined amplitude threshold
   • Enforce duration and separation constraints between events

5. Classify REM microstates
   REM periods are divided into fixed windows (default 4 s) and classified as:

   • Phasic REM
     Window contains ≥2 eye movement events that pass timing rules

   • Tonic REM
     Window contains no large eye movement activity
     (maximum EOG amplitude below tonic threshold)

   • Transition REM
     Windows inside REM that are neither phasic nor tonic

6. Merge adjacent windows
   • Windows separated by small gaps are merged into longer bouts

7. Compute summary statistics
   For each subject:
   • Total REM duration
   • Total phasic, tonic, and transition time
   • Percent of REM spent in each state
   • Number of bouts per REM period

8. Spectral analysis (EEG)
   • Extract EEG epochs aligned to detected REM states
   • Compute power spectral density (Welch method)
   • Calculate bandpower in canonical bands:
       delta (1–4 Hz)
       theta (4–8 Hz)
       alpha (8–12 Hz)
       beta (12–30 Hz)
       gamma (30–50 Hz)

9. Output results
   The program saves:
   • Per-subject REM state classification tables (CSV)
   • Summary statistics across recordings
   • REM period bout counts
   • EEG spectral plots comparing phasic, tonic, and transition REM

10. Visualization
   Figures include:
   • EEG PSD per subject and combined across subjects
   • Bandpower comparison across REM states
   • Theta-band comparison across EEG channels
   • Example EEG/EOG segments for each REM state
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import welch
import pyedflib


# Config

# Debug option: limit number of paired recordings processed (set to None for all)
MAX_RECORDINGS = 10


@dataclass
class Cfg:
    folder: Path
    edf_files: List[str]
    hypnogram_map: Dict[str, str]

    # EOG (used for ALL phasic/tonic/transition detection)
    eog_channels: List[str] = ("EOG horizontal",)

    # EEG (for theta plot + examples only)
    eeg_channels: Tuple[str, ...] = ("Fpz-Cz", "Pz-Oz")
    eeg_reference: str = "mean"

    # Preprocessing
    filter_lfreq_hz: float = 0.1
    filter_hfreq_hz: float = 60.0

    # Windowing
    window_s: float = 4.0

    # EM event definition
    phasic_peak_threshold_uV: float = 100.0
    max_event_duration_s: float = 0.5
    min_event_separation_s: float = 0.25

    # Edge guard (Rule 3)
    edge_guard_first_s: float = 2.0
    bin_s: float = 1.0

    # TONIC
    tonic_max_abs_uV: float = 60.0

    # Buffer rule
    buffer_s: float = 0.0

    # Merge rule (combine epochs separated by < 2 s)
    merge_gap_s: float = 2.0

    # Plot excerpt
    plot_span_s: float = 70.0
    plot_pad_left_s: float = 20.0
    plot_downsample: int = 5

    # Plot styling
    phasic_alpha: float = 0.30
    tonic_alpha: float = 0.22
    transition_alpha: float = 0.14

    # PSD
    psd_epoch_s: float = 4.0
    psd_epoch_overlap: float = 0.0
    welch_nperseg_s: float = 2.0
    welch_overlap: float = 0.5
    psd_fmax_hz: float = 50.0
    psd_to_db: bool = True
    combined_psd_df_hz: float = 0.5

    # Bandpower (for bar plot)
    bandpower_bands: Dict[str, Tuple[float, float]] = None

    # Outputs
    per_csv_suffix: str = ".rem_states_onset_aligned.csv"
    summary_csv: str = "REM_summary.csv"
    rem_period_counts_all_csv: str = "REM_period_counts_all_subjects.csv"

    # CLEAN FIGURES (new names to avoid EOG/EEG confusion)
    fig_eeg_pzoz_psd_png: str = "fig1_EEG_PzOz_PSD_per_subject_and_combined.png"
    fig_eeg_pzoz_bandpower_png: str = "fig2_EEG_PzOz_bandpower.png"
    theta_combined_png: str = "fig3_EEG_theta_FpzCz_vs_PzOz__PSD_and_thetaBandpower__same_epochs.png"
    eeg_examples_all_png: str = "fig4_EEG_examples_all_subjects__PzOzOnly.png"

    # Which EEG channel should be treated as the “main” downstream channel
    main_theta_channel: str = "Pz-Oz"

    def __post_init__(self):
        if self.bandpower_bands is None:
            self.bandpower_bands = {
                "delta (1–4)": (1.0, 4.0),
                "theta (4–8)": (4.0, 8.0),
                "alpha (8–12)": (8.0, 12.0),
                "beta (12–30)": (12.0, 30.0),
                "gamma (30–50)": (30.0, 50.0),
            }


# Small utilities

def runs_true(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = mask.astype(bool)
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]
    return list(zip(starts.tolist(), ends.tolist()))


def merge_intervals(intervals: List[Tuple[float, float]], gap_s: float) -> List[Tuple[float, float]]:
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


def subtract_intervals(base: List[Tuple[float, float]], cuts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
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


def tile_from_gap_starts(intervals: List[Tuple[float, float]], win_s: float) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for s, e in intervals:
        cur = float(s)
        while cur + win_s <= e:
            out.append((cur, cur + win_s))
            cur += win_s
    return out


def chunk(intervals: List[Tuple[float, float]], epoch_s: float, overlap_frac: float) -> List[Tuple[float, float]]:
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

        n_ph = _count(phasic_bouts)
        n_to = _count(tonic_bouts)
        n_tr = _count(transition_bouts)

        rows.append({
            "rem_period_idx": int(i),
            "rem_start_s": round3(rs),
            "rem_end_s": round3(re_),
            "rem_duration_s": round3(re_ - rs),
            "n_phasic_bouts": int(n_ph),
            "n_tonic_bouts": int(n_to),
            "n_transition_bouts": int(n_tr),
            "n_total_bouts": int(n_ph + n_to + n_tr),
        })
    return pd.DataFrame(rows)


# Robust channel resolution

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
                f"Missing channel '{w}'.\nAvailable channels:\n{list(raw.ch_names)}"
            )
        resolved.append(hit)
    return resolved


# Hypnogram -> REM intervals

def hyp_path_for_psg(psg_path: Path, cfg: Cfg) -> Path:
    key = psg_path.name
    if key not in cfg.hypnogram_map:
        raise KeyError(
            f"No hypnogram mapping for PSG file: {key}\n"
            f"Known PSG keys: {list(cfg.hypnogram_map.keys())}"
        )
    return psg_path.with_name(cfg.hypnogram_map[key])


def _norm_stage_label(desc) -> str:
    if desc is None:
        return ""
    if isinstance(desc, bytes):
        desc = desc.decode("utf-8", errors="replace")
    return " ".join(str(desc).strip().split()).lower()


def get_rem_intervals_from_hypnogram(hyp_path: Path, t_end_s: float) -> List[Tuple[float, float]]:
    f = pyedflib.EdfReader(str(hyp_path))
    try:
        onsets, durations, descriptions = f.readAnnotations()
    finally:
        f.close()

    if descriptions is None or len(descriptions) == 0:
        return []

    rem_ints: List[Tuple[float, float]] = []
    for onset, dur, desc in zip(onsets, durations, descriptions):
        d = _norm_stage_label(desc)
        is_rem = (d == "sleep stage r") or (d == "r")
        if not is_rem:
            continue

        s = float(onset)
        e = float(onset + float(dur))
        if e <= 0:
            continue
        s = max(0.0, s)
        e = min(float(t_end_s), e)
        if e > s:
            rem_ints.append((s, e))

    return merge_intervals(sorted(rem_ints), gap_s=0.0)


# Preprocessing helpers

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


# Core signal loading

def load_eog(edf_path: Path, cfg: Cfg, chs: List[str]) -> Tuple[np.ndarray, np.ndarray, float]:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw.rename_channels(lambda c: c.strip())
    _apply_filter(raw, cfg)

    resolved = pick_channels_robust(raw, list(chs))
    picks = mne.pick_channels(raw.ch_names, include=resolved)
    if len(picks) == 0:
        raise ValueError(f"Channels not found: {chs}\nAvailable: {raw.ch_names}")

    x = raw.get_data(picks=picks).mean(axis=0) * 1e6  # -> µV
    sf = float(raw.info["sfreq"])
    t = np.arange(x.size) / sf
    return t, x, sf


def load_eeg(edf_path: Path, cfg: Cfg) -> Tuple[np.ndarray, np.ndarray, float]:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw.rename_channels(lambda c: c.strip())
    _apply_filter(raw, cfg)

    if cfg.eeg_reference.lower() != "mean":
        wanted = [cfg.eeg_reference]
        resolved = pick_channels_robust(raw, wanted)
        picks = mne.pick_channels(raw.ch_names, include=resolved)
        x = raw.get_data(picks=picks).mean(axis=0) * 1e6
    else:
        wanted = list(cfg.eeg_channels)
        resolved = pick_channels_robust(raw, wanted)
        picks = mne.pick_channels(raw.ch_names, include=resolved)
        x = raw.get_data(picks=picks).mean(axis=0) * 1e6

    sf = float(raw.info["sfreq"])
    t = np.arange(x.size) / sf
    return t, x, sf


def load_eeg_channel(edf_path: Path, cfg: Cfg, ch: str) -> Tuple[np.ndarray, np.ndarray, float]:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw.rename_channels(lambda c: c.strip())
    _apply_filter(raw, cfg)

    resolved = pick_channels_robust(raw, [ch])
    picks = mne.pick_channels(raw.ch_names, include=resolved)
    if len(picks) != 1:
        raise RuntimeError(f"Expected exactly 1 pick for {ch}, got {len(picks)}: {resolved}")

    x = raw.get_data(picks=picks)[0] * 1e6  # -> µV
    sf = float(raw.info["sfreq"])
    t = np.arange(x.size) / sf
    return t, x, sf


# Detection

def detect_em_events(x_uV: np.ndarray, sf: float, cfg: Cfg) -> np.ndarray:
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


def edge_guard_ok(ev_rel: np.ndarray, cfg: Cfg) -> bool:
    if ev_rel.size == 0:
        return False
    all_in_first = np.all(ev_rel < cfg.edge_guard_first_s)
    bins = np.floor(ev_rel / cfg.bin_s).astype(int)
    in_two_bins = np.unique(bins).size >= 2
    return (not all_in_first) or in_two_bins


def detect_phasic(event_times_s: np.ndarray, rem: List[Tuple[float, float]], cfg: Cfg) -> List[Tuple[float, float]]:
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
    x_uV: np.ndarray, sf: float, rem: List[Tuple[float, float]], cfg: Cfg
) -> List[Tuple[float, float]]:
    x_abs = np.abs(x_uV)

    win_n = int(round(cfg.window_s * sf))
    if win_n <= 1:
        return []

    tonic_windows: List[Tuple[float, float]] = []
    for a, b in rem:
        cur = float(a)
        end = float(b)
        while cur + cfg.window_s <= end:
            i0 = int(round(cur * sf))
            i1 = i0 + win_n
            if i1 > x_abs.size:
                break
            mx = float(np.max(x_abs[i0:i1]))
            if mx < cfg.tonic_max_abs_uV:
                tonic_windows.append((cur, cur + cfg.window_s))
            cur += cfg.window_s

    return tonic_windows


def apply_buffer_rule(
    tonic_windows: List[Tuple[float, float]],
    phasic_windows: List[Tuple[float, float]],
    buffer_s: float,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    if buffer_s <= 0 or not tonic_windows or not phasic_windows:
        return tonic_windows, []

    expanded = [(ps - buffer_s, pe + buffer_s) for ps, pe in phasic_windows]

    kept, relabeled = [], []
    for tw in tonic_windows:
        (relabeled if any(overlap(tw, ex) for ex in expanded) else kept).append(tw)
    return kept, relabeled


def transition_between_phasic_and_tonic(
    phasic_bouts: List[Tuple[float, float]],
    tonic_bouts: List[Tuple[float, float]],
    rem: List[Tuple[float, float]],
    win_s: float,
) -> List[Tuple[float, float]]:
    occupied = sorted(phasic_bouts + tonic_bouts)
    rem_gaps = subtract_intervals(rem, occupied)
    return tile_from_gap_starts(rem_gaps, win_s)


def win_features(x_uV: np.ndarray, sf: float, s: float, e: float, event_times_s: np.ndarray) -> Tuple[float, int]:
    i0, i1 = int(round(s * sf)), int(round(e * sf))
    win_max = float(np.max(np.abs(x_uV[i0:i1]))) if i1 > i0 else float("nan")
    n_ev = int(np.sum((event_times_s >= s) & (event_times_s < e)))
    return win_max, n_ev

# PSD helpers

def welch_mean_sem(x_uV: np.ndarray, sf: float, epochs: List[Tuple[float, float]], cfg: Cfg):
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
        f, p = welch(
            seg, fs=sf, nperseg=nperseg, noverlap=noverlap,
            detrend="linear",
            scaling="density", window="hann", average="mean"
        )
        keep = f <= cfg.psd_fmax_hz
        f, p = f[keep], p[keep]
        if freqs_ref is None:
            freqs_ref = f
        else:
            if f.shape != freqs_ref.shape or not np.allclose(f, freqs_ref):
                raise RuntimeError("Welch frequency bins mismatch within a subject.")
        psds.append(p)

    if not psds:
        return np.array([]), np.array([]), np.array([]), 0

    psds = np.asarray(psds)
    mean = psds.mean(axis=0)
    sem = psds.std(axis=0, ddof=1) / np.sqrt(psds.shape[0]) if psds.shape[0] > 1 else np.zeros_like(mean)
    return freqs_ref, mean, sem, int(psds.shape[0])


def interp_to(freqs, y, target):
    if freqs.size == 0 or y.size == 0:
        return np.full_like(target, np.nan, dtype=float)
    return np.interp(target, freqs, y, left=np.nan, right=np.nan)


def bandpower_from_psd(freqs: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    if freqs.size == 0 or psd.size == 0:
        return np.nan
    m = (freqs >= f_lo) & (freqs < f_hi)
    if not np.any(m):
        return np.nan
    return float(np.trapz(psd[m], freqs[m]))


def compute_bandpower_table_channel(payloads: List[Dict], cfg: Cfg, channel: str) -> pd.DataFrame:
    rows = []
    for p in payloads:
        if "eeg_psd_channels" not in p or channel not in p["eeg_psd_channels"]:
            continue
        subj = p["name"]
        for state in ["phasic", "tonic", "transition"]:
            f = p["eeg_psd_channels"][channel][state]["freqs"]
            m = p["eeg_psd_channels"][channel][state]["mean"]
            for band_name, (lo, hi) in cfg.bandpower_bands.items():
                bp = bandpower_from_psd(f, m, lo, hi)
                rows.append({
                    "subject": subj,
                    "state": state,
                    "band": band_name,
                    "band_lo_hz": lo,
                    "band_hi_hz": hi,
                    "bandpower_uV2": bp,
                })
    return pd.DataFrame(rows)


# Plotting helpers

STATE_COLORS = {
    "phasic": "tab:red",
    "tonic": "tab:blue",
    "transition": "tab:gray",
}
STATE_PRETTY = {
    "phasic": "Phasic REM",
    "tonic": "Tonic REM",
    "transition": "Transition REM",
}


def _add_state_shading(ax, bouts, cfg: Cfg, tmin: float, tmax: float):
    for label in ["transition", "tonic", "phasic"]:  # draw in this order (phasic on top)
        alpha = cfg.transition_alpha if label == "transition" else cfg.tonic_alpha if label == "tonic" else cfg.phasic_alpha
        color = STATE_COLORS[label]
        for s, e in bouts[label]:
            if e > tmin and s < tmax:
                ax.axvspan(max(s, tmin), min(e, tmax), alpha=alpha, color=color, lw=0)


def _state_legend(cfg: Cfg):
    return [
        Patch(facecolor=STATE_COLORS["phasic"], alpha=cfg.phasic_alpha, label="Phasic REM"),
        Patch(facecolor=STATE_COLORS["tonic"], alpha=cfg.tonic_alpha, label="Tonic REM"),
        Patch(facecolor=STATE_COLORS["transition"], alpha=cfg.transition_alpha, label="Transition REM"),
    ]


def subject_excerpt(cfg: Cfg, t: np.ndarray, phasic_bouts: List[Tuple[float, float]]):
    if phasic_bouts:
        anchor = phasic_bouts[0][0]
        tmin = max(0.0, anchor - cfg.plot_pad_left_s)
        tmax = min(float(t[-1]), tmin + cfg.plot_span_s)
    else:
        tmin = 0.0
        tmax = min(float(t[-1]), cfg.plot_span_s)
    return tmin, tmax


def plot_subject_eog_preview(ax, payload: Dict, cfg: Cfg, show_legend: bool = False):
    """EOG preview ONLY (classification sanity check)."""
    t, x = payload["t"], payload["x_uV"]
    bouts = {
        "phasic": payload["phasic_bouts"],
        "tonic": payload["tonic_bouts"],
        "transition": payload["transition_bouts"],
    }
    name = payload["name"]

    tmin, tmax = subject_excerpt(cfg, t, bouts["phasic"])
    m = (t >= tmin) & (t <= tmax)
    tt = t[m][::cfg.plot_downsample]
    xx = x[m][::cfg.plot_downsample]

    ax.plot(tt, xx, lw=1)
    _add_state_shading(ax, bouts, cfg, tmin, tmax)

    ax.set_title(f"{name} — EOG preview (labels from EOG)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EOG (µV)")
    if show_legend:
        ax.legend(handles=_state_legend(cfg), loc="upper right", frameon=True)


def _plot_psd_curves(ax, psd_dict: Dict, cfg: Cfg, title: str, ylabel_prefix: str):
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"{ylabel_prefix} PSD (dB, µV²/Hz)" if cfg.psd_to_db else f"{ylabel_prefix} PSD (µV²/Hz)")
    ax.set_xlim(0, cfg.psd_fmax_hz)

    eps = 1e-20

    for label in ["phasic", "tonic", "transition"]:
        f = psd_dict[label]["freqs"]
        mean = psd_dict[label]["mean"]
        sem = psd_dict[label]["sem"]
        n = psd_dict[label]["n"]
        if f.size == 0:
            continue

        if cfg.psd_to_db:
            md = 10 * np.log10(mean + eps)
            lo = 10 * np.log10(np.maximum(mean - sem, 0) + eps)
            hi = 10 * np.log10(mean + sem + eps)
        else:
            md = mean
            lo = np.maximum(mean - sem, 0)
            hi = mean + sem

        ax.plot(f, md, label=f"{STATE_PRETTY[label]} (n={n})")
        ax.fill_between(f, lo, hi, alpha=0.18)

    ax.legend(loc="upper right")


def plot_subject_eeg_pzoz_psd(ax, payload: Dict, cfg: Cfg):
    """EEG PSD (Pz-Oz) for downstream analysis."""
    ch = cfg.main_theta_channel
    if "eeg_psd_channels" not in payload or ch not in payload["eeg_psd_channels"]:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"Missing EEG channel: {ch}", ha="center", va="center")
        return
    psd_dict = payload["eeg_psd_channels"][ch]
    _plot_psd_curves(
        ax,
        psd_dict=psd_dict,
        cfg=cfg,
        title=f"{payload['name']} — EEG PSD ({ch})",
        ylabel_prefix=f"EEG ({ch})"
    )


def plot_combined_psd_from_channel(ax, payloads: List[Dict], cfg: Cfg, channel: str):
    """Combined PSD across subjects for EEG[channel]."""
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

        curves = np.vstack(curves)
        mean = np.nanmean(curves, axis=0)
        n_eff = np.sum(np.isfinite(curves), axis=0)
        sd = np.nanstd(curves, axis=0, ddof=1)
        sem = np.where(n_eff > 0, sd / np.sqrt(n_eff), np.nan)

        if cfg.psd_to_db:
            md = 10 * np.log10(mean + eps)
            lo = 10 * np.log10(np.maximum(mean - sem, 0) + eps)
            hi = 10 * np.log10(mean + sem + eps)
        else:
            md = mean
            lo = np.maximum(mean - sem, 0)
            hi = mean + sem

        ax.plot(target, md, label=pretty)
        ax.fill_between(target, lo, hi, alpha=0.18)

    ax.legend(loc="upper right")


def plot_bandpower_bar_single_channel(ax, bp_df: pd.DataFrame, channel: str):
    states = ["phasic", "tonic", "transition"]
    bands = list(bp_df["band"].dropna().unique())

    stats = []
    for band in bands:
        for state in states:
            vals = bp_df[(bp_df["band"] == band) & (bp_df["state"] == state)]["bandpower_uV2"].astype(float).values
            vals = vals[np.isfinite(vals)]
            mean = float(np.mean(vals)) if vals.size else np.nan
            sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
            stats.append((band, state, mean, sem))
    stats_df = pd.DataFrame(stats, columns=["band", "state", "mean", "sem"])

    x = np.arange(len(bands))
    width = 0.25
    offsets = {"phasic": -width, "tonic": 0.0, "transition": width}

    for state in states:
        means = [stats_df[(stats_df.band == b) & (stats_df.state == state)]["mean"].values[0] for b in bands]
        sems = [stats_df[(stats_df.band == b) & (stats_df.state == state)]["sem"].values[0] for b in bands]
        ax.bar(x + offsets[state], means, width=width, label=STATE_PRETTY[state], yerr=sems, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(bands, rotation=0)
    ax.set_ylabel("Bandpower (µV²)")
    ax.set_title(f"EEG bandpower by REM state (mean ± SEM) [{channel}]")
    ax.legend(loc="upper right")


# Theta PSD plots (EEG)

def plot_theta_only_channel(ax, payloads: List[Dict], cfg: Cfg, channel: str):
    ax.set_title(f"Theta-range EEG PSD across subjects (mean ± SEM) [{channel}]")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB, µV²/Hz)" if cfg.psd_to_db else "PSD (µV²/Hz)")
    f_lo, f_hi = 3.0, 10.0
    ax.set_xlim(f_lo, f_hi)

    target = np.arange(f_lo, f_hi + 1e-9, cfg.combined_psd_df_hz)
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

        curves = np.vstack(curves)
        mean = np.nanmean(curves, axis=0)
        n_eff = np.sum(np.isfinite(curves), axis=0)
        sd = np.nanstd(curves, axis=0, ddof=1)
        sem = np.where(n_eff > 0, sd / np.sqrt(n_eff), np.nan)

        if cfg.psd_to_db:
            md = 10 * np.log10(mean + eps)
            lo = 10 * np.log10(np.maximum(mean - sem, 0) + eps)
            hi = 10 * np.log10(mean + sem + eps)
        else:
            md = mean
            lo = np.maximum(mean - sem, 0)
            hi = mean + sem

        ax.plot(target, md, label=pretty)
        ax.fill_between(target, lo, hi, alpha=0.18)

    ax.legend(loc="best")


def plot_theta_bandpower_bar(ax, bp_df: pd.DataFrame, title: str):
    states = ["phasic", "tonic", "transition"]
    pretty = {"phasic": "Phasic REM", "tonic": "Tonic REM", "transition": "Transition REM"}

    theta_rows = bp_df[bp_df["band"].astype(str).str.lower().str.contains("theta")].copy()
    if theta_rows.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No theta bandpower data found.", ha="center", va="center")
        return

    means, sems = [], []
    for s in states:
        vals = theta_rows[theta_rows["state"] == s]["bandpower_uV2"].astype(float).values
        vals = vals[np.isfinite(vals)]
        mu = float(np.mean(vals)) if vals.size else np.nan
        se = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        means.append(mu)
        sems.append(se)

    x = np.arange(len(states))
    ax.bar(x, means, yerr=sems, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[s] for s in states], rotation=0)
    ax.set_ylabel("Theta bandpower (µV²)")
    ax.set_title(title)


# EEG examples

def _pick_example_window(bouts: List[Tuple[float, float]], dur_s: float = 10.0) -> Tuple[float, float] | None:
    if not bouts:
        return None
    for s, e in bouts:
        if (e - s) >= dur_s:
            return (s, s + dur_s)
    s, e = bouts[0]
    return (s, min(e, s + dur_s))


def _slice(t: np.ndarray, x: np.ndarray, s: float, e: float) -> Tuple[np.ndarray, np.ndarray]:
    m = (t >= s) & (t <= e)
    if not np.any(m):
        return np.array([]), np.array([])
    return (t[m] - s), x[m]


def save_all_eeg_examples_grid(payloads: List[Dict], cfg: Cfg, out_path: Path, dur_s: float = 10.0):
    states = [
        ("phasic_bouts", "Phasic REM", "tab:red"),
        ("tonic_bouts", "Tonic REM", "tab:blue"),
        ("transition_bouts", "Transition REM", "tab:gray"),
    ]

    n_subj = len(payloads)
    n_cols = len(states)
    fig, axes = plt.subplots(n_subj, n_cols, figsize=(5.6 * n_cols, 2.8 * n_subj), sharex=False, sharey=False)

    if n_subj == 1:
        axes = np.array([axes])

    for r, p in enumerate(payloads):
        name = p["name"]
        t_eog, eog = p["t"], p["x_uV"]
        t_eeg, eeg = p["t_eeg"], p["eeg_uV"]

        for c, (key, pretty, color) in enumerate(states):
            ax = axes[r, c]
            bouts = p.get(key, [])
            win = _pick_example_window(bouts, dur_s=dur_s)
            if win is None:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"{name}\nNo {pretty}", ha="center", va="center")
                continue

            s, e = win
            tt_eeg, xx_eeg = _slice(t_eeg, eeg, s, e)
            tt_eog, xx_eog = _slice(t_eog, eog, s, e)

            ax.axvspan(0, e - s, alpha=0.08, color=color, lw=0)
            ax.plot(tt_eeg, xx_eeg, lw=0.9, label="EEG (µV)" if (r == 0 and c == 0) else None)
            ax.plot(tt_eog, xx_eog, lw=0.9, alpha=0.65, label="EOG (µV)" if (r == 0 and c == 0) else None)
            if r == 0:
                ax.set_title(pretty)
            if c == 0:
                ax.set_ylabel(f"{name}\nµV")
            ax.set_xlabel("Time (s)")
            ax.grid(False)

    fig.suptitle(f"EEG+EOG example windows — EEG={cfg.main_theta_channel} (downstream), EOG used for labeling", y=0.995)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# Analysis wrapper

def analyze_subject(edf_path: Path, cfg: Cfg):
    # EOG (detection)
    t, x_uV, sf = load_eog(edf_path, cfg, list(cfg.eog_channels))
    t_end = float(t[-1])

    # EEG (for analysis/examples)
    t_eeg, eeg_uV, sf_eeg = load_eeg(edf_path, cfg)

    hyp_path = hyp_path_for_psg(edf_path, cfg)
    if not hyp_path.exists():
        raise FileNotFoundError(f"Missing Hypnogram EDF+: {hyp_path}")

    rem = get_rem_intervals_from_hypnogram(hyp_path, t_end_s=t_end)
    if not rem:
        raise RuntimeError(f"No REM intervals found in hypnogram: {hyp_path.name}")

    total_rem_true = total_duration(rem)

    # EM events (EOG)
    ev_idx = detect_em_events(x_uV, sf, cfg)
    ev_t = ev_idx / sf

    # PHASIC
    ph = detect_phasic(ev_t, rem, cfg)

    # TONIC
    to_all = detect_tonic_maxabs(x_uV, sf, rem, cfg)

    # Ensure phasic/tonic disjoint
    to_all = [tw for tw in to_all if not any(overlap(tw, pw) for pw in ph)]

    # Buffer: tonic near phasic -> transition
    to_kept, to_relabel = apply_buffer_rule(to_all, ph, cfg.buffer_s)

    # TRANSITION
    transition = transition_between_phasic_and_tonic(ph, to_kept, rem, cfg.window_s)
    transition = sorted(transition + to_relabel)

    # Merge bouts if separated by < merge_gap_s
    ph_b = merge_intervals(ph, cfg.merge_gap_s)
    to_b = merge_intervals(to_kept, cfg.merge_gap_s)
    tr_b = merge_intervals(transition, cfg.merge_gap_s)

    # Per-REM-period bout counts
    rem_period_df = rem_period_counts_table(rem, ph_b, to_b, tr_b)

    total_ph = total_duration(ph_b)
    total_to = total_duration(to_b)
    total_tr = total_duration(tr_b)

    pct_ph = 100.0 * total_ph / total_rem_true if total_rem_true > 0 else np.nan
    pct_to = 100.0 * total_to / total_rem_true if total_rem_true > 0 else np.nan
    pct_tr = 100.0 * total_tr / total_rem_true if total_rem_true > 0 else np.nan

    n_rem_periods = len(rem)
    n_phasic_bouts = len(ph_b)
    n_tonic_bouts = len(to_b)
    n_transition_bouts = len(tr_b)

    # Bout-level CSV rows
    rows = []
    for label, bouts in [("phasic", ph_b), ("tonic", to_b), ("transition", tr_b)]:
        for s, e in bouts:
            rp = rem_period_index(s, rem)
            rem_start, rem_end = (rem[rp] if rp >= 0 else (np.nan, np.nan))
            mx, nev = win_features(x_uV, sf, s, e, ev_t)
            rows.append({
                "start_s": round3(s),
                "end_s": round3(e),
                "duration_s": round3(e - s),
                "label": label,
                "win_max_abs_uV": round3(mx),
                "n_em_events": int(nev),

                "rem_period_idx": int(rp) if rp >= 0 else np.nan,
                "rem_period_start_s": round3(rem_start) if rp >= 0 else np.nan,
                "rem_period_end_s": round3(rem_end) if rp >= 0 else np.nan,

                "total_rem_s": round3(total_rem_true),
                "total_phasic_s": round3(total_ph),
                "total_tonic_s": round3(total_to),
                "total_transition_s": round3(total_tr),
                "pct_phasic_of_rem": round3(pct_ph),
                "pct_tonic_of_rem": round3(pct_to),
                "pct_transition_of_rem": round3(pct_tr),
            })

    df = pd.DataFrame(rows).sort_values(["start_s", "end_s"]).reset_index(drop=True)
    globals_cols = [
        "total_rem_s", "total_phasic_s", "total_tonic_s", "total_transition_s",
        "pct_phasic_of_rem", "pct_tonic_of_rem", "pct_transition_of_rem",
    ]
    if len(df) > 1:
        df.loc[1:, globals_cols] = np.nan

    summary = {
        "file_name": edf_path.name,
        "total_rem_s": round3(total_rem_true),
        "total_phasic_s": round3(total_ph),
        "total_tonic_s": round3(total_to),
        "total_transition_s": round3(total_tr),
        "pct_phasic_of_rem": round3(pct_ph),
        "pct_tonic_of_rem": round3(pct_to),
        "pct_transition_of_rem": round3(pct_tr),
        "n_rem_periods": int(n_rem_periods),
        "n_phasic_bouts": int(n_phasic_bouts),
        "n_tonic_bouts": int(n_tonic_bouts),
        "n_transition_bouts": int(n_transition_bouts),
        "mean_phasic_bouts_per_rem": round3(n_phasic_bouts / n_rem_periods) if n_rem_periods else np.nan,
        "mean_tonic_bouts_per_rem": round3(n_tonic_bouts / n_rem_periods) if n_rem_periods else np.nan,
        "mean_transition_bouts_per_rem": round3(n_transition_bouts / n_rem_periods) if n_rem_periods else np.nan,
    }

    # PSD epochs; same epochs used everywhere downstream
    ph_ep = chunk(ph_b, cfg.psd_epoch_s, cfg.psd_epoch_overlap)
    to_ep = chunk(to_b, cfg.psd_epoch_s, cfg.psd_epoch_overlap)
    tr_ep = chunk(tr_b, cfg.psd_epoch_s, cfg.psd_epoch_overlap)

    # EEG PSD per channel (EXACT same epochs)
    eeg_psd_channels = {}
    for ch_name in ("Fpz-Cz", "Pz-Oz"):
        try:
            _tch, _xch_uV, _sfch = load_eeg_channel(edf_path, cfg, ch_name)
        except Exception:
            continue

        f_ph_c, m_ph_c, se_ph_c, n_ph_c = welch_mean_sem(_xch_uV, _sfch, ph_ep, cfg)
        f_to_c, m_to_c, se_to_c, n_to_c = welch_mean_sem(_xch_uV, _sfch, to_ep, cfg)
        f_tr_c, m_tr_c, se_tr_c, n_tr_c = welch_mean_sem(_xch_uV, _sfch, tr_ep, cfg)

        eeg_psd_channels[ch_name] = {
            "phasic": {"freqs": f_ph_c, "mean": m_ph_c, "sem": se_ph_c, "n": n_ph_c},
            "tonic": {"freqs": f_to_c, "mean": m_to_c, "sem": se_to_c, "n": n_to_c},
            "transition": {"freqs": f_tr_c, "mean": m_tr_c, "sem": se_tr_c, "n": n_tr_c},
        }

    plot_payload = {
        "name": edf_path.stem,
        "t": t, "x_uV": x_uV,
        "t_eeg": t_eeg, "eeg_uV": eeg_uV,
        "phasic_bouts": ph_b, "tonic_bouts": to_b, "transition_bouts": tr_b,
        "eeg_psd_channels": eeg_psd_channels,
    }

    return df, summary, plot_payload, rem_period_df


# File pairing (SC: PSG ↔ Hypnogram)

_SC_PSG_RE = re.compile(r"^(?P<prefix>SC\d+[\w]+)-PSG\.edf$", re.IGNORECASE)
_SC_HYP_RE = re.compile(r"^(?P<prefix>SC\d+[\w]+)-Hypnogram\.edf$", re.IGNORECASE)

def _sc_core_key(prefix: str) -> str:
    prefix = prefix.strip()
    return prefix[:-1] if len(prefix) >= 2 else prefix

def build_sc_pairs(folder: Path) -> Tuple[List[str], Dict[str, str]]:
    psg: Dict[str, Path] = {}
    hyp: Dict[str, Path] = {}

    for f in sorted(folder.rglob("*.edf")):
        name = f.name

        m = _SC_PSG_RE.match(name)
        if m and name.upper().startswith("SC"):
            k = _sc_core_key(m.group("prefix")).upper()
            psg[k] = f
            continue

        m = _SC_HYP_RE.match(name)
        if m and name.upper().startswith("SC"):
            k = _sc_core_key(m.group("prefix")).upper()
            hyp[k] = f
            continue

    matched = sorted(set(psg.keys()) & set(hyp.keys()))
    edf_files = [psg[k].name for k in matched]
    hypnogram_map = {psg[k].name: hyp[k].name for k in matched}

    if not edf_files:
        raise RuntimeError(
            f"No SC PSG/Hypnogram pairs found in: {folder}\n"
            f"Check the folder path and naming patterns."
        )

    missing_hyp = sorted(set(psg.keys()) - set(hyp.keys()))
    missing_psg = sorted(set(hyp.keys()) - set(psg.keys()))
    if missing_hyp or missing_psg:
        print("\nWARNING: Incomplete SC pairing in folder:", folder)
        print("  PSG without Hypnogram:", len(missing_hyp))
        print("  Hypnogram without PSG:", len(missing_psg))

    return edf_files, hypnogram_map


# Main

def main():
    folder = Path("/Users/hanna/Documents/UCD/classes/semester 2/Internship/dataset/sleep-cassette")
    edf_files, hypnogram_map = build_sc_pairs(folder)

    cfg = Cfg(
        folder=folder,
        edf_files=edf_files,
        hypnogram_map=hypnogram_map,
        eog_channels=("EOG horizontal",),

        tonic_max_abs_uV=50.0,
        buffer_s=0.0,
        merge_gap_s=2.0,

        # downstream theta EEG computed from Pz-Oz
        eeg_channels=("Pz-Oz",),
        eeg_reference="mean",

        filter_lfreq_hz=0.1,
        filter_hfreq_hz=60.0,
        main_theta_channel="Pz-Oz",
    )

    # results folder
    results_dir = cfg.folder.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    edf_paths = sorted([cfg.folder / f for f in cfg.edf_files], key=lambda p: p.name)

    if MAX_RECORDINGS is not None:
        edf_paths = edf_paths[:MAX_RECORDINGS]
    print("\nRunning only first {} recordings (DEBUG MODE)\n".format(len(edf_paths)))

    for p in edf_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing PSG EDF: {p}")
        hyp = hyp_path_for_psg(p, cfg)
        if not hyp.exists():
            raise FileNotFoundError(f"Missing Hypnogram EDF+: {hyp}")

    summaries: List[Dict] = []
    payloads: List[Dict] = []
    rem_period_tables: List[pd.DataFrame] = []
    failed: List[Dict] = []

    for edf in edf_paths:
        try:
            df, summary, payload, rem_period_df = analyze_subject(edf, cfg)
        except Exception as e:
            print(f"SKIP (error): {edf.name} -> {type(e).__name__}: {e}")
            failed.append({"file": edf.name, "error_type": type(e).__name__, "error": str(e)})
            continue

        df = df.copy()
        df["theta_eeg_channel"] = cfg.main_theta_channel
        df["theta_eeg_note"] = f"Theta (4–8 Hz) computed from EEG {cfg.main_theta_channel}; labels from EOG."
        df["preproc_note"] = f"Bandpass {cfg.filter_lfreq_hz}–{cfg.filter_hfreq_hz} Hz; Welch detrend=linear."

        out_csv = results_dir / f"{edf.stem}{cfg.per_csv_suffix}"
        df.to_csv(out_csv, index=False)

        summaries.append(summary)
        payloads.append(payload)

        rem_period_df = rem_period_df.copy()
        rem_period_df.insert(0, "subject", edf.stem)
        rem_period_df["theta_eeg_channel"] = cfg.main_theta_channel
        rem_period_df["preproc_note"] = f"Bandpass {cfg.filter_lfreq_hz}–{cfg.filter_hfreq_hz} Hz; Welch detrend=linear."
        rem_period_tables.append(rem_period_df)

        print(f"Done: {edf.name} -> {out_csv.name}")

    if failed:
        failed_csv = results_dir / "failed_files.csv"
        pd.DataFrame(failed).to_csv(failed_csv, index=False)
        print(f"\nSaved failure log: {failed_csv}")

    # Summary CSV
    df_sum = pd.DataFrame(summaries)
    if not df_sum.empty:
        df_sum["theta_eeg_channel"] = cfg.main_theta_channel
        df_sum["theta_eeg_note"] = f"Theta computed from EEG {cfg.main_theta_channel}; labels from EOG."
        df_sum["preproc_note"] = f"Bandpass {cfg.filter_lfreq_hz}–{cfg.filter_hfreq_hz} Hz; Welch detrend=linear."
    out_sum = results_dir / cfg.summary_csv
    df_sum.to_csv(out_sum, index=False)
    print("Saved summary CSV:", out_sum.resolve())

    # REM period table
    rem_all = pd.concat(rem_period_tables, ignore_index=True) if rem_period_tables else pd.DataFrame()
    out_rem_all = results_dir / cfg.rem_period_counts_all_csv
    rem_all.to_csv(out_rem_all, index=False)
    print("Saved REM period counts (all subjects):", out_rem_all.resolve())

    if not payloads:
        print("\nNo payloads to plot (all files failed).")
        return


    # FIG 1: EEG Pz-Oz PSD per subject + combined (this answers your supervisor)
    # Left column = same EOG preview for context; Right column = EEG PSD (Pz-Oz)
    n = len(payloads)
    fig1 = plt.figure(figsize=(16, 3.0 * n + 4.5))
    gs2 = fig1.add_gridspec(n + 1, 2, height_ratios=[1.0] * n + [1.1], hspace=0.6, wspace=0.28)

    for i, p in enumerate(payloads):
        ax_prev = fig1.add_subplot(gs2[i, 0])
        ax_psd = fig1.add_subplot(gs2[i, 1])
        plot_subject_eog_preview(ax_prev, p, cfg, show_legend=(i == 0))
        plot_subject_eeg_pzoz_psd(ax_psd, p, cfg)

    ax_comb = fig1.add_subplot(gs2[n, :])
    plot_combined_psd_from_channel(ax_comb, payloads, cfg, channel=cfg.main_theta_channel)

    fig1.suptitle(
        f"FIG 1 — Downstream EEG analysis using EOG-labeled epochs\n"
        f"EEG PSD shown for {cfg.main_theta_channel}; labels from EOG. Preproc {cfg.filter_lfreq_hz}–{cfg.filter_hfreq_hz} Hz; Welch detrend=linear.",
        y=0.995
    )
    fig1.tight_layout()
    out1 = results_dir / cfg.fig_eeg_pzoz_psd_png
    fig1.savefig(out1, dpi=200)
    plt.close(fig1)
    print("Saved:", out1.resolve())

    # FIG 2: EEG Pz-Oz bandpower (delta/theta/alpha/beta/gamma) — clean and explicit
    bp_df_eeg = compute_bandpower_table_channel(payloads, cfg, cfg.main_theta_channel)
    fig2, ax3 = plt.subplots(1, 1, figsize=(16, 5.2))
    plot_bandpower_bar_single_channel(ax3, bp_df_eeg, channel=cfg.main_theta_channel)
    fig2.suptitle(
        f"FIG 2 — EEG bandpower across REM microstates [{cfg.main_theta_channel}]\n"
        "Labels from EOG (classification); bandpower from EEG (downstream).",
        y=0.995
    )
    fig2.tight_layout()
    out2 = results_dir / cfg.fig_eeg_pzoz_bandpower_png
    fig2.savefig(out2, dpi=200)
    plt.close(fig2)
    print("Saved:", out2.resolve())

    # FIG 3: Theta comparison (Fpz-Cz vs Pz-Oz) — same epochs/states
    bp_df_eeg_fpz = compute_bandpower_table_channel(payloads, cfg, "Fpz-Cz")
    bp_df_eeg_pz2 = compute_bandpower_table_channel(payloads, cfg, "Pz-Oz")

    fig3 = plt.figure(figsize=(16, 10))
    gs4 = fig3.add_gridspec(2, 2, wspace=0.25, hspace=0.35)

    ax_psd_f = fig3.add_subplot(gs4[0, 0])
    plot_theta_only_channel(ax_psd_f, payloads, cfg, "Fpz-Cz")

    ax_psd_p = fig3.add_subplot(gs4[0, 1])
    plot_theta_only_channel(ax_psd_p, payloads, cfg, "Pz-Oz")

    ax_bar_f = fig3.add_subplot(gs4[1, 0])
    plot_theta_bandpower_bar(ax_bar_f, bp_df_eeg_fpz, "Theta bandpower by REM state (mean ± SEM) [Fpz-Cz]")

    ax_bar_p = fig3.add_subplot(gs4[1, 1])
    plot_theta_bandpower_bar(ax_bar_p, bp_df_eeg_pz2, "Theta bandpower by REM state (mean ± SEM) [Pz-Oz]")

    fig3.suptitle(
        "FIG 3 — Theta comparison using the same EOG-labeled epochs/states\n"
        f"Preproc: {cfg.filter_lfreq_hz}–{cfg.filter_hfreq_hz} Hz; Welch detrend=linear.",
        y=0.995
    )
    fig3.tight_layout()
    out3 = results_dir / cfg.theta_combined_png
    fig3.savefig(out3, dpi=200)
    plt.close(fig3)
    print("Saved:", out3.resolve())

    # FIG 4: EEG examples grid
    out4 = results_dir / cfg.eeg_examples_all_png
    save_all_eeg_examples_grid(payloads, cfg, out4, dur_s=10.0)
    print("Saved:", out4.resolve())


if __name__ == "__main__":
    main()