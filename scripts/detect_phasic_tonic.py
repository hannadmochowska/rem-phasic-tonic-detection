"""
PHASIC / TONIC REM DETECTION (threshold method) + CSV + SHORT PREVIEW PLOT

RULES:
1) Window length: 4 seconds. Windows are onset-aligned
   - PHASIC windows start at the time of a triggering eye-movement event: [t_event, t_event+4]
   - TONIC windows start at the onset of a quiet EOG run: [t_run_start, t_run_start+4] (tiled)
   - UNCLASSIFIED windows fill remaining REM time, tiled from gap starts.

2) Eye-movement (EM) event definition from EOG:
   - peak |EOG| >= 100 µV
   - event duration above threshold < 500 ms
   - enforce min separation between events: 250 ms

3) PHASIC window:
   - contains >= 2 EM events within the 4 s window
   - edge guard:
       (NOT all events occur in first 2 s) OR (events occur in >= 2 separate 1 s bins)

4) TONIC window:
   - max |EOG| < 25 µV across the entire 4 s window

5) Otherwise: UNCLASSIFIED

6) Buffer rule:
   - TONIC windows within ±8 seconds of any PHASIC window are re-labeled as UNCLASSIFIED

OUTPUTS (saved next to the EDF):
- CSV: <edf_stem>.rem_states_onset_aligned.csv
  columns: start_s, end_s, duration_s, label, win_max_abs_uV, n_em_events
- PNG preview plot (short): <edf_stem>.phasic_tonic_preview.png
  shows a ~70 s excerpt with both phasic + tonic shaded.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

# Config

@dataclass
class DetectorConfig:
    edf_path: Path
    eog_channels: List[str]

    # Windows
    window_s: float = 4.0

    # EM event thresholds (µV)
    phasic_peak_threshold_uV: float = 100.0
    max_event_duration_s: float = 0.5
    min_event_separation_s: float = 0.25

    # Edge guard
    edge_guard_first_s: float = 2.0
    bin_s: float = 1.0

    # Tonic threshold (µV)
    tonic_max_abs_uV: float = 25.0

    # Buffer around phasic
    buffer_s: float = 8.0

    # Signal handling
    use_abs: bool = True
    scale_to_uV: bool = True

    # Output
    out_suffix: str = ".rem_states_onset_aligned.csv"
    plot_png_suffix: str = ".phasic_tonic_preview.png"

    # Plot
    plot_span_s: float = 70.0
    plot_pad_left_s: float = 20.0
    plot_downsample: int = 5


# EDF loading + signal

def load_raw(edf_path: Path) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw.rename_channels(lambda ch: ch.strip())
    return raw


def get_signal(
    raw: mne.io.BaseRaw,
    channels: List[str],
    scale_to_uV: bool = True
) -> Tuple[np.ndarray, np.ndarray, float]:
    picks = mne.pick_channels(raw.ch_names, include=channels)
    if len(picks) == 0:
        raise ValueError(f"Channels not found: {channels}\nAvailable: {raw.ch_names}")

    x = raw.get_data(picks=picks).mean(axis=0)

    # V -> µV
    if scale_to_uV:
        x = x * 1e6

    sfreq = float(raw.info["sfreq"])
    t = np.arange(x.size) / sfreq
    return t, x, sfreq

# Helpers

def contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return (start_idx, end_idx_exclusive) for contiguous True runs."""
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


def enforce_min_separation(idxs: np.ndarray, sfreq: float, min_sep_s: float) -> np.ndarray:
    """Keep only events separated by at least min_sep_s."""
    if idxs.size == 0:
        return idxs
    min_sep = int(round(min_sep_s * sfreq))
    keep = [int(idxs[0])]
    for i in idxs[1:]:
        if int(i) - keep[-1] >= min_sep:
            keep.append(int(i))
    return np.asarray(keep, dtype=int)


def edge_guard_ok(event_times_rel_s: np.ndarray, first_s: float, bin_s: float) -> bool:
    if event_times_rel_s.size == 0:
        return False
    all_in_first = np.all(event_times_rel_s < first_s)
    bins = np.floor(event_times_rel_s / bin_s).astype(int)
    in_two_bins = (np.unique(bins).size >= 2)
    return (not all_in_first) or in_two_bins


def interval_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] < b[1] and a[1] > b[0]


def subtract_intervals(base: List[Tuple[float, float]], cuts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not base:
        return []
    if not cuts:
        return base[:]

    cuts_sorted = sorted(cuts)
    out: List[Tuple[float, float]] = []

    for bs, be in base:
        cur = bs
        for cs, ce in cuts_sorted:
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

    return [(s, e) for s, e in out if e - s > 0]


def tile_from_onset(intervals: List[Tuple[float, float]], window_s: float) -> List[Tuple[float, float]]:
    windows: List[Tuple[float, float]] = []
    for s, e in intervals:
        cur = s
        while cur + window_s <= e:
            windows.append((cur, cur + window_s))
            cur += window_s
    return windows


# Step 1: EM event detection (peak>=100µV, duration<500ms)

def detect_em_events(x_uV: np.ndarray, sfreq: float, cfg: DetectorConfig) -> np.ndarray:
    x_use = np.abs(x_uV) if cfg.use_abs else x_uV
    above = x_use >= cfg.phasic_peak_threshold_uV

    runs = contiguous_true_runs(above)
    max_len = int(round(cfg.max_event_duration_s * sfreq))
    peaks: List[int] = []
    for s, e in runs:
        if (e - s) < max_len:
            seg = x_use[s:e]
            peaks.append(s + int(np.argmax(seg)))

    if not peaks:
        return np.array([], dtype=int)

    peaks = np.asarray(sorted(peaks), dtype=int)
    return enforce_min_separation(peaks, sfreq, cfg.min_event_separation_s)


# Step 2: PHASIC windows

def detect_phasic_windows_event_anchored(
    event_times_s: np.ndarray,
    rem_intervals_s: List[Tuple[float, float]],
    cfg: DetectorConfig
) -> List[Tuple[float, float]]:
    if event_times_s.size < 2:
        return []

    phasic: List[Tuple[float, float]] = []
    i = 0
    while i < event_times_s.size - 1:
        start = float(event_times_s[i])
        end = start + cfg.window_s

        if not any((start >= a) and (end <= b) for a, b in rem_intervals_s):
            i += 1
            continue

        ev = event_times_s[(event_times_s >= start) & (event_times_s < end)]
        if ev.size >= 2 and edge_guard_ok(ev - start, cfg.edge_guard_first_s, cfg.bin_s):
            phasic.append((start, end))
            i = int(np.searchsorted(event_times_s, end, side="left"))
        else:
            i += 1

    return phasic


# Step 3: TONIC windows

def detect_tonic_windows_run_anchored(
    t: np.ndarray,
    x_uV: np.ndarray,
    sfreq: float,
    rem_intervals_s: List[Tuple[float, float]],
    cfg: DetectorConfig
) -> List[Tuple[float, float]]:
    x_use = np.abs(x_uV) if cfg.use_abs else x_uV

    tonic_intervals: List[Tuple[float, float]] = []
    for a, b in rem_intervals_s:
        i0 = int(round(a * sfreq))
        i1 = int(round(b * sfreq))
        if i1 <= i0:
            continue

        quiet = x_use[i0:i1] < cfg.tonic_max_abs_uV
        for rs, re in contiguous_true_runs(quiet):
            s = float(t[i0 + rs])
            e = float(t[i0 + re - 1]) + (1.0 / sfreq)
            if e - s >= cfg.window_s:
                tonic_intervals.append((s, e))

    return tile_from_onset(tonic_intervals, cfg.window_s)


# Step 4: Buffer rule

def apply_buffer_to_tonic(
    tonic_windows: List[Tuple[float, float]],
    phasic_windows: List[Tuple[float, float]],
    buffer_s: float
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    if not tonic_windows or not phasic_windows:
        return tonic_windows, []

    expanded = [(ps - buffer_s, pe + buffer_s) for ps, pe in phasic_windows]

    kept, dropped = [], []
    for tw in tonic_windows:
        contaminated = any(interval_overlap(tw, ex) for ex in expanded)
        (dropped if contaminated else kept).append(tw)
    return kept, dropped


# Step 5: UNCLASSIFIED windows

def build_unclassified_windows(
    rem_intervals_s: List[Tuple[float, float]],
    phasic_windows: List[Tuple[float, float]],
    tonic_windows: List[Tuple[float, float]],
    window_s: float
) -> List[Tuple[float, float]]:
    occupied = sorted(phasic_windows + tonic_windows)
    remaining = subtract_intervals(rem_intervals_s, occupied)
    return tile_from_onset(remaining, window_s)


# Window features

def window_features(
    x_uV: np.ndarray,
    sfreq: float,
    win: Tuple[float, float],
    event_times_s: np.ndarray,
    cfg: DetectorConfig
) -> Tuple[float, int]:
    s, e = win
    i0 = int(round(s * sfreq))
    i1 = int(round(e * sfreq))
    x_use = np.abs(x_uV) if cfg.use_abs else x_uV
    win_max = float(np.max(x_use[i0:i1])) if i1 > i0 else float("nan")
    n_events = int(np.sum((event_times_s >= s) & (event_times_s < e)))
    return win_max, n_events


# Preview plot

def plot_detection_preview(
    cfg: DetectorConfig,
    t: np.ndarray,
    x_uV: np.ndarray,
    phasic: List[Tuple[float, float]],
    tonic: List[Tuple[float, float]],
    out_png: Path
) -> None:
    if phasic:
        anchor = phasic[0][0]
        tmin = max(0.0, anchor - cfg.plot_pad_left_s)
        tmax = min(float(t[-1]), tmin + cfg.plot_span_s)
    else:
        tmin = 0.0
        tmax = min(float(t[-1]), cfg.plot_span_s)

    m = (t >= tmin) & (t <= tmax)
    tt = t[m][::cfg.plot_downsample]
    xx = x_uV[m][::cfg.plot_downsample]

    plt.figure(figsize=(12, 3.6))
    plt.plot(tt, xx, linewidth=1, label="EOG")
    plt.xlabel("Time (s)")
    plt.ylabel("EOG (µV)")
    plt.title(f"Detect phasic–tonic - {cfg.edf_path.name}")

    def shade(intervals, label, alpha):
        first = True
        for s, e in intervals:
            if e <= tmin or s >= tmax:
                continue
            plt.axvspan(max(s, tmin), min(e, tmax), alpha=alpha,
                        label=label if first else None)
            first = False

    shade(phasic, "phasic", 0.25)
    shade(tonic, "tonic", 0.20)

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# Main

def main():
    cfg = DetectorConfig(
        edf_path=Path("/Users/hanna/Documents/UCD/classes/semester 2/Internship/dataset/sleep-cassette/SC4001E0-PSG.edf"),
        eog_channels=["EOG horizontal"],
        plot_span_s=70.0,
        plot_pad_left_s=20.0,
        plot_downsample=5,
    )

    raw = load_raw(cfg.edf_path)
    t, x_uV, sfreq = get_signal(raw, cfg.eog_channels, scale_to_uV=cfg.scale_to_uV)
    rem_intervals_s: List[Tuple[float, float]] = [(0.0, float(t[-1]))]

    # 1) EM events (peak>=100µV, duration<500ms, min sep)
    event_peaks_idx = detect_em_events(x_uV, sfreq, cfg)
    event_times_s = event_peaks_idx / sfreq

    # 2) PHASIC windows: [event, event+4] with >=2 events + edge guard
    phasic_windows = detect_phasic_windows_event_anchored(event_times_s, rem_intervals_s, cfg)

    # 3) TONIC windows: tiled from runs where |EOG|<25µV
    tonic_windows_all = detect_tonic_windows_run_anchored(t, x_uV, sfreq, rem_intervals_s, cfg)

    # 4) Buffer: tonic near phasic -> unclassified
    tonic_windows, tonic_dropped = apply_buffer_to_tonic(tonic_windows_all, phasic_windows, cfg.buffer_s)

    # 5) UNCLASSIFIED: remaining REM after (phasic + tonic), tiled + buffer-dropped tonic
    unclassified_windows = build_unclassified_windows(rem_intervals_s, phasic_windows, tonic_windows, cfg.window_s)
    unclassified_windows = sorted(unclassified_windows + tonic_dropped)

    # 6) Build one dataframe
    rows = []
    for label, wins in [("phasic", phasic_windows), ("tonic", tonic_windows), ("unclassified", unclassified_windows)]:
        for s, e in wins:
            win_max, n_ev = window_features(x_uV, sfreq, (s, e), event_times_s, cfg)
            rows.append({
                "start_s": float(s),
                "end_s": float(e),
                "duration_s": float(e - s),
                "label": label,
                "win_max_abs_uV": win_max,
                "n_em_events": n_ev,
            })

    df = pd.DataFrame(rows).sort_values(["start_s", "end_s"]).reset_index(drop=True)

    # CSV
    out_csv = cfg.edf_path.parent / f"{cfg.edf_path.stem}{cfg.out_suffix}"
    df.to_csv(out_csv, index=False)

    # Plot
    out_png = cfg.edf_path.parent / f"{cfg.edf_path.stem}{cfg.plot_png_suffix}"
    plot_detection_preview(cfg, t, x_uV, phasic_windows, tonic_windows, out_png)

    print("Saved CSV:", out_csv.resolve())
    print("Saved plot:", out_png.resolve())
    print("Label counts:\n", df["label"].value_counts())


if __name__ == "__main__":
    main()x