"""
Unsupervised Phasic/Tonic REM Classification — Extracranial EEG
================================================================
Dataset  : Sleep-EDF Cassette + Telemetry
Channels : EOG horizontal, EEG Fpz-Cz, EEG Pz-Oz
Model    : Per-subject Hidden Markov Model (K=2 primary, K=3 exploratory)
           Gaussian emissions, k-means initialisation on EOG envelope

Segmentation: event-based (not fixed windows)
  - Phasic events  : EOG bursts detected via Hilbert envelope > 90th percentile
                     of REM envelope (MIN_BURST_DUR = 0.5 s, merged if < 0.5 s apart)
  - Tonic segments : burst-free REM intervals >= MIN_TONIC_GAP_S (5 s)
  This threshold pre-processing defines candidate events; the HMM classifier
  is applied subsequently on EEG feature vectors (EOG not used as HMM input).

Features per event/segment (3 scalars, per-subject z-scored within REM):
  1. Sawtooth power — 2–6 Hz bandpower, EEG Fpz-Cz
  2. Spindle power  — 12–15 Hz bandpower, EEG Pz-Oz  (lower = spindle suppression)
  3. Theta power    — 4–8 Hz bandpower, EEG Pz-Oz

  EOG envelope is computed per event and used post-hoc to label HMM states
  (highest mean EOG = phasic) but is NOT included in the HMM feature matrix.

Validation:
  - Rule-based labels (burst=phasic, tonic_seg=tonic) used as baseline for ARI/NMI
  - Feature HMM compared against threshold-based segmentation labels
  - Feature profile per HMM state (mean ± SD)
  - PCA scatter coloured by HMM state and rule-based label

Reference: Quinn et al. (2019) Brain Topography 32:1020–1034.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
import pyedflib
from scipy.signal import butter, filtfilt, hilbert
from scipy.integrate import simpson
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")


# ── Configuration ─────────────────────────────────────────────────────────────

# Paths
SC_DIR = Path(__file__).parent / "dataset" / "sleep-cassette"
ST_DIR = Path(__file__).parent / "dataset" / "sleep-telemetry"
OUT_DIR = Path(__file__).parent / "dataset" / "results" / "hmm_unsupervised"

# Preprocessing
FILTER_LO = 0.1
FILTER_HI = 60.0

# EOG burst detection
BURST_BAND     = (0.3, 35.0)   # Hz — broadband EOG for envelope
BURST_PCT      = 90.0          # percentile of REM EOG envelope → phasic threshold
MIN_BURST_DUR  = 0.5           # s  — minimum burst duration
MERGE_BURST_GAP = 0.5          # s  — merge bursts closer than this

# Tonic segment definition
MIN_TONIC_GAP_S = 5.0          # s  — burst-free gap required to define tonic

# Feature bands
SAWTOOTH_BAND  = (2.0,  6.0)   # Hz — Fpz-Cz (phasic-specific sawtooth waves)
SPINDLE_BAND   = (12.0, 15.0)  # Hz — Pz-Oz   (sigma / sleep spindles)
THETA_BAND     = (4.0,  8.0)   # Hz — Pz-Oz
DELTA_BAND     = (1.0,  4.0)   # Hz — Pz-Oz

# HMM
N_STATES_PRIMARY     = 2
N_STATES_EXPLORATORY = 3
N_ITER               = 200
N_INIT               = 10      # HMM restarts — keep best log-likelihood

# Recordings to skip (e.g. known bad)
SKIP_RECORDINGS: List[str] = []

# Limit for debugging (set to None to run all)
MAX_RECORDINGS: Optional[int] = None


# ── Helpers: signal processing ────────────────────────────────────────────────

def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    lo_ = max(lo, 0.01)
    hi_ = min(hi, nyq - 0.5)
    if lo_ >= hi_:
        return x.copy()
    b, a = butter(order, [lo_ / nyq, hi_ / nyq], btype="band")
    return filtfilt(b, a, x)


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(x))


def bandpower(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """Mean power spectral density in band via Welch, integrated with Simpson's rule."""
    from scipy.signal import welch
    nperseg = min(int(fs * 2), len(x))
    if nperseg < 4:
        return np.nan
    f, psd = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="linear")
    idx = (f >= lo) & (f <= hi)
    if not np.any(idx):
        return np.nan
    return float(simpson(psd[idx], x=f[idx]))


def extract_segment(x: np.ndarray, fs: float, t0: float, t1: float) -> np.ndarray:
    """Slice signal between t0 and t1 seconds."""
    i0 = max(0, int(round(t0 * fs)))
    i1 = min(len(x), int(round(t1 * fs)))
    return x[i0:i1]


# ── Helpers: interval utilities ───────────────────────────────────────────────

def merge_intervals(ivs: List[Tuple[float, float]], gap: float = 0.0) -> List[Tuple[float, float]]:
    if not ivs:
        return []
    ivs = sorted(ivs)
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def subtract_intervals(
    base: List[Tuple[float, float]],
    cuts: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Remove cut intervals from base intervals."""
    out = []
    for bs, be in base:
        cur = bs
        for cs, ce in sorted(cuts):
            if ce <= cur:
                continue
            if cs >= be:
                break
            if cs > cur:
                out.append((cur, min(cs, be)))
            cur = max(cur, ce)
        if cur < be:
            out.append((cur, be))
    return [(s, e) for s, e in out if e - s > 0]


# ── Data loading ──────────────────────────────────────────────────────────────

def _norm_stage(desc) -> str:
    if isinstance(desc, bytes):
        desc = desc.decode("utf-8", errors="replace")
    return " ".join(str(desc).strip().split()).lower()


def load_rem_intervals(hyp_path: Path, t_end_s: float) -> List[Tuple[float, float]]:
    f = pyedflib.EdfReader(str(hyp_path))
    try:
        onsets, durations, descriptions = f.readAnnotations()
    finally:
        f.close()
    ints = []
    for onset, dur, desc in zip(onsets, durations, descriptions):
        d = _norm_stage(desc)
        if d in ("sleep stage r", "r"):
            s = float(max(0.0, onset))
            e = float(min(t_end_s, onset + float(dur)))
            if e > s:
                ints.append((s, e))
    return merge_intervals(ints)


def load_raw_channels(
    edf_path: Path,
    channels: List[str],
) -> Tuple[Dict[str, np.ndarray], float]:
    """Load, filter, and return requested channels as µV arrays."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw.rename_channels(lambda c: c.strip())

    # Bandpass
    nyq = raw.info["sfreq"] / 2.0
    h = min(FILTER_HI, nyq - 0.5)
    raw.filter(l_freq=FILTER_LO, h_freq=h, fir_design="firwin", verbose="ERROR")

    fs = float(raw.info["sfreq"])
    data = {}
    for ch in channels:
        # Robust channel matching
        match = None
        for c in raw.ch_names:
            if c.strip().lower() == ch.strip().lower():
                match = c
                break
        if match is None:
            for c in raw.ch_names:
                if ch.lower() in c.lower():
                    match = c
                    break
        if match is None:
            raise ValueError(f"Channel '{ch}' not found. Available: {raw.ch_names}")
        idx = raw.ch_names.index(match)
        data[ch] = raw.get_data(picks=[idx])[0] * 1e6  # → µV
    return data, fs


_PSG_RE = re.compile(r"^(?P<prefix>[A-Z]{2}\d+[\w]+)-PSG\.edf$", re.IGNORECASE)
_HYP_RE = re.compile(r"^(?P<prefix>[A-Z]{2}\d+[\w]+)-Hypnogram\.edf$", re.IGNORECASE)


def _core_key(prefix: str) -> str:
    """Strip last character to create the shared PSG↔Hypnogram match key.

    e.g. SC4001E0 → SC4001E  (PSG)
         SC4001EC → SC4001E  (Hypnogram)
         ST7011J0 → ST7011J  (PSG)
         ST7011JP → ST7011J  (Hypnogram)
    """
    prefix = prefix.strip()
    return prefix[:-1].upper() if len(prefix) >= 2 else prefix.upper()


def find_pairs(folder: Path) -> List[Tuple[Path, Path]]:
    """Return sorted (PSG_path, Hypnogram_path) pairs using the same
    prefix-stripping key as the original pipeline."""
    psg_map: Dict[str, Path] = {}
    hyp_map: Dict[str, Path] = {}

    for f in sorted(folder.rglob("*.edf")):
        m = _PSG_RE.match(f.name)
        if m:
            psg_map[_core_key(m.group("prefix"))] = f
            continue
        m = _HYP_RE.match(f.name)
        if m:
            hyp_map[_core_key(m.group("prefix"))] = f

    matched = sorted(set(psg_map) & set(hyp_map))
    if not matched:
        print(f"  [WARN] No PSG/Hypnogram pairs found in {folder}")

    unmatched_psg = sorted(set(psg_map) - set(hyp_map))
    if unmatched_psg:
        print(f"  [WARN] {len(unmatched_psg)} PSG file(s) without a hypnogram in {folder.name}")

    return [(psg_map[k], hyp_map[k]) for k in matched]


# ── Event-based segmentation ──────────────────────────────────────────────────

def detect_eog_bursts(
    eog: np.ndarray,
    fs: float,
    rem_intervals: List[Tuple[float, float]],
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Detect phasic EOG bursts within REM using Hilbert envelope threshold
    (90th percentile of REM envelope). Used for initial event segmentation;
    the HMM classifier is applied subsequently on multi-feature vectors.
    """
    eog_bp = bandpass(eog, fs, *BURST_BAND)
    env = hilbert_envelope(eog_bp)

    mask = np.zeros(len(eog), dtype=bool)
    for s, e in rem_intervals:
        i0, i1 = int(round(s * fs)), int(round(e * fs))
        mask[i0:min(i1, len(mask))] = True

    rem_env = env[mask]
    if rem_env.size == 0:
        return [], np.nan

    threshold = float(np.percentile(rem_env, BURST_PCT))

    bursts = []
    for rem_s, rem_e in rem_intervals:
        i0 = max(0, int(round(rem_s * fs)))
        i1 = min(len(env), int(round(rem_e * fs)))
        seg_env = env[i0:i1]
        above = seg_env >= threshold

        in_burst = False
        b_start = 0
        for k, val in enumerate(above):
            if val and not in_burst:
                b_start = k
                in_burst = True
            elif not val and in_burst:
                dur = (k - b_start) / fs
                if dur >= MIN_BURST_DUR:
                    bursts.append((rem_s + b_start / fs, rem_s + k / fs))
                in_burst = False
        if in_burst:
            dur = (len(above) - b_start) / fs
            if dur >= MIN_BURST_DUR:
                bursts.append((rem_s + b_start / fs, rem_e))

    bursts = merge_intervals(bursts, gap=MERGE_BURST_GAP)
    return bursts, threshold


def define_tonic_segments(
    rem_intervals: List[Tuple[float, float]],
    bursts: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """
    Tonic segments = REM intervals minus bursts, keeping only gaps >= MIN_TONIC_GAP_S.
    """
    burst_padded = [(s - 0.1, e + 0.1) for s, e in bursts]
    gaps = subtract_intervals(rem_intervals, burst_padded)
    return [(s, e) for s, e in gaps if (e - s) >= MIN_TONIC_GAP_S]


# ── Feature extraction ────────────────────────────────────────────────────────

def features_for_interval(
    t0: float,
    t1: float,
    eog: np.ndarray,
    fpz: np.ndarray,
    pzoz: np.ndarray,
    fs: float,
) -> np.ndarray:
    """
    Extract feature vector for a single phasic event or tonic segment.
    Returns array of shape (5,) — NaN if segment too short.
    All 5 values are stored in the event DataFrame for reference.
    Only [sawtooth_power, spindle_power, theta_power] (FEATURE_COLS) are fed
    to the HMM; eog_envelope is used for state labeling; delta_theta_ratio
    is retained for inspection but excluded from both.
    """
    min_samples = int(fs * 0.5)

    eog_seg  = extract_segment(eog,  fs, t0, t1)
    fpz_seg  = extract_segment(fpz,  fs, t0, t1)
    pzoz_seg = extract_segment(pzoz, fs, t0, t1)

    if len(eog_seg) < min_samples:
        return np.full(5, np.nan)

    # 1. EOG envelope (mean Hilbert amplitude of bandpassed EOG)
    eog_env = float(np.mean(hilbert_envelope(bandpass(eog_seg, fs, *BURST_BAND))))

    # 2. Sawtooth power (2–6 Hz, Fpz-Cz)
    sawtooth = bandpower(fpz_seg, fs, *SAWTOOTH_BAND) if len(fpz_seg) >= min_samples else np.nan

    # 3. Spindle power (12–15 Hz, Pz-Oz)
    spindle = bandpower(pzoz_seg, fs, *SPINDLE_BAND) if len(pzoz_seg) >= min_samples else np.nan

    # 4. Theta power (4–8 Hz, Pz-Oz)
    theta = bandpower(pzoz_seg, fs, *THETA_BAND) if len(pzoz_seg) >= min_samples else np.nan

    # 5. Delta/theta ratio (Pz-Oz)
    delta = bandpower(pzoz_seg, fs, *DELTA_BAND) if len(pzoz_seg) >= min_samples else np.nan
    delta_theta = (delta / theta) if (theta and np.isfinite(theta) and theta > 0) else np.nan

    return np.array([eog_env, sawtooth, spindle, theta, delta_theta])


def extract_features(
    eog: np.ndarray,
    fpz: np.ndarray,
    pzoz: np.ndarray,
    fs: float,
    bursts: List[Tuple[float, float]],
    tonic_segs: List[Tuple[float, float]],
) -> pd.DataFrame:
    """
    Build ordered DataFrame of events (phasic + tonic), sorted by start time.
    Columns: start_s, end_s, duration_s, event_type, + 5 feature columns
    (eog_envelope, sawtooth_power, spindle_power, theta_power, delta_theta_ratio).
    Only sawtooth_power, spindle_power, theta_power are fed to the HMM.
    """
    FEATURE_NAMES = ["eog_envelope", "sawtooth_power", "spindle_power",
                     "theta_power", "delta_theta_ratio"]

    rows = []
    for s, e in bursts:
        f = features_for_interval(s, e, eog, fpz, pzoz, fs)
        rows.append({"start_s": s, "end_s": e, "duration_s": e - s,
                     "event_type": "phasic", **dict(zip(FEATURE_NAMES, f))})

    for s, e in tonic_segs:
        f = features_for_interval(s, e, eog, fpz, pzoz, fs)
        rows.append({"start_s": s, "end_s": e, "duration_s": e - s,
                     "event_type": "tonic", **dict(zip(FEATURE_NAMES, f))})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("start_s").reset_index(drop=True)
    return df


# ── Per-subject z-scoring ─────────────────────────────────────────────────────

FEATURE_COLS = ["eog_envelope", "sawtooth_power", "spindle_power", "theta_power"]
# delta_theta_ratio is computed and stored in the DataFrame for inspection
# but excluded from the HMM feature matrix (supervisor: remove as feature).


def zscored_features(df: pd.DataFrame) -> np.ndarray:
    """Return z-scored feature matrix, dropping rows with any NaN."""
    X = df[FEATURE_COLS].values.astype(float)
    valid = np.all(np.isfinite(X), axis=1)
    X = X[valid]
    if X.shape[0] == 0:
        return X, valid
    scaler = StandardScaler()
    return scaler.fit_transform(X), valid


# ── HMM fitting ───────────────────────────────────────────────────────────────

def fit_hmm(X: np.ndarray, n_states: int) -> GaussianHMM:
    """
    Fit Gaussian HMM with k-means initialisation (on sawtooth power, column 0).
    Runs N_INIT restarts, returns the model with best log-likelihood.
    """
    if X.shape[0] < n_states * 2:
        raise ValueError(f"Too few observations ({X.shape[0]}) for {n_states} states")

    best_model, best_score = None, -np.inf

    for seed in range(N_INIT):
        try:
            # K-means init on sawtooth power (column 0) to guide starting means
            km = KMeans(n_clusters=n_states, random_state=seed, n_init=3)
            km_labels = km.fit_predict(X[:, :1])

            model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=N_ITER,
                tol=1e-4,
                random_state=seed,
                init_params="",  # manual init below
                params="stmc",
            )
            # Initialise means from k-means cluster centres (broadcast to full feature space)
            means_init = np.array([X[km_labels == k].mean(axis=0)
                                   if np.any(km_labels == k) else X.mean(axis=0)
                                   for k in range(n_states)])
            model.means_ = means_init
            model.covars_ = np.array([np.cov(X.T) + np.eye(X.shape[1]) * 1e-3
                                      for _ in range(n_states)])
            model.startprob_ = np.ones(n_states) / n_states
            model.transmat_ = (np.ones((n_states, n_states)) * 0.1 +
                               np.eye(n_states) * 0.9 * (1 - 0.1))
            model.transmat_ /= model.transmat_.sum(axis=1, keepdims=True)

            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("All HMM restarts failed")

    return best_model


def label_states(model: GaussianHMM, X: np.ndarray,
                 eog_envelope: np.ndarray = None) -> Tuple[np.ndarray, bool]:
    """
    Decode most-likely state sequence and re-label so state 0 = tonic
    (lower EOG envelope mean) and state 1 = phasic (higher EOG envelope mean).
    For K=3, state 2 = highest EOG = phasic.

    eog_envelope : if provided, use this for state ordering (EOG not in X).
                   If None, uses column 0 of X (EOG envelope as feature 0).

    Returns (remapped_states, label_flipped).
    label_flipped=True means phasic > 50% after EOG-sort — the assignment is
    unreliable (likely poor threshold calibration or low-quality recording).
    """
    states = model.predict(X)
    eog = eog_envelope if eog_envelope is not None else X[:, 0]
    means = [eog[states == k].mean() if np.any(states == k) else 0.0
             for k in range(model.n_components)]

    order = np.argsort(means)       # ascending: 0 = lowest EOG = tonic
    remap = {old: new for new, old in enumerate(order)}
    remapped = np.array([remap[s] for s in states])

    # Flag if tonic does not dominate (physiologically tonic > phasic in REM)
    phasic_idx = model.n_components - 1   # highest EOG state = phasic
    label_flipped = bool(np.mean(remapped == phasic_idx) > 0.50)
    return remapped, label_flipped


# ── Validation ────────────────────────────────────────────────────────────────

STATE_NAMES_K2 = {0: "tonic", 1: "phasic"}
STATE_NAMES_K3 = {0: "tonic", 1: "transition", 2: "phasic"}


def load_rule_labels(
    df_events: pd.DataFrame,
    csv_path: Optional[Path],
) -> Optional[np.ndarray]:
    """
    Match HMM events to rule-based labels from the existing per-recording CSV.
    Returns array of integer labels (0=tonic, 1=phasic, 2=transition) or None.
    """
    if csv_path is None or not csv_path.exists():
        return None

    rule = pd.read_csv(csv_path)
    rule = rule.dropna(subset=["start_s", "end_s", "label"])
    label_map = {"tonic": 0, "phasic": 1, "transition": 2}

    matched = []
    for _, row in df_events.iterrows():
        mid = (row["start_s"] + row["end_s"]) / 2.0
        # Find rule-based interval that contains the midpoint
        hit = rule[(rule["start_s"] <= mid) & (rule["end_s"] > mid)]
        if len(hit) == 1:
            lbl = label_map.get(hit.iloc[0]["label"], -1)
        else:
            lbl = -1
        matched.append(lbl)
    return np.array(matched)


def feature_profile(X: np.ndarray, states: np.ndarray, n_states: int) -> pd.DataFrame:
    names = FEATURE_COLS
    rows = []
    for k in range(n_states):
        mask = states == k
        if not np.any(mask):
            continue
        sub = X[mask]
        for j, name in enumerate(names):
            rows.append({"state": k, "feature": name,
                         "mean": round(float(sub[:, j].mean()), 3),
                         "std":  round(float(sub[:, j].std()),  3)})
    return pd.DataFrame(rows)


def plot_pca(
    X: np.ndarray,
    hmm_states: np.ndarray,
    rule_labels: Optional[np.ndarray],
    rec_name: str,
    n_states: int,
    out_dir: Path,
):
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_

    n_panels = 2 if rule_labels is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    state_names = STATE_NAMES_K2 if n_states == 2 else STATE_NAMES_K3
    colors_hmm = ["steelblue", "tomato", "seagreen"]
    for k in range(n_states):
        mask = hmm_states == k
        axes[0].scatter(Xp[mask, 0], Xp[mask, 1],
                        c=colors_hmm[k], label=state_names[k], alpha=0.6, s=20)
    axes[0].set_title(f"HMM states (K={n_states})")
    axes[0].set_xlabel(f"PC1 ({ev[0]:.0%})")
    axes[0].set_ylabel(f"PC2 ({ev[1]:.0%})")
    axes[0].legend()

    if rule_labels is not None:
        colors_rule = {0: "steelblue", 1: "tomato", 2: "gold"}
        names_rule = {0: "tonic", 1: "phasic", 2: "transition"}
        for lbl, col in colors_rule.items():
            mask = rule_labels == lbl
            if np.any(mask):
                axes[1].scatter(Xp[mask, 0], Xp[mask, 1],
                                c=col, label=names_rule[lbl], alpha=0.6, s=20)
        axes[1].set_title("Rule-based labels (90/20 EOG)")
        axes[1].set_xlabel(f"PC1 ({ev[0]:.0%})")
        axes[1].set_ylabel(f"PC2 ({ev[1]:.0%})")
        axes[1].legend()

    fig.suptitle(rec_name, fontsize=11)
    plt.tight_layout()
    path = out_dir / f"{rec_name}_pca_K{n_states}.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)


def plot_temporal(
    df_events: pd.DataFrame,
    valid_mask: np.ndarray,
    hmm_states: np.ndarray,
    rec_name: str,
    n_states: int,
    out_dir: Path,
    max_s: float = 1200.0,
):
    """Plot HMM state sequence over the first max_s seconds of REM events."""
    df_v = df_events[valid_mask].reset_index(drop=True)
    # Restrict to first max_s seconds of recording shown
    t_start = df_v["start_s"].min() if len(df_v) else 0
    df_v = df_v[df_v["end_s"] <= t_start + max_s]
    if len(df_v) == 0:
        return

    states_v = hmm_states[:len(df_v)]
    state_names = STATE_NAMES_K2 if n_states == 2 else STATE_NAMES_K3
    colors = {0: "steelblue", 1: "tomato", 2: "seagreen"}

    fig, ax = plt.subplots(figsize=(14, 3))
    for i, (_, row) in enumerate(df_v.iterrows()):
        if i >= len(states_v):
            break
        s = states_v[i]
        ax.barh(0, row["end_s"] - row["start_s"], left=row["start_s"],
                color=colors.get(s, "grey"), height=0.6, alpha=0.8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k])
               for k in range(n_states)]
    ax.legend(handles, [state_names[k] for k in range(n_states)],
              loc="upper right")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title(f"{rec_name} — HMM state sequence (K={n_states})")
    plt.tight_layout()
    path = out_dir / f"{rec_name}_temporal_K{n_states}.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)


# ── Group-level summary plot ──────────────────────────────────────────────────

def plot_group_summary(
    summary: pd.DataFrame,
    X_concat: np.ndarray,
    st_concat: np.ndarray,
    n_states: int,
    out_dir: Path,
):
    """
    Three-panel group summary figure:
      A) State proportions per subject (stacked bar, sorted by % phasic)
      B) ARI and NMI per subject (dot plot, only if validation ran)
      C) Pooled feature profile per HMM state (mean ± SD bar chart)
    """
    state_names = STATE_NAMES_K2 if n_states == 2 else STATE_NAMES_K3
    colors = ["steelblue", "tomato", "seagreen"]  # tonic, phasic, (transition)

    has_validation = "ari" in summary.columns and summary["ari"].notna().any()
    n_cols = 3 if has_validation else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    # ── A: State proportions per subject ────────────────────────────────────
    ax = axes[0]
    pct_cols = [f"pct_{state_names[k]}" for k in range(n_states)]
    # Only keep columns that exist
    pct_cols = [c for c in pct_cols if c in summary.columns]
    df_pct = summary[["recording"] + pct_cols].copy()
    # Sort by % phasic (last state = highest EOG = phasic)
    phasic_col = f"pct_{state_names[n_states - 1]}"
    if phasic_col in df_pct.columns:
        df_pct = df_pct.sort_values(phasic_col, ascending=True)
    short_names = [r.replace("SC4", "").replace("ST7", "").replace("-PSG", "")
                   for r in df_pct["recording"]]
    bottom = np.zeros(len(df_pct))
    for k, col in enumerate(pct_cols):
        state_k = k  # index into state_names
        # map pct_col name back to state index
        label = state_names.get(k, col)
        vals = df_pct[col].values
        ax.barh(range(len(df_pct)), vals, left=bottom,
                color=colors[k], label=label, alpha=0.85)
        bottom += vals
    ax.set_yticks(range(len(df_pct)))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("% of events")
    ax.set_title(f"State proportions per subject (K={n_states})")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 100)

    # ── B: ARI / NMI per subject (if available) ─────────────────────────────
    if has_validation:
        ax = axes[1]
        valid = summary.dropna(subset=["ari"])
        x = np.arange(len(valid))
        ax.scatter(x, valid["ari"].values, color="steelblue", s=40, label="ARI", zorder=3)
        ax.scatter(x, valid["nmi"].values, color="tomato", marker="s", s=40,
                   label="NMI", zorder=3)
        ax.axhline(valid["ari"].mean(), color="steelblue", lw=1, ls="--", alpha=0.6)
        ax.axhline(valid["nmi"].mean(), color="tomato",    lw=1, ls="--", alpha=0.6)
        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [r.replace("SC4", "").replace("ST7", "").replace("-PSG", "")
             for r in valid["recording"]],
            rotation=90, fontsize=7
        )
        ax.set_ylabel("Score")
        ax.set_ylim(-0.1, 1.05)
        ax.set_title(f"ARI / NMI vs rule-based labels (K={n_states})\n"
                     f"mean ARI={valid['ari'].mean():.2f}  "
                     f"mean NMI={valid['nmi'].mean():.2f}")
        ax.legend(fontsize=8)
        feature_ax = axes[2]
    else:
        feature_ax = axes[1]

    # ── C: Pooled feature profile per state ─────────────────────────────────
    ax = feature_ax
    feat_labels = ["EOG\nenvelope", "Sawtooth\n(2–6 Hz)", "Spindle\npower", "Theta\npower"]
    n_feats = len(FEATURE_COLS)
    x = np.arange(n_feats)
    width = 0.8 / n_states

    for k in range(n_states):
        mask = st_concat == k
        if not np.any(mask):
            continue
        means = X_concat[mask].mean(axis=0)
        sds   = X_concat[mask].std(axis=0)
        offset = (k - (n_states - 1) / 2) * width
        ax.bar(x + offset, means, width=width * 0.9,
               color=colors[k], alpha=0.85, label=state_names[k], zorder=3)
        ax.errorbar(x + offset, means, yerr=sds,
                    fmt="none", color="black", capsize=3, lw=1, zorder=4)

    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(feat_labels, fontsize=9)
    ax.set_ylabel("Mean z-score (pooled across subjects)")
    ax.set_title(f"Feature profile per HMM state (K={n_states})")
    ax.legend(fontsize=8)

    fig.suptitle(f"Group summary — HMM K={n_states}  (n={len(summary)} subjects)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = out_dir / f"group_summary_K{n_states}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Group summary plot → {path.name}")


# ── Per-recording pipeline ────────────────────────────────────────────────────

def process_recording(
    psg_path: Path,
    hyp_path: Path,
    results_dir: Path,
    n_states: int = N_STATES_PRIMARY,
) -> Optional[Dict]:
    rec_name = psg_path.stem.replace("-PSG", "")
    print(f"\n{'─'*60}")
    print(f"  {rec_name}")

    # ── Load channels ────────────────────────────────────────────────────────
    try:
        channels, fs = load_raw_channels(
            psg_path,
            ["EOG horizontal", "EEG Fpz-Cz", "EEG Pz-Oz"]
        )
    except Exception as e:
        print(f"  [ERROR] Loading channels: {e}")
        return None

    eog  = channels["EOG horizontal"]
    fpz  = channels["EEG Fpz-Cz"]
    pzoz = channels["EEG Pz-Oz"]
    t_end = len(eog) / fs

    # ── REM intervals ────────────────────────────────────────────────────────
    rem_intervals = load_rem_intervals(hyp_path, t_end)
    if not rem_intervals:
        print("  [WARN] No REM intervals found, skipping")
        return None

    total_rem_s = sum(e - s for s, e in rem_intervals)
    print(f"  REM: {total_rem_s:.0f}s across {len(rem_intervals)} periods")

    # ── Burst detection ──────────────────────────────────────────────────────
    bursts, threshold = detect_eog_bursts(eog, fs, rem_intervals)
    tonic_segs = define_tonic_segments(rem_intervals, bursts)

    print(f"  Bursts: {len(bursts)}  |  Tonic segments: {len(tonic_segs)}"
          f"  |  EOG threshold: {threshold:.1f} µV")

    if len(bursts) + len(tonic_segs) < n_states * 3:
        print(f"  [SKIP] Too few events for K={n_states} HMM")
        return None

    # ── Feature extraction ───────────────────────────────────────────────────
    df_events = extract_features(eog, fpz, pzoz, fs, bursts, tonic_segs)
    if df_events.empty:
        print("  [SKIP] No events with valid features")
        return None

    X_raw, valid = zscored_features(df_events)
    if X_raw.shape[0] < n_states * 3:
        print(f"  [SKIP] Too few valid observations ({X_raw.shape[0]})")
        return None

    df_valid = df_events[valid].reset_index(drop=True)

    # ── Fit HMM ──────────────────────────────────────────────────────────────
    print(f"  Fitting HMM K={n_states} on {X_raw.shape[0]} observations...")
    try:
        model = fit_hmm(X_raw, n_states)
        hmm_states, label_flipped = label_states(model, X_raw)
    except Exception as e:
        print(f"  [ERROR] HMM failed: {e}")
        return None

    state_names = STATE_NAMES_K2 if n_states == 2 else STATE_NAMES_K3
    for k in range(n_states):
        pct = np.mean(hmm_states == k) * 100
        print(f"    State {k} ({state_names[k]}): {pct:.1f}%")
    if label_flipped:
        print(f"  [WARN] Phasic > 50% — possible threshold/quality issue")

    # ── Attach HMM labels to events ──────────────────────────────────────────
    df_valid = df_valid.copy()
    df_valid["hmm_state"] = hmm_states
    df_valid["hmm_label"] = [state_names[s] for s in hmm_states]
    df_valid.insert(0, "recording", rec_name)

    # ── Validation vs. rule-based labels ────────────────────────────────────
    # Use event_type (burst→phasic, tonic_seg→tonic) as universal rule baseline.
    # This gives ARI/NMI for every subject, measuring agreement between the
    # unsupervised HMM and the simple EOG-threshold classifier.
    # If a per-recording CSV exists, use it instead (more detailed labels).
    rule_map = {"phasic": 1, "tonic": 0, "transition": 2}
    rule_labels = np.array([rule_map.get(t, -1) for t in df_valid["event_type"]])

    rule_csv = psg_path.parent / f"{psg_path.stem}.rem_states_onset_aligned.csv"
    csv_labels = load_rule_labels(df_valid, rule_csv)
    if csv_labels is not None:
        rule_labels = csv_labels  # override with richer CSV labels if available

    ari, nmi = np.nan, np.nan
    if n_states == 2:
        # Binary comparison: exclude transitions (label==2)
        valid_rule = (rule_labels == 0) | (rule_labels == 1)
    else:
        valid_rule = rule_labels >= 0

    if np.sum(valid_rule) > n_states:
        ari = adjusted_rand_score(rule_labels[valid_rule], hmm_states[valid_rule])
        nmi = normalized_mutual_info_score(rule_labels[valid_rule], hmm_states[valid_rule])
        df_valid["rule_label"] = [
            {0: "tonic", 1: "phasic", 2: "transition"}.get(int(l), "unknown")
            if l >= 0 else "unmatched"
            for l in rule_labels
        ]
        print(f"  Validation  ARI={ari:.3f}  NMI={nmi:.3f}")

    # Summary row for this recording
    state_names_local = STATE_NAMES_K2 if n_states == 2 else STATE_NAMES_K3
    pct = {state_names_local[k]: round(float(np.mean(hmm_states == k) * 100), 1)
           for k in range(n_states)}

    metrics = {
        "recording":        rec_name,
        "n_states":         n_states,
        "total_rem_s":      round(total_rem_s, 1),
        "n_rem_periods":    len(rem_intervals),
        "n_bursts":         len(bursts),
        "n_tonic_segs":     len(tonic_segs),
        "n_events":         int(X_raw.shape[0]),
        "eog_threshold_uV": round(float(threshold), 2),
        **{f"pct_{k}": v for k, v in pct.items()},
        "label_flipped":    label_flipped,
        "ari":              round(float(ari), 3) if np.isfinite(ari) else np.nan,
        "nmi":              round(float(nmi), 3) if np.isfinite(nmi) else np.nan,
    }

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_pca(X_raw, hmm_states, rule_labels, rec_name, n_states, results_dir)
    plot_temporal(df_events, valid, hmm_states, rec_name, n_states, results_dir)

    return metrics, df_valid, X_raw, hmm_states


# ── Batch runner ──────────────────────────────────────────────────────────────

def run(folders: List[Path], n_states: int = N_STATES_PRIMARY):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    for folder in folders:
        if folder.exists():
            all_pairs.extend(find_pairs(folder))

    if not all_pairs:
        print("[ERROR] No PSG/hypnogram pairs found.")
        return

    if MAX_RECORDINGS is not None:
        all_pairs = all_pairs[:MAX_RECORDINGS]

    print(f"Found {len(all_pairs)} recordings. Running HMM K={n_states}...\n")

    all_metrics: List[Dict] = []
    all_events:  List[pd.DataFrame] = []
    all_X:       List[np.ndarray] = []
    all_states:  List[np.ndarray] = []

    for psg, hyp in all_pairs:
        rec_name = psg.stem.replace("-PSG", "")
        if rec_name in SKIP_RECORDINGS:
            continue
        try:
            result = process_recording(psg, hyp, OUT_DIR, n_states=n_states)
        except OSError as e:
            print(f"  [SKIP] {rec_name}: {e}")
            continue
        if result is None:
            continue
        metrics, df_valid, X_raw, hmm_states = result
        all_metrics.append(metrics)
        all_events.append(df_valid)
        all_X.append(X_raw)
        all_states.append(hmm_states)

    if not all_metrics:
        print("[WARN] No recordings processed successfully.")
        return

    K = n_states

    # ── 1. Summary CSV: one row per recording ────────────────────────────────
    summary = pd.DataFrame(all_metrics)
    summary_path = OUT_DIR / f"hmm_K{K}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"[1/3] Summary ({len(summary)} recordings) → {summary_path.name}")
    print(summary.to_string(index=False))

    if "ari" in summary.columns:
        valid_ari = summary["ari"].dropna()
        if len(valid_ari):
            print(f"\n  Mean ARI: {valid_ari.mean():.3f} ± {valid_ari.std():.3f}")
            print(f"  Mean NMI: {summary['nmi'].dropna().mean():.3f} ± {summary['nmi'].dropna().std():.3f}")

    # ── 2. All-events CSV: one row per event across all recordings ───────────
    df_all = pd.concat(all_events, ignore_index=True)
    events_path = OUT_DIR / f"hmm_K{K}_all_events.csv"
    df_all.to_csv(events_path, index=False)
    print(f"\n[2/3] All events ({len(df_all)} rows) → {events_path.name}")

    # ── 3. Feature profile CSV: group-level mean±SD per state per feature ────
    X_concat  = np.vstack(all_X)
    st_concat = np.concatenate(all_states)
    fp = feature_profile(X_concat, st_concat, n_states)
    state_names = STATE_NAMES_K2 if n_states == 2 else STATE_NAMES_K3
    fp["state_label"] = fp["state"].map(state_names)
    profile_path = OUT_DIR / f"hmm_K{K}_feature_profile.csv"
    fp.to_csv(profile_path, index=False)
    print(f"[3/3] Feature profile → {profile_path.name}")
    print(fp.to_string(index=False))

    # ── Group summary plot ───────────────────────────────────────────────────
    plot_group_summary(summary, X_concat, st_concat, n_states, OUT_DIR)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Primary run: K=2 (phasic + tonic)
    run([SC_DIR, ST_DIR], n_states=N_STATES_PRIMARY)

    # Exploratory run: K=3 (phasic + transition + tonic)
    run([SC_DIR, ST_DIR], n_states=N_STATES_EXPLORATORY)
