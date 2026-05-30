import socket
import threading
import queue
import time
from pathlib import Path
from dataclasses import dataclass
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# Version 1:
# Robust front human presence detector using background-subtracted range-Doppler evidence.
#
# Goal:
#   Classify the front scene into EMPTY / DRONE / HUMAN using dB-based RD features.
#
# Key idea:
#   Doppler decides "is something human-like moving?"
#   Range decides "where roughly is it?"
#   Temporal voting decides "should we trust it?"
#
# This version intentionally does NOT use angle FFT / point cloud.
# It is meant to be more robust than single-frame point-cloud peak picking.
# ============================================================


# ============================================================
# Network config
# ============================================================

PC_IP = "192.168.33.30"
DCA_IP = "192.168.33.180"
DATA_PORT = 4098
SOCKET_RCVBUF = 64 * 1024 * 1024


# ============================================================
# Radar config
# Must match your current mmWave Studio setting:
#   FrameConfig(0, 2, 300, 64, 200, 0, 1)
# ============================================================

NUM_FRAMES = 300              # 60 sec at 200 ms
FRAME_PERIOD_S = 0.200

NUM_LOOPS = 64
NUM_TX = 3
NUM_RX = 4
NUM_SAMPLES = 256

BYTES_PER_SAMPLE = 2          # int16
IQ = 2                        # I + Q

FRAME_BYTES = NUM_LOOPS * NUM_TX * NUM_SAMPLES * NUM_RX * IQ * BYTES_PER_SAMPLE
EXPECTED_BYTES = NUM_FRAMES * FRAME_BYTES


# ============================================================
# RF / FFT config
# ============================================================

SAMPLE_RATE_HZ = 10e6
SLOPE_HZ_PER_S = 29.982e12
START_FREQ_HZ = 77e9
C = 299_792_458.0

IDLE_TIME_S = 100e-6
RAMP_END_TIME_S = 60e-6
CHIRP_TIME_S = IDLE_TIME_S + RAMP_END_TIME_S

# For 3Tx TDM-MIMO, same-Tx slow-time spacing is one full TDM cycle.
TDM_CYCLE_S = NUM_TX * CHIRP_TIME_S


# ============================================================
# Detector ROI / thresholds
# ============================================================

# Front human detection range.
# Do not include too-near leakage / coupling region.
DETECT_RANGE_MIN_M = 0.70
DETECT_RANGE_MAX_M = 3.50

# Moving Doppler gate.
# Lower MIN_ABS_VELOCITY_MPS if slow drone hover / slow human motion is missed.
# Raise it if static clutter leaks through.
# For Tello hover, 0.03 is more sensitive than 0.08.
MIN_ABS_VELOCITY_MPS = 0.03

# The current 3Tx TDM setting has unambiguous velocity around +/-2.0 m/s.
# Keep this slightly below the edge.
MAX_ABS_VELOCITY_MPS = 1.90

# Background learning:
# Keep the scene empty during the first N frames.
BACKGROUND_CALIBRATION_FRAMES = 25
BACKGROUND_METHOD = "median"     # "median" is more robust than "mean"
BACKGROUND_SCALE = 1.10          # >1 subtracts static clutter more aggressively

# Per-frame CFAR-ish thresholding on background-subtracted RD power.
# The detector computes a robust noise floor from the moving ROI:
#   threshold = median + THRESHOLD_MAD_K * MAD
THRESHOLD_MAD_K = 6.0

# Component filtering.
MIN_COMPONENT_BINS = 3
MIN_COMPONENT_RANGE_BINS = 1
MIN_COMPONENT_DOPPLER_BINS = 1

# Energy gate.
# Component total energy must be at least this many dB above ROI robust noise level.
MIN_COMPONENT_ENERGY_DB_OVER_NOISE = 6.0

# Temporal confirmation.
# With 200 ms frame period:
#   3 of last 5 frames = confirmed after roughly 0.6~1.0 sec.
TRACK_HISTORY_FRAMES = 5
CONFIRM_HITS = 3
LOST_MISSES = 5

# Smooth displayed range / velocity estimate after confirmation.
TRACK_SMOOTH_ALPHA = 0.35

# ============================================================
# Event classification by absolute heatmap dB + blob shape
# Classes:
#   EMPTY : background-subtracted RD heatmap is not bright enough
#   DRONE : medium-bright and compact moving blob
#   HUMAN : very bright or wider/larger moving blob
#
# Important:
#   peak_db / top10_db are from the displayed background-subtracted heatmap:
#     rd_db = 10log10(max(current_RD_power - background_RD_power, 0) + 1e-12)
#
# This means these thresholds match what you visually see on the heatmap colorbar.
# ============================================================

ENABLE_EVENT_CLASSIFICATION = True

# Absolute heatmap dB thresholds.
# From your observation:
#   EMPTY usually around 60~90 dB
#   HUMAN often reaches 110~130 dB
EMPTY_MAX_PEAK_DB = 95.0
EMPTY_MAX_TOP10_DB = 90.0

HUMAN_MIN_PEAK_DB = 110.0
HUMAN_MIN_TOP10_DB = 105.0

# Drone is assumed to be between EMPTY and HUMAN in heatmap brightness,
# and more compact than a human body in RD. Tune after measuring Tello.
DRONE_MIN_PEAK_DB = 95.0
DRONE_MAX_PEAK_DB = 135.0
DRONE_MIN_TOP10_DB = 90.0
DRONE_MAX_TOP10_DB = 125.0

# Blob shape thresholds.
HUMAN_MIN_ACTIVE_BINS = 35
HUMAN_MIN_RANGE_SPREAD_M = 0.35
HUMAN_MIN_VELOCITY_SPREAD_MPS = 0.25

DRONE_MAX_ACTIVE_BINS = 18
DRONE_MAX_RANGE_SPREAD_M = 0.18
DRONE_MAX_VELOCITY_SPREAD_MPS = 0.45

# Special drone hover / slow-motion mode:
# Hovering Tello may have very small radial velocity, so the RD blob can be small.
# If it is compact and still brighter than empty, classify as DRONE.
DRONE_HOVER_MIN_PEAK_DB = 92.0
DRONE_HOVER_MIN_TOP10_DB = 88.0
DRONE_HOVER_MAX_ACTIVE_BINS = 12
DRONE_HOVER_MAX_RANGE_SPREAD_M = 0.16
DRONE_HOVER_MAX_VELOCITY_SPREAD_MPS = 0.20

# This E/noise gate only decides whether a detected component is credible.
# It is NOT the same thing as heatmap colorbar dB.
MIN_COMPONENT_ENERGY_DB_OVER_NOISE = 6.0

# Temporal event smoothing. Keeps the displayed class stable.
EVENT_HISTORY_FRAMES = 5
EVENT_CONFIRM_HITS = 3


# ============================================================
# Display config
# ============================================================

# Fixed magnitude bar for the background-subtracted range-Doppler heatmap.
# This is intentionally fixed, not auto-scaled per frame, so brightness is comparable over time.
#
# The plotted value is:
#   10*log10(max(current_RD_power - BACKGROUND_SCALE*background_RD_power, 0) + 1e-12)
#
# If the heatmap is mostly dark even when a person moves, lower these numbers.
# If the heatmap is saturated, raise these numbers.
FIXED_HEATMAP_VMIN_DB = 60.0
FIXED_HEATMAP_VMAX_DB = 130.0

PLOT_UPDATE_INTERVAL_MS = 100


# ============================================================
# Optional raw backup
# ============================================================

WRITE_RAW_BACKUP = True
OUT_PATH = Path(
    r"C:\Users\user\Desktop\ADPAR-final-project\adc\live_event_classifier_v4_drone_hover_sensitive_60s_200ms_Raw_0.bin"
)


# ============================================================
# DCA1000 packet format
# ============================================================

DCA_HEADER_BYTES = 10


# ============================================================
# Thread comms
# ============================================================

result_queue = queue.Queue(maxsize=1000)
stop_event = threading.Event()


@dataclass
class HumanDetectorFrame:
    frame_idx: int
    t_sec: float

    rd_db: np.ndarray                 # [valid_range, valid_doppler], background-subtracted
    detected_raw: bool                # single-frame detection before temporal confirmation
    confirmed: bool                   # temporal confirmed detection

    confidence: float                 # 0~1 rough confidence
    range_m: float
    velocity_mps: float
    range_spread_m: float
    velocity_spread_mps: float
    energy_db_over_noise: float
    peak_db: float
    top10_db: float
    active_bins: int

    event_label: str
    event_reason: str

    status: str


class PresenceTracker:
    """
    Tiny temporal tracker / voting filter.

    It does not track angle because Version 1 intentionally avoids angle FFT.
    It confirms human presence only when detections are temporally consistent.
    """

    def __init__(self):
        self.history = deque(maxlen=TRACK_HISTORY_FRAMES)
        self.confirmed = False
        self.miss_count = 0

        self.range_m = np.nan
        self.velocity_mps = np.nan
        self.confidence = 0.0

    def update(self, detected: bool, range_m: float, velocity_mps: float, confidence: float):
        self.history.append(bool(detected))

        hits = sum(self.history)
        if hits >= CONFIRM_HITS:
            self.confirmed = True
            self.miss_count = 0
        elif detected:
            self.miss_count = 0
        else:
            self.miss_count += 1
            if self.miss_count >= LOST_MISSES:
                self.confirmed = False

        if detected:
            if np.isnan(self.range_m):
                self.range_m = range_m
            else:
                self.range_m = (
                    (1.0 - TRACK_SMOOTH_ALPHA) * self.range_m
                    + TRACK_SMOOTH_ALPHA * range_m
                )

            if np.isnan(self.velocity_mps):
                self.velocity_mps = velocity_mps
            else:
                self.velocity_mps = (
                    (1.0 - TRACK_SMOOTH_ALPHA) * self.velocity_mps
                    + TRACK_SMOOTH_ALPHA * velocity_mps
                )

            self.confidence = (
                (1.0 - TRACK_SMOOTH_ALPHA) * self.confidence
                + TRACK_SMOOTH_ALPHA * confidence
            )
        else:
            self.confidence *= 0.90

        return self.confirmed, self.range_m, self.velocity_mps, self.confidence


tracker = PresenceTracker()


# ============================================================
# Axes
# ============================================================

def range_axis_m():
    freqs = np.fft.fftfreq(NUM_SAMPLES, d=1.0 / SAMPLE_RATE_HZ)[: NUM_SAMPLES // 2]
    return C * freqs / (2.0 * SLOPE_HZ_PER_S)


def doppler_axis_mps():
    lam = C / START_FREQ_HZ
    fd = np.fft.fftshift(np.fft.fftfreq(NUM_LOOPS, d=TDM_CYCLE_S))
    return fd * lam / 2.0


RANGE_AXIS = range_axis_m()
DOPPLER_AXIS = doppler_axis_mps()

RANGE_MASK = (RANGE_AXIS >= DETECT_RANGE_MIN_M) & (RANGE_AXIS <= DETECT_RANGE_MAX_M)
RANGE_IDXS = np.where(RANGE_MASK)[0]

DOPPLER_MASK = (
    (np.abs(DOPPLER_AXIS) >= MIN_ABS_VELOCITY_MPS)
    & (np.abs(DOPPLER_AXIS) <= MAX_ABS_VELOCITY_MPS)
)
DOPPLER_IDXS = np.where(DOPPLER_MASK)[0]

VALID_RANGE_AXIS = RANGE_AXIS[RANGE_IDXS]
VALID_DOPPLER_AXIS = DOPPLER_AXIS[DOPPLER_IDXS]


# ============================================================
# Background model
# ============================================================

bg_learning = True
bg_frames = []
background_rd_power = None


def reset_background():
    global bg_learning, bg_frames, background_rd_power, tracker, event_smoother
    bg_learning = True
    bg_frames = []
    background_rd_power = None
    tracker = PresenceTracker()
    event_smoother = EventSmoother()
    print()
    print("=== Background reset ===")
    print(f"Keep scene empty for {BACKGROUND_CALIBRATION_FRAMES} frames.")
    print()


# ============================================================
# DCA1000 parsing / raw reshape
# ============================================================

def parse_dca_packet(data: bytes):
    if len(data) < DCA_HEADER_BYTES:
        return None

    seq = int.from_bytes(data[0:4], byteorder="little", signed=False)
    byte_count = int.from_bytes(data[4:10], byteorder="little", signed=False)
    raw = data[10:]
    return seq, byte_count, raw


def raw_frame_to_adc(frame_bytes: bytes) -> np.ndarray:
    """
    Output shape:
      [loop, tx, sample, rx]

    Assumed DCA1000 sample ordering:
      Rx0I Rx1I Rx2I Rx3I Rx0Q Rx1Q Rx2Q Rx3Q
    """
    raw = np.frombuffer(frame_bytes, dtype=np.int16)

    expected_words = NUM_LOOPS * NUM_TX * NUM_SAMPLES * NUM_RX * IQ
    if raw.size != expected_words:
        raise ValueError(f"raw words={raw.size}, expected={expected_words}")

    data = raw.reshape(NUM_LOOPS * NUM_TX, NUM_SAMPLES, IQ, NUM_RX)

    i_data = data[:, :, 0, :].astype(np.float32)
    q_data = data[:, :, 1, :].astype(np.float32)

    adc = i_data + 1j * q_data
    adc = adc.reshape(NUM_LOOPS, NUM_TX, NUM_SAMPLES, NUM_RX)

    return adc


# ============================================================
# Signal processing
# ============================================================

def compute_range_doppler_power(adc: np.ndarray) -> np.ndarray:
    """
    Returns:
      rd_power_valid: [valid_range, valid_doppler] in linear power

    Processing:
      range FFT over fast time
      Doppler FFT over loops
      non-coherent sum over Tx/Rx
      keep only front range ROI and moving Doppler bins
    """

    # Remove DC across fast time per chirp/Rx to reduce leakage.
    adc = adc - np.mean(adc, axis=2, keepdims=True)

    range_win = np.hanning(NUM_SAMPLES)[None, None, :, None]
    range_fft = np.fft.fft(adc * range_win, axis=2)[:, :, : NUM_SAMPLES // 2, :]
    # [loop, tx, range, rx]

    doppler_win = np.hanning(NUM_LOOPS)[:, None, None, None]
    rd = np.fft.fftshift(
        np.fft.fft(range_fft * doppler_win, axis=0),
        axes=0,
    )
    # [doppler, tx, range, rx]

    # Non-coherent sum over Tx/Rx is robust for presence detection.
    rd_power = np.sum(np.abs(rd) ** 2, axis=(1, 3))  # [doppler, range]

    # Transpose to [range, doppler], then ROI crop.
    rd_power = rd_power.T
    rd_power_valid = rd_power[np.ix_(RANGE_IDXS, DOPPLER_IDXS)]

    return rd_power_valid.astype(np.float64)


def robust_noise_threshold(power: np.ndarray):
    """
    Robust threshold on linear power.
    Uses median + K * MAD.
    """
    vals = power[np.isfinite(power)]
    vals = vals[vals > 0]

    if vals.size == 0:
        return 0.0, 0.0, 0.0

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-12
    threshold = med + THRESHOLD_MAD_K * mad
    return threshold, med, mad


def connected_components(mask: np.ndarray):
    """
    Simple 8-connected components without scipy.
    mask shape: [range, doppler]
    Returns list of lists of (r, d).
    """
    visited = np.zeros_like(mask, dtype=bool)
    comps = []

    nr, nd = mask.shape

    for r in range(nr):
        for d in range(nd):
            if not mask[r, d] or visited[r, d]:
                continue

            stack = [(r, d)]
            visited[r, d] = True
            comp = []

            while stack:
                cr, cd = stack.pop()
                comp.append((cr, cd))

                for dr in (-1, 0, 1):
                    for dd in (-1, 0, 1):
                        if dr == 0 and dd == 0:
                            continue

                        rr = cr + dr
                        cc = cd + dd

                        if rr < 0 or rr >= nr or cc < 0 or cc >= nd:
                            continue

                        if mask[rr, cc] and not visited[rr, cc]:
                            visited[rr, cc] = True
                            stack.append((rr, cc))

            comps.append(comp)

    return comps


def analyze_components(power_sub: np.ndarray, threshold: float, noise_median: float):
    """
    Finds the strongest human-like moving blob.
    Returns:
      detected, metrics dict
    """

    mask = power_sub > threshold
    comps = connected_components(mask)

    best = None

    for comp in comps:
        if len(comp) < MIN_COMPONENT_BINS:
            continue

        rr = np.array([p[0] for p in comp], dtype=int)
        dd = np.array([p[1] for p in comp], dtype=int)

        range_span_bins = int(rr.max() - rr.min() + 1)
        doppler_span_bins = int(dd.max() - dd.min() + 1)

        if range_span_bins < MIN_COMPONENT_RANGE_BINS:
            continue

        if doppler_span_bins < MIN_COMPONENT_DOPPLER_BINS:
            continue

        weights = power_sub[rr, dd]
        total_energy = float(np.sum(weights))
        peak_energy = float(np.max(weights))

        # Approximate component energy above robust ROI noise.
        noise_energy = max(noise_median * len(comp), 1e-12)
        energy_db_over_noise = 10.0 * np.log10(total_energy / noise_energy + 1e-12)

        if energy_db_over_noise < MIN_COMPONENT_ENERGY_DB_OVER_NOISE:
            continue

        range_vals = VALID_RANGE_AXIS[rr]
        vel_vals = VALID_DOPPLER_AXIS[dd]

        wsum = float(np.sum(weights)) + 1e-12
        range_centroid = float(np.sum(weights * range_vals) / wsum)
        vel_centroid = float(np.sum(weights * vel_vals) / wsum)

        range_spread = float(np.sqrt(np.sum(weights * (range_vals - range_centroid) ** 2) / wsum))
        vel_spread = float(np.sqrt(np.sum(weights * (vel_vals - vel_centroid) ** 2) / wsum))

        # Confidence is intentionally simple and bounded.
        # It combines energy margin and blob size.
        energy_score = np.clip((energy_db_over_noise - MIN_COMPONENT_ENERGY_DB_OVER_NOISE) / 15.0, 0.0, 1.0)
        size_score = np.clip(len(comp) / 30.0, 0.0, 1.0)
        spread_score = np.clip(doppler_span_bins / 6.0, 0.0, 1.0)

        confidence = float(0.60 * energy_score + 0.25 * size_score + 0.15 * spread_score)

        candidate = {
            "rr": rr,
            "dd": dd,
            "total_energy": total_energy,
            "peak_energy": peak_energy,
            "energy_db_over_noise": energy_db_over_noise,
            "range_m": range_centroid,
            "velocity_mps": vel_centroid,
            "range_spread_m": range_spread,
            "velocity_spread_mps": vel_spread,
            "active_bins": len(comp),
            "confidence": confidence,
        }

        if best is None or candidate["total_energy"] > best["total_energy"]:
            best = candidate

    if best is None:
        return False, {
            "range_m": np.nan,
            "velocity_mps": np.nan,
            "range_spread_m": np.nan,
            "velocity_spread_mps": np.nan,
            "energy_db_over_noise": np.nan,
            "active_bins": 0,
            "confidence": 0.0,
        }

    return True, best



def compute_heatmap_db_features(rd_db: np.ndarray):
    """
    Compute absolute heatmap dB features from the same RD heatmap shown on screen.

    peak_db:
      Brightest single RD cell.

    top10_db:
      Average of the top 10 brightest RD cells. This is usually more stable
      than peak_db because it is less sensitive to one hot pixel.
    """
    vals = rd_db[np.isfinite(rd_db)].ravel()
    if vals.size == 0:
        return float("nan"), float("nan")

    peak_db = float(np.max(vals))
    k = min(10, vals.size)
    topk = np.partition(vals, -k)[-k:]
    top10_db = float(np.mean(topk))
    return peak_db, top10_db


class EventSmoother:
    """Majority-vote event smoother for EMPTY / DRONE / HUMAN labels."""

    def __init__(self):
        self.history = deque(maxlen=EVENT_HISTORY_FRAMES)

    def update(self, label: str) -> str:
        self.history.append(label)
        labels = [x for x in self.history if x not in ("LEARNING_BG", "DISABLED")]
        if not labels:
            return label

        candidates = sorted(set(labels))
        best = max(candidates, key=lambda x: labels.count(x))
        if labels.count(best) >= EVENT_CONFIRM_HITS:
            return best
        return label


event_smoother = EventSmoother()


def classify_event_from_db_features(
    detected_raw: bool,
    metrics: dict,
    peak_db: float,
    top10_db: float,
):
    """
    Event classifier V4.

    Fix:
      Drone hover / slow drone was detected as EMPTY.

    New behavior:
      - Lower Doppler/component thresholds are used globally.
      - If a compact RD blob is above empty brightness but slow/small,
        classify it as DRONE_HOVER / DRONE instead of EMPTY.
      - Human still requires large/wide blob:
          HUMAN_MIN_ACTIVE_BINS = 35
          HUMAN_MIN_RANGE_SPREAD_M = 0.35
    """
    if not ENABLE_EVENT_CLASSIFICATION:
        return "DISABLED", "event classification disabled"

    e_db = metrics.get("energy_db_over_noise", np.nan)
    bins = int(metrics.get("active_bins", 0))
    r_spread = metrics.get("range_spread_m", np.nan)
    v_spread = metrics.get("velocity_spread_mps", np.nan)

    if not detected_raw:
        return (
            "EMPTY",
            f"no moving component; peak={peak_db:.1f}dB, top10={top10_db:.1f}dB",
        )

    compact_drone_shape = (
        bins <= DRONE_MAX_ACTIVE_BINS
        and (not np.isfinite(r_spread) or r_spread <= DRONE_MAX_RANGE_SPREAD_M)
        and (not np.isfinite(v_spread) or v_spread <= DRONE_MAX_VELOCITY_SPREAD_MPS)
    )

    compact_hover_shape = (
        bins <= DRONE_HOVER_MAX_ACTIVE_BINS
        and (not np.isfinite(r_spread) or r_spread <= DRONE_HOVER_MAX_RANGE_SPREAD_M)
        and (not np.isfinite(v_spread) or v_spread <= DRONE_HOVER_MAX_VELOCITY_SPREAD_MPS)
    )

    # Strong empty only when the heatmap is weak AND it does not look like a compact drone return.
    if (
        np.isfinite(peak_db)
        and np.isfinite(top10_db)
        and peak_db < EMPTY_MAX_PEAK_DB
        and top10_db < EMPTY_MAX_TOP10_DB
    ):
        # Hover drone exception:
        # Tello can be compact and only slightly above empty brightness.
        if (
            compact_hover_shape
            and peak_db >= DRONE_HOVER_MIN_PEAK_DB
            and top10_db >= DRONE_HOVER_MIN_TOP10_DB
        ):
            return (
                "DRONE",
                f"hover/slow compact drone: peak={peak_db:.1f}dB, top10={top10_db:.1f}dB, "
                f"bins={bins}, Rspread={r_spread:.2f}m, Vspread={v_spread:.2f}m/s, "
                f"E/noise={e_db:.1f}dB",
            )

        return (
            "EMPTY",
            f"low heatmap power: peak={peak_db:.1f}dB<{EMPTY_MAX_PEAK_DB:.1f}, "
            f"top10={top10_db:.1f}dB<{EMPTY_MAX_TOP10_DB:.1f}",
        )

    in_drone_power_band = (
        np.isfinite(peak_db)
        and DRONE_MIN_PEAK_DB <= peak_db <= DRONE_MAX_PEAK_DB
        and np.isfinite(top10_db)
        and DRONE_MIN_TOP10_DB <= top10_db <= DRONE_MAX_TOP10_DB
    )

    # Drone first: compact bright blobs are drone even if peak is high.
    if compact_drone_shape and in_drone_power_band:
        return (
            "DRONE",
            f"compact blob first: peak={peak_db:.1f}dB, top10={top10_db:.1f}dB, "
            f"bins={bins}, Rspread={r_spread:.2f}m, Vspread={v_spread:.2f}m/s, "
            f"E/noise={e_db:.1f}dB",
        )

    # Drone hover / slow-motion fallback:
    # If compact and above hover brightness, accept as drone.
    if (
        compact_hover_shape
        and peak_db >= DRONE_HOVER_MIN_PEAK_DB
        and top10_db >= DRONE_HOVER_MIN_TOP10_DB
    ):
        return (
            "DRONE",
            f"hover/slow compact drone: peak={peak_db:.1f}dB, top10={top10_db:.1f}dB, "
            f"bins={bins}, Rspread={r_spread:.2f}m, Vspread={v_spread:.2f}m/s, "
            f"E/noise={e_db:.1f}dB",
        )

    large_human_shape = (
        bins >= HUMAN_MIN_ACTIVE_BINS
        or (np.isfinite(r_spread) and r_spread >= HUMAN_MIN_RANGE_SPREAD_M)
        or (np.isfinite(v_spread) and v_spread >= HUMAN_MIN_VELOCITY_SPREAD_MPS)
    )

    bright_human_power = (
        (np.isfinite(peak_db) and peak_db >= HUMAN_MIN_PEAK_DB)
        or (np.isfinite(top10_db) and top10_db >= HUMAN_MIN_TOP10_DB)
    )

    # Human now needs wide/large evidence, not just brightness.
    if bright_human_power and large_human_shape:
        return (
            "HUMAN",
            f"bright AND wide blob: peak={peak_db:.1f}dB, top10={top10_db:.1f}dB, "
            f"bins={bins}, Rspread={r_spread:.2f}m, Vspread={v_spread:.2f}m/s, "
            f"E/noise={e_db:.1f}dB",
        )

    if large_human_shape:
        return (
            "HUMAN",
            f"wide blob: peak={peak_db:.1f}dB, top10={top10_db:.1f}dB, "
            f"bins={bins}, Rspread={r_spread:.2f}m, Vspread={v_spread:.2f}m/s, "
            f"E/noise={e_db:.1f}dB",
        )

    if compact_drone_shape:
        return (
            "DRONE",
            f"compact moving blob: peak={peak_db:.1f}dB, top10={top10_db:.1f}dB, "
            f"bins={bins}, Rspread={r_spread:.2f}m, Vspread={v_spread:.2f}m/s, "
            f"E/noise={e_db:.1f}dB",
        )

    return (
        "UNKNOWN_MOVING",
        f"moving but unmatched: peak={peak_db:.1f}dB, top10={top10_db:.1f}dB, "
        f"bins={bins}, Rspread={r_spread:.2f}m, Vspread={v_spread:.2f}m/s, "
        f"E/noise={e_db:.1f}dB",
    )


def process_frame_human_detector(frame_bytes: bytes, frame_idx: int) -> HumanDetectorFrame:
    global bg_learning, bg_frames, background_rd_power

    adc = raw_frame_to_adc(frame_bytes)
    rd_power = compute_range_doppler_power(adc)

    # -----------------------------
    # Background learning
    # -----------------------------
    if bg_learning:
        bg_frames.append(rd_power)

        rd_db = 10.0 * np.log10(rd_power + 1e-12)

        if len(bg_frames) >= BACKGROUND_CALIBRATION_FRAMES:
            stack = np.stack(bg_frames, axis=0)

            if BACKGROUND_METHOD == "median":
                background_rd_power = np.median(stack, axis=0)
            elif BACKGROUND_METHOD == "mean":
                background_rd_power = np.mean(stack, axis=0)
            else:
                raise ValueError(f"Unknown BACKGROUND_METHOD: {BACKGROUND_METHOD}")

            bg_learning = False
            bg_frames = []

            print()
            print("=== Background ready ===")
            print("Human detector is now active.")
            print()

        peak_db, top10_db = compute_heatmap_db_features(rd_db)

        return HumanDetectorFrame(
            frame_idx=frame_idx,
            t_sec=frame_idx * FRAME_PERIOD_S,
            rd_db=rd_db,
            detected_raw=False,
            confirmed=False,
            confidence=0.0,
            range_m=np.nan,
            velocity_mps=np.nan,
            range_spread_m=np.nan,
            velocity_spread_mps=np.nan,
            energy_db_over_noise=np.nan,
            peak_db=peak_db,
            top10_db=top10_db,
            active_bins=0,
            event_label="LEARNING_BG",
            event_reason="collecting empty-scene background",
            status=f"Learning background {len(bg_frames)}/{BACKGROUND_CALIBRATION_FRAMES}",
        )

    # -----------------------------
    # Background subtraction
    # -----------------------------
    if background_rd_power is not None:
        rd_sub = rd_power - BACKGROUND_SCALE * background_rd_power
        rd_sub = np.maximum(rd_sub, 0.0)
    else:
        rd_sub = rd_power

    threshold, noise_median, noise_mad = robust_noise_threshold(rd_sub)

    detected_raw, metrics = analyze_components(
        power_sub=rd_sub,
        threshold=threshold,
        noise_median=max(noise_median, 1e-12),
    )

    confirmed, track_range, track_vel, track_conf = tracker.update(
        detected=detected_raw,
        range_m=metrics["range_m"],
        velocity_mps=metrics["velocity_mps"],
        confidence=metrics["confidence"],
    )

    rd_db = 10.0 * np.log10(rd_sub + 1e-12)
    peak_db, top10_db = compute_heatmap_db_features(rd_db)

    event_label, event_reason = classify_event_from_db_features(
        detected_raw=detected_raw,
        metrics=metrics,
        peak_db=peak_db,
        top10_db=top10_db,
    )
    event_label_smoothed = event_smoother.update(event_label)

    if confirmed:
        status = f"CONFIRMED {event_label_smoothed}"
        display_range = track_range
        display_vel = track_vel
        display_conf = track_conf
    elif detected_raw:
        status = f"Tentative {event_label_smoothed}"
        display_range = metrics["range_m"]
        display_vel = metrics["velocity_mps"]
        display_conf = metrics["confidence"]
    else:
        status = event_label_smoothed
        display_range = np.nan
        display_vel = np.nan
        display_conf = 0.0

    return HumanDetectorFrame(
        frame_idx=frame_idx,
        t_sec=frame_idx * FRAME_PERIOD_S,
        rd_db=rd_db,
        detected_raw=detected_raw,
        confirmed=confirmed,
        confidence=float(display_conf),
        range_m=float(display_range),
        velocity_mps=float(display_vel),
        range_spread_m=float(metrics["range_spread_m"]),
        velocity_spread_mps=float(metrics["velocity_spread_mps"]),
        energy_db_over_noise=float(metrics["energy_db_over_noise"]),
        peak_db=float(peak_db),
        top10_db=float(top10_db),
        active_bins=int(metrics["active_bins"]),
        event_label=event_label_smoothed,
        event_reason=event_reason,
        status=status,
    )


# ============================================================
# UDP worker
# ============================================================

def udp_receiver_worker():
    print("=== DCA1000 live event classifier V4: drone hover sensitive ===")
    print(f"Bind: {PC_IP}:{DATA_PORT}")
    print(f"DCA IP: {DCA_IP}")
    print(f"FRAME_BYTES: {FRAME_BYTES}")
    print(f"EXPECTED_BYTES: {EXPECTED_BYTES}")
    print()
    print("Detector ROI:")
    print(f"  range: {DETECT_RANGE_MIN_M:.2f} ~ {DETECT_RANGE_MAX_M:.2f} m")
    print(f"  |velocity|: {MIN_ABS_VELOCITY_MPS:.2f} ~ {MAX_ABS_VELOCITY_MPS:.2f} m/s")
    print()
    print("Background subtraction:")
    print(f"  calibration frames: {BACKGROUND_CALIBRATION_FRAMES}")
    print(f"  method: {BACKGROUND_METHOD}")
    print(f"  scale: {BACKGROUND_SCALE}")
    print()
    print("Fixed heatmap magnitude bar:")
    print(f"  vmin = {FIXED_HEATMAP_VMIN_DB:.1f} dB")
    print(f"  vmax = {FIXED_HEATMAP_VMAX_DB:.1f} dB")
    print()
    print("Event classification thresholds:")
    print("  Main thresholds use absolute heatmap dB + blob shape.")
    print("  Drone-priority + hover-sensitive logic is enabled.")
    print(f"  Doppler gate: |v| >= {MIN_ABS_VELOCITY_MPS:.2f} m/s")
    print(f"  EMPTY: peak < {EMPTY_MAX_PEAK_DB:.1f} dB and top10 < {EMPTY_MAX_TOP10_DB:.1f} dB")
    print(f"  DRONE hover: compact, peak >= {DRONE_HOVER_MIN_PEAK_DB:.1f} dB, top10 >= {DRONE_HOVER_MIN_TOP10_DB:.1f} dB")
    print(f"  DRONE moving: compact, peak {DRONE_MIN_PEAK_DB:.1f}~{DRONE_MAX_PEAK_DB:.1f} dB")
    print(f"  HUMAN: bright AND wide, or clearly wide/large blob")
    print(f"         HUMAN_MIN_ACTIVE_BINS={HUMAN_MIN_ACTIVE_BINS}, Rspread>={HUMAN_MIN_RANGE_SPREAD_M:.2f}m")
    print()
    print("IMPORTANT:")
    print(f"  Keep scene empty for first {BACKGROUND_CALIBRATION_FRAMES} frames")
    print(f"  = about {BACKGROUND_CALIBRATION_FRAMES * FRAME_PERIOD_S:.1f} seconds.")
    print()
    print("Run order:")
    print("  1. Run this script.")
    print("  2. Wait until it says Listening for DCA1000 packets.")
    print("  3. In mmWave Studio: StartRecord.")
    print("  4. Trigger StartFrame.")
    print()

    if WRITE_RAW_BACKUP:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUT_PATH.open("wb") as f:
            f.truncate(EXPECTED_BYTES)
        backup_f = OUT_PATH.open("r+b")
        print(f"Raw backup enabled: {OUT_PATH}")
    else:
        backup_f = None

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_RCVBUF)
    sock.bind((PC_IP, DATA_PORT))
    sock.settimeout(0.2)

    actual_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print(f"Actual SO_RCVBUF = {actual_buf}")
    print("Listening for DCA1000 packets...")
    print()

    stream_buffer = bytearray()

    packet_count = 0
    raw_byte_count = 0
    frame_count = 0

    first_seq = None
    last_seq = None
    gap_count = 0

    t0 = time.time()
    last_report = t0

    try:
        while not stop_event.is_set():
            if frame_count >= NUM_FRAMES:
                print("Receiver: expected frame count reached.")
                break

            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue

            src_ip, src_port = addr
            if src_ip != DCA_IP:
                continue

            parsed = parse_dca_packet(data)
            if parsed is None:
                continue

            seq, byte_count, raw = parsed

            if first_seq is None:
                first_seq = seq
                print(
                    f"First packet: seq={seq}, "
                    f"byte_count={byte_count}, raw_len={len(raw)}"
                )

            if last_seq is not None and seq != last_seq + 1:
                gap = seq - last_seq - 1
                if gap > 0:
                    gap_count += gap
                    print(
                        f"WARNING packet gap: "
                        f"last_seq={last_seq}, seq={seq}, missing={gap}"
                    )

            last_seq = seq
            packet_count += 1
            raw_byte_count += len(raw)

            if backup_f is not None and byte_count < EXPECTED_BYTES:
                remaining = EXPECTED_BYTES - byte_count
                backup_f.seek(byte_count)
                backup_f.write(raw[:remaining])

            # For live processing, assume continuous ordered stream.
            # Your previous tests showed no missing packets for stable 60 s capture.
            stream_buffer.extend(raw)

            while len(stream_buffer) >= FRAME_BYTES:
                frame_bytes = bytes(stream_buffer[:FRAME_BYTES])
                del stream_buffer[:FRAME_BYTES]

                frame_idx = frame_count
                frame_count += 1

                try:
                    result = process_frame_human_detector(frame_bytes, frame_idx)
                    try:
                        result_queue.put_nowait(result)
                    except queue.Full:
                        print("WARNING result_queue full; dropping display result.")
                except Exception as e:
                    print(f"ERROR processing frame {frame_idx}: {e}")

            now = time.time()
            if now - last_report >= 2.0:
                elapsed = now - t0
                mbps = (raw_byte_count * 8 / 1e6) / max(elapsed, 1e-9)

                print(
                    f"rx: pkts={packet_count:7d}, "
                    f"seq={seq:7d}, "
                    f"raw_MB={raw_byte_count/1e6:8.2f}, "
                    f"frames={frame_count:4d}/{NUM_FRAMES}, "
                    f"gaps={gap_count:4d}, "
                    f"rate={mbps:5.1f} Mbps"
                )

                last_report = now

    finally:
        sock.close()

        if backup_f is not None:
            backup_f.flush()
            backup_f.close()

        print()
        print("=== UDP receiver worker stopped ===")
        print(f"Packets       : {packet_count}")
        print(f"First seq     : {first_seq}")
        print(f"Last seq      : {last_seq}")
        print(f"Gaps          : {gap_count}")
        print(f"Raw bytes     : {raw_byte_count}")
        print(f"Frames made   : {frame_count}/{NUM_FRAMES}")

        if WRITE_RAW_BACKUP:
            print(f"Raw backup    : {OUT_PATH}")
            print(f"Backup size   : {OUT_PATH.stat().st_size}")


# ============================================================
# Plot
# ============================================================

def run_plot():
    plt.ion()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    init_heat = np.zeros((len(RANGE_IDXS), len(DOPPLER_IDXS)))

    im = ax.imshow(
        init_heat,
        aspect="auto",
        origin="lower",
        extent=[
            VALID_DOPPLER_AXIS[0],
            VALID_DOPPLER_AXIS[-1],
            VALID_RANGE_AXIS[0],
            VALID_RANGE_AXIS[-1],
        ],
        vmin=FIXED_HEATMAP_VMIN_DB,
        vmax=FIXED_HEATMAP_VMAX_DB,
    )

    ax.set_xlabel("Radial velocity (m/s)")
    ax.set_ylabel("Range (m)")
    ax.set_title("Event Classifier V4: drone hover sensitive classifier")

    det_marker, = ax.plot([], [], "o", markersize=10, fillstyle="none", markeredgewidth=2)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Background-subtracted power (dB)")

    status_text = fig.text(0.02, 0.02, "Waiting for frames...", fontsize=10)
    fig.text(
        0.02,
        0.955,
        "Keyboard: 'b' reset background | '['/']' shift bar | '-/=' widen/narrow bar",
        fontsize=9,
    )

    latest_result = {"value": None}

    heatbar = {
        "vmin": float(FIXED_HEATMAP_VMIN_DB),
        "vmax": float(FIXED_HEATMAP_VMAX_DB),
    }

    def on_key(event):
        if event.key == "b":
            reset_background()
            return

        # Shift fixed bar down/up by 5 dB.
        if event.key == "[":
            heatbar["vmin"] -= 5.0
            heatbar["vmax"] -= 5.0
        elif event.key == "]":
            heatbar["vmin"] += 5.0
            heatbar["vmax"] += 5.0

        # Widen/narrow the fixed display range around its center.
        elif event.key == "-":
            center = 0.5 * (heatbar["vmin"] + heatbar["vmax"])
            half = 0.5 * (heatbar["vmax"] - heatbar["vmin"]) + 5.0
            heatbar["vmin"] = center - half
            heatbar["vmax"] = center + half
        elif event.key in ("=", "+"):
            center = 0.5 * (heatbar["vmin"] + heatbar["vmax"])
            half = max(5.0, 0.5 * (heatbar["vmax"] - heatbar["vmin"]) - 5.0)
            heatbar["vmin"] = center - half
            heatbar["vmax"] = center + half
        else:
            return

        print(
            f"Fixed heatmap bar: "
            f"{heatbar['vmin']:.1f} dB ~ {heatbar['vmax']:.1f} dB"
        )

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        while True:
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                break

            latest_result["value"] = result

        if latest_result["value"] is not None:
            r = latest_result["value"]

            heat = r.rd_db
            im.set_data(heat)

            # Fixed magnitude bar:
            # Do NOT auto-scale by current heatmap max.
            # This makes frame-to-frame intensity visually meaningful.
            im.set_clim(heatbar["vmin"], heatbar["vmax"])

            if np.isfinite(r.range_m) and np.isfinite(r.velocity_mps):
                det_marker.set_data([r.velocity_mps], [r.range_m])
            else:
                det_marker.set_data([], [])

            if r.confirmed:
                state = "YES"
            elif r.detected_raw:
                state = "tentative"
            else:
                state = "no"

            status_text.set_text(
                f"Frame {r.frame_idx:03d}/{NUM_FRAMES - 1}, "
                f"t={r.t_sec:.1f}s | "
                f"event={r.event_label} | "
                f"confirmed={state} | "
                f"range={r.range_m:.2f} m | "
                f"v={r.velocity_mps:.2f} m/s | "
                f"conf={r.confidence:.2f} | "
                f"peak={r.peak_db:.1f}dB | "
                f"top10={r.top10_db:.1f}dB | "
                f"E/noise={r.energy_db_over_noise:.1f}dB | "
                f"bins={r.active_bins} | "
                f"bar={heatbar['vmin']:.0f}~{heatbar['vmax']:.0f}dB | "
                f"{r.event_reason}"
            )

        return im, det_marker, status_text

    ani = FuncAnimation(fig, update, interval=PLOT_UPDATE_INTERVAL_MS, blit=False)

    try:
        plt.show(block=True)
    finally:
        stop_event.set()


def main():
    worker = threading.Thread(target=udp_receiver_worker, daemon=True)
    worker.start()

    run_plot()

    stop_event.set()
    worker.join(timeout=2.0)


if __name__ == "__main__":
    main()
