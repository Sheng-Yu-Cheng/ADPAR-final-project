import csv
import time
from pathlib import Path
from datetime import datetime

import numpy as np


# ====== Paths ======
ADC_DIR = Path(r"C:\\Users\\user\\Desktop\\ADPAR-final-project\\adc")
EMPTY_PATH = ADC_DIR / "manual_64f_Raw_0.bin"

DECISION_PATH = ADC_DIR / "latest_decision.txt"
CSV_PATH = ADC_DIR / "decisions_64f.csv"


# ====== Radar config: must match mmWave Studio settings ======
NUM_FRAMES = 64
NUM_CHIRPS = 128
NUM_SAMPLES = 256
NUM_RX = 4

SAMPLE_RATE_HZ = 10e6          # 10000 ksps
SLOPE_HZ_PER_S = 29.982e12     # 29.982 MHz/us
C = 299_792_458.0

EXPECTED_BYTES = NUM_FRAMES * NUM_CHIRPS * NUM_SAMPLES * NUM_RX * 2 * 2
EXPECTED_WORDS = EXPECTED_BYTES // 2


def wait_until_file_stable(
    path: Path,
    stable_checks: int = 4,
    interval_s: float = 0.5,
    timeout_s: float = 60.0,
) -> bool:
    start = time.time()
    last_size = -1
    stable_count = 0

    while time.time() - start < timeout_s:
        if path.exists():
            size = path.stat().st_size
            if size == last_size and size > 0:
                stable_count += 1
                if stable_count >= stable_checks:
                    return True
            else:
                stable_count = 0
                last_size = size

        time.sleep(interval_s)

    return False


def read_xwr22xx_complex_4rx(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int16)

    if raw.size < EXPECTED_WORDS:
        raise ValueError(
            f"{path.name} too small: words={raw.size}, expected={EXPECTED_WORDS}"
        )

    raw = raw[:EXPECTED_WORDS]

    # xWR22xx complex 4RX 4-lane order:
    # Rx0I Rx1I Rx2I Rx3I Rx0Q Rx1Q Rx2Q Rx3Q
    data = raw.reshape(NUM_FRAMES, NUM_CHIRPS, NUM_SAMPLES, 2, NUM_RX)

    i_data = data[:, :, :, 0, :].astype(np.float32)
    q_data = data[:, :, :, 1, :].astype(np.float32)

    return i_data + 1j * q_data
    # shape: [frame, chirp, sample, rx]


def range_axis_m() -> np.ndarray:
    freqs = np.fft.fftfreq(NUM_SAMPLES, d=1.0 / SAMPLE_RATE_HZ)[: NUM_SAMPLES // 2]
    return C * freqs / (2.0 * SLOPE_HZ_PER_S)


def range_doppler_db_per_frame(adc: np.ndarray, rx_idx: int = 0) -> np.ndarray:
    """
    Return RD map for every frame.
    Shape: [frame, doppler_bin, range_bin]
    """
    range_win = np.hanning(NUM_SAMPLES)[None, :]
    dop_win = np.hanning(NUM_CHIRPS)[:, None]

    rd_frames = []

    for frame_idx in range(NUM_FRAMES):
        x = adc[frame_idx, :, :, rx_idx]  # [chirp, sample]

        range_fft = np.fft.fft(x * range_win, axis=1)
        range_fft = range_fft[:, : NUM_SAMPLES // 2]

        rd = np.fft.fftshift(np.fft.fft(range_fft * dop_win, axis=0), axes=0)
        rd_db = 20.0 * np.log10(np.abs(rd) + 1e-12)

        rd_frames.append(rd_db)

    return np.stack(rd_frames, axis=0)


def motion_score_from_diff(rd_db_frames: np.ndarray, baseline_db_frames: np.ndarray) -> float:
    """
    Simple moving-vs-static score.
    Ignores zero-Doppler clutter and focuses on 0.4m ~ 6m.
    """
    rd_db = np.mean(rd_db_frames, axis=0)
    baseline_db = np.mean(baseline_db_frames, axis=0)

    diff = rd_db - baseline_db

    ranges = range_axis_m()
    doppler_bins = np.arange(-NUM_CHIRPS // 2, NUM_CHIRPS // 2)

    range_mask = (ranges >= 0.4) & (ranges <= 6.0)
    doppler_mask = np.abs(doppler_bins) >= 3

    roi = diff[np.ix_(doppler_mask, range_mask)]

    # High percentile avoids one-pixel noise spikes.
    return float(np.percentile(roi, 99.5))


def micro_doppler_energy_feature(rd_db_frames: np.ndarray) -> dict:
    """
    Very early feature extraction for later zombie-hop vs walk classification.
    Not final, but useful for logging.
    """
    ranges = range_axis_m()
    doppler_bins = np.arange(-NUM_CHIRPS // 2, NUM_CHIRPS // 2)

    range_mask = (ranges >= 0.4) & (ranges <= 6.0)
    moving_mask = np.abs(doppler_bins) >= 3

    # Convert dB to linear-ish magnitude proxy.
    rd_lin = 10 ** (rd_db_frames / 20.0)

    roi = rd_lin[:, :, range_mask]          # [frame, doppler, range]
    moving_roi = roi[:, moving_mask, :]     # remove static Doppler

    energy_per_frame = np.sum(moving_roi, axis=(1, 2))

    return {
        "energy_mean": float(np.mean(energy_per_frame)),
        "energy_std": float(np.std(energy_per_frame)),
        "energy_peak": float(np.max(energy_per_frame)),
    }


def classify_motion(score: float) -> str:
    # For 64-frame data, tune this using your new CSV.
    if score >= 15.0:
        return "MOVING"
    return "EMPTY_OR_STATIC"


def write_decision(scan_name: str, score: float, decision: str, features: dict) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")

    DECISION_PATH.write_text(
        (
            f"{decision}\n"
            f"scan={scan_name}\n"
            f"score={score:.2f}\n"
            f"time={timestamp}\n"
            f"energy_mean={features['energy_mean']:.3e}\n"
            f"energy_std={features['energy_std']:.3e}\n"
            f"energy_peak={features['energy_peak']:.3e}\n"
        ),
        encoding="utf-8",
    )

    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "time",
                "scan",
                "score_db",
                "decision",
                "energy_mean",
                "energy_std",
                "energy_peak",
            ])
        writer.writerow([
            timestamp,
            scan_name,
            f"{score:.2f}",
            decision,
            f"{features['energy_mean']:.6e}",
            f"{features['energy_std']:.6e}",
            f"{features['energy_peak']:.6e}",
        ])


def main() -> None:
    print("Expected bytes per 64-frame capture:", EXPECTED_BYTES)

    if not EMPTY_PATH.exists():
        raise FileNotFoundError(
            f"Missing baseline file: {EMPTY_PATH}\n"
            "Record or rename a 64-frame empty scene capture to empty64_Raw_0.bin."
        )

    print("Loading empty baseline:", EMPTY_PATH)
    empty_adc = read_xwr22xx_complex_4rx(EMPTY_PATH)
    baseline_rd_frames = range_doppler_db_per_frame(empty_adc, rx_idx=0)
    print("Baseline loaded.")

    print("Watching for cl64_*_Raw_0.bin in:", ADC_DIR)

    seen = set()

    while True:
        files = sorted(ADC_DIR.glob("cl64_*_Raw_0.bin"))

        for path in files:
            if path in seen:
                continue

            print(f"\nNew candidate: {path.name}")

            if not wait_until_file_stable(path):
                print(f"WARNING: file not stable, skip for now: {path.name}")
                continue

            size = path.stat().st_size
            if size != EXPECTED_BYTES:
                print(
                    f"WARNING: unexpected size {size}, expected {EXPECTED_BYTES}, skip: {path.name}"
                )
                seen.add(path)
                continue

            try:
                adc = read_xwr22xx_complex_4rx(path)
                rd_frames = range_doppler_db_per_frame(adc, rx_idx=0)

                score = motion_score_from_diff(rd_frames, baseline_rd_frames)
                decision = classify_motion(score)
                features = micro_doppler_energy_feature(rd_frames)

                print(
                    f"{path.name}: score={score:.2f} dB, "
                    f"decision={decision}, "
                    f"energy_std={features['energy_std']:.3e}"
                )

                write_decision(path.name, score, decision, features)

            except Exception as e:
                print(f"ERROR analyzing {path.name}: {e}")

            seen.add(path)

        time.sleep(0.5)


if __name__ == "__main__":
    main()