import csv
import time
from pathlib import Path
from datetime import datetime

import numpy as np


# ====== Paths ======
ADC_DIR = Path(r"C:\\Users\\user\\Desktop\\ADPAR-final-project\\adc")
EMPTY_PATH = Path(r"C:\\Users\\user\\Desktop\\ADPAR-final-project\\adc\\empty.bin")

DECISION_PATH = ADC_DIR / "latest_decision.txt"
CSV_PATH = ADC_DIR / "decisions.csv"


# ====== Radar config: match your mmWave Studio settings ======
NUM_FRAMES = 8
NUM_CHIRPS = 128
NUM_SAMPLES = 256
NUM_RX = 4

SAMPLE_RATE_HZ = 10e6          # 10000 ksps
SLOPE_HZ_PER_S = 29.982e12     # 29.982 MHz/us
C = 299_792_458.0

EXPECTED_BYTES = NUM_FRAMES * NUM_CHIRPS * NUM_SAMPLES * NUM_RX * 2 * 2
EXPECTED_WORDS = EXPECTED_BYTES // 2


def wait_until_file_stable(path: Path, stable_checks: int = 3, interval_s: float = 0.25, timeout_s: float = 20.0) -> bool:
    """
    Wait until file size stops changing.
    This avoids reading while DCA1000/mmWave Studio is still writing.
    """
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
        raise ValueError(f"{path.name} too small: words={raw.size}, expected={EXPECTED_WORDS}")

    raw = raw[:EXPECTED_WORDS]

    # DCA1000 xWR22xx complex 4RX 4-lane order:
    # Rx0I Rx1I Rx2I Rx3I Rx0Q Rx1Q Rx2Q Rx3Q per ADC sample
    data = raw.reshape(NUM_FRAMES, NUM_CHIRPS, NUM_SAMPLES, 2, NUM_RX)

    i_data = data[:, :, :, 0, :].astype(np.float32)
    q_data = data[:, :, :, 1, :].astype(np.float32)

    adc = i_data + 1j * q_data
    # shape = [frame, chirp, sample, rx]
    return adc


def range_axis_m() -> np.ndarray:
    freqs = np.fft.fftfreq(NUM_SAMPLES, d=1.0 / SAMPLE_RATE_HZ)[: NUM_SAMPLES // 2]
    ranges = C * freqs / (2.0 * SLOPE_HZ_PER_S)
    return ranges


def range_doppler_db(adc: np.ndarray, rx_idx: int = 0) -> np.ndarray:
    """
    Average range-Doppler magnitude across frames.
    Output shape: [doppler_bin, range_bin]
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

    return np.mean(np.stack(rd_frames, axis=0), axis=0)


def motion_score_from_diff(rd_db: np.ndarray, baseline_db: np.ndarray) -> float:
    """
    Simple motion score:
    Compare moving file against empty baseline.
    Ignore zero-Doppler clutter and focus on near/mid range.
    """
    diff = rd_db - baseline_db

    ranges = range_axis_m()
    doppler_bins = np.arange(-NUM_CHIRPS // 2, NUM_CHIRPS // 2)

    # Focus on useful range. Adjust later for your setup.
    range_mask = (ranges >= 0.4) & (ranges <= 6.0)

    # Ignore center Doppler bins dominated by static clutter.
    doppler_mask = np.abs(doppler_bins) >= 3

    roi = diff[np.ix_(doppler_mask, range_mask)]

    # Robust high-percentile score instead of raw max.
    return float(np.percentile(roi, 99.5))


def classify_score(score: float) -> str:
    """
    First-pass threshold.
    You will tune this after collecting more data.
    """
    if score >= 12.0:
        return "MOVING"
    return "EMPTY_OR_STATIC"


def write_decision(scan_name: str, score: float, decision: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")

    DECISION_PATH.write_text(
        f"{decision}\nscan={scan_name}\nscore={score:.2f}\ntime={timestamp}\n",
        encoding="utf-8",
    )

    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["time", "scan", "score_db", "decision"])
        writer.writerow([timestamp, scan_name, f"{score:.2f}", decision])


def main() -> None:
    ADC_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading empty baseline:", EMPTY_PATH)
    empty_adc = read_xwr22xx_complex_4rx(EMPTY_PATH)
    baseline_db = range_doppler_db(empty_adc, rx_idx=0)

    print("Baseline loaded.")
    print("Watching:", ADC_DIR)

    seen = set()

    while True:
        files = sorted(ADC_DIR.glob("cl_*_Raw_0.bin"))

        for path in files:
            if path in seen:
                continue

            print(f"\nNew candidate: {path.name}")

            if not wait_until_file_stable(path):
                print(f"WARNING: file not stable, skip for now: {path.name}")
                continue

            size = path.stat().st_size
            if size != EXPECTED_BYTES:
                print(f"WARNING: unexpected size {size}, expected {EXPECTED_BYTES}, skip: {path.name}")
                seen.add(path)
                continue

            try:
                adc = read_xwr22xx_complex_4rx(path)
                rd_db = range_doppler_db(adc, rx_idx=0)
                score = motion_score_from_diff(rd_db, baseline_db)
                decision = classify_score(score)

                print(f"{path.name}: score={score:.2f} dB, decision={decision}")
                write_decision(path.name, score, decision)

            except Exception as e:
                print(f"ERROR analyzing {path.name}: {e}")

            seen.add(path)

        time.sleep(0.5)


if __name__ == "__main__":
    main()