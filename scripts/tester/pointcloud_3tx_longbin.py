from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# =========================
# User config
# =========================

BIN_PATH = Path(r"C:\Users\user\Desktop\ADPAR-final-project\adc\socket_capture_Raw_0.bin")

NUM_FRAMES = 25
NUM_LOOPS = 64
NUM_TX = 3
NUM_RX = 4
NUM_SAMPLES = 256

SAMPLE_RATE_HZ = 10e6
SLOPE_HZ_PER_S = 29.982e12
START_FREQ_HZ = 77e9
C = 299_792_458.0

FRAME_PERIOD_S = 0.200

EXPECTED_BYTES = NUM_FRAMES * NUM_LOOPS * NUM_TX * NUM_SAMPLES * NUM_RX * 2 * 2
EXPECTED_WORDS = EXPECTED_BYTES // 2


# =========================
# Basic axes
# =========================

def range_axis_m():
    freqs = np.fft.fftfreq(NUM_SAMPLES, d=1.0 / SAMPLE_RATE_HZ)[: NUM_SAMPLES // 2]
    return C * freqs / (2.0 * SLOPE_HZ_PER_S)


def doppler_axis_mps():
    """
    Approximate Doppler axis for TDM-MIMO.
    One Doppler slow-time sample is one full Tx0/Tx1/Tx2 cycle.
    """
    lambda_m = C / START_FREQ_HZ

    chirp_time_s = (100e-6 + 60e-6)  # from your current profile
    tdm_cycle_s = NUM_TX * chirp_time_s

    fd = np.fft.fftshift(np.fft.fftfreq(NUM_LOOPS, d=tdm_cycle_s))
    v = fd * lambda_m / 2.0
    return v


# =========================
# Reader
# =========================

def read_3tx_longbin(path: Path):
    raw = np.fromfile(path, dtype=np.int16)

    print("Raw int16 words:", raw.size)
    print("Expected words:", EXPECTED_WORDS)
    print("File bytes:", path.stat().st_size)
    print("Expected bytes:", EXPECTED_BYTES)

    if raw.size < EXPECTED_WORDS:
        raise ValueError("File is smaller than expected. Check FrameConfig or capture completion.")

    raw = raw[:EXPECTED_WORDS]

    # Per ADC sample order:
    # Rx0I Rx1I Rx2I Rx3I Rx0Q Rx1Q Rx2Q Rx3Q
    data = raw.reshape(NUM_FRAMES, NUM_LOOPS * NUM_TX, NUM_SAMPLES, 2, NUM_RX)

    i_data = data[:, :, :, 0, :].astype(np.float32)
    q_data = data[:, :, :, 1, :].astype(np.float32)

    adc = i_data + 1j * q_data
    # shape: [frame, chirp_index, sample, rx]

    # chirp_index order:
    # 0: Tx0 loop0
    # 1: Tx1 loop0
    # 2: Tx2 loop0
    # 3: Tx0 loop1
    # ...
    adc = adc.reshape(NUM_FRAMES, NUM_LOOPS, NUM_TX, NUM_SAMPLES, NUM_RX)

    return adc
    # shape: [frame, loop, tx, sample, rx]


# =========================
# Range-Doppler processing
# =========================

def range_doppler_for_virtual_channel(adc_frame, tx_idx=0, rx_idx=0):
    """
    adc_frame shape: [loop, tx, sample, rx]
    Return RD map: [doppler_bin, range_bin]
    """
    x = adc_frame[:, tx_idx, :, rx_idx]  # [loop, sample]

    range_win = np.hanning(NUM_SAMPLES)[None, :]
    dop_win = np.hanning(NUM_LOOPS)[:, None]

    rfft = np.fft.fft(x * range_win, axis=1)
    rfft = rfft[:, : NUM_SAMPLES // 2]

    rd = np.fft.fftshift(np.fft.fft(rfft * dop_win, axis=0), axes=0)
    rd_db = 20 * np.log10(np.abs(rd) + 1e-12)

    return rd_db


def simple_cfar_like_detections(rd_db, max_points=20):
    """
    Not real CFAR yet.
    First-pass peak picker:
    - remove near-zero Doppler clutter
    - ignore too-near range bins
    - pick top peaks above threshold
    """
    ranges = range_axis_m()
    velocities = doppler_axis_mps()

    rd = rd_db.copy()

    # Ignore near range and far range for first test
    range_mask = (ranges >= 0.4) & (ranges <= 8.0)

    # Ignore zero-Doppler clutter bins
    doppler_mask = np.abs(velocities) >= 0.15

    roi = rd[np.ix_(doppler_mask, range_mask)]

    if roi.size == 0:
        return []

    noise_floor = np.median(roi)
    threshold = noise_floor + 14.0

    candidates = np.argwhere(roi > threshold)

    points = []
    doppler_indices = np.where(doppler_mask)[0]
    range_indices = np.where(range_mask)[0]

    for local_dop_idx, local_rng_idx in candidates:
        d_idx = doppler_indices[local_dop_idx]
        r_idx = range_indices[local_rng_idx]
        mag = rd[d_idx, r_idx]

        points.append({
            "range_m": float(ranges[r_idx]),
            "velocity_mps": float(velocities[d_idx]),
            "mag_db": float(mag),
        })

    points.sort(key=lambda p: p["mag_db"], reverse=True)
    return points[:max_points]


def process_frame(adc, frame_idx):
    adc_frame = adc[frame_idx]  # [loop, tx, sample, rx]

    # First version: use Tx0-Rx0 channel for range-Doppler detection.
    rd_db = range_doppler_for_virtual_channel(adc_frame, tx_idx=0, rx_idx=0)

    points = simple_cfar_like_detections(rd_db)

    return rd_db, points


def range_fft_all_channels_one_frame(adc_frame):
    """
    adc_frame shape: [loop, tx, sample, rx]
    Return range FFT:
        shape [loop, tx, range_bin, rx]
    """
    range_win = np.hanning(NUM_SAMPLES)[None, None, :, None]
    x = adc_frame * range_win

    rfft = np.fft.fft(x, axis=2)
    rfft = rfft[:, :, : NUM_SAMPLES // 2, :]

    return rfft


def doppler_fft_all_channels_one_frame(rfft):
    """
    rfft shape: [loop, tx, range_bin, rx]
    Return RD cube:
        shape [doppler_bin, tx, range_bin, rx]
    """
    dop_win = np.hanning(NUM_LOOPS)[:, None, None, None]
    rd = np.fft.fftshift(np.fft.fft(rfft * dop_win, axis=0), axes=0)
    return rd


def find_static_target_range_bin(rd_cube, min_range_m=0.5, max_range_m=5.0):
    """
    For static metal calibration:
    Look at zero-Doppler bin and find strongest range bin.
    Use Tx0/Rx0 initially.
    """
    ranges = range_axis_m()
    zero_doppler_idx = NUM_LOOPS // 2

    mag = np.abs(rd_cube[zero_doppler_idx, 0, :, 0])
    mag_db = 20 * np.log10(mag + 1e-12)

    mask = (ranges >= min_range_m) & (ranges <= max_range_m)
    candidate_bins = np.where(mask)[0]

    best_local = np.argmax(mag_db[candidate_bins])
    best_bin = int(candidate_bins[best_local])

    return best_bin, float(ranges[best_bin]), float(mag_db[best_bin])


def tdm_phase_compensation(doppler_bin_shifted, tx_idx):
    """
    For static target doppler_bin_shifted = 0, correction = 1.
    Keep this function now so moving target AoA can use it later.

    doppler_bin_shifted ranges roughly:
        -NUM_LOOPS/2 ... +NUM_LOOPS/2-1
    """
    return np.exp(-1j * 2.0 * np.pi * doppler_bin_shifted * tx_idx / (NUM_LOOPS * NUM_TX))


def get_azimuth_virtual_vector_8ula(rd_cube, range_bin, doppler_bin_idx=None):
    """
    Build 8-element horizontal virtual array:
        Tx0 Rx0~Rx3 => x = 0,1,2,3
        Tx2 Rx0~Rx3 => x = 4,5,6,7

    rd_cube shape: [doppler_bin, tx, range_bin, rx]
    """
    if doppler_bin_idx is None:
        doppler_bin_idx = NUM_LOOPS // 2

    doppler_bin_shifted = doppler_bin_idx - NUM_LOOPS // 2

    # Tx0 row
    v_tx0 = rd_cube[doppler_bin_idx, 0, range_bin, :]  # 4 RX

    # Tx2 row, needs TDM phase compensation for moving targets.
    # For static calibration, this is 1.
    comp_tx2 = tdm_phase_compensation(doppler_bin_shifted, tx_idx=2)
    v_tx2 = rd_cube[doppler_bin_idx, 2, range_bin, :] * comp_tx2

    v8 = np.concatenate([v_tx0, v_tx2], axis=0)
    return v8


def estimate_azimuth_fft(v8, nfft=256):
    """
    v8 is an 8-element lambda/2 ULA.
    Angle FFT maps spatial frequency u = sin(theta).
    """
    # optional remove common magnitude imbalance lightly
    # v8 = v8 / (np.abs(v8) + 1e-12)

    spectrum = np.fft.fftshift(np.fft.fft(v8, n=nfft))
    power = np.abs(spectrum) ** 2

    bins = np.arange(-nfft // 2, nfft // 2)
    u = bins / (nfft / 2.0)  # u = sin(theta), valid [-1, 1]

    peak_idx = int(np.argmax(power))
    peak_u = float(np.clip(u[peak_idx], -1.0, 1.0))
    angle_deg = float(np.degrees(np.arcsin(peak_u)))

    return angle_deg, u, 10 * np.log10(power + 1e-12)


def static_angle_calibration(adc, frame_idx=0):
    """
    One-frame static AoA calibration:
    1. Compute RD cube for this frame
    2. Find strongest static target range bin
    3. Build Tx0+Tx2 8-element azimuth ULA
    4. Estimate azimuth
    """
    adc_frame = adc[frame_idx]  # [loop, tx, sample, rx]

    rfft = range_fft_all_channels_one_frame(adc_frame)
    rd_cube = doppler_fft_all_channels_one_frame(rfft)

    range_bin, range_m, mag_db = find_static_target_range_bin(
        rd_cube,
        min_range_m=0.5,
        max_range_m=5.0,
    )

    zero_doppler_idx = NUM_LOOPS // 2
    v8 = get_azimuth_virtual_vector_8ula(
        rd_cube,
        range_bin=range_bin,
        doppler_bin_idx=zero_doppler_idx,
    )

    angle_deg, u, angle_spectrum_db = estimate_azimuth_fft(v8, nfft=256)

    print("\n=== Static angle calibration ===")
    print(f"Frame index: {frame_idx}")
    print(f"Detected static target range bin: {range_bin}")
    print(f"Detected static target range: {range_m:.2f} m")
    print(f"Static target mag: {mag_db:.1f} dB")
    print(f"Estimated azimuth angle: {angle_deg:.2f} deg")
    print("v8 phase deg:", np.round(np.degrees(np.unwrap(np.angle(v8))), 1))

    return {
        "range_bin": range_bin,
        "range_m": range_m,
        "mag_db": mag_db,
        "angle_deg": angle_deg,
        "v8": v8,
        "u": u,
        "angle_spectrum_db": angle_spectrum_db,
    }

def range_fft_all_channels_one_frame(adc_frame):
    """
    adc_frame shape: [loop, tx, sample, rx]
    Return shape: [loop, tx, range_bin, rx]
    """
    range_win = np.hanning(NUM_SAMPLES)[None, None, :, None]
    rfft = np.fft.fft(adc_frame * range_win, axis=2)
    return rfft[:, :, : NUM_SAMPLES // 2, :]


def doppler_fft_all_channels_one_frame(rfft):
    """
    rfft shape: [loop, tx, range_bin, rx]
    Return RD cube shape: [doppler_bin, tx, range_bin, rx]
    """
    dop_win = np.hanning(NUM_LOOPS)[:, None, None, None]
    rd = np.fft.fftshift(np.fft.fft(rfft * dop_win, axis=0), axes=0)
    return rd


def find_static_target_range_bin(rd_cube, min_range_m=0.7, max_range_m=3.0):
    """
    Static laptop calibration:
    Find strongest zero-Doppler range bin in a reasonable range.
    """
    ranges = range_axis_m()
    zero_doppler_idx = NUM_LOOPS // 2

    # Use Tx0/Rx0 first as a simple robust detector.
    mag = np.abs(rd_cube[zero_doppler_idx, 0, :, 0])
    mag_db = 20 * np.log10(mag + 1e-12)

    mask = (ranges >= min_range_m) & (ranges <= max_range_m)
    candidate_bins = np.where(mask)[0]

    best_local = np.argmax(mag_db[candidate_bins])
    best_bin = int(candidate_bins[best_local])

    return best_bin, float(ranges[best_bin]), float(mag_db[best_bin])


def get_azimuth_virtual_vector_8ula(rd_cube, range_bin, doppler_bin_idx=None):
    """
    Build 8-element horizontal ULA:
      Tx0Rx0~Rx3 -> x = 0,1,2,3
      Tx2Rx0~Rx3 -> x = 4,5,6,7

    We intentionally skip Tx1 because physical TX2 is the elevated row.
    """
    if doppler_bin_idx is None:
        doppler_bin_idx = NUM_LOOPS // 2

    doppler_bin_shifted = doppler_bin_idx - NUM_LOOPS // 2

    v_tx0 = rd_cube[doppler_bin_idx, 0, range_bin, :]  # Tx0 Rx0~Rx3

    # TDM compensation for Tx2. For static target, doppler_bin_shifted=0, so this is 1.
    comp_tx2 = np.exp(-1j * 2.0 * np.pi * doppler_bin_shifted * 2 / (NUM_LOOPS * NUM_TX))
    v_tx2 = rd_cube[doppler_bin_idx, 2, range_bin, :] * comp_tx2

    return np.concatenate([v_tx0, v_tx2], axis=0)


def estimate_azimuth_fft(v8, nfft=256, calib=None):
    """
    v8: 8-element lambda/2 ULA.
    calib: optional complex calibration vector, shape [8].
    """
    if calib is not None:
        v8 = v8 * calib

    spectrum = np.fft.fftshift(np.fft.fft(v8, n=nfft))
    power = np.abs(spectrum) ** 2

    bins = np.arange(-nfft // 2, nfft // 2)
    u = bins / (nfft / 2.0)  # u = sin(theta)

    peak_idx = int(np.argmax(power))
    peak_u = float(np.clip(u[peak_idx], -1.0, 1.0))
    angle_deg = float(np.degrees(np.arcsin(peak_u)))

    theta_deg = np.degrees(np.arcsin(np.clip(u, -1, 1)))
    spectrum_db = 10 * np.log10(power + 1e-12)

    return angle_deg, theta_deg, spectrum_db


def static_angle_calibration_one_frame(adc, frame_idx):
    adc_frame = adc[frame_idx]  # [loop, tx, sample, rx]

    rfft = range_fft_all_channels_one_frame(adc_frame)
    rd_cube = doppler_fft_all_channels_one_frame(rfft)

    range_bin, range_m, mag_db = find_static_target_range_bin(
        rd_cube,
        min_range_m=0.7,
        max_range_m=3.0,
    )

    zero_doppler_idx = NUM_LOOPS // 2

    v8 = get_azimuth_virtual_vector_8ula(
        rd_cube,
        range_bin=range_bin,
        doppler_bin_idx=zero_doppler_idx,
    )

    angle_deg, theta_deg, spectrum_db = estimate_azimuth_fft(v8, nfft=256)

    phase_deg = np.degrees(np.unwrap(np.angle(v8)))

    print(f"\nFrame {frame_idx:03d}")
    print(f"  detected range bin = {range_bin}")
    print(f"  detected range     = {range_m:.2f} m")
    print(f"  magnitude          = {mag_db:.1f} dB")
    print(f"  estimated angle    = {angle_deg:.2f} deg")
    print(f"  v8 phase deg       = {np.round(phase_deg, 1)}")

    return {
        "frame_idx": frame_idx,
        "range_bin": range_bin,
        "range_m": range_m,
        "mag_db": mag_db,
        "angle_deg": angle_deg,
        "v8": v8,
        "theta_deg": theta_deg,
        "spectrum_db": spectrum_db,
        "phase_deg": phase_deg,
    }

# =========================
# Main
# =========================
def main():
    adc = read_3tx_longbin(BIN_PATH)

    print("ADC shape:", adc.shape)
    print("[frame, loop, tx, sample, rx]")

    # Check several frames across the 5-second capture.
    frame_indices = [0, 5, 10, 15, 20, 24]

    results = []
    for frame_idx in frame_indices:
        result = static_angle_calibration_one_frame(adc, frame_idx)
        results.append(result)

    angles = np.array([r["angle_deg"] for r in results])
    ranges = np.array([r["range_m"] for r in results])

    print("\n=== Static front laptop summary ===")
    print("Frame indices:", frame_indices)
    print("Ranges m:", np.round(ranges, 3))
    print("Angles deg:", np.round(angles, 2))
    print(f"Mean range = {np.mean(ranges):.3f} m")
    print(f"Mean angle = {np.mean(angles):.2f} deg")
    print(f"Angle std  = {np.std(angles):.2f} deg")

    # Plot angle spectrum from middle frame.
    mid = results[len(results) // 2]

    plt.figure()
    plt.plot(mid["theta_deg"], mid["spectrum_db"])
    plt.xlabel("Azimuth angle (deg)")
    plt.ylabel("Power (dB)")
    plt.title(
        f"Front laptop static AoA, "
        f"R={mid['range_m']:.2f} m, angle={mid['angle_deg']:.1f} deg"
    )
    plt.grid(True)
    plt.show()

    # Plot angle over time.
    plt.figure()
    times = np.array(frame_indices) * FRAME_PERIOD_S
    plt.plot(times, angles, marker="o")
    plt.xlabel("Time (s)")
    plt.ylabel("Estimated azimuth angle (deg)")
    plt.title("Static target angle stability")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()