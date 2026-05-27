import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ====== 修改這兩個路徑 ======
empty_path = Path(r"C:\\ti\\mmwave_studio_03_00_00_14\\mmWaveStudio\\PostProc\\empty.bin")
move_path  = Path(r"C:\\ti\\mmwave_studio_03_00_00_14\\mmWaveStudio\\PostProc\\metal_ruler_move.bin")

# ====== 你的 mmWave Studio 設定 ======
num_frames = 8
num_chirps = 128
num_samples = 256
num_rx = 4

sample_rate_hz = 10e6          # 10000 ksps
slope_hz_per_s = 29.982e12     # 29.982 MHz/us
c = 299_792_458.0

def read_xwr22xx_complex_4rx(path):
    raw = np.fromfile(path, dtype=np.int16)

    expected_words = num_frames * num_chirps * num_samples * num_rx * 2
    print(f"{path.name}: raw words={raw.size}, expected={expected_words}")

    if raw.size < expected_words:
        raise ValueError(f"{path.name} too small. Check capture settings or file path.")

    raw = raw[:expected_words]

    # File order per sample:
    # Rx0I Rx1I Rx2I Rx3I Rx0Q Rx1Q Rx2Q Rx3Q
    data = raw.reshape(num_frames, num_chirps, num_samples, 2, num_rx)

    I = data[:, :, :, 0, :].astype(np.float32)
    Q = data[:, :, :, 1, :].astype(np.float32)

    adc = I + 1j * Q
    # shape: [frame, chirp, sample, rx]
    return adc

def range_fft(adc, frame_idx=0, rx_idx=0):
    # adc shape: [frame, chirp, sample, rx]
    x = adc[frame_idx, :, :, rx_idx]  # [chirp, sample]
    win = np.hanning(num_samples)[None, :]
    X = np.fft.fft(x * win, axis=1)
    X = X[:, :num_samples // 2]
    return X

def range_doppler(adc, frame_idx=0, rx_idx=0):
    Xr = range_fft(adc, frame_idx, rx_idx)  # [chirp, range_bin]

    # Doppler FFT across chirps
    dop_win = np.hanning(num_chirps)[:, None]
    RD = np.fft.fftshift(np.fft.fft(Xr * dop_win, axis=0), axes=0)

    RD_db = 20 * np.log10(np.abs(RD) + 1e-12)
    return RD_db

def range_axis_m():
    # Beat frequency bins
    freqs = np.fft.fftfreq(num_samples, d=1 / sample_rate_hz)[:num_samples // 2]
    ranges = c * freqs / (2 * slope_hz_per_s)
    return ranges

def plot_1d_range(adc, title, frame_idx=0, rx_idx=0):
    Xr = range_fft(adc, frame_idx, rx_idx)
    mag = 20 * np.log10(np.mean(np.abs(Xr), axis=0) + 1e-12)
    r = range_axis_m()

    plt.figure()
    plt.plot(r, mag)
    plt.title(title)
    plt.xlabel("Range (m)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)

def plot_rd(adc, title, frame_idx=0, rx_idx=0):
    RD_db = range_doppler(adc, frame_idx, rx_idx)
    r = range_axis_m()

    # 先用 Doppler bin，不急著換成 m/s
    doppler_bins = np.arange(-num_chirps // 2, num_chirps // 2)

    plt.figure()
    plt.imshow(
        RD_db,
        aspect="auto",
        origin="lower",
        extent=[r[0], r[-1], doppler_bins[0], doppler_bins[-1]],
    )
    plt.title(title)
    plt.xlabel("Range (m)")
    plt.ylabel("Doppler bin")
    plt.colorbar(label="Magnitude (dB)")

empty_adc = read_xwr22xx_complex_4rx(empty_path)
move_adc = read_xwr22xx_complex_4rx(move_path)

# 看 raw time domain sanity check
x = move_adc[0, 0, :, 0]
plt.figure()
plt.plot(np.real(x), label="I")
plt.plot(np.imag(x), label="Q")
plt.title("Time-domain: move, frame 0 chirp 0 RX0")
plt.xlabel("Sample")
plt.ylabel("ADC code")
plt.legend()
plt.grid(True)

# Range FFT
plot_1d_range(empty_adc, "Empty scene range FFT, RX0")
plot_1d_range(move_adc, "Metal ruler move range FFT, RX0")

# Range-Doppler
plot_rd(empty_adc, "Empty scene range-Doppler, RX0")
plot_rd(move_adc, "Metal ruler move range-Doppler, RX0")

# Simple difference map
RD_empty = range_doppler(empty_adc, frame_idx=0, rx_idx=0)
RD_move = range_doppler(move_adc, frame_idx=0, rx_idx=0)
diff = RD_move - RD_empty

r = range_axis_m()
doppler_bins = np.arange(-num_chirps // 2, num_chirps // 2)

plt.figure()
plt.imshow(
    diff,
    aspect="auto",
    origin="lower",
    extent=[r[0], r[-1], doppler_bins[0], doppler_bins[-1]],
)
plt.title("Move - Empty range-Doppler difference, RX0")
plt.xlabel("Range (m)")
plt.ylabel("Doppler bin")
plt.colorbar(label="dB difference")

plt.show()