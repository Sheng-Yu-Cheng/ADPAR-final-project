import socket
import threading
import queue
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# Network config
# ============================================================

PC_IP = "192.168.33.30"
DCA_IP = "192.168.33.180"
DATA_PORT = 4098
SOCKET_RCVBUF = 64 * 1024 * 1024


# ============================================================
# Radar config
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


# ============================================================
# 2D azimuth config
# ============================================================

# For your AWR2243BOOST antenna geometry:
# Tx0 + Tx2 form the clean horizontal 8-element virtual array.
AZIMUTH_TXS = [0, 2]

ANGLE_FFT = 64

RANGE_MIN_M = 0.40
RANGE_MAX_M = 6.00

# Static point cloud peak picking:
MAX_POINTS_PER_FRAME = 40
PEAK_REL_DB = 12.0
MIN_SEP_RANGE_BINS = 2
MIN_SEP_ANGLE_BINS = 2

# Optional absolute floor after background subtraction.
# Set to None to use only relative threshold.
# If you get too many tiny residual speckles, try 5~15.
MIN_BG_SUB_DB_ABOVE_NOISE = None

# Heatmap display:
HEATMAP_DYNAMIC_RANGE_DB = 35


# ============================================================
# Background subtraction config
# ============================================================

# First N frames are treated as empty/background.
# At 200 ms per frame:
#   30 frames = 6 seconds
#   50 frames = 10 seconds
BACKGROUND_CALIBRATION_FRAMES = 30

# "median" is more robust if one or two bad frames appear during calibration.
# "mean" is smoother but easier to contaminate.
BACKGROUND_METHOD = "median"      # "median" or "mean"

# Multiply background before subtraction.
# 1.00 = normal subtraction
# 0.80 = gentler, leaves more static residual
# 1.20 = more aggressive, may erase weak targets near clutter
BACKGROUND_SCALE = 1.00

# Residual floor to avoid log(0).
POWER_EPS = 1e-18

# If True, after calibration the plot shows background-subtracted heatmap.
# If False, it shows raw static range-angle heatmap.
ENABLE_BG_SUBTRACTION = True


# ============================================================
# Optional raw backup
# ============================================================

WRITE_RAW_BACKUP = True
OUT_PATH = Path(
    r"C:\Users\user\Desktop\ADPAR-final-project\adc\live_pointcloud_bgsub_60s_200ms_Raw_0.bin"
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
background_reset_event = threading.Event()


@dataclass
class StaticBGPointCloudFrame:
    frame_idx: int
    t_sec: float

    heatmap_db: np.ndarray        # [valid_range_bin, angle_bin], display heatmap
    raw_heatmap_db: np.ndarray    # before background subtraction
    bg_heatmap_db: np.ndarray | None

    points_x: np.ndarray
    points_y: np.ndarray
    points_mag_db: np.ndarray

    strongest_range_m: float
    strongest_angle_deg: float
    strongest_mag_db: float

    background_ready: bool
    background_progress: int
    background_total: int
    mode: str


def range_axis_m():
    freqs = np.fft.fftfreq(NUM_SAMPLES, d=1.0 / SAMPLE_RATE_HZ)[: NUM_SAMPLES // 2]
    return C * freqs / (2.0 * SLOPE_HZ_PER_S)


def angle_axis_deg():
    # For lambda/2 ULA, spatial u = sin(theta)
    k = np.arange(ANGLE_FFT) - ANGLE_FFT / 2
    u = 2.0 * k / ANGLE_FFT
    u = np.clip(u, -1.0, 1.0)
    theta = np.degrees(np.arcsin(u))
    return theta


RANGE_AXIS = range_axis_m()
ANGLE_AXIS_DEG = angle_axis_deg()

RANGE_MASK = (RANGE_AXIS >= RANGE_MIN_M) & (RANGE_AXIS <= RANGE_MAX_M)
RANGE_IDXS = np.where(RANGE_MASK)[0]


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
    """
    raw = np.frombuffer(frame_bytes, dtype=np.int16)

    expected_words = NUM_LOOPS * NUM_TX * NUM_SAMPLES * NUM_RX * IQ
    if raw.size != expected_words:
        raise ValueError(f"raw words={raw.size}, expected={expected_words}")

    # Assumed DCA1000 sample ordering:
    # Rx0I Rx1I Rx2I Rx3I Rx0Q Rx1Q Rx2Q Rx3Q
    data = raw.reshape(NUM_LOOPS * NUM_TX, NUM_SAMPLES, IQ, NUM_RX)

    i_data = data[:, :, 0, :].astype(np.float32)
    q_data = data[:, :, 1, :].astype(np.float32)

    adc = i_data + 1j * q_data
    adc = adc.reshape(NUM_LOOPS, NUM_TX, NUM_SAMPLES, NUM_RX)

    return adc


def build_static_virtual_azimuth_vector(range_cube: np.ndarray, range_bin: int):
    """
    range_cube shape:
      [tx, range, rx]

    Build 8-element azimuth virtual array:
      Tx0Rx0~Rx3 + Tx2Rx0~Rx3

    For static point cloud, we do not need Doppler phase compensation.
    """
    parts = []
    for tx_idx in AZIMUTH_TXS:
        parts.append(range_cube[tx_idx, range_bin, :])  # [rx]

    v8 = np.concatenate(parts, axis=0)
    return v8


def compute_static_range_angle_power(frame_bytes: bytes) -> np.ndarray:
    """
    Returns:
      heatmap_power [valid_range_bin, angle_bin]

    Pipeline:
      raw ADC -> range FFT -> coherent average over loops -> angle FFT
    """
    adc = raw_frame_to_adc(frame_bytes)

    # Optional light DC removal across fast-time samples.
    # This helps reduce leakage near zero range without touching mmWave settings.
    adc = adc - np.mean(adc, axis=2, keepdims=True)

    # Range FFT
    range_win = np.hanning(NUM_SAMPLES)[None, None, :, None]
    range_fft = np.fft.fft(adc * range_win, axis=2)[:, :, : NUM_SAMPLES // 2, :]
    # [loop, tx, range, rx]

    # Static heatmap: coherent average over loops.
    # This makes stationary objects stable and reduces random noise.
    range_cube = np.mean(range_fft, axis=0)
    # [tx, range, rx]

    heatmap_rows = []

    angle_win = np.hanning(len(AZIMUTH_TXS) * NUM_RX)

    for rb in RANGE_IDXS:
        v8 = build_static_virtual_azimuth_vector(range_cube, rb)
        angle_spec = np.fft.fftshift(np.fft.fft(v8 * angle_win, n=ANGLE_FFT))

        # Use power domain for background subtraction.
        angle_power = np.abs(angle_spec) ** 2
        heatmap_rows.append(angle_power)

    return np.asarray(heatmap_rows, dtype=np.float64)


def power_to_db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, POWER_EPS))


def subtract_background_power(current_power: np.ndarray, background_power: np.ndarray) -> np.ndarray:
    """
    Practical background subtraction in power domain:
      residual = max(current - scale * background, eps)

    This removes stable walls/tables/fixed reflectors while keeping new objects.
    """
    residual = current_power - BACKGROUND_SCALE * background_power
    residual = np.maximum(residual, POWER_EPS)
    return residual


def local_max_2d(mat: np.ndarray, r: int, a: int) -> bool:
    r0 = max(0, r - 1)
    r1 = min(mat.shape[0], r + 2)
    a0 = max(0, a - 1)
    a1 = min(mat.shape[1], a + 2)

    patch = mat[r0:r1, a0:a1]
    return mat[r, a] >= np.max(patch)


def extract_points_from_heatmap(heatmap_db: np.ndarray):
    """
    heatmap_db:
      [valid_range_idx, angle_bin]
    """
    if heatmap_db.size == 0 or not np.isfinite(heatmap_db).any():
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            [],
        )

    global_max = float(np.nanmax(heatmap_db))
    cand = []

    for r in range(heatmap_db.shape[0]):
        for a in range(heatmap_db.shape[1]):
            val = heatmap_db[r, a]

            if not np.isfinite(val):
                continue

            if val < global_max - PEAK_REL_DB:
                continue

            if MIN_BG_SUB_DB_ABOVE_NOISE is not None:
                # Because residual floor is POWER_EPS, this threshold is only a rough practical guard.
                if val < MIN_BG_SUB_DB_ABOVE_NOISE:
                    continue

            if not local_max_2d(heatmap_db, r, a):
                continue

            cand.append((val, r, a))

    cand.sort(reverse=True, key=lambda x: x[0])

    chosen = []
    used = []

    for val, r, a in cand:
        ok = True

        for rr, aa in used:
            if abs(r - rr) <= MIN_SEP_RANGE_BINS and abs(a - aa) <= MIN_SEP_ANGLE_BINS:
                ok = False
                break

        if ok:
            chosen.append((val, r, a))
            used.append((r, a))

        if len(chosen) >= MAX_POINTS_PER_FRAME:
            break

    xs = []
    ys = []
    mags = []

    for val, r, a in chosen:
        rb = RANGE_IDXS[r]
        rng = RANGE_AXIS[rb]

        angle_deg = ANGLE_AXIS_DEG[a]
        angle_rad = np.radians(angle_deg)

        x = rng * np.sin(angle_rad)
        y = rng * np.cos(angle_rad)

        xs.append(x)
        ys.append(y)
        mags.append(val)

    return np.array(xs), np.array(ys), np.array(mags), chosen


class BackgroundSubtractor:
    def __init__(self):
        self.samples = []
        self.background_power = None

    def reset(self):
        self.samples = []
        self.background_power = None

    @property
    def ready(self) -> bool:
        return self.background_power is not None

    @property
    def progress(self) -> int:
        if self.ready:
            return BACKGROUND_CALIBRATION_FRAMES
        return len(self.samples)

    def update(self, heatmap_power: np.ndarray):
        """
        During calibration, collect empty-scene heatmaps.
        Once enough frames are collected, freeze background.
        """
        if self.ready:
            return

        self.samples.append(heatmap_power.copy())

        if len(self.samples) >= BACKGROUND_CALIBRATION_FRAMES:
            stack = np.stack(self.samples, axis=0)

            if BACKGROUND_METHOD.lower() == "mean":
                self.background_power = np.mean(stack, axis=0)
            elif BACKGROUND_METHOD.lower() == "median":
                self.background_power = np.median(stack, axis=0)
            else:
                raise ValueError(f"Unknown BACKGROUND_METHOD={BACKGROUND_METHOD}")

            self.samples = []


def process_frame_bg_sub(
    frame_bytes: bytes,
    frame_idx: int,
    bg: BackgroundSubtractor,
) -> StaticBGPointCloudFrame:
    raw_power = compute_static_range_angle_power(frame_bytes)
    raw_db = power_to_db(raw_power)

    bg.update(raw_power)

    if ENABLE_BG_SUBTRACTION and bg.ready:
        display_power = subtract_background_power(raw_power, bg.background_power)
        display_db = power_to_db(display_power)
        bg_db = power_to_db(bg.background_power)
        mode = "BG-SUB"
    else:
        display_db = raw_db
        bg_db = power_to_db(bg.background_power) if bg.background_power is not None else None
        mode = "CALIBRATING" if not bg.ready else "RAW"

    # During calibration, do not output point cloud, because empty scene peaks are not meaningful.
    if not bg.ready:
        xs = np.array([])
        ys = np.array([])
        mags = np.array([])
        chosen = []
    else:
        xs, ys, mags, chosen = extract_points_from_heatmap(display_db)

    if len(chosen) > 0:
        best_mag, best_r, best_a = chosen[0]
        best_rb = RANGE_IDXS[best_r]
        best_range = float(RANGE_AXIS[best_rb])
        best_angle = float(ANGLE_AXIS_DEG[best_a])
        best_mag = float(best_mag)
    else:
        best_range = float("nan")
        best_angle = float("nan")
        best_mag = float("nan")

    return StaticBGPointCloudFrame(
        frame_idx=frame_idx,
        t_sec=frame_idx * FRAME_PERIOD_S,

        heatmap_db=display_db,
        raw_heatmap_db=raw_db,
        bg_heatmap_db=bg_db,

        points_x=xs,
        points_y=ys,
        points_mag_db=mags,

        strongest_range_m=best_range,
        strongest_angle_deg=best_angle,
        strongest_mag_db=best_mag,

        background_ready=bg.ready,
        background_progress=bg.progress,
        background_total=BACKGROUND_CALIBRATION_FRAMES,
        mode=mode,
    )


def udp_receiver_worker():
    print("=== UDP receiver worker started ===")
    print(f"Bind: {PC_IP}:{DATA_PORT}")
    print(f"DCA IP: {DCA_IP}")
    print(f"FRAME_BYTES: {FRAME_BYTES}")
    print(f"EXPECTED_BYTES: {EXPECTED_BYTES}")
    print(f"AZIMUTH_TXS: {AZIMUTH_TXS}")
    print(f"BACKGROUND_CALIBRATION_FRAMES: {BACKGROUND_CALIBRATION_FRAMES}")
    print(f"BACKGROUND_METHOD: {BACKGROUND_METHOD}")
    print(f"BACKGROUND_SCALE: {BACKGROUND_SCALE}")
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
    print("Now press StartRecord, then StartFrame in mmWave Studio.")
    print("Important: keep the scene EMPTY during the first background calibration frames.")
    print("Press keyboard key 'b' in the plot window to re-learn background.")
    print()

    stream_buffer = bytearray()
    bg = BackgroundSubtractor()

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

            if background_reset_event.is_set():
                bg.reset()
                background_reset_event.clear()
                print("Background reset requested. Re-learning empty scene now.")

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
            # This matches your current stable 60 s DCA1000 UDP behavior.
            stream_buffer.extend(raw)

            while len(stream_buffer) >= FRAME_BYTES:
                frame_bytes = bytes(stream_buffer[:FRAME_BYTES])
                del stream_buffer[:FRAME_BYTES]

                frame_idx = frame_count
                frame_count += 1

                try:
                    result = process_frame_bg_sub(frame_bytes, frame_idx, bg)
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
                    f"rate={mbps:5.1f} Mbps, "
                    f"bg={bg.progress:2d}/{BACKGROUND_CALIBRATION_FRAMES}, "
                    f"ready={bg.ready}"
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


def run_plot():
    plt.ion()

    fig, (ax_heat, ax_pc) = plt.subplots(1, 2, figsize=(13, 6))

    init_heat = np.zeros((len(RANGE_IDXS), ANGLE_FFT))

    im = ax_heat.imshow(
        init_heat,
        aspect="auto",
        origin="lower",
        extent=[ANGLE_AXIS_DEG[0], ANGLE_AXIS_DEG[-1], RANGE_MIN_M, RANGE_MAX_M],
    )

    ax_heat.set_xlabel("Angle (deg)")
    ax_heat.set_ylabel("Range (m)")
    ax_heat.set_title("Static Range-Angle Heatmap, Background-subtracted")

    cbar = fig.colorbar(im, ax=ax_heat)
    cbar.set_label("Magnitude (dB)")

    scat = ax_pc.scatter([], [])
    ax_pc.set_xlim(-4, 4)
    ax_pc.set_ylim(0, 6)
    ax_pc.set_xlabel("Lateral x (m)")
    ax_pc.set_ylabel("Forward y (m)")
    ax_pc.set_title("Static 2D Point Cloud after BG Sub")
    ax_pc.grid(True)

    status_text = fig.text(0.02, 0.02, "Waiting for frames...", fontsize=10)

    latest_result = {"value": None}

    def on_key(event):
        if event.key == "b":
            background_reset_event.set()
            status_text.set_text("Background reset requested. Keep scene empty...")

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

            heat = r.heatmap_db
            im.set_data(heat)

            heat_max = float(np.nanmax(heat))
            im.set_clim(heat_max - HEATMAP_DYNAMIC_RANGE_DB, heat_max)

            if len(r.points_x) > 0:
                pts = np.column_stack([r.points_x, r.points_y])
                scat.set_offsets(pts)

                # Color by point magnitude.
                scat.set_array(r.points_mag_db)
                mags = r.points_mag_db
                scat.set_clim(float(np.min(mags)), float(np.max(mags)) if np.max(mags) > np.min(mags) else float(np.min(mags) + 1.0))

                # Size by magnitude.
                sizes = 25 + 8 * (mags - np.min(mags) + 1.0)
                scat.set_sizes(sizes)
            else:
                scat.set_offsets(np.empty((0, 2)))
                scat.set_array(np.array([]))
                scat.set_sizes([])

            if r.background_ready:
                bg_msg = "ready"
            else:
                bg_msg = f"learning {r.background_progress}/{r.background_total}"

            status_text.set_text(
                f"Frame {r.frame_idx:03d}/{NUM_FRAMES - 1}, "
                f"t={r.t_sec:.1f}s, "
                f"mode={r.mode}, "
                f"background={bg_msg}, "
                f"R={r.strongest_range_m:.2f}m, "
                f"angle={r.strongest_angle_deg:.1f}deg, "
                f"mag={r.strongest_mag_db:.1f}dB, "
                f"points={len(r.points_x)} | "
                f"Press 'b' to re-learn background"
            )

        return im, scat, status_text

    # Add colorbar for point magnitude.
    scat.set_array(np.array([0.0]))
    cbar2 = fig.colorbar(scat, ax=ax_pc)
    cbar2.set_label("Point magnitude (dB)")

    ani = FuncAnimation(fig, update, interval=100, blit=False)

    try:
        plt.show(block=True)
    finally:
        stop_event.set()


def main():
    print("=== DCA1000 live 2D static point cloud, background-subtracted ===")
    print("Run order:")
    print("  1. Run this script.")
    print("  2. Wait until it says Listening for DCA1000 packets.")
    print("  3. Keep scene EMPTY for background calibration.")
    print("  4. In mmWave Studio: StartRecord.")
    print("  5. Trigger StartFrame.")
    print("  6. After background is ready, place/person enters scene.")
    print()
    print("Use stable streaming setup:")
    print("  FrameConfig(0, 2, 300, 64, 200, 0, 1)")
    print()
    print("Background subtraction:")
    print(f"  calibration frames = {BACKGROUND_CALIBRATION_FRAMES}")
    print(f"  calibration time   = {BACKGROUND_CALIBRATION_FRAMES * FRAME_PERIOD_S:.1f} s")
    print(f"  method             = {BACKGROUND_METHOD}")
    print(f"  scale              = {BACKGROUND_SCALE}")
    print("  press 'b' in plot window to re-learn background")
    print()

    worker = threading.Thread(target=udp_receiver_worker, daemon=True)
    worker.start()

    run_plot()

    stop_event.set()
    worker.join(timeout=2.0)


if __name__ == "__main__":
    main()
