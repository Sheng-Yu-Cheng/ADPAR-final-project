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

# From your current profile:
IDLE_TIME_S = 100e-6
RAMP_END_TIME_S = 60e-6
CHIRP_TIME_S = IDLE_TIME_S + RAMP_END_TIME_S
TDM_CYCLE_S = NUM_TX * CHIRP_TIME_S


# ============================================================
# 2D azimuth config
# ============================================================

# For your AWR2243BOOST antenna geometry:
# Tx0 + Tx2 form the clean horizontal 8-element virtual array.
AZIMUTH_TXS = [0, 2]

ANGLE_FFT = 64

RANGE_MIN_M = 0.40
RANGE_MAX_M = 6.00

# Doppler gate:
# Keep only moving bins with |v| >= MIN_ABS_VELOCITY_MPS.
# If too few points appear, lower this to 0.05.
# If too much clutter remains, raise this to 0.15~0.25.
MIN_ABS_VELOCITY_MPS = 0.10

# Ignore very high velocity bins if needed.
MAX_ABS_VELOCITY_MPS = 6.0

# Peak picking:
MAX_POINTS_PER_FRAME = 40
PEAK_REL_DB = 12.0
MIN_SEP_RANGE_BINS = 2
MIN_SEP_ANGLE_BINS = 2

# Heatmap display:
HEATMAP_DYNAMIC_RANGE_DB = 30


# ============================================================
# Optional raw backup
# ============================================================

WRITE_RAW_BACKUP = True
OUT_PATH = Path(
    r"C:\Users\user\Desktop\ADPAR-final-project\adc\live_pointcloud_doppler_60s_200ms_Raw_0.bin"
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
class MovingPointCloudFrame:
    frame_idx: int
    t_sec: float

    heatmap_db: np.ndarray        # [range_bin, angle_bin], Doppler-gated
    points_x: np.ndarray
    points_y: np.ndarray
    points_v: np.ndarray
    points_mag_db: np.ndarray

    strongest_range_m: float
    strongest_angle_deg: float
    strongest_velocity_mps: float
    strongest_mag_db: float


def range_axis_m():
    freqs = np.fft.fftfreq(NUM_SAMPLES, d=1.0 / SAMPLE_RATE_HZ)[: NUM_SAMPLES // 2]
    return C * freqs / (2.0 * SLOPE_HZ_PER_S)


def doppler_axis_mps():
    """
    Slow-time sample spacing is one full TDM cycle:
      Tx0, Tx1, Tx2
    """
    lam = C / START_FREQ_HZ
    fd = np.fft.fftshift(np.fft.fftfreq(NUM_LOOPS, d=TDM_CYCLE_S))
    v = fd * lam / 2.0
    return v


def angle_axis_deg():
    # For lambda/2 ULA, spatial u = sin(theta)
    k = np.arange(ANGLE_FFT) - ANGLE_FFT / 2
    u = 2.0 * k / ANGLE_FFT
    u = np.clip(u, -1.0, 1.0)
    theta = np.degrees(np.arcsin(u))
    return theta


RANGE_AXIS = range_axis_m()
DOPPLER_AXIS = doppler_axis_mps()
ANGLE_AXIS_DEG = angle_axis_deg()

RANGE_MASK = (RANGE_AXIS >= RANGE_MIN_M) & (RANGE_AXIS <= RANGE_MAX_M)
RANGE_IDXS = np.where(RANGE_MASK)[0]

DOPPLER_MASK = (
    (np.abs(DOPPLER_AXIS) >= MIN_ABS_VELOCITY_MPS)
    & (np.abs(DOPPLER_AXIS) <= MAX_ABS_VELOCITY_MPS)
)
DOPPLER_IDXS = np.where(DOPPLER_MASK)[0]


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


def tdm_phase_compensation(doppler_bin_idx: int, tx_idx: int):
    """
    TDM-MIMO Doppler phase compensation.

    doppler_bin_shifted:
      -NUM_LOOPS/2 ... +NUM_LOOPS/2-1

    Tx0 index = 0
    Tx1 index = 1
    Tx2 index = 2
    """
    doppler_bin_shifted = doppler_bin_idx - NUM_LOOPS // 2
    return np.exp(
        -1j * 2.0 * np.pi * doppler_bin_shifted * tx_idx / (NUM_LOOPS * NUM_TX)
    )


def build_virtual_azimuth_vector(rd_cube: np.ndarray, doppler_bin: int, range_bin: int):
    """
    rd_cube shape:
      [doppler, tx, range, rx]

    Build 8-element azimuth virtual array:
      Tx0Rx0~Rx3 + Tx2Rx0~Rx3
    """
    parts = []

    for tx_idx in AZIMUTH_TXS:
        v = rd_cube[doppler_bin, tx_idx, range_bin, :]  # [rx]
        comp = tdm_phase_compensation(doppler_bin, tx_idx)
        parts.append(v * comp)

    v8 = np.concatenate(parts, axis=0)
    return v8


def local_max_2d(mat: np.ndarray, r: int, a: int) -> bool:
    r0 = max(0, r - 1)
    r1 = min(mat.shape[0], r + 2)
    a0 = max(0, a - 1)
    a1 = min(mat.shape[1], a + 2)

    patch = mat[r0:r1, a0:a1]
    return mat[r, a] >= np.max(patch)


def extract_points_from_heatmap(heatmap_db: np.ndarray, best_velocity_idx_map: np.ndarray):
    """
    heatmap_db:
      [valid_range_idx, angle_bin]

    best_velocity_idx_map:
      [valid_range_idx, angle_bin], stores doppler bin index
    """
    global_max = float(np.max(heatmap_db))
    cand = []

    for r in range(heatmap_db.shape[0]):
        for a in range(heatmap_db.shape[1]):
            val = heatmap_db[r, a]

            if val < global_max - PEAK_REL_DB:
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
    vs = []
    mags = []

    for val, r, a in chosen:
        rb = RANGE_IDXS[r]
        rng = RANGE_AXIS[rb]

        angle_deg = ANGLE_AXIS_DEG[a]
        angle_rad = np.radians(angle_deg)

        dbin = int(best_velocity_idx_map[r, a])
        vel = DOPPLER_AXIS[dbin]

        x = rng * np.sin(angle_rad)
        y = rng * np.cos(angle_rad)

        xs.append(x)
        ys.append(y)
        vs.append(vel)
        mags.append(val)

    return np.array(xs), np.array(ys), np.array(vs), np.array(mags), chosen


def process_frame_doppler_gated(frame_bytes: bytes, frame_idx: int) -> MovingPointCloudFrame:
    adc = raw_frame_to_adc(frame_bytes)

    # Range FFT
    range_win = np.hanning(NUM_SAMPLES)[None, None, :, None]
    range_fft = np.fft.fft(adc * range_win, axis=2)[:, :, : NUM_SAMPLES // 2, :]
    # [loop, tx, range, rx]

    # Doppler FFT over loops
    doppler_win = np.hanning(NUM_LOOPS)[:, None, None, None]
    rd_cube = np.fft.fftshift(
        np.fft.fft(range_fft * doppler_win, axis=0),
        axes=0,
    )
    # [doppler, tx, range, rx]

    # Doppler-gated range-angle heatmap:
    # For each range and angle, take the maximum over moving Doppler bins.
    heatmap_rows = []
    velocity_idx_rows = []

    for rb in RANGE_IDXS:
        # angle spectra for all moving Doppler bins
        angle_spectra = []
        doppler_for_rows = []

        for dbin in DOPPLER_IDXS:
            v8 = build_virtual_azimuth_vector(rd_cube, dbin, rb)

            v8_win = v8 * np.hanning(v8.size)
            angle_spec = np.fft.fftshift(np.fft.fft(v8_win, n=ANGLE_FFT))
            angle_db = 20 * np.log10(np.abs(angle_spec) + 1e-12)

            angle_spectra.append(angle_db)
            doppler_for_rows.append(dbin)

        angle_spectra = np.asarray(angle_spectra)  # [moving_doppler, angle]
        doppler_for_rows = np.asarray(doppler_for_rows)

        # Max over Doppler dimension.
        best_doppler_local = np.argmax(angle_spectra, axis=0)
        best_angle_db = angle_spectra[best_doppler_local, np.arange(ANGLE_FFT)]
        best_doppler_global = doppler_for_rows[best_doppler_local]

        heatmap_rows.append(best_angle_db)
        velocity_idx_rows.append(best_doppler_global)

    heatmap_db = np.asarray(heatmap_rows)              # [range, angle]
    best_velocity_idx_map = np.asarray(velocity_idx_rows)

    xs, ys, vs, mags, chosen = extract_points_from_heatmap(
        heatmap_db,
        best_velocity_idx_map,
    )

    if len(chosen) > 0:
        best_mag, best_r, best_a = chosen[0]
        best_rb = RANGE_IDXS[best_r]
        best_range = float(RANGE_AXIS[best_rb])
        best_angle = float(ANGLE_AXIS_DEG[best_a])
        best_dbin = int(best_velocity_idx_map[best_r, best_a])
        best_vel = float(DOPPLER_AXIS[best_dbin])
        best_mag = float(best_mag)
    else:
        best_range = float("nan")
        best_angle = float("nan")
        best_vel = float("nan")
        best_mag = float("nan")

    return MovingPointCloudFrame(
        frame_idx=frame_idx,
        t_sec=frame_idx * FRAME_PERIOD_S,
        heatmap_db=heatmap_db,
        points_x=xs,
        points_y=ys,
        points_v=vs,
        points_mag_db=mags,
        strongest_range_m=best_range,
        strongest_angle_deg=best_angle,
        strongest_velocity_mps=best_vel,
        strongest_mag_db=best_mag,
    )


def udp_receiver_worker():
    print("=== UDP receiver worker started ===")
    print(f"Bind: {PC_IP}:{DATA_PORT}")
    print(f"DCA IP: {DCA_IP}")
    print(f"FRAME_BYTES: {FRAME_BYTES}")
    print(f"EXPECTED_BYTES: {EXPECTED_BYTES}")
    print(f"AZIMUTH_TXS: {AZIMUTH_TXS}")
    print(f"MIN_ABS_VELOCITY_MPS: {MIN_ABS_VELOCITY_MPS}")
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
            stream_buffer.extend(raw)

            while len(stream_buffer) >= FRAME_BYTES:
                frame_bytes = bytes(stream_buffer[:FRAME_BYTES])
                del stream_buffer[:FRAME_BYTES]

                frame_idx = frame_count
                frame_count += 1

                try:
                    result = process_frame_doppler_gated(frame_bytes, frame_idx)
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
    ax_heat.set_title("Moving Range-Angle Heatmap, Doppler-gated")

    cbar = fig.colorbar(im, ax=ax_heat)
    cbar.set_label("Magnitude (dB)")

    scat = ax_pc.scatter([], [])
    ax_pc.set_xlim(-4, 4)
    ax_pc.set_ylim(0, 6)
    ax_pc.set_xlabel("Lateral x (m)")
    ax_pc.set_ylabel("Forward y (m)")
    ax_pc.set_title("Moving 2D Point Cloud")
    ax_pc.grid(True)

    status_text = fig.text(0.02, 0.02, "Waiting for frames...", fontsize=10)

    latest_result = {"value": None}

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

                # Color by radial velocity.
                scat.set_array(r.points_v)
                scat.set_clim(-2.0, 2.0)

                # Size by magnitude.
                mags = r.points_mag_db
                sizes = 25 + 8 * (mags - np.min(mags) + 1.0)
                scat.set_sizes(sizes)
            else:
                scat.set_offsets(np.empty((0, 2)))
                scat.set_sizes([])

            status_text.set_text(
                f"Frame {r.frame_idx:03d}/{NUM_FRAMES - 1}, "
                f"t={r.t_sec:.1f}s, "
                f"R={r.strongest_range_m:.2f}m, "
                f"angle={r.strongest_angle_deg:.1f}deg, "
                f"v={r.strongest_velocity_mps:.2f}m/s, "
                f"mag={r.strongest_mag_db:.1f}dB, "
                f"points={len(r.points_x)}"
            )

        return im, scat, status_text

    # Add colorbar for velocity scatter.
    scat.set_array(np.array([0.0]))
    cbar2 = fig.colorbar(scat, ax=ax_pc)
    cbar2.set_label("Velocity (m/s)")

    ani = FuncAnimation(fig, update, interval=100, blit=False)

    try:
        plt.show(block=True)
    finally:
        stop_event.set()


def main():
    print("=== DCA1000 live 2D moving point cloud, Doppler-gated ===")
    print("Run order:")
    print("  1. Run this script.")
    print("  2. Wait until it says Listening for DCA1000 packets.")
    print("  3. In mmWave Studio: StartRecord.")
    print("  4. Trigger StartFrame.")
    print()
    print("Use stable streaming setup:")
    print("  FrameConfig(0, 2, 300, 64, 200, 0, 1)")
    print()
    print("Doppler gate:")
    print(f"  Keep |v| >= {MIN_ABS_VELOCITY_MPS} m/s")
    print()

    worker = threading.Thread(target=udp_receiver_worker, daemon=True)
    worker.start()

    run_plot()

    stop_event.set()
    worker.join(timeout=2.0)


if __name__ == "__main__":
    main()