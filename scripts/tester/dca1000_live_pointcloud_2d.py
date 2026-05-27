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
C = 299_792_458.0

# For first 2D azimuth point cloud, use 2 azimuth TXs only.
# If later you verify a different azimuth-TX pair on your board,
# only change this constant.
AZIMUTH_TXS = [0, 2]          # first attempt

ANGLE_FFT = 64
RANGE_MIN_M = 0.40
RANGE_MAX_M = 6.00

MAX_POINTS_PER_FRAME = 20
PEAK_REL_DB = 10.0            # keep points within (global max - PEAK_REL_DB)
MIN_SEP_RANGE_BINS = 2
MIN_SEP_ANGLE_BINS = 2

REMOVE_STATIC_CLUTTER = False

# ============================================================
# Optional raw backup
# ============================================================

WRITE_RAW_BACKUP = True
OUT_PATH = Path(
    r"C:\Users\user\Desktop\ADPAR-final-project\adc\live_pointcloud_60s_200ms_Raw_0.bin"
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
class PointCloudFrame:
    frame_idx: int
    t_sec: float
    heatmap_db: np.ndarray      # [range_bin, angle_bin]
    points_x: np.ndarray        # lateral (m)
    points_y: np.ndarray        # forward (m)
    points_mag_db: np.ndarray
    strongest_range_m: float
    strongest_angle_deg: float
    strongest_mag_db: float


def range_axis_m():
    freqs = np.fft.fftfreq(NUM_SAMPLES, d=1.0 / SAMPLE_RATE_HZ)[: NUM_SAMPLES // 2]
    return C * freqs / (2.0 * SLOPE_HZ_PER_S)


def angle_axis_deg():
    # For ULA with d = lambda/2, FFT bin maps approximately to u = sin(theta)
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

    # Assume DCA1000 sample ordering:
    # Rx0I Rx1I Rx2I Rx3I Rx0Q Rx1Q Rx2Q Rx3Q
    data = raw.reshape(NUM_LOOPS * NUM_TX, NUM_SAMPLES, IQ, NUM_RX)

    i_data = data[:, :, 0, :].astype(np.float32)
    q_data = data[:, :, 1, :].astype(np.float32)

    adc = i_data + 1j * q_data
    adc = adc.reshape(NUM_LOOPS, NUM_TX, NUM_SAMPLES, NUM_RX)

    return adc


def build_virtual_azimuth_snapshot(range_fft_cube: np.ndarray, rb: int) -> np.ndarray:
    """
    range_fft_cube shape:
      [loop, tx, range_bin, rx]

    Return 8-channel azimuth snapshot for one range bin:
      [virtual_channel]
    """
    # Average across loops first
    snap_tx_rx = np.mean(range_fft_cube[:, AZIMUTH_TXS, rb, :], axis=0)  # [2, 4]

    # Flatten into 8 virtual channels.
    # If later you find the board needs a different ordering, change here.
    v8 = snap_tx_rx.reshape(-1)

    return v8


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
      [num_valid_ranges, ANGLE_FFT]

    Returns selected peak points.
    """
    global_max = float(np.max(heatmap_db))
    cand = []

    for r in range(heatmap_db.shape[0]):
        for a in range(heatmap_db.shape[1]):
            v = heatmap_db[r, a]
            if v < global_max - PEAK_REL_DB:
                continue
            if not local_max_2d(heatmap_db, r, a):
                continue
            cand.append((v, r, a))

    cand.sort(reverse=True, key=lambda x: x[0])

    chosen = []
    used = []

    for v, r, a in cand:
        ok = True
        for rr, aa in used:
            if abs(r - rr) <= MIN_SEP_RANGE_BINS and abs(a - aa) <= MIN_SEP_ANGLE_BINS:
                ok = False
                break
        if ok:
            chosen.append((v, r, a))
            used.append((r, a))
        if len(chosen) >= MAX_POINTS_PER_FRAME:
            break

    xs = []
    ys = []
    mags = []

    for v, r, a in chosen:
        rb = RANGE_IDXS[r]
        rng = RANGE_AXIS[rb]
        ang_deg = ANGLE_AXIS_DEG[a]
        ang_rad = np.radians(ang_deg)

        x = rng * np.sin(ang_rad)   # lateral
        y = rng * np.cos(ang_rad)   # forward

        xs.append(x)
        ys.append(y)
        mags.append(v)

    return np.array(xs), np.array(ys), np.array(mags), chosen


def quick_process_frame(frame_bytes: bytes, frame_idx: int) -> PointCloudFrame:
    adc = raw_frame_to_adc(frame_bytes)

    # Optional clutter removal across loops
    if REMOVE_STATIC_CLUTTER:
        adc = adc - np.mean(adc, axis=0, keepdims=True)

    # Range FFT
    win_r = np.hanning(NUM_SAMPLES)[None, None, :, None]
    range_fft = np.fft.fft(adc * win_r, axis=2)[:, :, : NUM_SAMPLES // 2, :]
    # shape: [loop, tx, range_bin, rx]

    # Build range-angle heatmap
    heatmap_rows = []

    for rb in RANGE_IDXS:
        v8 = build_virtual_azimuth_snapshot(range_fft, rb)

        # spatial FFT over azimuth virtual array
        v8_win = v8 * np.hanning(v8.size)
        ang_spec = np.fft.fftshift(np.fft.fft(v8_win, n=ANGLE_FFT))
        ang_db = 20 * np.log10(np.abs(ang_spec) + 1e-12)
        heatmap_rows.append(ang_db)

    heatmap_db = np.array(heatmap_rows)  # [num_valid_ranges, ANGLE_FFT]

    # Extract point cloud peaks
    xs, ys, mags, chosen = extract_points_from_heatmap(heatmap_db)

    if len(chosen) > 0:
        best_mag, best_r, best_a = chosen[0]
        best_rb = RANGE_IDXS[best_r]
        strongest_range_m = float(RANGE_AXIS[best_rb])
        strongest_angle_deg = float(ANGLE_AXIS_DEG[best_a])
        strongest_mag_db = float(best_mag)
    else:
        strongest_range_m = float("nan")
        strongest_angle_deg = float("nan")
        strongest_mag_db = float("nan")

    return PointCloudFrame(
        frame_idx=frame_idx,
        t_sec=frame_idx * FRAME_PERIOD_S,
        heatmap_db=heatmap_db,
        points_x=xs,
        points_y=ys,
        points_mag_db=mags,
        strongest_range_m=strongest_range_m,
        strongest_angle_deg=strongest_angle_deg,
        strongest_mag_db=strongest_mag_db,
    )


def udp_receiver_worker():
    print("=== UDP receiver worker started ===")
    print(f"Bind: {PC_IP}:{DATA_PORT}")
    print(f"DCA IP: {DCA_IP}")
    print(f"FRAME_BYTES: {FRAME_BYTES}")
    print(f"EXPECTED_BYTES: {EXPECTED_BYTES}")
    print(f"AZIMUTH_TXS: {AZIMUTH_TXS}")
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

            # For live processing, assume in-order continuous stream.
            stream_buffer.extend(raw)

            while len(stream_buffer) >= FRAME_BYTES:
                frame_bytes = bytes(stream_buffer[:FRAME_BYTES])
                del stream_buffer[:FRAME_BYTES]

                frame_idx = frame_count
                frame_count += 1

                try:
                    result = quick_process_frame(frame_bytes, frame_idx)
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

    # Initial heatmap
    init_heat = np.zeros((len(RANGE_IDXS), ANGLE_FFT))
    im = ax_heat.imshow(
        init_heat,
        aspect="auto",
        origin="lower",
        extent=[ANGLE_AXIS_DEG[0], ANGLE_AXIS_DEG[-1], RANGE_MIN_M, RANGE_MAX_M],
    )
    ax_heat.set_xlabel("Angle (deg)")
    ax_heat.set_ylabel("Range (m)")
    ax_heat.set_title("Range-Angle Heatmap")
    cbar = fig.colorbar(im, ax=ax_heat)
    cbar.set_label("Magnitude (dB)")

    # Point cloud scatter
    scat = ax_pc.scatter([], [])
    ax_pc.set_xlim(-4, 4)
    ax_pc.set_ylim(0, 6)
    ax_pc.set_xlabel("Lateral x (m)")
    ax_pc.set_ylabel("Forward y (m)")
    ax_pc.set_title("2D Point Cloud")
    ax_pc.grid(True)

    status_text = fig.text(0.02, 0.02, "Waiting for frames...", fontsize=10)

    latest_result = {"value": None}

    def update(_):
        updated = False

        while True:
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                break
            latest_result["value"] = result
            updated = True

        if latest_result["value"] is not None:
            r = latest_result["value"]

            heat = r.heatmap_db
            im.set_data(heat)
            im.set_clim(np.max(heat) - 25, np.max(heat))

            if len(r.points_x) > 0:
                pts = np.column_stack([r.points_x, r.points_y])
                scat.set_offsets(pts)

                # use magnitude for size
                mags = r.points_mag_db
                sizes = 20 + 8 * (mags - np.min(mags) + 1.0)
                scat.set_sizes(sizes)
            else:
                scat.set_offsets(np.empty((0, 2)))
                scat.set_sizes([])

            status_text.set_text(
                f"Frame {r.frame_idx:03d}/{NUM_FRAMES - 1}, "
                f"t={r.t_sec:.1f}s, "
                f"strongest R={r.strongest_range_m:.2f}m, "
                f"strongest angle={r.strongest_angle_deg:.1f}deg, "
                f"mag={r.strongest_mag_db:.1f}dB, "
                f"points={len(r.points_x)}"
            )

        return im, scat, status_text

    ani = FuncAnimation(fig, update, interval=100, blit=False)

    try:
        plt.show(block=True)
    finally:
        stop_event.set()


def main():
    print("=== DCA1000 live 2D point cloud ===")
    print("Run order:")
    print("  1. Run this script.")
    print("  2. Wait until it says Listening for DCA1000 packets.")
    print("  3. In mmWave Studio: StartRecord.")
    print("  4. Trigger StartFrame.")
    print()
    print("Use the stable streaming setup:")
    print("  FrameConfig(0, 2, 300, 64, 200, 0, 1)")
    print()

    worker = threading.Thread(target=udp_receiver_worker, daemon=True)
    worker.start()

    run_plot()

    stop_event.set()
    worker.join(timeout=2.0)


if __name__ == "__main__":
    main()