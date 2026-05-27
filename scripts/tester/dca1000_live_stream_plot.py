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
# 3Tx 4Rx, 64 loops, 256 ADC samples
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

SAMPLE_RATE_HZ = 10e6
SLOPE_HZ_PER_S = 29.982e12
C = 299_792_458.0

# ============================================================
# Optional raw backup
# ============================================================

WRITE_RAW_BACKUP = True
OUT_PATH = Path(
    r"C:\Users\user\Desktop\ADPAR-final-project\adc\live_socket_60s_200ms_Raw_0.bin"
)

# ============================================================
# DCA1000 packet format
# ============================================================

DCA_HEADER_BYTES = 10

# ============================================================
# Thread communication
# ============================================================

result_queue = queue.Queue(maxsize=1000)
stop_event = threading.Event()


@dataclass
class FrameResult:
    frame_idx: int
    t_sec: float
    strongest_range_m: float
    strongest_mag_db: float
    range_profile_db: np.ndarray


def range_axis_m():
    freqs = np.fft.fftfreq(NUM_SAMPLES, d=1.0 / SAMPLE_RATE_HZ)[: NUM_SAMPLES // 2]
    return C * freqs / (2.0 * SLOPE_HZ_PER_S)


RANGE_AXIS = range_axis_m()


def parse_dca_packet(data: bytes):
    """
    DCA1000 UDP payload:
      [0:4]   sequence number
      [4:10]  byte_count / raw stream byte offset
      [10:]   raw ADC payload
    """
    if len(data) < DCA_HEADER_BYTES:
        return None

    seq = int.from_bytes(data[0:4], byteorder="little", signed=False)
    byte_count = int.from_bytes(data[4:10], byteorder="little", signed=False)
    raw = data[10:]

    return seq, byte_count, raw


def raw_frame_to_adc(frame_bytes: bytes) -> np.ndarray:
    """
    Convert one radar frame to adc:
      shape = [loop, tx, sample, rx]
    """
    raw = np.frombuffer(frame_bytes, dtype=np.int16)

    expected_words = NUM_LOOPS * NUM_TX * NUM_SAMPLES * NUM_RX * IQ
    if raw.size != expected_words:
        raise ValueError(f"raw words={raw.size}, expected={expected_words}")

    # Assumed DCA1000 order per ADC sample:
    # Rx0I Rx1I Rx2I Rx3I Rx0Q Rx1Q Rx2Q Rx3Q
    data = raw.reshape(NUM_LOOPS * NUM_TX, NUM_SAMPLES, IQ, NUM_RX)

    i_data = data[:, :, 0, :].astype(np.float32)
    q_data = data[:, :, 1, :].astype(np.float32)

    adc = i_data + 1j * q_data
    adc = adc.reshape(NUM_LOOPS, NUM_TX, NUM_SAMPLES, NUM_RX)

    return adc


def quick_process_frame(frame_bytes: bytes, frame_idx: int) -> FrameResult:
    """
    First live metric:
      Tx0/Rx0 averaged range FFT.
    Later we will replace/extend this with range-Doppler + AoA.
    """
    adc = raw_frame_to_adc(frame_bytes)

    # Use Tx0/Rx0 only for the first live display.
    x = adc[:, 0, :, 0]  # [loop, sample]

    win = np.hanning(NUM_SAMPLES)[None, :]
    rfft = np.fft.fft(x * win, axis=1)[:, : NUM_SAMPLES // 2]

    # Average magnitude over loops.
    mag = np.mean(np.abs(rfft), axis=0)
    profile_db = 20 * np.log10(mag + 1e-12)

    # Ignore too-near leakage and too-far low-SNR bins for first display.
    mask = (RANGE_AXIS >= 0.4) & (RANGE_AXIS <= 8.0)
    idxs = np.where(mask)[0]

    best_idx = int(idxs[np.argmax(profile_db[idxs])])

    return FrameResult(
        frame_idx=frame_idx,
        t_sec=frame_idx * FRAME_PERIOD_S,
        strongest_range_m=float(RANGE_AXIS[best_idx]),
        strongest_mag_db=float(profile_db[best_idx]),
        range_profile_db=profile_db,
    )


def udp_receiver_worker():
    print("=== UDP receiver worker started ===")
    print(f"Bind: {PC_IP}:{DATA_PORT}")
    print(f"DCA IP: {DCA_IP}")
    print(f"FRAME_BYTES: {FRAME_BYTES}")
    print(f"EXPECTED_BYTES: {EXPECTED_BYTES}")
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
    first_packet_seen = False

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
                first_packet_seen = True

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

            # Write raw backup by DCA byte_count offset.
            if backup_f is not None and byte_count < EXPECTED_BYTES:
                remaining = EXPECTED_BYTES - byte_count
                backup_f.seek(byte_count)
                backup_f.write(raw[:remaining])

            # For live processing we assume packet order is continuous.
            # We already warn if a seq gap occurs.
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


# ============================================================
# Matplotlib display
# ============================================================

def run_plot():
    plt.ion()

    fig, (ax_profile, ax_range) = plt.subplots(2, 1, figsize=(10, 7))

    # Current range profile
    profile_line, = ax_profile.plot(RANGE_AXIS, np.zeros_like(RANGE_AXIS))
    ax_profile.set_xlim(0, 8)
    ax_profile.set_ylim(0, 120)
    ax_profile.set_xlabel("Range (m)")
    ax_profile.set_ylabel("Magnitude (dB)")
    ax_profile.set_title("Current range profile: Tx0/Rx0")

    # Strongest range over time
    time_hist = []
    range_hist = []
    mag_hist = []

    range_line, = ax_range.plot([], [], marker="o", markersize=3)
    ax_range.set_xlim(0, NUM_FRAMES * FRAME_PERIOD_S)
    ax_range.set_ylim(0, 8)
    ax_range.set_xlabel("Time (s)")
    ax_range.set_ylabel("Strongest range (m)")
    ax_range.set_title("Strongest reflector range over time")

    status_text = fig.text(0.02, 0.02, "Waiting for frames...", fontsize=10)

    latest_result = {"value": None}

    def update(_):
        updated = False

        # Drain queue; keep latest profile, append all range points.
        while True:
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                break

            latest_result["value"] = result
            time_hist.append(result.t_sec)
            range_hist.append(result.strongest_range_m)
            mag_hist.append(result.strongest_mag_db)
            updated = True

        if latest_result["value"] is not None:
            r = latest_result["value"]

            profile_line.set_ydata(r.range_profile_db)

            range_line.set_data(time_hist, range_hist)

            status_text.set_text(
                f"Frame {r.frame_idx:03d}/{NUM_FRAMES - 1}, "
                f"t={r.t_sec:.1f}s, "
                f"strongest R={r.strongest_range_m:.2f}m, "
                f"mag={r.strongest_mag_db:.1f}dB"
            )

            # Auto-scale magnitude a little.
            ymax = max(80, float(np.nanmax(r.range_profile_db)) + 10)
            ax_profile.set_ylim(0, ymax)

        return profile_line, range_line, status_text

    ani = FuncAnimation(fig, update, interval=100, blit=False)

    try:
        plt.show(block=True)
    finally:
        stop_event.set()


def main():
    print("=== DCA1000 live stream plot ===")
    print("Run order:")
    print("  1. Run this script.")
    print("  2. Wait until it says Listening for DCA1000 packets.")
    print("  3. In mmWave Studio: StartRecord.")
    print("  4. Trigger Frame.")
    print()

    worker = threading.Thread(target=udp_receiver_worker, daemon=True)
    worker.start()

    run_plot()

    stop_event.set()
    worker.join(timeout=2.0)


if __name__ == "__main__":
    main()