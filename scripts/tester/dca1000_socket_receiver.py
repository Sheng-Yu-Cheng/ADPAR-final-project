import socket
import time
from pathlib import Path
from dataclasses import dataclass


# ============================================================
# Network config
# ============================================================

PC_IP = "192.168.33.30"
DCA_IP = "192.168.33.180"
DATA_PORT = 4098

# ============================================================
# Radar config: your current 3Tx 4Rx setup
# ============================================================

NUM_FRAMES = 400
NUM_LOOPS = 64
NUM_TX = 3
NUM_RX = 4
NUM_SAMPLES = 256

BYTES_PER_SAMPLE = 2      # int16
IQ = 2                    # I + Q

FRAME_BYTES = NUM_LOOPS * NUM_TX * NUM_SAMPLES * NUM_RX * IQ * BYTES_PER_SAMPLE
EXPECTED_BYTES = NUM_FRAMES * FRAME_BYTES

# ============================================================
# Output
# ============================================================

OUT_PATH = Path(r"C:\Users\user\Desktop\ADPAR-final-project\adc\socket_capture_80s_200ms_Raw_0.bin")

# ============================================================
# DCA1000 packet config
# ============================================================

DCA_HEADER_BYTES = 10
NOMINAL_RAW_PAYLOAD_BYTES = 1456

# ============================================================
# Receiver behavior
# ============================================================

SOCKET_RCVBUF = 64 * 1024 * 1024

# Stop after this much silence once first packet has arrived.
IDLE_TIMEOUT_S = 5.0

# Hard safety timeout.
MAX_WAIT_S = 120.0

# Set True only if you know what you are doing.
# On Windows this may allow bind even if another socket uses 4098,
# but packet delivery may be ambiguous. Prefer False first.
USE_REUSEADDR = False


@dataclass
class Stats:
    packets: int = 0
    useful_packets: int = 0
    duplicate_packets: int = 0
    out_of_range_packets: int = 0
    bytes_written: int = 0
    first_seq: int | None = None
    last_seq: int | None = None
    min_offset: int | None = None
    max_offset: int | None = None
    first_packet_time: float | None = None
    last_packet_time: float | None = None


def parse_dca_packet(data: bytes):
    """
    DCA1000 raw UDP payload:
      [0:4]   sequence number, little-endian uint32
      [4:10]  byte_count / raw byte offset, little-endian 6-byte integer
      [10:]   raw ADC payload
    """
    if len(data) < DCA_HEADER_BYTES:
        return None

    seq = int.from_bytes(data[0:4], byteorder="little", signed=False)
    byte_count = int.from_bytes(data[4:10], byteorder="little", signed=False)
    raw = data[10:]

    return seq, byte_count, raw


def make_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if USE_REUSEADDR:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_RCVBUF)

    # Bind specifically to DCA1000 data port on your Ethernet NIC.
    sock.bind((PC_IP, DATA_PORT))
    sock.settimeout(0.2)

    actual_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print(f"Actual SO_RCVBUF = {actual_buf}")

    return sock


def preallocate_file(path: Path, size: int):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        f.truncate(size)


def main():
    print("=== DCA1000 Python socket receiver ===")
    print(f"Bind address     : {PC_IP}:{DATA_PORT}")
    print(f"DCA1000 IP       : {DCA_IP}")
    print(f"Output           : {OUT_PATH}")
    print(f"FRAME_BYTES      : {FRAME_BYTES}")
    print(f"NUM_FRAMES       : {NUM_FRAMES}")
    print(f"EXPECTED_BYTES   : {EXPECTED_BYTES}")
    print(f"USE_REUSEADDR    : {USE_REUSEADDR}")
    print()
    print("Run order:")
    print("  1. Start this Python receiver first.")
    print("  2. In mmWave Studio, StartRecord / ARM DCA1000.")
    print("  3. Trigger frame.")
    print("  4. Wait until Python says Done.")
    print()

    print("Preallocating output file...")
    preallocate_file(OUT_PATH, EXPECTED_BYTES)

    try:
        sock = make_socket()
    except OSError as e:
        print()
        print("ERROR: could not bind UDP socket.")
        print(e)
        print()
        print("Most likely mmWave Studio / DCA1000 DLL already owns port 4098.")
        print("Close any existing recording, or try running this before StartRecord.")
        print("If it still conflicts, we need DCA1000 CLI / direct control next.")
        return

    # Track received nominal packet slots.
    # 25-frame test is ~13504 packets. This is small.
    num_slots = (EXPECTED_BYTES + NOMINAL_RAW_PAYLOAD_BYTES - 1) // NOMINAL_RAW_PAYLOAD_BYTES
    received = bytearray(num_slots)

    stats = Stats()
    t0 = time.time()
    last_report = t0

    print("Listening for DCA1000 UDP data...")

    with OUT_PATH.open("r+b") as f:
        while True:
            now = time.time()

            if now - t0 > MAX_WAIT_S:
                print("Stop: MAX_WAIT_S reached.")
                break

            if stats.first_packet_time is not None and stats.last_packet_time is not None:
                if now - stats.last_packet_time > IDLE_TIMEOUT_S:
                    print("Stop: idle timeout after last packet.")
                    break

            if stats.bytes_written >= EXPECTED_BYTES:
                print("Stop: expected bytes received.")
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
            stats.packets += 1

            now = time.time()
            if stats.first_packet_time is None:
                stats.first_packet_time = now
                stats.first_seq = seq
                print(
                    f"First packet: src={src_ip}:{src_port}, "
                    f"seq={seq}, byte_count={byte_count}, raw_len={len(raw)}"
                )

            stats.last_packet_time = now
            stats.last_seq = seq

            if stats.min_offset is None or byte_count < stats.min_offset:
                stats.min_offset = byte_count
            if stats.max_offset is None or byte_count > stats.max_offset:
                stats.max_offset = byte_count

            # Ignore data outside expected capture size.
            if byte_count >= EXPECTED_BYTES:
                stats.out_of_range_packets += 1
                continue

            remaining = EXPECTED_BYTES - byte_count
            raw_to_write = raw[:remaining]

            slot = byte_count // NOMINAL_RAW_PAYLOAD_BYTES
            if 0 <= slot < num_slots:
                if received[slot]:
                    stats.duplicate_packets += 1
                else:
                    received[slot] = 1
                    stats.useful_packets += 1
                    stats.bytes_written += len(raw_to_write)

            f.seek(byte_count)
            f.write(raw_to_write)

            # Low-rate report only.
            if now - last_report >= 2.0:
                elapsed = now - t0
                mbps = (stats.bytes_written * 8 / 1e6) / max(elapsed, 1e-9)
                frames_equiv = stats.bytes_written / FRAME_BYTES
                print(
                    f"pkts={stats.packets:6d} "
                    f"useful={stats.useful_packets:6d} "
                    f"seq={seq:6d} "
                    f"offset={byte_count:9d} "
                    f"raw_MB={stats.bytes_written / 1e6:7.2f} "
                    f"frames={frames_equiv:5.2f}/{NUM_FRAMES} "
                    f"rate={mbps:6.1f} Mbps"
                )
                last_report = now

    sock.close()

    # Analyze missing slots.
    missing_slots = [i for i, v in enumerate(received) if not v]

    print()
    print("=== Done ===")
    print(f"Output file          : {OUT_PATH}")
    print(f"Expected bytes       : {EXPECTED_BYTES}")
    print(f"Useful bytes written : {stats.bytes_written}")
    print(f"Packets seen         : {stats.packets}")
    print(f"Useful packets       : {stats.useful_packets}")
    print(f"Duplicate packets    : {stats.duplicate_packets}")
    print(f"Out-of-range packets : {stats.out_of_range_packets}")
    print(f"First seq            : {stats.first_seq}")
    print(f"Last seq             : {stats.last_seq}")
    print(f"Min byte offset      : {stats.min_offset}")
    print(f"Max byte offset      : {stats.max_offset}")
    print(f"Missing packet slots : {len(missing_slots)} / {num_slots}")

    if missing_slots:
        print("First missing slots  :", missing_slots[:20])
        print("Status               : NOT COMPLETE")
    else:
        print("Status               : COMPLETE")

    actual_size = OUT_PATH.stat().st_size
    print(f"File size            : {actual_size}")

    if stats.bytes_written == EXPECTED_BYTES and not missing_slots:
        print("Capture completeness : OK")
    else:
        print("Capture completeness : NOT OK")


if __name__ == "__main__":
    main()