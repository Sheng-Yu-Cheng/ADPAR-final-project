from scapy.all import sniff, UDP, IP, conf
import time

DCA_IP = "192.168.33.180"
DATA_PORT = 4098
IFACE = r"\Device\NPF_{E6A439F8-29B9-49FF-8BAD-E8141FD0C9B1}"

conf.use_pcap = True

packet_count = 0
first_seq = None
last_seq = None
gap_count = 0
first_time = None
last_time = None

def handle(pkt):
    global packet_count, first_seq, last_seq, gap_count, first_time, last_time

    if IP not in pkt or UDP not in pkt:
        return

    payload = bytes(pkt[UDP].payload)

    if len(payload) < 10:
        return

    seq = int.from_bytes(payload[0:4], byteorder="little", signed=False)

    now = time.time()

    if first_time is None:
        first_time = now
        first_seq = seq

    if last_seq is not None and seq != last_seq + 1:
        gap = seq - last_seq - 1
        if gap > 0:
            gap_count += gap

    last_seq = seq
    last_time = now
    packet_count += 1

print("Minimal Scapy packet counter")
print("Start mmWave Studio StartRecord + StartFrame now.")
print("Press Ctrl+C after Record completed.\n")

try:
    sniff(
        iface=IFACE,
        filter=f"udp and src host {DCA_IP} and dst port {DATA_PORT}",
        prn=handle,
        store=False,
        promisc=True,
    )
except KeyboardInterrupt:
    pass

print("\n=== Summary ===")
print(f"packet_count = {packet_count}")
print(f"first_seq    = {first_seq}")
print(f"last_seq     = {last_seq}")
print(f"gap_count    = {gap_count}")

if first_seq is not None and last_seq is not None:
    expected_packets = last_seq - first_seq + 1
    print(f"expected     = {expected_packets}")
    print(f"received %   = {packet_count / expected_packets * 100:.2f}%")

if first_time is not None and last_time is not None:
    dt = last_time - first_time
    print(f"duration     = {dt:.3f} s")