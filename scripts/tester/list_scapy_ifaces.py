from scapy.arch.windows import get_windows_if_list

for idx, iface in enumerate(get_windows_if_list()):
    print("=" * 80)
    print("Index:", idx)
    print("Name:", iface.get("name"))
    print("Description:", iface.get("description"))
    print("GUID:", iface.get("guid"))
    print("MAC:", iface.get("mac"))
    print("IPs:", iface.get("ips"))