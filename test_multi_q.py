#!/usr/bin/env python3
from scapy.all import IP, UDP, Ether, sendp

packet = Ether() / IP(dst="10.10.100.2") / UDP(sport=4095, dport=4095) / ("X" * (1050 - 20 - 8))
packet2 = Ether() / IP(dst="10.10.100.2") / UDP(sport=4096, dport=4096) / ("X" * (1050 - 20 - 8))
sendp(packet, iface="enP5p3s0f0np0", count=1, verbose=1)
sendp(packet2, iface="enP5p3s0f0np0", count=1, verbose=1)
