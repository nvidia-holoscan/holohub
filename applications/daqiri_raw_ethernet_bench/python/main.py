#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import ipaddress
import signal
import struct
import sys
import threading
import time
from dataclasses import dataclass

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required: python3 -m pip install pyyaml") from exc

import daqiri

ETH_HEADER_LEN = 14
IPV4_HEADER_LEN = 20
IPPROTO_UDP = 17


@dataclass
class RawTxConfig:
    interface_name: str = "tx_port"
    batch_size: int = 1024
    payload_size: int = 1000
    header_size: int = 64
    udp_src_port: str = "4096"
    udp_dst_port: str = "4096"
    ip_src_addr: str = "1.2.3.4"
    ip_dst_addr: str = "5.6.7.8"
    eth_dst_addr: str = "00:00:00:00:00:00"


@dataclass
class RawRxConfig:
    interface_name: str = "rx_port"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAQIRI raw GPUDirect TX/RX benchmark")
    parser.add_argument(
        "config", help="YAML config, e.g. daqiri_raw_ethernet_bench_default_tx_rx.yaml"
    )
    parser.add_argument(
        "--seconds", type=int, default=10, help="seconds to run; <=0 runs until Ctrl-C"
    )
    parser.add_argument(
        "--mode",
        choices=("both", "sender", "receiver", "tx", "rx"),
        default="both",
        help="which benchmark workers to run",
    )
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_rx(root: dict) -> RawRxConfig:
    data = root.get("bench_rx") or {}
    return RawRxConfig(interface_name=str(data.get("interface_name", "rx_port")))


def parse_tx(root: dict) -> RawTxConfig:
    data = root.get("bench_tx") or {}
    return RawTxConfig(
        interface_name=str(data.get("interface_name", "tx_port")),
        batch_size=int(data.get("batch_size", 1024)),
        payload_size=int(data.get("payload_size", 1000)),
        header_size=int(data.get("header_size", 64)),
        udp_src_port=str(data.get("udp_src_port", "4096")),
        udp_dst_port=str(data.get("udp_dst_port", "4096")),
        ip_src_addr=str(data.get("ip_src_addr", "1.2.3.4")),
        ip_dst_addr=str(data.get("ip_dst_addr", "5.6.7.8")),
        eth_dst_addr=str(data.get("eth_dst_addr", "00:00:00:00:00:00")),
    )


def parse_udp_ports(spec: str) -> list[int]:
    if "-" not in spec:
        return [int(spec)]
    begin, end = (int(part) for part in spec.split("-", 1))
    if begin > end:
        raise ValueError(f"invalid UDP port range: {spec}")
    return list(range(begin, end + 1))


def parse_mac(addr: str) -> bytes:
    parts = addr.split(":")
    if len(parts) != 6:
        raise ValueError(f"invalid MAC address: {addr}")
    return bytes(int(part, 16) for part in parts)


def build_udp_ipv4_packet_template(cfg: RawTxConfig, src_port: int, dst_port: int) -> bytes:
    packet = bytearray(cfg.header_size + cfg.payload_size)
    packet[0:6] = parse_mac(cfg.eth_dst_addr)
    packet[12:14] = struct.pack("!H", 0x0800)

    ip_offset = ETH_HEADER_LEN
    udp_offset = ETH_HEADER_LEN + IPV4_HEADER_LEN
    payload_offset = cfg.header_size

    ip_total_len = cfg.payload_size + cfg.header_size - ETH_HEADER_LEN
    udp_total_len = cfg.payload_size + cfg.header_size - (ETH_HEADER_LEN + IPV4_HEADER_LEN)

    packet[ip_offset] = 0x45
    packet[ip_offset + 2 : ip_offset + 4] = struct.pack("!H", ip_total_len)
    packet[ip_offset + 9] = IPPROTO_UDP
    packet[ip_offset + 12 : ip_offset + 16] = ipaddress.IPv4Address(cfg.ip_src_addr).packed
    packet[ip_offset + 16 : ip_offset + 20] = ipaddress.IPv4Address(cfg.ip_dst_addr).packed

    packet[udp_offset : udp_offset + 2] = struct.pack("!H", src_port)
    packet[udp_offset + 2 : udp_offset + 4] = struct.pack("!H", dst_port)
    packet[udp_offset + 4 : udp_offset + 6] = struct.pack("!H", udp_total_len)
    packet[payload_offset:] = bytes(i & 0xFF for i in range(cfg.payload_size))
    return bytes(packet)


def tx_worker(cfg: RawTxConfig, stop: threading.Event) -> None:
    port_id = daqiri.get_port_id(cfg.interface_name)
    if port_id < 0:
        print(f"Invalid TX interface_name: {cfg.interface_name}", file=sys.stderr)
        stop.set()
        return

    src_ports = parse_udp_ports(cfg.udp_src_port)
    dst_ports = parse_udp_ports(cfg.udp_dst_port)
    src_idx = 0
    dst_idx = 0
    sent_bursts = 0
    sent_packets = 0
    sent_bytes = 0
    packet_size = cfg.header_size + cfg.payload_size
    initialized_tx_buffers: set[int] = set()
    can_reuse_packet_templates = len(src_ports) == 1 and len(dst_ports) == 1
    packet_templates: dict[tuple[int, int], bytes] = {}
    start = time.monotonic()

    while not stop.is_set():
        burst = daqiri.create_tx_burst_params()
        daqiri.set_header(burst, port_id, 0, cfg.batch_size, 1)

        if not daqiri.is_tx_burst_available(burst):
            daqiri.free_tx_metadata(burst)
            time.sleep(0.0001)
            continue

        status = daqiri.get_tx_packet_burst(burst)
        if status != daqiri.Status.SUCCESS:
            daqiri.free_tx_metadata(burst)
            time.sleep(0.0001)
            continue

        failed = False
        num_pkts = daqiri.get_num_packets(burst)
        for pkt_idx in range(num_pkts):
            src_port = src_ports[src_idx]
            dst_port = dst_ports[dst_idx]
            src_idx = (src_idx + 1) % len(src_ports)
            dst_idx = (dst_idx + 1) % len(dst_ports)

            key = (src_port, dst_port)
            packet_template = packet_templates.get(key)
            if packet_template is None:
                packet_template = build_udp_ipv4_packet_template(cfg, src_port, dst_port)
                packet_templates[key] = packet_template

            pkt_ptr = daqiri.get_segment_packet_ptr(burst, 0, pkt_idx)
            if not can_reuse_packet_templates or pkt_ptr not in initialized_tx_buffers:
                status = daqiri.copy_buffer_to_segment_packet(burst, 0, pkt_idx, packet_template)
                if status != daqiri.Status.SUCCESS:
                    failed = True
                    break
                initialized_tx_buffers.add(pkt_ptr)

            status = daqiri.set_packet_lengths(burst, pkt_idx, [packet_size])
            if status != daqiri.Status.SUCCESS:
                failed = True
                break

        if failed:
            daqiri.free_all_packets_and_burst_tx(burst)
            continue

        status = daqiri.send_tx_burst(burst)
        if status == daqiri.Status.SUCCESS:
            sent_bursts += 1
            sent_packets += num_pkts
            sent_bytes += num_pkts * packet_size

    elapsed = max(time.monotonic() - start, 1e-9)
    print(
        "TX complete: "
        f"packets={sent_packets} bytes={sent_bytes} bursts={sent_bursts} "
        f"seconds={elapsed:.3f} pps={sent_packets / elapsed:.3f} "
        f"gbps={sent_bytes * 8 / elapsed / 1e9:.3f}"
    )


def rx_worker(cfg: RawRxConfig, stop: threading.Event) -> None:
    port_id = daqiri.get_port_id(cfg.interface_name)
    if port_id < 0:
        print(f"Invalid RX interface_name: {cfg.interface_name}", file=sys.stderr)
        stop.set()
        return

    packets = 0
    total_bytes = 0
    bursts = 0
    start = time.monotonic()

    while not stop.is_set():
        got_any = False
        for queue_id in range(daqiri.get_num_rx_queues(port_id)):
            status, burst = daqiri.get_rx_burst(port_id, queue_id)
            if status != daqiri.Status.SUCCESS or burst is None:
                continue

            got_any = True
            packets += daqiri.get_num_packets(burst)
            total_bytes += daqiri.get_burst_tot_byte(burst)
            bursts += 1
            daqiri.free_all_packets_and_burst_rx(burst)

        if not got_any:
            time.sleep(0.0001)

    elapsed = max(time.monotonic() - start, 1e-9)
    print(
        "RX complete: "
        f"packets={packets} bytes={total_bytes} bursts={bursts} "
        f"seconds={elapsed:.3f} pps={packets / elapsed:.3f} "
        f"gbps={total_bytes * 8 / elapsed / 1e9:.3f}"
    )


def should_run_rx(mode: str, root: dict) -> bool:
    return mode in ("both", "receiver", "rx") and bool(
        (root.get("bench_rx") or {}).get("interface_name")
    )


def should_run_tx(mode: str, root: dict) -> bool:
    return mode in ("both", "sender", "tx") and bool(
        (root.get("bench_tx") or {}).get("interface_name")
    )


def main() -> int:
    args = parse_args()
    root = load_yaml(args.config)

    status = daqiri.daqiri_init(args.config)
    if status != daqiri.Status.SUCCESS:
        print(f"daqiri_init failed: {status}", file=sys.stderr)
        return 1

    run_rx = should_run_rx(args.mode, root)
    run_tx = should_run_tx(args.mode, root)
    if not run_rx and not run_tx:
        print("Config and mode did not select bench_rx or bench_tx", file=sys.stderr)
        daqiri.shutdown()
        return 1

    stop = threading.Event()

    def request_stop(signum, frame):
        del signum, frame
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    threads: list[threading.Thread] = []
    if run_rx:
        threads.append(threading.Thread(target=rx_worker, args=(parse_rx(root), stop), daemon=True))
    if run_tx:
        threads.append(threading.Thread(target=tx_worker, args=(parse_tx(root), stop), daemon=True))

    for thread in threads:
        thread.start()

    try:
        start = time.monotonic()
        while not stop.is_set():
            if args.seconds > 0 and time.monotonic() - start >= args.seconds:
                break
            time.sleep(0.1)
    finally:
        stop.set()
        for thread in threads:
            thread.join()
        daqiri.print_stats()
        daqiri.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
