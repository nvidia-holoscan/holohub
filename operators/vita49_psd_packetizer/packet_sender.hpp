/*
 * SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/ip.h>
#include <arpa/inet.h>
#include "holoscan/holoscan.hpp"

class PacketSender {
 public:
    PacketSender(
        const char *dest_host,
        const unsigned short dest_port,
        const uint32_t manufacturer_oui,
        const uint32_t device_code);
    int send_signal_data(
        const uint8_t *psd_data,
        const size_t psd_data_size,
        std::shared_ptr<holoscan::MetadataDictionary> meta);
    int send_context_packet(std::shared_ptr<holoscan::MetadataDictionary> meta);
    bool is_time_for_context();

    PacketSender() = default;

    std::string dest_host;
    unsigned short dest_port;

 private:
    uint64_t packet_count = 0;
    uint64_t context_packet_count = 0;
    static std::string lookup_host(const char *host);
    std::string dest_host_resolved;
    int sock;
    sockaddr_in dest;
    uint32_t manufacturer_oui;
    uint32_t device_code;
};
