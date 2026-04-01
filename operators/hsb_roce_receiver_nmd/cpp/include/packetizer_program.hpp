/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOHUB_HSB_ROCE_RECEIVER_NMD_PACKETIZER_PROGRAM_SHIM
#define HOLOHUB_HSB_ROCE_RECEIVER_NMD_PACKETIZER_PROGRAM_SHIM

// Compatibility shim:
// Some Hololink installs expose hololink/core/data_channel.hpp but do not ship
// packetizer_program.hpp. DataChannel only needs forward declarations here.
namespace hololink {
class PacketizerProgram;
class NullPacketizerProgram;
class Csi10ToPacked10;
class Csi12ToPacked12;
}  // namespace hololink

#endif /* HOLOHUB_HSB_ROCE_RECEIVER_NMD_PACKETIZER_PROGRAM_SHIM */
