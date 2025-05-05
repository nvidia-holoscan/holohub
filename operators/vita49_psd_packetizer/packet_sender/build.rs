// SPDX-FileCopyrightText: 2025 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
fn main() {
    let _build = cxx_build::bridge("src/packet_sender.rs");
    println!("cargo:rerun-if-changed=src/packet_sender.rs");
}
