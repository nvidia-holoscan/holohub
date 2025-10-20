// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "data_writer.hpp"

using in_t = std::tuple<tensor_t<complex, 2>, cudaStream_t>;

namespace holoscan::ops {

void DataWriter::setup(OperatorSpec& spec) {
    spec.input<in_t>("in");
    spec.param(burst_size,
        "burst_size",
        "Burst size"
        "Number of samples to process in each burst");
    spec.param(num_bursts,
        "num_bursts",
        "Number of bursts"
        "Number of sample bursts to process at once");
}

void DataWriter::initialize() {
    holoscan::Operator::initialize();
    make_tensor(data_host, {num_bursts.get(), burst_size.get()}, MATX_HOST_MEMORY);
}

void DataWriter::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
    auto data = op_input.receive<in_t>("in").value();
    copy(data_host, std::get<0>(data));

    // TODO: the MatX write_csv function segfaults for some reason
    // io::write_csv(psd_host, "psdOut.csv", ",");
    auto meta = metadata();
    std::stringstream _file_path;
    _file_path << "data_writer_out";
    _file_path << "_ch" << meta->get<uint16_t>("channel_number", 0);
    _file_path << "_bw" << meta->get<double>("bandwidth_hz", 0.0) / 1.0e6;
    _file_path << "_freq" << meta->get<double>("rf_ref_freq_hz", 0.0) / 1.0e6;
    _file_path << ".dat";
    auto file_path = _file_path.str();

    HOLOSCAN_LOG_INFO("Writing {} bytes to {}",
                      sizeof(complex) * num_bursts * burst_size, file_path);
    std::ofstream out_file(file_path,
        std::ios::out | std::ios::binary | std::ios::trunc);
    out_file.write(reinterpret_cast<const char*>(data_host.Data()),
                   sizeof(complex) * num_bursts * burst_size);
    out_file.close();
}

}  // namespace holoscan::ops
