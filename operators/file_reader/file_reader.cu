// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "file_reader.hpp"

namespace holoscan::ops {

void FileReader::setup(OperatorSpec& spec) {
    spec.output<tensor_t<complex, 1>>("out");
    spec.param(file_name,
               "file_name",
               "Data file"
               "Raw data file to send");
    spec.param(burst_size,
               "burst_size",
               "Burst size"
               "Number of samples to send in each burst");
    spec.param(stream_id,
               "stream_id",
               "Stream ID",
               "VITA 49 stream ID to pass along in metadata");
    spec.param(rf_ref_freq_hz,
               "rf_ref_freq_hz",
               "Tune frequency"
               "Center tune frequency to pass along in metadata");
    spec.param(bandwidth_hz,
               "bandwidth_hz",
               "Bandwidth"
               "Bandwidth to pass along in metadata");
    spec.param(sample_rate_sps,
               "sample_rate_sps",
               "Sample rate"
               "Sample rate to derive burst rate and pass along in metadata");
    spec.param(ref_level_dbm,
               "ref_level_dbm",
               "Reference level"
               "Reference level to pass along in metadata");
    spec.param(gain_db,
               "gain_db",
               "Gain"
               "Gain to pass along in metadata");
}

static std::complex<float> cast_to_complex_float(int16_t real, int16_t imag) {
    // Scale the int16 values to -1.0 thru +1.0 by dividing by 2^15 - 1 (0x7FFF)
    constexpr float scalar = 1.0 / 0x7FFF;
    return std::complex<float>(
        static_cast<float>(real) * scalar,
        static_cast<float>(imag) * scalar);
}

void FileReader::initialize() {
    holoscan::Operator::initialize();

    file.open(file_name.get(), std::ios::binary);
    if (file.fail()) {
        HOLOSCAN_LOG_ERROR("could not open file {}", file_name.get());
        exit(1);
    }
    file.seekg(0, file.end);
    size_t length = file.tellg();
    file.seekg(0, file.beg);

    num_chunks = length / (burst_size.get() * sizeof(int16_t) * 2);

    if (num_chunks < 1) {
        HOLOSCAN_LOG_ERROR("File not large enough: {}", length);
        file.close();
        exit(1);
    }

    HOLOSCAN_LOG_INFO("Loading {} sample bursts into memory from {}", num_chunks, file_name.get());

    auto h_rf_data = make_tensor<complex, 2>({num_chunks, burst_size.get()}, MATX_HOST_MEMORY);
    make_tensor(rf_data, {num_chunks, burst_size.get()}, MATX_DEVICE_MEMORY);

    size_t interleaved_samples_size = burst_size.get() * 2 * sizeof(int16_t);
    int16_t *interleaved_int_array = reinterpret_cast<int16_t *>(malloc(interleaved_samples_size));
    if (!interleaved_int_array && interleaved_samples_size) {
        throw std::bad_alloc();
    }

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        file.read(reinterpret_cast<char*>(interleaved_int_array), interleaved_samples_size);

        if (file.eof() || file.fail()) {
            free(interleaved_int_array);
            throw std::runtime_error("Failed to read file data into tensor");
        }

        for (int samp = 0; samp < burst_size.get(); samp++) {
            h_rf_data(chunk, samp) = cast_to_complex_float(
                interleaved_int_array[samp * 2],
                interleaved_int_array[(samp * 2) + 1]);
        }
    }
    free(interleaved_int_array);

    copy(rf_data, h_rf_data);

    file.close();
}

void FileReader::compute(InputContext&, OutputContext& op_output, ExecutionContext&) {
    auto meta = metadata();
    meta->set("integer_timestamp", vrt_time.intTime());
    meta->set("fractional_timestamp", vrt_time.fracTime());
    op_output.emit(
        slice<1>(rf_data, {chunk_index++ % num_chunks, 0}, {matxDropDim, matxEnd}), "out");
    vrt_time.increment(burst_size.get(), sample_rate_sps.get());
}
}  // namespace holoscan::ops
