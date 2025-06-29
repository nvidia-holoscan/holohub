/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPERATORS_ADVANCED_NETWORK_MEDIA_RX_FRAME_PROVIDER_H_
#define OPERATORS_ADVANCED_NETWORK_MEDIA_RX_FRAME_PROVIDER_H_

#include <memory>
#include "../common/frame_buffer.h"

namespace holoscan::ops {

/**
 * @brief Interface for frame buffer allocation and management
 *
 * This interface abstracts frame buffer allocation, allowing different
 * components to provide frame buffers without coupling to specific
 * allocation strategies. Implementations typically handle memory pool
 * management and frame lifecycle.
 */
class IFrameProvider {
 public:
  virtual ~IFrameProvider() = default;

  /**
   * @brief Get a new frame buffer for processing
   * @return Frame buffer or nullptr if allocation failed
   */
  virtual std::shared_ptr<FrameBufferBase> get_new_frame() = 0;

  /**
   * @brief Get expected frame size
   * @return Frame size in bytes
   */
  virtual size_t get_frame_size() const = 0;

  /**
   * @brief Check if frames are available for allocation
   * @return True if frames are available, false if pool is empty
   */
  virtual bool has_available_frames() const = 0;

  /**
   * @brief Return a frame back to the pool
   * @param frame Frame to return to pool
   */
  virtual void return_frame_to_pool(std::shared_ptr<FrameBufferBase> frame) = 0;
};

}  // namespace holoscan::ops

#endif  // OPERATORS_ADVANCED_NETWORK_MEDIA_RX_FRAME_PROVIDER_H_
