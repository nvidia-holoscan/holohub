/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "xr_swapchain_cuda.hpp"

#include <memory>
#include <utility>
#include <vector>

#include "holoscan/holoscan.hpp"
#include "xr_session.hpp"

namespace holoscan {

// Derived types from swapchain format.
int channel_count(XrSwapchainCuda::Format format);
DLDataType dl_data_type(XrSwapchainCuda::Format format);
VkFormat vk_format(XrSwapchainCuda::Format format);
vk::ImageAspectFlags image_aspect(XrSwapchainCuda::Format format);
vk::ImageLayout original_image_layout(XrSwapchainCuda::Format format);
XrSwapchainUsageFlags swapchain_usage_flags(XrSwapchainCuda::Format format);

XrSwapchainCuda::XrSwapchainCuda(XrSession& session, Format format, uint32_t width,
                                 uint32_t height)
    : session_(session),
      format_(format),
      width_(width),
      height_(height),
      command_pool_(session.vk_device(),
                    {
                        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                        .queueFamilyIndex = session.vk_queue_family_index(),
                    }) {
  swapchain_ = session.get().createSwapchainUnique(xr::SwapchainCreateInfo({
      .type = XR_TYPE_SWAPCHAIN_CREATE_INFO,
      .createFlags = 0,
      .usageFlags = swapchain_usage_flags(format),
      .format = vk_format(format),
      .sampleCount = 1,
      .width = width,
      .height = height,
      .faceCount = 1,
      .arraySize = 1,
      .mipCount = 1,
  }));

  // Enumerate and persist Vulkan images from the OpenXR swapchain.
  std::vector<xr::SwapchainImageVulkanKHR> vulkan_images =
      swapchain_->enumerateSwapchainImagesToVector<xr::SwapchainImageVulkanKHR>();
  images_.resize(vulkan_images.size());
  for (int i = 0; i < vulkan_images.size(); i++) {
    images_[i].vk_image = vk::raii::Image(session_.vk_device(), vulkan_images[i].image);
  }

  // Create a transfer buffer per swapchain image for upstream renderers to
  // write into; buffers are allocated as Vulkan external memory and exported to
  // CUDA.
  for (int i = 0; i < vulkan_images.size(); i++) {
    vk::MemoryRequirements image_memory_requirements = images_[i].vk_image.getMemoryRequirements();
    uint32_t vk_queue_family_index = session_.vk_queue_family_index();
    vk::StructureChain<vk::BufferCreateInfo, vk::ExternalMemoryBufferCreateInfo> buffer_create_info{
        vk::BufferCreateInfo({
            .flags = {},
            .size = image_memory_requirements.size,
            .usage = vk::BufferUsageFlagBits::eTransferSrc,
            .sharingMode = vk::SharingMode::eExclusive,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vk_queue_family_index,
        }),
        vk::ExternalMemoryBufferCreateInfo({
            .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd,
        }),
    };
    images_[i].transfer_buffer =
        session.vk_device().createBuffer(buffer_create_info.get<vk::BufferCreateInfo>());
    vk::MemoryRequirements buffer_memory_requirements =
        images_[i].transfer_buffer.getMemoryRequirements();
    vk::PhysicalDeviceMemoryProperties device_memory_props =
        session_.vk_physical_device().getMemoryProperties();
    uint32_t memory_type_index = 0;
    for (uint32_t i = 0; i < device_memory_props.memoryTypeCount; i++) {
      if ((buffer_memory_requirements.memoryTypeBits & (1 << i)) &&
          (device_memory_props.memoryTypes[i].propertyFlags &
           vk::MemoryPropertyFlagBits::eDeviceLocal)) {
        memory_type_index = i;
      }
    }
    vk::StructureChain<vk::MemoryAllocateInfo, vk::ExportMemoryAllocateInfo> memory_alloc_info{
        vk::MemoryAllocateInfo({
            .allocationSize = buffer_memory_requirements.size,
            .memoryTypeIndex = memory_type_index,
        }),
        vk::ExportMemoryAllocateInfo({
            .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd,
        }),
    };
    images_[i].transfer_memory =
        session.vk_device().allocateMemory(memory_alloc_info.get<vk::MemoryAllocateInfo>());
    uint32_t kMemoryOffset = 0;
    images_[i].transfer_buffer.bindMemory(*images_[i].transfer_memory, kMemoryOffset);
    int transfer_memory_fd = session.vk_device().getMemoryFdKHR({
        .memory = *images_[i].transfer_memory,
        .handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd,
    });
    cudaExternalMemoryHandleDesc external_memory_swapchain_desc = {
        .type = cudaExternalMemoryHandleTypeOpaqueFd,
        .handle = {.fd = transfer_memory_fd},
        .size = buffer_memory_requirements.size,
    };
    if (cudaImportExternalMemory(&images_[i].cuda_transfer_memory,
                                 &external_memory_swapchain_desc) != cudaSuccess) {
      throw std::runtime_error("cudaImportExternalMemory failed");
    }
    cudaExternalMemoryBufferDesc external_memory_buffer_desc = {
        .offset = kMemoryOffset,
        .size = buffer_memory_requirements.size,
        .flags = 0,
    };
    if (cudaExternalMemoryGetMappedBuffer(&images_[i].cuda_transfer_buffer,
                                          images_[i].cuda_transfer_memory,
                                          &external_memory_buffer_desc) != cudaSuccess) {
      throw std::runtime_error("cudaExternalMemoryGetMappedBuffer failed");
    }
    if (cudaMemset(images_[i].cuda_transfer_buffer, 0, buffer_memory_requirements.size) !=
        cudaSuccess) {
      throw std::runtime_error("cudaMemset failed");
    }
    cudaPointerAttributes attributes = {};
    if (cudaPointerGetAttributes(&attributes, images_[i].cuda_transfer_buffer) != cudaSuccess) {
      throw std::runtime_error("cudaPointerGetAttributes failed");
    }
    images_[i].cuda_device = attributes.device;
  }


  // Create and export a Vulkan semaphore / CUDA external semaphore to signal
  // when upstream rendering finishes.
  for (int i = 0; i < images_.size(); i++) {
    vk::StructureChain<vk::SemaphoreCreateInfo, vk::ExportSemaphoreCreateInfoKHR> create_info{
        vk::SemaphoreCreateInfo(),
        vk::ExportSemaphoreCreateInfoKHR({
            .handleTypes = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd,
        }),
    };
    images_[i].render_done_vk_semaphore =
        session_.vk_device().createSemaphore(create_info.get<vk::SemaphoreCreateInfo>());
    vk::SemaphoreGetFdInfoKHR semaphore_get_fd_info{
        .semaphore = *images_[i].render_done_vk_semaphore,
        .handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd,
    };
    int semaphore_fd = session_.vk_device().getSemaphoreFdKHR(semaphore_get_fd_info);
    cudaExternalSemaphoreHandleDesc external_swapchain_desc{
        .type = cudaExternalSemaphoreHandleTypeOpaqueFd,
        .handle = {.fd = semaphore_fd},
    };
    if (cudaImportExternalSemaphore(&images_[i].render_done_cuda_semaphore,
                                    &external_swapchain_desc) != cudaSuccess) {
      throw std::runtime_error("cudaImportExternalSemaphore failed");
    }
  }

  // Record Vulkan command buffers to copy data from each transfer buffer to its
  // corresponding Vulkan image in the OpenXR swapchain.
  for (int i = 0; i < vulkan_images.size(); i++) {
    vk::CommandBufferAllocateInfo command_buffer_alloc_info({
        .commandPool = *command_pool_,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    });
    vk::raii::CommandBuffers command_buffers(session_.vk_device(), command_buffer_alloc_info);
    images_[i].command_buffer = std::move(command_buffers[0]);
    vk::raii::CommandBuffer& command_buffer = images_[i].command_buffer;
    command_buffer.begin({.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse});
    vk::ImageSubresourceRange imageSubresourceRange({image_aspect(format), 0, 1, 0, 1});
    vk::ImageMemoryBarrier transfer_layout_barrier({
        .srcAccessMask = vk::AccessFlagBits::eNone,
        .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
        .oldLayout = original_image_layout(format),
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = *images_[i].vk_image,
        .subresourceRange = imageSubresourceRange,
    });
    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eBottomOfPipe,
                                   vk::PipelineStageFlagBits::eTransfer,
                                   {},
                                   nullptr,
                                   nullptr,
                                   transfer_layout_barrier);
    vk::BufferImageCopy buffer_image_copy({
        .imageSubresource{
            .aspectMask = image_aspect(format),
            .layerCount = 1,
        },
        .imageExtent{
            .width = width,
            .height = height,
            .depth = 1,
        },
    });
    command_buffer.copyBufferToImage(*images_[i].transfer_buffer,
                                     *images_[i].vk_image,
                                     vk::ImageLayout::eTransferDstOptimal,
                                     {buffer_image_copy});
    vk::ImageMemoryBarrier attachment_layout_barrier({
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eNone,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = original_image_layout(format),
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = *images_[i].vk_image,
        .subresourceRange = imageSubresourceRange,
    });
    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eTopOfPipe,
                                   {},
                                   nullptr,
                                   nullptr,
                                   attachment_layout_barrier);
    command_buffer.end();
  }
}

XrSwapchainCuda::~XrSwapchainCuda() {
  for (XrSwapchainCuda::Image& image : images_) {
    cudaFree(image.cuda_transfer_buffer);
    cudaDestroyExternalMemory(image.cuda_transfer_memory);
    cudaDestroyExternalSemaphore(image.render_done_cuda_semaphore);

    // Let OpenXR destroy the swapchain images.
    image.vk_image.release();
  }
}

holoscan::Tensor XrSwapchainCuda::acquire() {
  current_image_index_ = swapchain_->acquireSwapchainImage({});
  Image& image = images_[current_image_index_];

  swapchain_->waitSwapchainImage({xr::Duration::infinite()});

  // Create a Holoscan Tensor for the image with DLPack.
  auto dl_context = std::make_shared<DLManagedTensorContext>();
  dl_context->dl_shape = {height_, width_, channel_count(format_)};
  dl_context->tensor = {
      .dl_tensor{
          .data = image.cuda_transfer_buffer,
          .device = {.device_type = kDLCUDA, .device_id = image.cuda_device},
          .ndim = static_cast<int32_t>(dl_context->dl_shape.size()),
          .dtype = dl_data_type(format_),
          .shape = dl_context->dl_shape.data(),
      },
  };
  return holoscan::Tensor(dl_context);
}

void XrSwapchainCuda::release(cudaStream_t cuda_stream) {
  Image& image = images_[current_image_index_];

  // Ensure all cuda_stream processing is complete prior to copying the buffer
  // by signaling the render done semaphore on the CUDA stream then waiting for
  // it on the Vulkan queue.
  cudaExternalSemaphoreSignalParams signal_params{};
  if (cudaSignalExternalSemaphoresAsync(
          &image.render_done_cuda_semaphore, &signal_params, 1, cuda_stream) != cudaSuccess) {
    throw std::runtime_error("cudaSignalExternalSemaphoresAsync failed");
  }

  vk::SubmitInfo submitInfo = {
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &*image.render_done_vk_semaphore,
      .commandBufferCount = 1,
      .pCommandBuffers = &*image.command_buffer,
  };

  session_.vk_queue().submit(submitInfo, /*fence=*/VK_NULL_HANDLE);

  swapchain_->releaseSwapchainImage({});
}

int channel_count(XrSwapchainCuda::Format format) {
  switch (format) {
    case XrSwapchainCuda::Format::R8G8B8A8_SRGB:
    case XrSwapchainCuda::Format::R8G8B8A8_UNORM:
      return 4;
    case XrSwapchainCuda::Format::D16_UNORM:
    case XrSwapchainCuda::Format::D32_SFLOAT:
      return 1;
    default:
      throw std::runtime_error("Swapchain format not supported");
  }
}

DLDataType dl_data_type(XrSwapchainCuda::Format format) {
  switch (format) {
    case XrSwapchainCuda::Format::R8G8B8A8_SRGB:
    case XrSwapchainCuda::Format::R8G8B8A8_UNORM:
      return {.code = kDLUInt, .bits = 8, .lanes = 1};
    case XrSwapchainCuda::Format::D16_UNORM:
      return {.code = kDLFloat, .bits = 16, .lanes = 1};
    case XrSwapchainCuda::Format::D32_SFLOAT:
      return {.code = kDLFloat, .bits = 32, .lanes = 1};
    default:
      throw std::runtime_error("Swapchain format not supported");
  }
}

VkFormat vk_format(XrSwapchainCuda::Format format) {
  switch (format) {
    case XrSwapchainCuda::Format::R8G8B8A8_SRGB:
      return VK_FORMAT_R8G8B8A8_SRGB;
    case XrSwapchainCuda::Format::R8G8B8A8_UNORM:
      return VK_FORMAT_R8G8B8A8_UNORM;
    case XrSwapchainCuda::Format::D16_UNORM:
      return VK_FORMAT_D16_UNORM;
    case XrSwapchainCuda::Format::D32_SFLOAT:
      return VK_FORMAT_D32_SFLOAT;
    default:
      throw std::runtime_error("Swapchain format not supported");
  }
}

vk::ImageAspectFlags image_aspect(XrSwapchainCuda::Format format) {
  switch (format) {
    case XrSwapchainCuda::Format::R8G8B8A8_SRGB:
    case XrSwapchainCuda::Format::R8G8B8A8_UNORM:
      return vk::ImageAspectFlagBits::eColor;
    case XrSwapchainCuda::Format::D16_UNORM:
    case XrSwapchainCuda::Format::D32_SFLOAT:
      return vk::ImageAspectFlagBits::eDepth;
    default:
      throw std::runtime_error("Swapchain format not supported");
  }
}

vk::ImageLayout original_image_layout(XrSwapchainCuda::Format format) {
  switch (format) {
    case XrSwapchainCuda::Format::R8G8B8A8_SRGB:
    case XrSwapchainCuda::Format::R8G8B8A8_UNORM:
      return vk::ImageLayout::eColorAttachmentOptimal;
    case XrSwapchainCuda::Format::D16_UNORM:
    case XrSwapchainCuda::Format::D32_SFLOAT:
      return vk::ImageLayout::eDepthStencilAttachmentOptimal;
    default:
      throw std::runtime_error("Swapchain format not supported");
  }
}

XrSwapchainUsageFlags swapchain_usage_flags(XrSwapchainCuda::Format format) {
  switch (format) {
    case XrSwapchainCuda::Format::R8G8B8A8_SRGB:
    case XrSwapchainCuda::Format::R8G8B8A8_UNORM:
      return XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
    case XrSwapchainCuda::Format::D16_UNORM:
    case XrSwapchainCuda::Format::D32_SFLOAT:
      return XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    default:
      throw std::runtime_error("Swapchain format not supported");
  }
}

}  // namespace holoscan
