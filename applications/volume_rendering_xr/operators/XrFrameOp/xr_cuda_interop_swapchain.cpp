/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "xr_cuda_interop_swapchain.hpp"
#include "xr_session.hpp"

#include "holoscan/holoscan.hpp"

namespace holoscan::openxr {

namespace {
vk::ImageAspectFlags imageAspect(xr::SwapchainCreateInfo swapchain_create_info) {
  switch (swapchain_create_info.format) {
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_R8G8B8A8_UNORM:
      return vk::ImageAspectFlagBits::eColor;
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D32_SFLOAT:
      return vk::ImageAspectFlagBits::eDepth;
    default:
      throw std::runtime_error("Swapchain format not supported");
  }
}

vk::ImageLayout originalImageLayout(xr::SwapchainCreateInfo swapchain_create_info) {
  switch (swapchain_create_info.format) {
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_R8G8B8A8_UNORM:
      return vk::ImageLayout::eColorAttachmentOptimal;
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D32_SFLOAT:
      return vk::ImageLayout::eDepthStencilAttachmentOptimal;
    default:
      throw std::runtime_error("Swapchain format not supported");
  }
}
}  // namespace

std::unique_ptr<XrCudaInteropSwapchain> XrCudaInteropSwapchain::create(
    XrSession& session, xr::SwapchainCreateInfo swapchain_create_info) {
  return std::make_unique<XrCudaInteropSwapchain>(session, swapchain_create_info);
}

XrCudaInteropSwapchain::XrCudaInteropSwapchain(XrSession& session,
                                               xr::SwapchainCreateInfo swapchain_create_info)
    : session_(session),
      swapchain_create_info_(swapchain_create_info),
      handle_(session.handle().createSwapchainUnique(swapchain_create_info)),
      command_pool_(session.vk_device(),
                    {.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                     .queueFamilyIndex = session.vk_queue_family_index()}) {
  // Enumerate and persist Vulkan images from the OpenXR swapchain.
  std::vector<xr::SwapchainImageVulkanKHR> vulkan_images =
      handle_->enumerateSwapchainImagesToVector<xr::SwapchainImageVulkanKHR>();
  images_.resize(vulkan_images.size());
  for (int i = 0; i < vulkan_images.size(); i++) {
    images_[i].vk_image = vulkan_images[i].image;
  }

  // Create a transfer buffer per swapchain image for upstream renderers to
  // write into; buffers are allocated as Vulkan external memory and exported to
  // CUDA.
  for (int i = 0; i < vulkan_images.size(); i++) {
    vk::MemoryRequirements image_memory_requirements =
        (*session_.vk_device()).getImageMemoryRequirements(images_[i].vk_image);
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
    cudaExternalMemoryHandleDesc external_memory_handle_desc = {
        .type = cudaExternalMemoryHandleTypeOpaqueFd,
        .handle = {.fd = transfer_memory_fd},
        .size = buffer_memory_requirements.size,
    };
    if (cudaImportExternalMemory(&images_[i].cuda_transfer_memory, &external_memory_handle_desc) !=
        cudaSuccess) {
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
    cudaExternalSemaphoreHandleDesc external_handle_desc{
        .type = cudaExternalSemaphoreHandleTypeOpaqueFd,
        .handle = {.fd = semaphore_fd},
    };
    if (cudaImportExternalSemaphore(&images_[i].render_done_cuda_semaphore,
                                    &external_handle_desc) != cudaSuccess) {
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
    images_[i].command_buffer.begin({.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse});
    vk::ImageSubresourceRange imageSubresourceRange(
        {imageAspect(swapchain_create_info_), 0, 1, 0, 1});
    vk::ImageMemoryBarrier transfer_layout_barrier({
        .srcAccessMask = vk::AccessFlagBits::eNone,
        .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
        .oldLayout = originalImageLayout(swapchain_create_info_),
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = images_[i].vk_image,
        .subresourceRange = imageSubresourceRange,
    });
    images_[i].command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eBottomOfPipe,
                                              vk::PipelineStageFlagBits::eTransfer,
                                              {},
                                              nullptr,
                                              nullptr,
                                              transfer_layout_barrier);
    vk::BufferImageCopy buffer_image_copy({
        .imageSubresource = {
                .aspectMask = imageAspect(swapchain_create_info_),
                .layerCount = 1,
            },
        .imageExtent = {
                .width = swapchain_create_info.width,
                .height = swapchain_create_info.height,
                .depth = 1,
            },
    });
    images_[i].command_buffer.copyBufferToImage(*images_[i].transfer_buffer,
                                                images_[i].vk_image,
                                                vk::ImageLayout::eTransferDstOptimal,
                                                {buffer_image_copy});
    vk::ImageMemoryBarrier attachment_layout_barrier({
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eNone,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = originalImageLayout(swapchain_create_info_),
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = images_[i].vk_image,
        .subresourceRange = imageSubresourceRange,
    });
    images_[i].command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                              vk::PipelineStageFlagBits::eTopOfPipe,
                                              {},
                                              nullptr,
                                              nullptr,
                                              attachment_layout_barrier);
    images_[i].command_buffer.end();
  }
}

XrCudaInteropSwapchain::~XrCudaInteropSwapchain() {
  for (XrCudaInteropSwapchain::Image& image : images_) {
    cudaFree(image.cuda_transfer_buffer);
    cudaDestroyExternalMemory(image.cuda_transfer_memory);
    cudaDestroyExternalSemaphore(image.render_done_cuda_semaphore);
  }
}

namespace {
template <nvidia::gxf::VideoFormat C>
nvidia::gxf::VideoBufferInfo createVideoBufferInfo(
    const xr::SwapchainCreateInfo& swapchain_create_info) {
  nvidia::gxf::VideoTypeTraits<C> color_format;
  nvidia::gxf::VideoFormatSize<C> format_size;
  return nvidia::gxf::VideoBufferInfo{
      .width = swapchain_create_info.width,
      .height = swapchain_create_info.height,
      .color_format = color_format.value,
      .color_planes = format_size.getDefaultColorPlanes(swapchain_create_info.width,
                                                        swapchain_create_info.height),
      .surface_layout = nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR,
  };
}
}  // namespace

void XrCudaInteropSwapchain::acquire(holoscan::gxf::Entity entity) {
  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> buffer =
      static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::VideoBuffer>().value();

  current_image_index_ = handle_->acquireSwapchainImage({});

  handle_->waitSwapchainImage({xr::Duration::infinite()});

  nvidia::gxf::VideoBufferInfo video_buffer_info;
  switch (swapchain_create_info_.format) {
    case VK_FORMAT_D16_UNORM:
      video_buffer_info = createVideoBufferInfo<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY16>(
          swapchain_create_info_);
      break;
    case VK_FORMAT_D32_SFLOAT:
      video_buffer_info = createVideoBufferInfo<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32F>(
          swapchain_create_info_);
      break;
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_R8G8B8A8_UNORM:
      video_buffer_info = createVideoBufferInfo<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
          swapchain_create_info_);
      break;
    default:
      throw std::runtime_error("Swapchain format not supported");
  }
  vk::MemoryRequirements memory_requirements =
      images_[current_image_index_].transfer_buffer.getMemoryRequirements();
  buffer->wrapMemory(video_buffer_info,
                     memory_requirements.size,
                     nvidia::gxf::MemoryStorageType::kDevice,
                     images_[current_image_index_].cuda_transfer_buffer,
                     {});
}

void XrCudaInteropSwapchain::release(holoscan::gxf::Entity entity) {
  Image& image = images_[current_image_index_];

  // Fence the buffer copy operation on upstream CUDA events by first waiting
  // for the events on the session CUDA stream and second signaling the
  // semaphore on the CUDA stream then waiting for it on the Vulkan queue.
  auto cuda_events = entity.findAll<nvidia::gxf::CudaEvent>().value();
  for (auto cuda_event : cuda_events) {
    if (cuda_event.has_value() && cuda_event.value()->event().has_value()) {
      if (cudaStreamWaitEvent(session_.cuda_stream(), cuda_event.value()->event().value()) !=
          cudaSuccess) {
        throw std::runtime_error("cudaStreamWaitEvent failed");
      }
    }
  }
  cudaExternalSemaphoreSignalParams signal_params{};
  if (cudaSignalExternalSemaphoresAsync(
          &image.render_done_cuda_semaphore, &signal_params, 1, session_.cuda_stream()) !=
      cudaSuccess) {
    throw std::runtime_error("cudaSignalExternalSemaphoresAsync failed");
  }

  vk::SubmitInfo submitInfo = {
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &*image.render_done_vk_semaphore,
      .commandBufferCount = 1,
      .pCommandBuffers = &*image.command_buffer,
  };

  session_.vk_queue().submit(submitInfo);

  handle_->releaseSwapchainImage({});
}

}  // namespace holoscan::openxr
