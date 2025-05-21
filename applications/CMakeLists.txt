# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Make sure to order by alphabetical list
add_holohub_application(adv_networking_bench DEPENDS
                        OPERATORS advanced_network)

add_holohub_application(aja_video_capture DEPENDS
                        OPERATORS aja_source)

add_holohub_application(basic_networking_ping DEPENDS
                        OPERATORS basic_network)

add_holohub_application(body_pose_estimation DEPENDS
                        OPERATORS OPTIONAL dds_video_subscriber dds_video_publisher)

add_holohub_application(colonoscopy_segmentation DEPENDS OPERATORS aja_source)

add_holohub_application(cvcuda_basic DEPENDS OPERATORS cvcuda_holoscan_interop)

add_subdirectory(dds)

add_holohub_application(deltacast_transmitter DEPENDS
                        OPERATORS deltacast_videomaster
                        )

add_holohub_application(depth_anything_v2)

add_subdirectory(distributed)

add_holohub_application(endoscopy_depth_estimation)

add_holohub_application(endoscopy_out_of_body_detection DEPENDS OPERATORS aja_source)

add_holohub_application(endoscopy_tool_tracking DEPENDS
                        OPERATORS lstm_tensor_rt_inference
                                  tool_tracking_postprocessor
                                  aja_source
                                  OPTIONAL deltacast_videomaster yuan_qcap vtk_renderer)

add_subdirectory(h264)

add_holohub_application(high_speed_endoscopy DEPENDS
                        OPERATORS emergent_source)

add_subdirectory(holoviz)

add_holohub_application(imaging_ai_segmentator)

add_holohub_application(hyperspectral_segmentation)

add_subdirectory(laser_detection_latency)

add_holohub_application(monai_endoscopic_tool_seg DEPENDS OPERATORS aja_source)
add_holohub_application(multiai_endoscopy DEPENDS OPERATORS aja_source)
add_holohub_application(multiai_ultrasound DEPENDS OPERATORS aja_source)

add_holohub_application(simple_radar_pipeline)
add_holohub_application(simple_pdw_pipeline DEPENDS
                        OPERATORS basic_network)

add_holohub_application(object_detection_torch DEPENDS OPERATORS aja_source)

add_holohub_application(openigtlink_3dslicer DEPENDS OPERATORS openigtlink)

add_holohub_application(pva_video_filter)

add_holohub_application(prohawk_video_replayer DEPENDS OPERATORS prohawk_video_processing)

add_holohub_application(qt_video_replayer DEPENDS OPERATORS qt_video npp_filter)

add_holohub_application(realsense_visualizer DEPENDS OPERATORS realsense_camera)

add_holohub_application(stereo_vision)

add_holohub_application(tao_peoplenet)

add_holohub_application(network_radar_pipeline DEPENDS
                        OPERATORS basic_network
                                  advanced_network)

add_holohub_application(psd_pipeline DEPENDS
                        OPERATORS advanced_network
                                  fft
                                  high_rate_psd
                                  low_rate_psd
                                  vita49_psd_packetizer
                                  data_writer)

add_holohub_application(ultrasound_segmentation DEPENDS OPERATORS aja_source)

add_holohub_application(velodyne_lidar_app DEPENDS
                        OPERATORS velodyne_lidar
                                  basic_network
                                  )

add_holohub_application(volume_rendering DEPENDS
                        OPERATORS volume_loader
                                  volume_renderer)

add_holohub_application(volume_rendering_xr DEPENDS
                        OPERATORS volume_loader volume_renderer)

add_holohub_application(vpi_stereo)

add_holohub_application(video_deidentification)

add_holohub_application(webrtc_video_server)

add_holohub_application(yolo_model_deployment)
add_holohub_application(vila_live)
add_holohub_application(xr_holoviz DEPENDS OPERATORS xr)
add_subdirectory(orsi)
add_subdirectory(nvidia_nim)
add_subdirectory(ehr_query_llm)
