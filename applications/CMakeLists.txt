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

add_holohub_application(basic_networking_ping DEPENDS
                        OPERATORS basic_network)

add_holohub_application(body_pose_estimation DEPENDS
                        OPERATORS OPTIONAL dds_video_subscriber dds_video_publisher)

add_holohub_application(colonoscopy_segmentation)

add_holohub_application(cvcuda_basic DEPENDS OPERATORS cvcuda_holoscan_interop)

add_holohub_application(dds_video DEPENDS
                        OPERATORS dds_shapes_subscriber
                                  dds_video_publisher
                                  dds_video_subscriber)

add_holohub_application(deltacast_transmitter DEPENDS
                        OPERATORS deltacast_videomaster
                        )

add_holohub_application(endoscopy_depth_estimation)

add_holohub_application(endoscopy_out_of_body_detection)

add_holohub_application(endoscopy_tool_tracking DEPENDS
                        OPERATORS lstm_tensor_rt_inference
                                  tool_tracking_postprocessor
                                  OPTIONAL deltacast_videomaster yuan_qcap vtk_renderer)

add_holohub_application(endoscopy_tool_tracking_distributed DEPENDS
                        OPERATORS lstm_tensor_rt_inference
                                  tool_tracking_postprocessor)
add_subdirectory(h264)

add_holohub_application(high_speed_endoscopy DEPENDS
                        OPERATORS emergent_source)

add_subdirectory(holoviz)

add_holohub_application(imaging_ai_segmentator)

add_holohub_application(hyperspectral_segmentation)

add_holohub_application(multiai_endoscopy)

add_holohub_application(multiai_ultrasound DEPENDS
                        OPERATORS visualizer_icardio)

add_holohub_application(simple_radar_pipeline)
add_holohub_application(simple_pdw_pipeline DEPENDS
                        OPERATORS basic_network)

add_holohub_application(object_detection_torch)

add_holohub_application(openigtlink_3dslicer DEPENDS OPERATORS openigtlink)

add_holohub_application(pva_video_filter)

add_holohub_application(prohawk_video_replayer DEPENDS OPERATORS prohawk_video_processing)

add_holohub_application(qt_video_replayer DEPENDS OPERATORS qt_video npp_filter)

add_holohub_application(realsense_visualizer DEPENDS OPERATORS realsense_camera)

add_holohub_application(tao_peoplenet)

add_holohub_application(network_radar_pipeline DEPENDS
                        OPERATORS basic_network
                                  advanced_network)

add_holohub_application(ultrasound_segmentation)

add_holohub_application(velodyne_lidar_app DEPENDS
                        OPERATORS velodyne_lidar
                                  basic_network
                                  )

add_holohub_application(volume_rendering DEPENDS
                        OPERATORS volume_loader
                                  volume_renderer)

add_holohub_application(volume_rendering_xr DEPENDS
                        OPERATORS volume_loader
                                  volume_renderer
                                  XrFrameOp
                                  XrTransformOp)

add_holohub_application(xr_hello_holoscan DEPENDS
                        OPERATORS volume_renderer # for nlohmann_json dependency
                                  XrFrameOp
                                  XrTransformOp
                                  xr_basic_render)

add_holohub_application(video_deidentification)

add_holohub_application(webrtc_video_server)

add_subdirectory(orsi)
add_subdirectory(nvidia_nim)
add_subdirectory(ehr_query_llm/domain_specific/fhir)
