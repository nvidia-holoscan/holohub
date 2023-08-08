R"(

#version 450

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

in vec2 tex_coords;



layout(binding = 0) uniform sampler2D surgical_video;
layout(binding = 1) uniform sampler2D preop_mesh_layer;
layout(binding = 2) uniform sampler2D surgical_tool_mask;

layout(location = 1) uniform bool swizzleVideo = false;
layout(location = 2) uniform bool applySurgicalToolOverlayEffect = true;
layout(location = 3) uniform bool applyAnonymizationEffect = true;
layout(location = 4) uniform bool toggleAnonymization = true;

layout(location = 0) out vec4 out_color;

//#define DEBUG_VIS

void main() {

  vec2 video_tex_coords = tex_coords;
  // surgical video
  
  if(applyAnonymizationEffect) {
    vec2 pixelation_f = vec2(40.0 / 1920.0, 40.0 / 1080.0);
    video_tex_coords = pixelation_f * floor(video_tex_coords / pixelation_f);
  }


  vec4 video_rgba = texture2D(surgical_video, video_tex_coords);
  if(swizzleVideo) {
    video_rgba = video_rgba.bgra;
  }
  // pre-op mesh rendering via VTK
  vec4 preop_mesh_layer = texture2D(preop_mesh_layer, tex_coords);

  // standard opengl like blending
  // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  //  dst  = dest*(1-src.alpha) + src*(src.alpha)
  out_color  = video_rgba * (1 - preop_mesh_layer.a);
  out_color += preop_mesh_layer;

  if(applySurgicalToolOverlayEffect) {
    out_color = texture2D(surgical_tool_mask, tex_coords).r > 0.0 ? video_rgba : out_color;
  }
  

// change to #if 1 to enable debug display of tool segmentation mask
#if 1
  if(texture2D(surgical_tool_mask, tex_coords).r > 0.0) {
    out_color.g = 1.0;
  }
#endif


#ifdef DEBUG_VIS
  vec2 debug_rect_size = vec2(0.33);  

  if(tex_coords.x > 1.0 - debug_rect_size.x && tex_coords.y > 1.0 - debug_rect_size.y) 
  {
    vec2 debug_texcoords = tex_coords;
    debug_texcoords -= (vec2(1.0) - debug_rect_size);
    debug_texcoords /= debug_rect_size;
    vec4 org_video_rgba = texture2D(surgical_video, debug_texcoords);
    out_color = org_video_rgba;
  }
#endif

  //out_color = vec4(tex_coords.xy, 0.0, 1.0);

};
)"