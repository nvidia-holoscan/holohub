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

out vec2 tex_coords;

void main()
{
        /*
           This vertex shader is used for rendering a single fullscreen
           covering triangle. The vertex position in clip space and
           2D texture coordinates are computed from the vertex id.

           Vertex ID   |   Vertex Pos  |  Texture Coords
           ----------------------------------------------
                0      |    [-1, -1]   |   [ 0,  0 ]
                1      |    [ 3, -1]   |   [ 2,  0 ]
                2      |    [-1,  3]   |   [ 0, -2 ]

           The triangle fully covers the clip space region and creates
           fragments with vertex positions in the range [-1, 1] and
           texture coordinates in the range X: [0, 1], Y: [0, -1].
           Please note that for display of surgical video the texture
           coordinates y - component is flipped.

           The main benefit of using this shader is that no vertex and
           vertex attribute buffers are required. Only a single call to
           glDrawArrays to render one triangle is needed.

         */
        float x = -1.0 + float((gl_VertexID & 1) << 2);
        float y = -1.0 + float((gl_VertexID & 2) << 1);
        tex_coords =  0.5 * vec2(x,y) + vec2(0.5);
        tex_coords.y = 1.0 - tex_coords.y;
        gl_Position = vec4(x, y, 0.0, 1.0);
}

)"