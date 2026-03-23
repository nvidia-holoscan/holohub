#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -euo pipefail

# Convert a local MP4 into GXF entities consumable by Holoscan's VideoStreamReplayerOp.
# Usage: ./prepare_video_gxf.sh [path/to/video.mp4] [output_directory]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOLOHUB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_VIDEO="${1:?Usage: $0 <path/to/video.mp4> [output_directory]}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/data/converted_video}"
BASENAME="video_stream"
META_PATH="${OUTPUT_DIR}/${BASENAME}.meta.json"

CONVERTER="${HOLOHUB_ROOT}/utilities/convert_video_to_gxf_entities.py"

if [ ! -f "${INPUT_VIDEO}" ]; then
  echo "Input video not found: ${INPUT_VIDEO}" >&2
  exit 1
fi

if [ ! -f "${CONVERTER}" ]; then
  echo "GXF converter not found at: ${CONVERTER}" >&2
  echo "Make sure you are running from within the holohub repository." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required. Please install ffmpeg and retry." >&2
  exit 1
fi

if ! python3 - <<'PY' 2>/dev/null
import numpy  # noqa: F401
PY
then
  echo "Python3 with numpy is required. Install numpy and retry." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Gathering video metadata from ${INPUT_VIDEO}..."
PROBE_OUT="$(ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,sample_aspect_ratio \
  -of default=nw=1:nk=1 "${INPUT_VIDEO}")"

mapfile -t PROBE_LINES <<<"${PROBE_OUT}"
WIDTH="${PROBE_LINES[0]:-}"
HEIGHT="${PROBE_LINES[1]:-}"
SAR_RAW="${PROBE_LINES[2]:-}"
FPS_RAW="${PROBE_LINES[3]:-}"

if [[ -z "${WIDTH}" || -z "${HEIGHT}" || -z "${FPS_RAW}" ]]; then
  echo "Failed to read video metadata. ffprobe output:" >&2
  echo "${PROBE_OUT}" >&2
  exit 1
fi

if [[ -z "${SAR_RAW}" || "${SAR_RAW}" == "N/A" ]]; then
  SAR_RAW="1:1"
fi

read FPS_FLOAT FPS_INT <<<"$(FPS_RAW="${FPS_RAW}" python3 - <<'PY'
import os, math
raw = os.environ["FPS_RAW"]
try:
    num, den = raw.split("/")
    val = float(num) / float(den)
except Exception:
    val = float(raw)
fps_int = max(1, int(round(val)))
print(f"{val:.3f} {fps_int}")
PY
)"

read OUTPUT_WIDTH OUTPUT_HEIGHT SAR_FLOAT <<<"$(WIDTH="${WIDTH}" HEIGHT="${HEIGHT}" SAR="${SAR_RAW}" python3 - <<'PY'
import os, math
width = int(os.environ["WIDTH"])
height = int(os.environ["HEIGHT"])
sar_raw = os.environ["SAR"]
try:
    num, den = sar_raw.split(":")
    sar_val = float(num) / float(den)
    if sar_val <= 0:
        raise ValueError
except Exception:
    sar_val = 1.0

scaled_width = int(round(width * sar_val))
scaled_height = height

if scaled_width % 2:
    scaled_width += 1
if scaled_height % 2:
    scaled_height += 1

print(f"{scaled_width} {scaled_height} {sar_val:.6f}")
PY
)"

FFMPEG_FILTER_ARGS=()
if [[ "${OUTPUT_WIDTH}" != "${WIDTH}" || "${OUTPUT_HEIGHT}" != "${HEIGHT}" ]]; then
  echo "Rescaling for pixel aspect ratio ${SAR_RAW} -> ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT}"
  FFMPEG_FILTER_ARGS=(-vf "scale=${OUTPUT_WIDTH}:${OUTPUT_HEIGHT}")
else
  echo "Using square pixels (SAR=${SAR_RAW}); keeping ${WIDTH}x${HEIGHT}"
fi

TARGET_WIDTH="${TARGET_WIDTH:-}"
TARGET_HEIGHT="${TARGET_HEIGHT:-}"
FINAL_WIDTH="${OUTPUT_WIDTH}"
FINAL_HEIGHT="${OUTPUT_HEIGHT}"
PAD_X=0
PAD_Y=0

if [[ -n "${TARGET_WIDTH}" || -n "${TARGET_HEIGHT}" ]]; then
  if [[ -z "${TARGET_WIDTH}" || -z "${TARGET_HEIGHT}" ]]; then
    echo "Both TARGET_WIDTH and TARGET_HEIGHT must be set together." >&2
    exit 1
  fi
  if ! [[ "${TARGET_WIDTH}" =~ ^[1-9][0-9]*$ && "${TARGET_HEIGHT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TARGET_WIDTH/TARGET_HEIGHT must be positive integers." >&2
    exit 1
  fi
  if (( TARGET_WIDTH % 2 )); then TARGET_WIDTH=$((TARGET_WIDTH + 1)); fi
  if (( TARGET_HEIGHT % 2 )); then TARGET_HEIGHT=$((TARGET_HEIGHT + 1)); fi

  FINAL_WIDTH="${TARGET_WIDTH}"
  FINAL_HEIGHT="${TARGET_HEIGHT}"
  scale_num=0
  scale_den=0
  if (( OUTPUT_WIDTH * TARGET_HEIGHT <= TARGET_WIDTH * OUTPUT_HEIGHT )); then
    scale_num="${TARGET_HEIGHT}"
    scale_den="${OUTPUT_HEIGHT}"
  else
    scale_num="${TARGET_WIDTH}"
    scale_den="${OUTPUT_WIDTH}"
  fi
  SCALED_WIDTH=$((OUTPUT_WIDTH * scale_num / scale_den))
  SCALED_HEIGHT=$((OUTPUT_HEIGHT * scale_num / scale_den))
  PAD_X=$(((FINAL_WIDTH - SCALED_WIDTH) / 2))
  PAD_Y=$(((FINAL_HEIGHT - SCALED_HEIGHT) / 2))

  echo "Letterboxing to ${FINAL_WIDTH}x${FINAL_HEIGHT}"
  FFMPEG_FILTER_ARGS=(-vf "scale=${FINAL_WIDTH}:${FINAL_HEIGHT}:force_original_aspect_ratio=decrease,pad=${FINAL_WIDTH}:${FINAL_HEIGHT}:(ow-iw)/2:(oh-ih)/2")
fi

echo "Converting ${INPUT_VIDEO} -> ${OUTPUT_DIR}/${BASENAME}.* (WxH=${FINAL_WIDTH}x${FINAL_HEIGHT}, FPS=${FPS_FLOAT})"

rm -f "${OUTPUT_DIR}/${BASENAME}.gxf_index" "${OUTPUT_DIR}/${BASENAME}.gxf_entities" "${OUTPUT_DIR}/${BASENAME}.stamp"
rm -f "${META_PATH}"

ffmpeg -y -v error -i "${INPUT_VIDEO}" "${FFMPEG_FILTER_ARGS[@]}" -f rawvideo -pix_fmt rgb24 - \
  | python3 "${CONVERTER}" \
      --width "${FINAL_WIDTH}" \
      --height "${FINAL_HEIGHT}" \
      --channels 3 \
      --framerate "${FPS_INT}" \
      --basename "${BASENAME}" \
      --directory "${OUTPUT_DIR}"

echo "Conversion complete. GXF assets at ${OUTPUT_DIR}"

cat > "${META_PATH}" <<EOF
{
  "source_width": ${WIDTH},
  "source_height": ${HEIGHT},
  "sample_aspect_ratio": "${SAR_RAW}",
  "content_width": ${OUTPUT_WIDTH},
  "content_height": ${OUTPUT_HEIGHT},
  "output_width": ${FINAL_WIDTH},
  "output_height": ${FINAL_HEIGHT},
  "pad_x": ${PAD_X},
  "pad_y": ${PAD_Y}
}
EOF
echo "Wrote metadata to ${META_PATH}"
