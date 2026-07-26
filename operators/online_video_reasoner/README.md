# Online Video Reasoner Operator

`OnlineVideoReasonerOp` turns a live RGB stream into bounded multimodal
reasoning requests. It supports:

- `image` mode, which submits one sampled frame as JPEG; and
- `video` mode, which submits an ordered rolling window as one MP4.

The video defaults match the
[Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) recommendation of
4 frames per second. The operator sends either an OpenAI-compatible
`image_url` or `video_url` data URL and emits text events from the response.

Use video mode for ongoing scene narration with models such as Cosmos3-Edge.
Each request preserves the ordered frames in a bounded window so the model can
describe activity and change instead of unrelated still images.

## Requirements

- Holoscan SDK 4.4.0 or later.
- NumPy, Requests, and an `ffmpeg` executable.
- An OpenAI-compatible `/v1/chat/completions` endpoint that accepts a base64
  `image_url` or MP4 `video_url`. The
  [`nvidia/Cosmos3-Edge` model card](https://huggingface.co/nvidia/Cosmos3-Edge#vllm)
  provides NVIDIA's current vLLM container and reasoner deployment command.

## Ports

- `input`: an `HxWx3` RGB `uint8` tensor in host or CUDA memory.
- `events`: dictionaries with `request_id`, `kind`, and kind-specific fields.
  `kind` is `started`, `delta`, `completed`, or `error`.
  A `started` event indicates that encoding is complete and HTTP dispatch is
  starting. Text and error events also carry `sequence` and either `text` or
  `message`.
  A `delta` is an SSE text chunk, not a guaranteed model-token boundary.

## Parameters

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `endpoint` | Required | Full HTTPS chat-completions URL, or HTTP URL on `localhost` or a literal loopback address. |
| `model` | Required | Model identifier sent in each request. |
| `prompt` | Required | Text instruction sent with every observation. |
| `mode` | `video` | `video` sends an MP4 clip; `image` sends one JPEG. |
| `tensor_name` | `""` | Name of the RGB tensor in each input entity. |
| `sample_fps` | `4.0` | Frame sampling rate and MP4 frame rate. |
| `clip_duration_s` | `4.0` | Target temporal window; video mode retains `round(sample_fps * clip_duration_s)` frames. |
| `request_interval_s` | `4.0` | Minimum interval between accepted requests. |
| `max_frame_gap_s` | `2 / sample_fps` | Maximum interval between sampled video frames before the rolling window is reset. |
| `max_tokens` | `128` | Maximum generated tokens requested from the model. |
| `connect_timeout_s` | `10.0` | Endpoint connection timeout in seconds. |
| `timeout_s` | `60.0` | Response read timeout in seconds. |
| `api_key_env` | `REASONER_API_KEY` | Environment variable read for an optional bearer token. |
| `stream` | `true` | Consume SSE deltas when true; otherwise consume one JSON response. |
| `request_options` | `{}` | Additional top-level request fields; `max_tokens`, `messages`, `model`, and `stream` cannot be overridden. |
| `ffmpeg` | `ffmpeg` | FFmpeg executable name or path. |

## Behaviour

Only one request is active at a time. Video frames continue to update a
fixed-size rolling window while that request runs. A capture gap longer than
`max_frame_gap_s`, or a window whose elapsed time differs from the requested
span by more than one sampling period, resets the window. This prevents frames
from opposite sides of a source interruption being encoded as continuous
video.

SSE deltas are bounded; the final `completed` event always contains the full
response and sets `deltas_dropped` if backpressure discarded any intermediate
chunks. Stopping the operator closes an active HTTP response before waiting for
the worker to finish. Local cleartext HTTP requests ignore environment proxy
settings so their media and credentials cannot be forwarded outside the host.

Pass a `PeriodicCondition` when constructing the operator. Its input is
optional so periodic ticks can drain response events after a finite source
stops.

```python
from datetime import timedelta

reasoner = OnlineVideoReasonerOp(
    fragment,
    PeriodicCondition(fragment, recess_period=timedelta(seconds=1 / 60)),
    endpoint="http://localhost:8000/v1/chat/completions",
    model="nvidia/Cosmos3-Edge",
    prompt="Describe what changes during this clip in one sentence.",
    mode="video",
    sample_fps=4,
    clip_duration_s=4,
    stream=True,
)
```

The API key is read from `REASONER_API_KEY` by default. Local vLLM endpoints
that do not require authentication can leave it unset. HTTP is accepted only
for `localhost` or a literal loopback IP address; use HTTPS for every non-local
endpoint.

## Validation status

Validated with Holoscan 4.4, replay input, and a live Cosmos3-Edge endpoint on
NVIDIA GB10 (`aarch64`) and an A6000 RTX workstation (`x86_64`).
