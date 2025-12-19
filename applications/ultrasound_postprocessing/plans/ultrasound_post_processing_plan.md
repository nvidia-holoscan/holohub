Ultrasound Post-Processing Library — Step-by-Step Development Plan
=================================================================

## Overview

A modular, CUDA-accelerated post-processing library for **beam-formed ultrasound images** stored in **UFF format**. Each operator is implemented as a Holoscan-style Python operator using **CuPy** and **cuCIM** (no OpenCV). The MVP focuses on a single **Gamma Compression** operator and an interactive **Streamlit** GUI.

### Scope

* **Input**: Beam-formed RF or B-mode images from UFF files.
* **Reader**: `pyuff_ustb` (Magnus Kvalevåg’s Python reader).
* **All processing**: On GPU with CuPy/cuCIM.
* **Out of scope**: Beamforming, scan conversion.
* **Frontend**: Streamlit-based, auto-generated parameter controls, live preview.

---

## Step 0 — Bootstrap (½ day)

**Goal:** Create a minimal, CUDA-only repository skeleton.

**Dependencies**
`cupy-cudaXX`, `cucim`, `pyuff_ustb`, `streamlit`, `pydantic`, `pyyaml`

**Structure:**

```
ultra_post/
  app/gui/streamlit_app.py
  core/base.py           # USOp, registry, tensor utils
  core/pipeline.py       # ordered list apply (GPU)
  io/yaml_io.py          # stub for later YAML support
  io/uff_loader.py       # UFF → CuPy tensor
  ops/gamma.py           # Step 1 operator
  examples/demo.uff      # One B-mode frame
```

**Acceptance:** `streamlit run app/gui/streamlit_app.py` launches an empty shell successfully.

---

## Step 1 — Gamma Compression MVP (1 day)

**Goal:** Load one B-mode image from UFF, apply a **Gamma Compression** filter, and view it live.

### 1.1 — UFF Loader

* Implement `io/uff_loader.py` using `pyuff_ustb`.
* Read beam-formed image data and normalize to `[0,1]` on GPU.
* Return: `{"data": cupy.ndarray, "meta": {...}}`.

### 1.2 — Gamma Compression Operator

File: `ops/gamma.py`

**Equation:** `y = x**(1/gamma)`  (with `x∈[0,1]`, `gamma<1` brightens)

**Parameters:**

```python
class Params(BaseModel):
    gamma: float = Field(1.0, ge=0.2, le=3.0, description="Gamma value", json_schema_extra={"widget": "slider", "step": 0.05})
    enable: bool = True
```

**Implementation:** Elementwise CuPy (`cp.power`), in-place safe.

### 1.3 — Streamlit UI

* File loader (uses bundled demo.uff).
* Gamma slider (0.2–3.0).
* Split view: Original vs processed.
* Live GPU updates; copy to CPU (`.get()`) only for display.

**Acceptance:**

* Real-time adjustment of gamma.
* Smooth preview (>15 FPS on 512×512).
* No CPU computations except display.

---

## Step 2 — Save/Load Preset (½ day)

**Goal:** Add pipeline persistence with YAML.

**YAML format:**

```yaml
version: 0.1
graph:
  - op: GammaCompression
    params: { gamma: 0.8, enable: true }
```

**UI additions:** “Save” and “Load” buttons.

**Acceptance:** Saved YAML reloads and reproduces identical output.

---

## Step 3 — Operator Palette & Reorder (1 day)

**Goal:** Support multiple ops (still just gamma in practice).

* Introduce `USOp` base class + registry.
* Implement `pipeline.py` with ordered op list and enable/disable.
* UI: Add/remove/reorder ops; toggle enable.

**Acceptance:** Two gamma ops with different params produce expected sequential results.

---

## Step 4 — Add Core Filters (2 days)

**Goal:** Expand GPU operator set.

1. **Median Filter** — `cupyx.scipy.ndimage.median_filter(ksize=3–7)`
2. **Gaussian/Low-Pass** — `cupyx.scipy.ndimage.gaussian_filter(sigma=...)`
3. **Unsharp Mask** — `y = x + amount * (x - gaussian(x, r))`

**Acceptance:** All filters run on GPU, togglable, reorderable, and interactively adjustable.

---

## Step 5 — CLAHE (1 day)

**Goal:** Implement adaptive gray mapping using cuCIM.

* Function: `cucim.skimage.exposure.equalize_adapthist(image, kernel_size=(8,8), clip_limit=0.01)`
* Params: `tiles`, `clip_limit`, `nbins` (expert-only).

**Acceptance:** CLAHE enhances local contrast smoothly at interactive rates.

---

## Step 6 — Holoscan-First Integration (1–2 days)

Goal: Use Holoscan as the primary runtime (avoid custom streaming loops), following Holohub patterns.

What we’ll add:

- Optional dependency: `holoscan` (as an extra); imports gated.
- Minimal Holoscan Python app (per Holohub hello_world/simple-pipeline patterns): UFFSource → PipelineOp (wraps `Pipeline.run`) → Holoviz display.
- Zero‑copy path: pass device buffers via CUDA Array Interface/DLPack to avoid host round‑trips.
- Version note: confirm decorator/function-operator API vs class-based `Operator` for the installed SDK and choose accordingly.

References from Holohub (structure only):
- hello_world/python (Application + Graph wiring)
- holoviz operator usage and tensor message types
- endoscopy_tool_tracking/python (pattern for Python operators and graph composition)

Acceptance:
- The Holoscan app processes demo UFF at ≈30 FPS with visual parity to Streamlit.
- No custom while-loops or timers; orchestration is via Holoscan scheduler.

---

## Step 7 — UFF Source Operator (1 day)

Goal: Implement a Holoscan source that emits frames from UFF on GPU.

What we’ll add:

- `UffSourceOp` (Python): reads via `io/uff_loader.py`/`io/uff_stream.py`, outputs device tensors compatible with Holoviz.
- Handles `--dataset`, looping, and start index; relies on Holoscan scheduling rather than custom sleeps.
- Ensures compatibility with CuPy buffers and Holoscan tensor message type (via DLPack/CUDA array interface per SDK version).

Acceptance:
- Frames flow into Holoviz with zero/minimal host copying.

---

## Step 8 — Pipeline Operator Wrapper (1 day)

Goal: Wrap our `Pipeline.run` as a Holoscan operator.

What we’ll add:

- `PipelineOp`: receives a device tensor, applies enabled ops in order, emits device tensor (2D grayscale or 3‑channel RGB).
- Operator parameters include dynamic range and optional colorization ordering (ColorMap last) to mirror UI behavior.

Acceptance:
- Output matches the Streamlit visualization on the same preset.

---

## Step 9 — Dynamic Reconfig via Control Plane (1–2 days)

Goal: Live updates of parameters and operator ordering without restarting the Holoscan app; driven by the Replit interface.

What we’ll add:

- Lightweight control server (WebSocket/REST) embedded in the Holoscan app or sidecar.
- Param-only updates applied lock-free or with brief lock inside `PipelineOp`; structural changes use atomic swap of the in‑memory `Pipeline`.
- Optional file-watcher fallback reading YAML for environments without a control port.

Acceptance:
- Param changes apply within <100 ms median latency; structural swaps within 1–2 frames with graceful handoff.

---

## Step 9 — Holoscan Function-Style Operators (2–3 days)

Goal: Integrate Holoscan using its function-operator decorator for a fast end‑to‑end streaming app built from the saved YAML pipeline.

What we’ll add:

- Dependency: add optional `holoscan` extra; gate imports so non-Holoscan flows still work.
- Function-style ops: implement source → pipeline → sink as function operators (decorator-based) for minimal boilerplate.
  - UFF source op: reads frames on GPU (CuPy) and emits device buffers + lightweight metadata.
  - Pipeline op: wraps our `Pipeline.run` (or color-map-last variant) on the incoming device tensor.
  - Display op: Holoviz sink (or headless FPS logger) to visualize/process results.
- App/CLI: build a `holoscan.core.Application` wiring the three ops; CLI `uspp-holoscan --preset pipeline.yaml --uff file.uff [--dataset beamformed] [--fps 30]`.
- Zero‑copy: prefer passing device buffers without host copies (DLPack/CUDA array interface as needed by the selected Holoscan version).
- Hot‑swap: reuse Step 8’s watcher to atomically replace the in‑memory `Pipeline` inside the pipeline op.

Notes:

- Confirm the exact decorator name/signature for the project’s Holoscan SDK version; fall back to class‑based `Operator` if required.

Acceptance:

- Runs at target FPS on demo data with visual parity to the Streamlit app.
- Pipeline changes from YAML are applied at runtime without restarting the Holoscan app.

---

## Step 10 — Replit Interface (1–2 days)

Goal: Use a Replit-hosted interface to edit pipeline YAML and parameters and push updates into the running Holoscan app.

What we’ll add:

- Minimal Replit UI (reuse existing Streamlit widgets or a small web form) that edits the same YAML schema and publishes updates over WS/HTTP to the control plane.
- Stable addressing: deterministic `op_id` in YAML for targeting specific operators.
- Auth/transport: configurable target URL/port; local-file fallback for dev when WS/HTTP is unavailable.

Acceptance:

- Editing in Replit updates the live Holoscan app without restart.
- Param changes reflect within <100 ms; preset swaps within 1–2 frames.

---

## Step 11 — Holoscan CLI Entrypoint (½ day)

Goal: Add a thin CLI to launch the Holoscan app with a preset and source options.

What we’ll add:

- `uspp-holoscan --preset pipeline.yaml --uff demo.uff [--dataset beamformed] [--loop] [--control :port]`.
- Keep `uspp-run` as a dev/debug path only; Holoscan is the primary runtime.

Acceptance:

- CLI spins up the graph and prints FPS; control plane available if enabled.

---
