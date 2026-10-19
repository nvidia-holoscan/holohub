# Spectrum Visualizer Operator

## Overview

The operator that converts a 1-D power spectrum (in dB) into Holoviz-compatible line geometry.

## Role In The Pipeline

`SpectrumVisualizerOp` takes the per-bin dB power values from `SpectrumMagnitudeOp` and maps them into screen coordinates. It builds one `(num_bins, 2)` tensor per channel, where each row is an `(x, y)` point for a `LINE_STRIP` layer in Holoviz.

It also optionally detects the strongest peak of each channel in the current spectrum and publishes the per-channel peak frequency and dB level back to the UI overlay through shared state.

## Diagram Notes

The visualizer diagrams are easiest to understand if they are read as two related workflows:

1. the normal rendering path, where input dB power bins become Holoviz line geometry
2. the optional peak path, which runs only when `show_peak` is enabled in `SpectrumViewState`

In the normal rendering path, `SpectrumVisualizerOp` receives one 1-D dB power spectrum from `SpectrumMagnitudeOp`, normalizes each dB bin into a screen-space y coordinate, recomputes the x coordinates from the current center frequency and bandwidth, and packs the final `(x, y)` coordinates into one device tensor per channel. It also emits matching `LINE_STRIP` draw specifications so `HolovizOp` knows how to render each channel.

When reading the visualizer data-flow diagram, it is important to distinguish between frequency-domain data and display geometry. The input to this operator is a power spectrum in dB, but the output is not another spectrum tensor. The output is render-ready geometry plus Holoviz metadata.

```mermaid
flowchart LR
    A["SpectrumMagnitudeOp"] -->|"power in dB per bin"| B["SpectrumVisualizerOp<br/>x axis: frequency bins<br/>y axis: power in dB per bin"]
    B -->|"Geometry tensors"| H["HolovizOp"]
    B -->|"LINE_STRIP input specs"| H
    B -->|"Write:<br/>peak_freq_mhz, peak_db"| S["SpectrumViewState"]
    S -->|"Read:<br/>center_mhz, bandwidth_mhz, show_peak"| B

    classDef white fill:#ffffff,stroke:#000,color:#000;
    class A,B,H,S white;
```

### Peak Detection Flow

The peak workflow starts by reading `show_peak` from `SpectrumViewState`. If the flag is disabled, the operator skips the extra peak-search work and only emits geometry. If the flag is enabled, the operator copies the current dB bins from device memory to the host, scans them on the CPU to find the maximum bin for that channel, converts that bin index into an absolute frequency using the reference center frequency and bandwidth, and writes the resulting `peak_freq_mhz` and `peak_db` values for that channel's slot back into `SpectrumViewState`.

```mermaid
flowchart LR
    A["SpectrumVisualizerOp reads<br/>show_peak from SpectrumViewState"] --> B{"Is show_peak<br/>enabled?"}
    B -->|YES| C["Copy dB values<br/>from GPU to host"]
    C --> D["Scan bins for the<br/>maximum dB (peak bin)"]
    D --> E["Convert peak bin index<br/>into peak frequency"]
    E --> F["Store peak_freq_mhz & peak_db<br/>into SpectrumViewState"]
    F --> G["Holoviz renders spectrum<br/>with peak annotation"]
    B -.->|NO| H["Skip peak search<br/>(no state update)"]
    H --> I["Holoviz renders spectrum<br/>without peak annotation"]

    classDef white fill:#ffffff,stroke:#000,color:#000;
    class A,B,C,D,E,F,G,H,I white;
```

If your diagram shows a peak annotation after the `show_peak == false` branch, that should be corrected. In the actual code, peak values are only recomputed and displayed when the peak toggle is enabled.

### Pan And Zoom Flow

The pan and zoom diagram should emphasize that changing the center frequency or bandwidth does not change the incoming dB values. Instead, the overlay writes the new view parameters into `SpectrumViewState`, and `SpectrumVisualizerOp` uses those values to recompute the x coordinates of the line geometry on the next `compute()` call. The y coordinates still come from the normalized dB spectrum.

```mermaid
flowchart TD
    subgraph User
        Start([Start]) --> U1["Change Center Frequency<br/>or Bandwidth"]
        U2["Sees zoomed or<br/>panned spectrum"] --> End([End])
    end
    subgraph SpectrumOverlay
        O1["Write values to<br/>SpectrumViewState"]
    end
    subgraph SpectrumViewState
        S1["Store center_mhz &<br/>bandwidth_mhz"]
    end
    subgraph SpectrumVisualizerOp
        V1["Read center_mhz &<br/>bandwidth_mhz"] --> V2["Recalculate X<br/>coordinates"]
    end
    subgraph Holoviz
        H1["Emit new geometry"] --> H2["Update displayed<br/>spectrum"]
    end
    subgraph DisplayWindow["Display Window"]
        D1["Render updated<br/>spectrum"]
    end
    U1 -->|changes| O1
    O1 -->|writes| S1
    S1 -->|reads| V1
    V2 -->|emits| H1
    H2 -->|renders| D1
    D1 --> U2

    classDef white fill:#ffffff,stroke:#000,color:#000;
    class Start,U1,U2,End,O1,S1,V1,V2,H1,H2,D1 white;
    style User fill:#c7f9cc,stroke:#000,color:#000;
    style SpectrumOverlay fill:#c7f9cc,stroke:#000,color:#000;
    style SpectrumViewState fill:#c7f9cc,stroke:#000,color:#000;
    style SpectrumVisualizerOp fill:#c7f9cc,stroke:#000,color:#000;
    style Holoviz fill:#c7f9cc,stroke:#000,color:#000;
    style DisplayWindow fill:#c7f9cc,stroke:#000,color:#000;
```

## Key Files

- `spectrum_visualizer.hpp`: operator declaration, shared state hook, and rendering constants.
- `spectrum_visualizer.cu`: dB normalization, pan/zoom mapping, Holoviz tensor packing, and peak detection.
- `spectrum_view_state.hpp`: shared state exchanged between the overlay callback and the visualizer operator.

## Notes

- The operator owns the data-to-geometry conversion.
- The UI itself is not implemented here; it lives in `../spectrum_overlay/`.
