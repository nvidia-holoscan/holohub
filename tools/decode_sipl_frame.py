#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Decode and visualize a frame saved by sipl_frame_saver.
#
# Each frame consists of:
#   frame_<seq>.raw  — packed device bytes (RAW10 or NV12)
#   frame_<seq>.txt  — ImageT descriptor metadata
#
# RAW10 frames report encoding=CUSTOM: NvSci's X2Rc10Rb10Ra10 packing (three 10-bit
# samples per 4-byte dword) doesn't satisfy a BAYER_* encoding's contract that one
# Tensor element is one sample. The Bayer phase travels separately as the sidecar's
# bayer_phase field instead. Sidecars from before that change still name the phase
# directly as encoding=BAYER_<phase>, which is still accepted here; sidecars that
# predate bayer_phase entirely (encoding=CUSTOM, no bayer_phase line) need --bayer-phase.
#
# RAW10's SIPL-embedded top/bottom lines (register dumps and blanking, not Bayer
# samples) are cropped by SIPLCaptureOp before publication; a current .raw never
# contains them, and roi_offset_y in the sidecar is where row 0 sits within the full
# sensor frame, not a row count to strip. A .raw captured before that crop existed
# still has the embedded rows baked in and will demosaic with a corrupted top/bottom
# edge; re-capture it.
#
# Dependencies:  numpy, opencv-python (pip install numpy opencv-python)
#
# Usage:
#   python3 decode_sipl_frame.py /tmp/sipl_frames/frame_00000003
#   python3 decode_sipl_frame.py /tmp/sipl_frames/frame_00000003 --out out.png
#   python3 decode_sipl_frame.py /tmp/sipl_frames/frame_00000003 --info

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# RAW10 (MIPI packed) unpacker
# ---------------------------------------------------------------------------


def unpack_raw10(data: bytes, width: int, pitch: int, height: int) -> np.ndarray:
    """Unpack NvSIPL ICP RAW10 buffer to uint16.

    NvSIPL stores RAW10 in NvSci's X2Rc10Rb10Ra10 layout: 3 pixels per
    32-bit dword, LSB-first, with 2 bits of padding at the top:
      bits[ 9: 0] = pixel 0  (10 bits)
      bits[19:10] = pixel 1  (10 bits)
      bits[29:20] = pixel 2  (10 bits)
      bits[31:30] = don't care

    This is NOT standard MIPI RAW10 (which packs 4 pixels into 5 bytes).
    pitch is the byte stride per row; pitch // 4 gives dwords per row.
    """
    pitch_u32 = pitch // 4
    buf = np.frombuffer(data, dtype=np.uint32).reshape(height, pitch_u32)

    n_dwords = (width + 2) // 3  # dwords needed for 'width' pixels
    row = buf[:, :n_dwords]  # shape: (height, n_dwords)

    out = np.zeros((height, width), dtype=np.uint16)
    col0 = np.arange(n_dwords) * 3  # column index of pixel 0 in each dword
    col1 = col0 + 1
    col2 = col0 + 2

    m0 = col0 < width
    m1 = col1 < width
    m2 = col2 < width

    out[:, col0[m0]] = (row[:, m0] & 0x3FF).astype(np.uint16)
    out[:, col1[m1]] = ((row[:, m1] >> 10) & 0x3FF).astype(np.uint16)
    out[:, col2[m2]] = ((row[:, m2] >> 20) & 0x3FF).astype(np.uint16)

    return out  # values in [0, 1023]


# ---------------------------------------------------------------------------
# NV12 extractor (Y plane only for grayscale preview)
# ---------------------------------------------------------------------------


def extract_nv12_luma(data: bytes, width: int, y_pitch: int, height: int) -> np.ndarray:
    buf = np.frombuffer(data, dtype=np.uint8)
    y = buf[: y_pitch * height].reshape(height, y_pitch)
    return y[:, :width]  # strip pitch padding


# ---------------------------------------------------------------------------
# Bayer demosaic (requires opencv)
# ---------------------------------------------------------------------------


def _bilinear_bayer(
    f: np.ndarray, gr: int, gc: int, br: int, bc: int, rr: int, rc: int
) -> tuple:
    """Proper per-pixel bilinear Bayer demosaic.

    g: G sub-grid at (gr::2, gc::2) AND the complementary sub-grid.
    b: B sub-grid at (br::2, bc::2).
    r: R sub-grid at (rr::2, rc::2).

    Returns (r_plane, g_plane, b_plane) all float32, shape of f.
    """
    h, w = f.shape

    g_out = np.zeros_like(f)
    b_out = np.zeros_like(f)
    r_out = np.zeros_like(f)

    # G sub-grids (two of them, both half-size)
    g0 = f[gr::2, gc::2]  # shape (h//2, w//2)  — primary G
    # The complementary G is at the other diagonal
    g1r, g1c = 1 - gr, 1 - gc
    g1 = f[g1r::2, g1c::2]  # shape (h//2, w//2)  — secondary G

    b = f[br::2, bc::2]  # shape (h//2, w//2)
    r = f[rr::2, rc::2]  # shape (h//2, w//2)

    # — Known values —
    g_out[gr::2, gc::2] = g0
    g_out[g1r::2, g1c::2] = g1
    b_out[br::2, bc::2] = b
    r_out[rr::2, rc::2] = r

    # — Green at B positions: horizontal average of the two adjacent primary-G values —
    # B is at (br::2, bc::2).  The nearest primary-G in the same row is
    # at col bc±1 (same row as B), which is in the g0 sub-grid (same even/odd row).
    # g0 columns: gc, gc+2, gc+4, ... so they bracket each B column.
    if bc == 0:
        # B at even cols; g0 at odd cols (gc=1)
        # G left of B[2i, 0] is clamped; G right is g0[i, 0] (col gc=1)
        g_at_b = np.empty_like(b)
        g_at_b[:, 0] = g0[:, 0]
        g_at_b[:, 1:] = (g0[:, :-1] + g0[:, 1:]) / 2
    else:
        # B at odd cols; g0 at even cols (gc=0) brackets each B col
        g_at_b = np.empty_like(b)
        g_at_b[:, :-1] = (g0[:, :-1] + g0[:, 1:]) / 2
        g_at_b[:, -1] = g0[:, -1]
    g_out[br::2, bc::2] = g_at_b

    # — Green at R positions: vertical average of the two adjacent primary-G values —
    # R is at (rr::2, rc::2).  g0 is at rows gr::2; nearest rows are above/below R.
    if rr == 0:
        # R at even rows; g0 also at even rows (gr=0): take column-adjacent g1
        g_at_r = np.empty_like(r)
        g_at_r[0, :] = g1[0, :]
        g_at_r[1:, :] = (g1[:-1, :] + g1[1:, :]) / 2
    else:
        # R at odd rows; g0 at even rows (gr=0): average rows above and below
        g_at_r = np.empty_like(r)
        g_at_r[:-1, :] = (g0[:-1, :] + g0[1:, :]) / 2
        g_at_r[-1, :] = g0[-1, :]
    g_out[rr::2, rc::2] = g_at_r

    # — Blue at G positions: average of the four adjacent B values —
    def fill_cross(src_half, dst_plane, known_r, known_c, fill_r, fill_c):
        """Fill dst at (fill_r::2, fill_c::2) by averaging 4 neighbours from src_half."""
        s = src_half
        out = np.empty_like(s)
        # interior
        out[1:, 1:] = (s[:-1, :-1] + s[:-1, 1:] + s[1:, :-1] + s[1:, 1:]) / 4
        out[0, 1:] = (s[0, :-1] + s[0, 1:]) / 2
        out[1:, 0] = (s[:-1, 0] + s[1:, 0]) / 2
        out[0, 0] = s[0, 0]
        dst_plane[fill_r::2, fill_c::2] = out

    # B at g0 position (br XOR 0) — the G subgrid that shares same rows as B
    fill_cross(b, b_out, br, bc, gr, gc)  # B at G-primary positions
    fill_cross(b, b_out, br, bc, g1r, g1c)  # B at G-secondary positions (same math)
    # B at R position: also diagonally adjacent to B sub-grid
    if rr != br:
        fill_cross(b, b_out, br, bc, rr, rc)

    # — Red at G and B positions: same pattern as Blue —
    fill_cross(r, r_out, rr, rc, gr, gc)
    fill_cross(r, r_out, rr, rc, g1r, g1c)
    if br != rr:
        fill_cross(r, r_out, rr, rc, br, bc)

    return r_out, g_out, b_out


def demosaic(bayer16: np.ndarray, encoding: str, stretch: bool = True) -> np.ndarray:
    # OpenCV cvtColor Bayer codes don't interpolate on this build.
    # Use a pure-numpy bilinear demosaic instead.
    #
    # Bayer layouts — top-left 2×2 pixel:
    #   GBRG:  G B / R G   → G at (0,0)+(1,1), B at (0,1), R at (1,0)
    #   GRBG:  G R / B G   → G at (0,0)+(1,1), R at (0,1), B at (1,0)
    #   BGGR:  B G / G R   → G at (0,1)+(1,0), B at (0,0), R at (1,1)
    #   RGGB:  R G / G B   → G at (0,1)+(1,0), R at (0,0), B at (1,1)
    CONFIGS = {
        # encoding → (gr, gc, br, bc, rr, rc)
        "BAYER_GBRG": (0, 0, 0, 1, 1, 0),
        "BAYER_GRBG": (0, 0, 1, 0, 0, 1),
        "BAYER_BGGR": (0, 1, 0, 0, 1, 1),
        "BAYER_RGGB": (0, 1, 1, 1, 0, 0),
    }
    if encoding not in CONFIGS:
        raise ValueError(f"Unknown Bayer encoding: {encoding}")

    try:
        import cv2  # noqa: F401 -- import used only to test availability
    except ImportError:
        print("opencv-python not installed — saving grayscale", file=sys.stderr)
        mono = _to_display(bayer16, stretch=False)
        return np.stack([mono] * 3, axis=-1)

    lo, hi = np.percentile(bayer16, (1, 99))
    hi = max(hi, lo + 1.0)
    raw = bayer16.astype(np.float32)

    # Column FPN correction: each Bayer channel samples different physical
    # sensor columns.  Without ISP dark-current subtraction, column bias
    # (fixed-pattern noise) differs per channel, producing colour stripes.
    # Subtract each sub-grid's per-column mean from itself so columns share
    # the same mean.  This is equivalent to estimating and removing the
    # column dark-current offset from a "flat" scene assumption.
    gr, gc, br, bc, rr, rc = CONFIGS[encoding]
    g1r, g1c = 1 - gr, 1 - gc
    for rs, cs in [(gr, gc), (g1r, g1c), (br, bc), (rr, rc)]:
        sub = raw[rs::2, cs::2]
        col_bias = sub.mean(axis=0, keepdims=True) - sub.mean()
        raw[rs::2, cs::2] -= col_bias

    f = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)

    r_plane, g_plane, b_plane = _bilinear_bayer(f, gr, gc, br, bc, rr, rc)

    bgr = np.clip(np.stack([b_plane, g_plane, r_plane], axis=2) * 255, 0, 255).astype(
        np.uint8
    )
    return bgr


def _to_display(arr: np.ndarray, stretch: bool) -> np.ndarray:
    """Convert uint16 (or uint16 3-channel) to uint8 with optional auto-stretch."""
    if not stretch:
        return (arr >> 8).astype(np.uint8)
    # Per-channel stretch to [0, 255] using 1st–99th percentile to avoid
    # blown highlights / crushed shadows dominating the scale.
    if arr.ndim == 2:
        lo, hi = np.percentile(arr, (1, 99))
        hi = max(hi, lo + 1)
        out = np.clip((arr.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255)
        return out.astype(np.uint8)
    # 3-channel: stretch each channel independently
    out = np.empty_like(arr, dtype=np.uint8)
    for ch in range(arr.shape[2]):
        lo, hi = np.percentile(arr[:, :, ch], (1, 99))
        hi = max(hi, lo + 1)
        out[:, :, ch] = np.clip(
            (arr[:, :, ch].astype(np.float32) - lo) / (hi - lo) * 255, 0, 255
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_sidecar(txt_path: Path) -> dict:
    meta = {}
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if "=" in line:
            k, _, v = line.partition("=")
            meta[k.strip()] = v.strip()
    return meta


def main():
    ap = argparse.ArgumentParser(
        description="Decode and visualize a sipl_frame_saver frame"
    )
    ap.add_argument(
        "base", help="Path without extension, e.g. /tmp/sipl_frames/frame_00000003"
    )
    ap.add_argument("--out", default=None, help="Output PNG path (default: <base>.png)")
    ap.add_argument(
        "--info",
        action="store_true",
        help="Print metadata and exit without writing image",
    )
    ap.add_argument(
        "--mono",
        action="store_true",
        help="Save grayscale (no demosaic) — useful to verify data before colour",
    )
    ap.add_argument(
        "--roi-offset",
        type=int,
        default=None,
        help="Override roi_offset_y from sidecar (embedded top lines to skip)",
    )
    ap.add_argument(
        "--bayer-phase",
        choices=["RGGB", "BGGR", "GRBG", "GBRG"],
        default=None,
        help="Bayer phase for a RAW10 (encoding=CUSTOM) frame, overriding or "
        "substituting for the sidecar's bayer_phase field",
    )
    args = ap.parse_args()

    base = Path(args.base)
    raw_path = base.with_suffix(".raw")
    txt_path = base.with_suffix(".txt")

    if not txt_path.exists():
        sys.exit(f"Sidecar not found: {txt_path}")
    if not raw_path.exists():
        sys.exit(f"Raw file not found: {raw_path}")

    meta = parse_sidecar(txt_path)

    print("=== Frame metadata ===")
    for k, v in meta.items():
        print(f"  {k:30s} = {v}")
    print()

    if args.info:
        return

    encoding = meta.get("encoding", "BAYER_GBRG")
    width = int(meta["width"])
    height = int(meta["height"])
    roi_offset_y = (
        args.roi_offset
        if args.roi_offset is not None
        else int(meta.get("roi_offset_y", 0))
    )
    sig_bits = int(meta.get("significant_bits", 10))
    plane_layouts = (
        [int(p) for p in meta["plane_layouts"].split(",")]
        if "plane_layouts" in meta
        else []
    )

    # RAW10's Bayer phase: named directly in `encoding` on an old sidecar (BAYER_<phase>), carried
    # separately as `bayer_phase` on a current one (encoding=CUSTOM), or supplied on the command
    # line for a sidecar that predates bayer_phase entirely.
    bayer_phase = args.bayer_phase or meta.get("bayer_phase")
    if encoding.startswith("BAYER_"):
        bayer_encoding = encoding
    elif encoding == "CUSTOM" and bayer_phase:
        bayer_encoding = f"BAYER_{bayer_phase}"
    else:
        bayer_encoding = None

    raw = raw_path.read_bytes()
    print(f"File: {raw_path.name}  ({len(raw):,} bytes)")
    print(
        f"Encoding: {encoding}  {width}×{height}  roi_offset_y={roi_offset_y}  "
        f"significant_bits={sig_bits}"
    )

    # --- decode ---
    if bayer_encoding is not None:
        # pitch is total row bytes; derive from file size or plane_layouts
        pitch = plane_layouts[0] if plane_layouts else (len(raw) // height)
        bayer = unpack_raw10(raw, width, pitch, height)

        # SIPLCaptureOp publishes only the active rows: it already skips embedded_top_lines_ /
        # embedded_bottom_lines_ rows at the copy source, so `raw` never contains them here.
        # roi_offset_y describes where row 0 of this (already-cropped) data sits within the full
        # sensor frame -- informational, not an instruction to crop further.
        if roi_offset_y > 0:
            print(
                f"Row 0 is full-sensor row {roi_offset_y} (embedded rows already excluded)"
            )

        # _bilinear_bayer slices every other row into same-shape sub-grids (2×2 CFA tiling),
        # which requires an even row count; an odd active height otherwise leaves one dangling
        # row whose sub-grids mismatch in shape and crash the demosaic. Drop it -- one row lost
        # off a multi-megapixel frame is not visually meaningful.
        if bayer.shape[0] % 2 != 0:
            bayer = bayer[:-1]

        image_height = bayer.shape[0]
        print(
            f"Pixel region: {width}×{image_height}  "
            f"bayer_phase={bayer_encoding[len('BAYER_') :]}"
        )
        print(
            f"Value range: min={bayer.min()}  max={bayer.max()}  "
            f"mean={bayer.mean():.1f}"
        )

        mono = _to_display(bayer, stretch=True)
        if args.mono:
            bgr = np.stack([mono] * 3, axis=-1)
        else:
            bgr = demosaic(bayer, bayer_encoding)
            # Always also save a mono alongside the colour attempt so the
            # user has a clean reference (RAW10 colour requires ISP calibration
            # for clean output; mono shows the scene without FPN/noise issues).
            mono_path = (
                args.out.replace(".png", "_mono.png")
                if args.out
                else str(base.with_suffix("")) + "_mono.png"
            )
            try:
                import cv2 as _cv2_mono

                _cv2_mono.imwrite(mono_path, np.stack([mono] * 3, axis=-1))
                print(f"Saved mono (clean reference): {mono_path}")
            except Exception:
                pass

    elif encoding == "NV12":
        y_pitch = plane_layouts[0] if len(plane_layouts) > 0 else width
        luma = extract_nv12_luma(raw, width, y_pitch, height)
        if roi_offset_y > 0:
            luma = luma[roi_offset_y:]
        bgr = np.stack([luma, luma, luma], axis=-1)
        print(f"NV12 luma-only preview: {width}×{luma.shape[0]}")

    elif encoding == "CUSTOM":
        sys.exit(
            f"{txt_path} has encoding=CUSTOM (RAW10) but no bayer_phase field, so the phase "
            "is unknown -- pass --bayer-phase RGGB|BGGR|GRBG|GBRG, or re-capture with the "
            "updated sipl_frame_saver, which now writes bayer_phase for this format."
        )

    else:
        sys.exit(f"Unsupported encoding for decode: {encoding}")

    # --- write ---
    try:
        import cv2

        out_path = args.out or str(base.with_suffix(".png"))
        cv2.imwrite(out_path, bgr)
        print(f"Saved: {out_path}")
    except ImportError:
        grey = bgr[:, :, 0]
        out_path = args.out or str(base.with_suffix(".pgm"))
        with open(out_path, "wb") as f:
            f.write(f"P5\n{width} {grey.shape[0]}\n255\n".encode())
            f.write(grey.tobytes())
        print(f"Saved (grayscale PGM — install opencv-python for colour): {out_path}")


if __name__ == "__main__":
    main()
