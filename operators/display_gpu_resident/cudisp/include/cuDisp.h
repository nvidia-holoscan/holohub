/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CUDISP_H
#define CUDISP_H

#include <stdint.h>
#include <cuda.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/**
 * \file cuDisp.h
 * \brief Header file for the cuDisp application programming interface.
 *
 * cuDisp provides swapchain creation, buffer management, and display present
 * (page-flip) for one or more layers (primary + overlay planes). It supports
 * both host-driven and GPU-driven present paths, HDR metadata, color
 * management, and per-plane compositing controls.
 */

/**
 * \defgroup CUDISP_TYPES Data types used by cuDisp driver
 * @{
 */

#define CUDISP_VER_MAJOR 1U
#define CUDISP_VER_MINOR 0U
#define CUDISP_VER_PATCH 0U

/**
 * Error codes.
 */
typedef enum cuDispStatus_t {
    /**
     * The API call returned with no errors.
     */
    cuDispSuccess                                           = 0,
    /**
     * This indicates that one or more parameters passed
     * to the API is/are incorrect.
     */
    cuDispErrorInvalidParam                                 = 1,
    /**
     * This indicates that the API call failed due to
     * lack of underlying resources.
     */
    cuDispErrorOutOfResources                              = 2,
    /**
     * This indicates that an internal error occurred
     * during initialization.
     */
    cuDispErrorCreationFailed                               = 3,
    /**
     * This indicates that an OS error occurred.
     */
    cuDispErrorOs                                           = 4,
    /**
     * This indicates that there was an error in a
     * CUDA operation as part of the API call.
     */
    cuDispErrorCuda                                         = 5,
    /**
     * This indicates that there was an error in the
     * underlying display driver.
     */
    cuDispErrorDisplay                                      = 7,
    /**
     * The requested feature or combination of options
     * is not supported by this implementation.
     */
    cuDispErrorNotSupported                                 = 8,
    /**
     * This indicates that an unknown error has
     * occurred.
     */
    cuDispErrorUnknown                                      = 0x7fffffff
} cuDispStatus;

/**
 * Opaque swapchain handle returned by cuDispCreateSwapchain().
 */
struct cuDispSwapchain_t;
typedef struct cuDispSwapchain_t* cuDispSwapchain;

/**
 * Surface pixel formats. Values are sequential; the library maps
 * them to backend-specific format codes internally.
 */
typedef enum cuDispSurfaceFormat_enum {
    /* RGB formats */
    CUDISP_SURFACE_FORMAT_ARGB1555 = 0,  /**< 16-bit ARGB, 1:5:5:5 */
    CUDISP_SURFACE_FORMAT_XRGB1555,      /**< 16-bit XRGB, x:5:5:5 */
    CUDISP_SURFACE_FORMAT_RGB565,        /**< 16-bit RGB, 5:6:5 */
    CUDISP_SURFACE_FORMAT_ARGB8888,      /**< 32-bit ARGB, 8:8:8:8 */
    CUDISP_SURFACE_FORMAT_XRGB8888,      /**< 32-bit XRGB, x:8:8:8 */
    CUDISP_SURFACE_FORMAT_ABGR8888,      /**< 32-bit ABGR, 8:8:8:8 */
    CUDISP_SURFACE_FORMAT_XBGR8888,      /**< 32-bit XBGR, x:8:8:8 */
    CUDISP_SURFACE_FORMAT_ABGR2101010,   /**< 32-bit ABGR, 2:10:10:10 */
    CUDISP_SURFACE_FORMAT_XBGR2101010,   /**< 32-bit XBGR, x:10:10:10 */
    CUDISP_SURFACE_FORMAT_ABGR16161616,  /**< 64-bit ABGR, 16:16:16:16 integer */
    CUDISP_SURFACE_FORMAT_ABGR16161616F, /**< 64-bit ABGR, 16:16:16:16 float */
    CUDISP_SURFACE_FORMAT_XBGR16161616F, /**< 64-bit XBGR, x:16:16:16 float */

    /* YUV packed */
    CUDISP_SURFACE_FORMAT_YUYV,          /**< Packed YUV 4:2:2, Y0 Cb Y1 Cr */
    CUDISP_SURFACE_FORMAT_UYVY,          /**< Packed YUV 4:2:2, Cb Y0 Cr Y1 */

    /* YUV semi-planar 4:4:4 */
    CUDISP_SURFACE_FORMAT_NV42,          /**< Semi-planar YUV 4:4:4, Y / CrCb */
    CUDISP_SURFACE_FORMAT_NV24,          /**< Semi-planar YUV 4:4:4, Y / CbCr */

    /* YUV semi-planar 4:2:2 */
    CUDISP_SURFACE_FORMAT_NV61,          /**< Semi-planar YUV 4:2:2, Y / CrCb */
    CUDISP_SURFACE_FORMAT_NV16,          /**< Semi-planar YUV 4:2:2, Y / CbCr */

    /* YUV semi-planar 4:2:0 */
    CUDISP_SURFACE_FORMAT_NV21,          /**< Semi-planar YUV 4:2:0, Y / CrCb */
    CUDISP_SURFACE_FORMAT_NV12,          /**< Semi-planar YUV 4:2:0, Y / CbCr */

    /* YUV semi-planar 10/12-bit */
    CUDISP_SURFACE_FORMAT_P210,          /**< Semi-planar YUV 4:2:2, 10-bit */
    CUDISP_SURFACE_FORMAT_P010,          /**< Semi-planar YUV 4:2:0, 10-bit */
    CUDISP_SURFACE_FORMAT_P012           /**< Semi-planar YUV 4:2:0, 12-bit */
} cuDispSurfaceFormat;

/**
 * Display colorspace. Applied per-plane; tells the display engine
 * how to interpret the pixel data on that plane.
 */
typedef enum cuDispColorspace_enum {
    CUDISP_COLORSPACE_DEFAULT = 0,
    CUDISP_COLORSPACE_RGB_WIDE_GAMUT_FIXED_POINT,
    CUDISP_COLORSPACE_RGB_WIDE_GAMUT_FLOATING_POINT,
    CUDISP_COLORSPACE_OPRGB,
    CUDISP_COLORSPACE_DCI_P3_RGB_D65,
    CUDISP_COLORSPACE_BT2020_RGB,
    CUDISP_COLORSPACE_BT601_YCC,
    CUDISP_COLORSPACE_BT709_YCC,
    CUDISP_COLORSPACE_XVYCC_601,
    CUDISP_COLORSPACE_XVYCC_709,
    CUDISP_COLORSPACE_SYCC_601,
    CUDISP_COLORSPACE_OPYCC_601,
    CUDISP_COLORSPACE_BT2020_CYCC,
    CUDISP_COLORSPACE_BT2020_YCC,
    CUDISP_COLORSPACE_SMPTE_170M_YCC,
    CUDISP_COLORSPACE_DCI_P3_RGB_THEATER
} cuDispColorspace;

/**
 * Maximum bits per channel for the display link output.
 * Higher values are needed for HDR content (typically 10 or 12 bpc).
 */
typedef enum cuDispMaxBpc_enum {
    CUDISP_MAX_BPC_DEFAULT = 0,   /**< Driver default */
    CUDISP_MAX_BPC_8       = 8,   /**< 8 bits per channel */
    CUDISP_MAX_BPC_10      = 10,  /**< 10 bits per channel */
    CUDISP_MAX_BPC_12      = 12,  /**< 12 bits per channel */
    CUDISP_MAX_BPC_16      = 16   /**< 16 bits per channel */
} cuDispMaxBpc;

/**
 * Per-plane alpha blending mode. Controls how the plane's pixel
 * values are composited with the layers below.
 *
 * None:          out = plane_alpha * fg + (1 - plane_alpha) * bg
 * Pre-multiplied: out = plane_alpha * fg + (1 - plane_alpha * fg.a) * bg
 * Coverage:       out = plane_alpha * fg.a * fg + (1 - plane_alpha * fg.a) * bg
 */
typedef enum cuDispBlendMode_enum {
    CUDISP_BLEND_MODE_DEFAULT       = 0,  /**< Use driver default */
    CUDISP_BLEND_MODE_NONE          = 1,  /**< Ignore pixel alpha, use plane alpha only */
    CUDISP_BLEND_MODE_PREMULTIPLIED = 2,  /**< Pixel colors are pre-multiplied with alpha */
    CUDISP_BLEND_MODE_COVERAGE      = 3   /**< Pixel colors are not pre-multiplied */
} cuDispBlendMode;

/**
 * Per-plane rotation and reflection. Mapped to backend-specific
 * rotation values internally.
 */
typedef enum cuDispRotation_enum {
    CUDISP_ROTATE_0   = 0,  /**< No rotation (default) */
    CUDISP_ROTATE_90  = 1,  /**< 90 degrees clockwise */
    CUDISP_ROTATE_180 = 2,  /**< 180 degrees */
    CUDISP_ROTATE_270 = 3,  /**< 270 degrees clockwise */
    CUDISP_REFLECT_X  = 4,  /**< Horizontal reflection */
    CUDISP_REFLECT_Y  = 5   /**< Vertical reflection */
} cuDispRotation;

/**
 * YCbCr color encoding standard. Tells the display engine which
 * matrix to use when converting YUV pixel data to RGB.
 * Only relevant for YUV surface formats.
 */
typedef enum cuDispColorEncoding_enum {
    CUDISP_COLOR_ENCODING_DEFAULT = 0,  /**< Use driver default */
    CUDISP_COLOR_ENCODING_BT601   = 1,  /**< ITU-R BT.601 (SDTV) */
    CUDISP_COLOR_ENCODING_BT709   = 2,  /**< ITU-R BT.709 (HDTV) */
    CUDISP_COLOR_ENCODING_BT2020  = 3   /**< ITU-R BT.2020 (UHDTV/HDR) */
} cuDispColorEncoding;

/**
 * YCbCr quantization range. Determines whether the luma/chroma
 * values use the limited (studio) or full (PC) range.
 * Only relevant for YUV surface formats.
 */
typedef enum cuDispColorRange_enum {
    CUDISP_COLOR_RANGE_DEFAULT  = 0,  /**< Use driver default */
    CUDISP_COLOR_RANGE_LIMITED  = 1,  /**< Limited / studio range (Y: 16-235) */
    CUDISP_COLOR_RANGE_FULL     = 2   /**< Full / PC range (Y: 0-255) */
} cuDispColorRange;

/**
 * HDR metadata type. Indicates whether the metadata describes
 * static content characteristics or dynamic per-frame values.
 */
typedef enum cuDispHDRMetadataType_enum {
    CUDISP_HDR_METADATA_STATIC  = 0,  /**< Static: fixed for the content session */
    CUDISP_HDR_METADATA_DYNAMIC = 1   /**< Dynamic: may change per frame */
} cuDispHDRMetadataType;

/**
 * Electro-Optical Transfer Function (EOTF) as defined by CTA-861-G.
 * PQ (Perceptual Quantizer) is SMPTE ST 2084.
 */
typedef enum cuDispEOTF_enum {
    CUDISP_EOTF_SDR_TRADITIONAL = 0,  /**< Traditional gamma — SDR */
    CUDISP_EOTF_HDR_TRADITIONAL = 1,  /**< Traditional gamma — HDR */
    CUDISP_EOTF_SMPTE_ST2084   = 2,  /**< SMPTE ST 2084 (PQ) */
    CUDISP_EOTF_HLG            = 3   /**< Hybrid Log-Gamma */
} cuDispEOTF;

/**
 * HDR metadata following CTA-861-G. One struct per layer; used
 * both at swapchain creation (static) and at present time (dynamic)
 * via the pHDRMetadata pointer in cuDispBufferMemory.
 *
 * Display primary and white-point coordinates use CTA-861-G units
 * (0.00002 increments; e.g. 0.708 encoded as 35400).
 */
typedef struct cuDispHDRMetadata_st {
    uint32_t layerIndex;                  /**< Layer this metadata applies to */
    cuDispHDRMetadataType metadataType;   /**< Static or dynamic */
    cuDispEOTF eotf;                      /**< Transfer function */
    uint16_t displayPrimaryRedX;          /**< Red primary X coordinate */
    uint16_t displayPrimaryRedY;          /**< Red primary Y coordinate */
    uint16_t displayPrimaryGreenX;        /**< Green primary X coordinate */
    uint16_t displayPrimaryGreenY;        /**< Green primary Y coordinate */
    uint16_t displayPrimaryBlueX;         /**< Blue primary X coordinate */
    uint16_t displayPrimaryBlueY;         /**< Blue primary Y coordinate */
    uint16_t whitePointX;                 /**< White point X coordinate */
    uint16_t whitePointY;                 /**< White point Y coordinate */
    uint32_t maxDisplayLuminance;         /**< Max mastering luminance (cd/m^2) */
    uint32_t minDisplayLuminance;         /**< Min mastering luminance (0.0001 cd/m^2) */
    uint16_t maxContentLightLevel;        /**< MaxCLL (cd/m^2) */
    uint16_t maxFrameAverageLightLevel;   /**< MaxFALL (cd/m^2) */
    char pad[24];                         /**< Reserved, must be zero */
} cuDispHDRMetadata;

/**
 * Single entry in a color lookup table.
 * Each channel is a 16-bit unsigned value.
 */
typedef struct cuDispColorLut_st {
    uint16_t red;       /**< Red channel value */
    uint16_t green;     /**< Green channel value */
    uint16_t blue;      /**< Blue channel value */
    uint16_t reserved;  /**< Reserved, must be zero */
} cuDispColorLut;

/**
 * Buffer memory descriptor. Returned by cuDispGetBuffer().
 *
 * The union holds a pointer-to-pointer for the buffer's backing memory;
 * the active member depends on the allocation type used by the library.
 * The application must set the desired pointer member to a valid storage
 * location before calling cuDispGetBuffer().
 */
typedef struct cuDispBufferMemory_st {
    union {
        CUdeviceptr *devicePtr;           /**< Linear device memory */
        CUarray *arrayPtr;                /**< CUDA array */
        CUmipmappedArray *mipmappedArrayPtr; /**< CUDA mipmapped array */
        CUexternalMemory *externalMemoryPtr; /**< Imported external memory */
    };
    uint64_t *size;               /**< Output: buffer size in bytes (optional, can be NULL) */
    uint32_t *stride;             /**< Output: row pitch in bytes (optional, can be NULL) */
    cuDispHDRMetadata *pHDRMetadata;  //!< Output: per-buffer HDR metadata
                                     //!< (device-mapped host memory). NULL if
                                     //!< HDR was not requested at creation.
                                     //!< Writable from both CPU and GPU.
} cuDispBufferMemory;

/**
 * Display mode configuration. Describes the display output timing,
 * independent of buffer dimensions. If not specified, the display's
 * native/preferred mode is used.
 */
typedef struct cuDispModeInfo_st {
    uint32_t modeWidth;         /**< Display horizontal resolution in pixels */
    uint32_t modeHeight;        /**< Display vertical resolution in pixels */
    uint32_t refreshRateMilliHz;  //!< Refresh rate in milliHz (e.g. 60000 = 60 Hz).
                                  //!< 0 = use display native refresh.
    uint32_t enableVrr;         /**< Non-zero to enable Variable Refresh Rate */
    cuDispMaxBpc maxBpc;        /**< Maximum bits per channel for the link. 0 = driver default. */
    char pad[44];               /**< Reserved, must be zero */
} cuDispModeInfo;

/**
 * Per-layer buffer and compositing configuration. One attribute entry
 * per layer; the layerIndex field identifies which layer (0 = primary,
 * 1+ = overlay planes).
 *
 * Buffer dimensions (width/height) define the render target size.
 * Position (posX/posY) and scale dimensions (scaleWidth/scaleHeight)
 * define where and how the buffer appears on the display:
 * - posX/posY: top-left corner on the display (0,0 for fullscreen primary).
 * - scaleWidth/scaleHeight: displayed size; 0 = same as buffer (no scaling).
 *   When different from width/height, the display engine scales the buffer.
 */
typedef struct cuDispBufferInfo_st {
    uint32_t layerIndex;              /**< 0 = primary, 1+ = overlay plane index */
    uint32_t numBuffers;              /**< Buffer count for this layer (2 = double-buffer) */
    cuDispSurfaceFormat format;       /**< Pixel format for this layer's buffers */
    uint32_t width;                   /**< Buffer width in pixels */
    uint32_t height;                  /**< Buffer height in pixels */
    uint32_t posX;                    /**< Display destination X position */
    uint32_t posY;                    /**< Display destination Y position */
    uint32_t scaleWidth;              /**< Display destination width (0 = same as buffer width) */
    uint32_t scaleHeight;             /**< Display destination height (0 = same as buffer height) */
    uint16_t alpha;                   /**< Plane alpha (0=transparent, 0xFFFF=opaque) */
    cuDispBlendMode blendMode;        /**< Alpha blending mode for compositing */
    cuDispRotation rotation;          /**< Rotation / reflection applied by the display engine */
    cuDispColorEncoding colorEncoding; /**< YCbCr encoding standard (for YUV formats) */
    cuDispColorRange colorRange;      /**< YCbCr quantization range (for YUV formats) */
} cuDispBufferInfo;

/**
 * GPU present handle. Used as an output attribute: after swapchain
 * creation, *handleGPUPresent points to an opaque structure that the
 * application passes to cuDispLaunchPresentKernel() / cuDispGPUPresent().
 */
typedef struct cuDispGPUPresent_st {
    void **handleGPUPresent;          /**< Output: pointer written by cuDispCreateSwapchain() */
} cuDispGPUPresent;

/**
 * Per-layer colorspace configuration. Tells the display engine / monitor
 * how to interpret the pixel values on a given plane.
 */
typedef struct cuDispColorspaceConfig_st {
    uint32_t layerIndex;              /**< Layer this colorspace applies to */
    cuDispColorspace colorspace;      /**< Colorspace for the layer */
    char pad[56];                     /**< Reserved, must be zero */
} cuDispColorspaceConfig;

/**
 * Color lookup table configuration. Used for both degamma (pre-CTM)
 * and gamma (post-CTM) LUTs. The application allocates the LUT array
 * and keeps it valid for the lifetime of the swapchain.
 */
typedef struct cuDispLutConfig_st {
    cuDispColorLut* pLut;             /**< App-owned LUT array. NULL = pass-through. */
    uint32_t lutSize;                 /**< Number of entries in the LUT array */
    char pad[52];                     /**< Reserved, must be zero */
} cuDispLutConfig;

/**
 * Color Transformation Matrix (CTM) configuration. The matrix is
 * a 3x3 array of 9 int64_t values in row-major order, using the
 * sign-magnitude fixed-point format (1 sign + 31 integer + 32
 * fractional bits). NULL = identity (no transform).
 */
typedef struct cuDispCtmConfig_st {
    int64_t* pCtm;                    /**< Pointer to app-owned 3x3 matrix (9 x int64_t). */
    char pad[56];                     /**< Reserved, must be zero */
} cuDispCtmConfig;

/**
 * Display selection. Allows the application to target a specific
 * display by connector name (e.g. "DP-2") or EDID monitor name
 * (e.g. "ASUS VG27A"). The library tries connector name match
 * first, then EDID match. If empty or unset, the first connected
 * display is used.
 */
typedef struct cuDispDisplaySelect_st {
    char name[60];  /**< Null-terminated connector or monitor name */
    char pad[4];    /**< Reserved, must be zero */
} cuDispDisplaySelect;

/**
 * Present flags. Passed to cuDispPresent().
 */
typedef enum cuDispPresentFlags_enum {
    CUDISP_PRESENT_FLAG_NONE      = 0x00,  /**< Default: synchronize to vblank */
    CUDISP_PRESENT_FLAG_VSYNC_OFF = 0x01   /**< Immediate flip, no vblank wait */
} cuDispPresentFlags;

/**
 * Swapchain creation attribute IDs.
 */
typedef enum cuDispCreateAttributeID_enum {
    CUDISP_CREATE_ATTRIBUTE_IGNORE       = 0,  /**< Ignored entry (array padding) */
    CUDISP_CREATE_ATTRIBUTE_MODE_INFO    = 1,  /**< Mode: resolution, refresh, VRR, bpc */
    CUDISP_CREATE_ATTRIBUTE_BUFFER_INFO  = 2,  /**< Per-layer buffer config (one entry per layer) */
    CUDISP_CREATE_ATTRIBUTE_GPU_PRESENT  = 3,  /**< GPU present handle (optional, output) */
    CUDISP_CREATE_ATTRIBUTE_HDR_METADATA = 4,  /**< HDR metadata per layer (optional) */
    CUDISP_CREATE_ATTRIBUTE_COLORSPACE   = 5,  /**< Colorspace per layer (optional) */
    CUDISP_CREATE_ATTRIBUTE_DEGAMMA_LUT  = 6,  /**< Pre-CTM degamma LUT (optional) */
    CUDISP_CREATE_ATTRIBUTE_GAMMA_LUT    = 7,  /**< Post-CTM gamma LUT (optional) */
    CUDISP_CREATE_ATTRIBUTE_CTM          = 8,  /**< Color transformation matrix (optional) */
    CUDISP_CREATE_ATTRIBUTE_DISPLAY_SELECT = 9  /**< Target display by name (optional) */
} cuDispCreateAttributeID;

/**
 * Union for attribute values (padded to 64 bytes for ABI stability across compilers)
 */
typedef union cuDispCreateAttributeValue_union {
    cuDispModeInfo modeInfo;
    cuDispBufferInfo bufferInfo;
    cuDispGPUPresent gpuPresent;
    cuDispHDRMetadata hdrMetadata;
    cuDispColorspaceConfig colorspace;
    cuDispLutConfig lutConfig;
    cuDispCtmConfig ctmConfig;
    cuDispDisplaySelect displaySelect;
    char pad[64];
} cuDispCreateAttributeValue;

/**
 * Creation attribute: pairs an attribute ID with its value.
 */
typedef struct cuDispCreateAttribute_st {
    cuDispCreateAttributeID id;                   /**< Attribute to set */
    char pad[8 - sizeof(cuDispCreateAttributeID)]; /**< Padding for 8-byte alignment of value */
    cuDispCreateAttributeValue value;             /**< Value of the attribute */
} cuDispCreateAttribute;

/** @} */ /* END CUDISP_TYPES */

/**
 * \defgroup CUDISP_API cuDisp API Functions
 * @{
 */

/**
 * Get the version of the cuDisp library.
 *
 * @param version  Output: Version number as 64-bit integer
 *                 Format: (major << 32) | (minor << 16) | patch
 *
 * @return cuDispSuccess on success, cuDispErrorInvalidParam if version is NULL
 */
cuDispStatus cuDispGetVersion(uint64_t* version);

/**
 * Create a swapchain with specified configuration attributes.
 *
 * @param swapchain      Output: Swapchain handle
 * @param attributes     Array of creation attributes
 * @param numAttributes  Number of attributes in array
 * @param flags          Creation flags (reserved for future use, must be 0)
 *
 * @return cuDispSuccess on success, error code on failure
 *
 * Required attributes:
 * - CUDISP_CREATE_ATTRIBUTE_BUFFER_INFO (at least one entry for the primary layer)
 *
 * Optional attributes:
 * - CUDISP_CREATE_ATTRIBUTE_MODE_INFO (defaults to display native mode)
 * - CUDISP_CREATE_ATTRIBUTE_GPU_PRESENT (output: opaque GPU present handle)
 * - CUDISP_CREATE_ATTRIBUTE_HDR_METADATA (HDR metadata per layer)
 * - CUDISP_CREATE_ATTRIBUTE_COLORSPACE (colorspace per layer)
 * - CUDISP_CREATE_ATTRIBUTE_DEGAMMA_LUT (pre-CTM lookup table)
 * - CUDISP_CREATE_ATTRIBUTE_GAMMA_LUT (post-CTM lookup table)
 * - CUDISP_CREATE_ATTRIBUTE_CTM (color transformation matrix)
 *
 * Multiple BUFFER_INFO entries define multiple layers (one per layerIndex).
 * Multiple HDR_METADATA and COLORSPACE entries configure individual layers.
 */
cuDispStatus cuDispCreateSwapchain(
    cuDispSwapchain* swapchain,
    cuDispCreateAttribute* attributes,
    uint32_t numAttributes,
    uint32_t flags
);

/**
 * Get the buffer memory for a specific buffer in a layer.
 *
 * Writes the buffer's device pointer into the storage pointed to by
 * the active union member of \p outBufferMemory. The application must
 * set the desired pointer member to a valid storage location before calling.
 *
 * If HDR metadata was requested at creation, \p outBufferMemory->pHDRMetadata
 * is set to a device-mapped host memory region for this buffer. The
 * application (or a CUDA kernel) may write per-frame HDR metadata there
 * before presenting.
 *
 * @param swapchain       Swapchain handle
 * @param layerIndex      Layer index (0 = primary, 1+ = overlay planes)
 * @param bufNum          Buffer index within the layer (0 to numBuffers-1)
 * @param outBufferMemory Output: buffer memory descriptor
 * @param flags           Flags (reserved for future use, must be 0)
 *
 * @return cuDispSuccess on success, error code on failure
 */
cuDispStatus cuDispGetBuffer(
    cuDispSwapchain swapchain,
    uint32_t layerIndex,
    uint64_t bufNum,
    cuDispBufferMemory* outBufferMemory,
    uint32_t flags
);

/**
 * Present one or more layer buffers to the display in a single atomic commit.
 *
 * The bufferMemory array contains one entry per layer, ordered by layer index
 * (bufferMemory[0] = primary, bufferMemory[1] = first overlay, etc.).
 * All layers are committed atomically in one page-flip request.
 *
 * If HDR metadata was requested at creation, the library reads the per-buffer
 * metadata from the pHDRMetadata region (set via cuDispGetBuffer) and includes
 * it in the atomic commit.
 *
 * @param swapchain     Swapchain handle
 * @param stream        CUDA stream for synchronization (can be NULL)
 * @param bufferMemory  Array of buffer descriptors, one per layer
 * @param numLayers     Number of entries in the bufferMemory array
 * @param flags         Present flags (cuDispPresentFlags bitmask)
 *
 * @return cuDispSuccess on success, error code on failure
 */
cuDispStatus cuDispPresent(
    cuDispSwapchain swapchain,
    void* stream,
    const cuDispBufferMemory* bufferMemory,
    uint32_t numLayers,
    uint32_t flags
);

/**
 * Destroy a swapchain and free all associated resources.
 *
 * @param swapchain  Swapchain handle to destroy
 *
 * @return cuDispSuccess on success, error code on failure
 */
cuDispStatus cuDispDestroySwapchain(cuDispSwapchain swapchain);

/** @} */ /* END CUDISP_API */

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#include "cuDispDevice.h"

#endif /* CUDISP_H */
