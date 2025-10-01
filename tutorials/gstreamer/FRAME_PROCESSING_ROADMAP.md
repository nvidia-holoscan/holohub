# GStreamer Frame Processing Roadmap

A comprehensive action list to add manual frame processing capabilities to the Holoscan GStreamer integration.

## Overview

This roadmap builds incrementally upon the existing promise-based buffer retrieval system to add comprehensive frame processing capabilities, supporting both CPU and GPU processing paths.

---

## Phase 1: Basic Buffer Data Access ✅ COMPLETED

### Foundation Components
- [x] **1. Re-implement GstMapGuard for CPU memory mapping**
  - ✅ Create RAII wrapper for `gst_buffer_map`/`gst_buffer_unmap`
  - ✅ Handle different map flags (READ/WRITE/READWRITE)
  - ✅ Add safety checks and error handling

- [x] **2. Add direct GstMapInfo usage** (removed generic template function)
  - ✅ Direct RAII buffer mapping for more control
  - ✅ Exception-safe buffer access
  - ✅ Support for different map modes

- [x] **3. Update example to demonstrate basic data access**
  - ✅ Show first few bytes of buffer data
  - ✅ Log buffer size and memory characteristics
  - ✅ Verify mapping works correctly

**Estimated Time:** 1 week ✅  
**Dependencies:** None  
**Deliverable:** ✅ Basic buffer data access functionality

---

## Phase 2: Format-Aware Processing

### Media Format Support
- [x] **4. Add format detection utilities** ✅ COMPLETED
  - ✅ Video format parsing (RGB, YUV, etc.)
  - ✅ Pixel/sample size calculations
  - ✅ Buffer size calculations for different formats

- [x] **5. Create format-specific data accessors** ✅ COMPLETED
  - ✅ Video: plane data access through MappedBuffer
  - ✅ Audio: sample access through MappedBuffer
  - ✅ Raw data: byte-level access through MappedBuffer

- [x] **6. Add format validation and safety checks** ✅ COMPLETED
  - ✅ Verify buffer size matches expected format
  - ✅ Handle stride/padding in video formats
  - ✅ Detect format mismatches

**Estimated Time:** 1 week  
**Dependencies:** Phase 1  
**Deliverable:** Format-aware data access with type safety

---

## Phase 3: Holoviz Integration for Video Visualization ✅ COMPLETED

### Video Frame Display Integration
- [x] **7. Integrate Holoviz operator for video frame visualization** ✅ COMPLETED
  - ✅ Add Holoviz operator to GStreamer example
  - ✅ Configure Holoviz for video frame display
  - ✅ Set up proper data flow from GStreamer to Holoviz

- [x] **8. Implement GStreamer to Holoviz data conversion** ✅ COMPLETED
  - ✅ Convert GStreamer buffer data to Holoviz-compatible format using GXF tensors
  - ✅ Handle different video formats (RGB, YUV, etc.) with proper memory management
  - ✅ Implement proper memory management with UnboundedAllocator

- [x] **9. Add comprehensive video visualization pipeline** ✅ COMPLETED
  - ✅ Real-time video frame display in Holoviz
  - ✅ Error handling and recovery for visualization
  - ✅ Performance optimization with proper tensor creation

**Estimated Time:** 1 week ✅  
**Dependencies:** Phase 2 ✅  
**Deliverable:** ✅ Real-time video frame visualization using Holoviz

---

## Phase 4: GPU Memory Detection & Access

### GPU Integration Foundation
- [ ] **10. Add GPU memory type detection**
  - CUDA memory detection
  - OpenGL memory detection  
  - NVMM memory detection (Jetson)

- [ ] **11. Implement GPU pointer extraction**
  - CUDA device pointer access
  - OpenGL texture ID access
  - Memory type-specific handlers

- [ ] **12. Add GPU/CPU processing decision logic**
  - Automatic path selection based on memory type
  - Performance optimization hints
  - Fallback mechanisms

**Estimated Time:** 1 week  
**Dependencies:** Phase 3  
**Deliverable:** GPU memory detection and access

---

## Phase 5: Advanced GPU Processing

### High-Performance GPU Operations
- [ ] **13. Implement basic CUDA kernel processing**
  - Simple image filters on GPU
  - Color space conversions
  - Basic computer vision operations

- [ ] **14. Add OpenGL-based processing**
  - Shader-based image effects
  - Texture-based operations
  - GPU compute shader support

- [ ] **15. Implement zero-copy GPU pipelines**
  - GPU-to-GPU processing chains
  - Minimize CPU/GPU transfers
  - Async processing with streams

**Estimated Time:** 2-3 weeks  
**Dependencies:** Phase 4  
**Deliverable:** Full GPU processing capabilities

---

## Phase 6: Output & Integration

### Data Export and Framework Integration
- [ ] **16. Add processed buffer creation**
  - Create new GstBuffers from processed data
  - Maintain proper reference counting
  - Handle different output formats

- [ ] **17. Implement data export capabilities**
  - Save to image files (PNG, JPEG)
  - Save to video files
  - Memory buffer extraction

- [ ] **18. Add integration with ML frameworks**
  - OpenCV Mat conversion
  - TensorFlow/PyTorch tensor conversion
  - CUDA memory sharing with frameworks

**Estimated Time:** 2 weeks  
**Dependencies:** Phase 5  
**Deliverable:** Complete output and integration support

---

## Phase 7: Performance & Polish

### Production-Ready Features
- [ ] **19. Add performance monitoring**
  - Processing time measurements
  - Memory usage tracking
  - Throughput analysis

- [ ] **20. Implement processing configuration**
  - Runtime parameter adjustment
  - Processing pipeline configuration
  - Quality vs performance trade-offs

- [ ] **21. Add comprehensive error handling**
  - Graceful degradation
  - Processing fallbacks
  - Detailed error reporting

**Estimated Time:** 1-2 weeks  
**Dependencies:** Phase 6  
**Deliverable:** Production-ready frame processing system

---

## Timeline Summary

| Phase | Duration | Total Time | Key Deliverable | Status |
|-------|----------|------------|-----------------|---------|
| 1 | 1 week | 1 week | Basic buffer access | ✅ COMPLETED |
| 2 | 1 week | 2 weeks | Format-aware processing | ✅ COMPLETED |
| 3 | 1 week | 3 weeks | Holoviz integration | ✅ COMPLETED |
| 4 | 1 week | 4 weeks | GPU memory access | 🔄 NEXT |
| 5 | 2-3 weeks | 6-7 weeks | GPU processing | ⏳ PENDING |
| 6 | 2 weeks | 8-9 weeks | Output & integration | ⏳ PENDING |
| 7 | 1-2 weeks | 9-11 weeks | Production polish | ⏳ PENDING |

**Total Estimated Time:** 9-11 weeks

---

## Key Design Principles

### Core Architectural Guidelines
1. **🔄 Incremental**: Each phase builds on the previous
2. **🛡️ RAII**: All memory management through smart wrappers  
3. **✅ Type Safety**: Format-aware processing with compile-time checks
4. **⚡ Performance**: Zero-copy when possible, efficient fallbacks
5. **🔧 Flexibility**: Support CPU and GPU processing paths
6. **🔗 Integration**: Easy to use with existing Holoscan operators

---

## Current Status

**Current Phase:** ✅ Phase 3 COMPLETED - Holoviz Integration Complete  
**Last Updated:** [Current Date]  
**Next Milestone:** Phase 4 - GPU Memory Detection & Access

### Progress Notes
- ✅ Initial roadmap created
- ✅ Foundation established with GstBufferGuard and GstCaps  
- ✅ Promise-based buffer retrieval working correctly
- ✅ Refactored to gst_common.hpp/cpp for better organization
- ✅ All RAII guards and helper functions properly separated
- ✅ **Phase 1 COMPLETED**: GstMapGuard implemented with full RAII memory mapping
- ✅ Direct GstMapInfo usage with RAII for safe buffer access
- ✅ Example updated to demonstrate actual buffer data access
- ✅ **Phase 2 Step 4 COMPLETED**: Video format detection utilities implemented
- ✅ VideoFormat enum with support for RGB, BGR, YUV variants, NV12/NV21, I420/YV12, GRAY
- ✅ Format detection, bytes-per-pixel calculation, buffer size calculation
- ✅ **Phase 2 Step 5 COMPLETED**: Format-specific data accessors implemented
- ✅ MappedBuffer class with RAII mapping and plane data access
- ✅ VideoInfo and AudioInfo classes with direct GStreamer structure access
- ✅ **Phase 2 Step 6 COMPLETED**: Format validation and safety checks implemented
- ✅ Buffer size validation, stride/padding validation, format mismatch detection
- ✅ Comprehensive validation reporting with detailed diagnostics
- ✅ **Phase 3 COMPLETED**: Holoviz integration with real-time video visualization
- ✅ GXF tensor creation with proper memory management using UnboundedAllocator
- ✅ Fixed segmentation fault issues with proper tensor lifecycle management
- ✅ Created reusable `create_tensor()` function for clean code organization
- ✅ Real-time video frame display working successfully in Holoviz window
- ✅ **Recent Improvements**: Fixed critical segmentation fault and memory management issues
- ✅ **Code Refactoring**: Created reusable `create_tensor()` function for better maintainability
- ✅ **Error Handling**: Added comprehensive error checking for all GXF operations
- ✅ All builds passing - **PHASE 3 COMPLETE** - ready for Phase 4!

---

## Implementation Notes

### Key Files to Modify
- `operators/gstreamer/gst_common.hpp` - RAII guards and helper functions
- `operators/gstreamer/gst_common.cpp` - Common utilities implementation
- `operators/gstreamer/gst_sink_resource.hpp` - Core sink resource interfaces
- `operators/gstreamer/gst_sink_resource.cpp` - Sink resource implementation
- `tutorials/gstreamer/holoscan_gst_example.cpp` - Examples and demonstrations

### Testing Strategy
- Unit tests for each processing function
- Integration tests with real GStreamer pipelines
- Performance benchmarks for CPU vs GPU paths
- Memory leak detection and validation

### Documentation Requirements
- API documentation for new processing functions
- Usage examples for different media types
- Performance tuning guide
- GPU setup and troubleshooting guide

---

## Future Enhancements (Post-Roadmap)

### Potential Advanced Features
- [ ] Real-time streaming output
- [ ] Multi-threaded CPU processing
- [ ] Custom shader language support
- [ ] WebRTC integration
- [ ] Cloud processing offload
- [ ] Machine learning model integration
- [ ] Advanced video codecs (AV1, VP9)
- [ ] HDR and wide color gamut support

---

*This roadmap is a living document. Update progress regularly and adjust timelines based on actual implementation experience.*




Add a SYNC-ing mechanism between the Holoscan pipe and the Gst pipe. Block the render function


./holoscan-gst-example --count 1000 --pipeline "videotestsrc pattern=0 ! videoconvert"