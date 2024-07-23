#ifndef VERSION_HELPER_MACROS_HPP
#define VERSION_HELPER_MACROS_HPP

// If GXF has gxf/std/dlpack_utils.hpp it has DLPack support
#if __has_include("gxf/std/dlpack_utils.hpp")
  #define GXF_HAS_DLPACK_SUPPORT 1
  #include "gxf/std/tensor.hpp"
#else
  #define GXF_HAS_DLPACK_SUPPORT 0
  #include "holoscan/core/gxf/gxf_tensor.hpp"
#endif

#endif /* VERSION_HELPER_MACROS_HPP */
