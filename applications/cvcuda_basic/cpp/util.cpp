// FMT_S8
// FMT_S16
// FMT_S32
// FMT_U8
// FMT_U16
// FMT_U32
// FMT_F16
// FMT_F32
// FMT_F64
// FMT_C64
// FMT_C128
// // also RGBA, BGR and BGRA variants of the below
// FMT_RGB8    // HWC
// FMT_RGBf16  // HWC
// FMT_RGBf32  // HWC
// FMT_RGB8p   // CHW
// FMT_RGBf16p // CHW
// FMT_RGBf32p // CHW

// ImageFormat fmt_from_dtype() {

// }

nvcv::DataType dldatatype_to_nvcvdtype(DLDataType dtype, int num_channels=0) {
  nvcv::DataType type;
  uint8_t bits = dtype.bits;
  uint16_t channels = (num_channels == 0) ? dtype.lanes : num_channels;

  switch (dtype.code) {
    case kDLInt:
      switch (bits) {
        case 8:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_S8
              break;
            case 2:
              type = nvcv::TYPE_2S8
              break;
            case 3:
              type = nvcv::TYPE_3S8
              break;
            case 4:
              type = nvcv::TYPE_4S8
              break;
            default:
              throw std::runtime_error(
                fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                             dtype.code,
                             dtype.bits,
                             channels));
          }
          break;
        case 16:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_S16
              break;
            case 2:
              type = nvcv::TYPE_2S16
              break;
            case 3:
              type = nvcv::TYPE_3S16
              break;
            case 4:
              type = nvcv::TYPE_4S16
              break;
            default:
              throw std::runtime_error(
                fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                             dtype.code,
                             dtype.bits,
                             channels));
          }
          break;
        case 32:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_S32
              break;
            case 2:
              type = nvcv::TYPE_2S32
              break;
            case 3:
              type = nvcv::TYPE_3S32
              break;
            case 4:
              type = nvcv::TYPE_4S32
              break;
            default:
              throw std::runtime_error(
                fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                             dtype.code,
                             dtype.bits,
                             channels));
          }
          break;
        case 64:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_S64
              break;
            case 2:
              type = nvcv::TYPE_2S64
              break;
            case 3:
              type = nvcv::TYPE_3S64
              break;
            case 4:
              type = nvcv::TYPE_4S64
              break;
            default:
              throw std::runtime_error(
                fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                             dtype.code,
                             dtype.bits,
                             channels));
          }
          break;
        default:
          throw std::runtime_error(
            fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                         dtype.code,
                         dtype.bits,
                         channels));
      }
      break;
    case kDLUInt:
      switch (bits) {
        case 8:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_U8
              break;
            case 2:
              type = nvcv::TYPE_2U8
              break;
            case 3:
              type = nvcv::TYPE_3U8
              break;
            case 4:
              type = nvcv::TYPE_4U8
              break;
            default:
              throw std::runtime_error(
                fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                             dtype.code,
                             dtype.bits,
                             channels));
          }
          break;
        case 16:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_U16
              break;
            case 2:
              type = nvcv::TYPE_2U16
              break;
            case 3:
              type = nvcv::TYPE_3U16
              break;
            case 4:
              type = nvcv::TYPE_4U16
              break;
            default:
              throw std::runtime_error(
                fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                             dtype.code,
                             dtype.bits,
                             channels));
          }
          break;
        case 32:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_U32
              break;
            case 2:
              type = nvcv::TYPE_2U32
              break;
            case 3:
              type = nvcv::TYPE_3U32
              break;
            case 4:
              type = nvcv::TYPE_4U32
              break;
            default:
              throw std::runtime_error(
                fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                             dtype.code,
                             dtype.bits,
                             channels));
          }
          break;
        case 64:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_U64
              break;
            case 2:
              type = nvcv::TYPE_2U64
              break;
            case 3:
              type = nvcv::TYPE_3U64
              break;
            case 4:
              type = nvcv::TYPE_4U64
              break;
            default:
              throw std::runtime_error(
                fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                             dtype.code,
                             dtype.bits,
                             channels));
          }
          break;
        default:
          throw std::runtime_error(
            fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                         dtype.code,
                         dtype.bits,
                         channels));
      }
      break;
    case kDLFloat:
      switch (bits) {
        case 16:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_F16
              break;
            case 2:
              type = nvcv::TYPE_2F16
              break;
            case 3:
              type = nvcv::TYPE_3F16
              break;
            case 4:
              type = nvcv::TYPE_4F16
              break;
            default:
              throw std::runtime_error(
                fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                             dtype.code,
                             dtype.bits,
                             channels));
          }
          break;
        case 32:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_F32
              break;
            case 2:
              type = nvcv::TYPE_2F32
              break;
            case 3:
              type = nvcv::TYPE_3F32
              break;
            case 4:
              type = nvcv::TYPE_4F32
              break;
            default:
              throw std::runtime_error(
                fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                             dtype.code,
                             dtype.bits,
                             channels));
          }
          break;
        case 64:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_F64
              break;
            case 2:
              type = nvcv::TYPE_2F64
              break;
            case 3:
              type = nvcv::TYPE_3F64
              break;
            case 4:
              type = nvcv::TYPE_4F64
              break;
            default:
              throw std::runtime_error(
                fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                             dtype.code,
                             dtype.bits,
                             channels));
          }
          break;
        default:
          throw std::runtime_error(
            fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                         dtype.code,
                         dtype.bits,
                         channels));
      }
      break;
    case kDLComplex:
      switch (bits) {
        case 64:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_C64
              break;
            case 2:
              type = nvcv::TYPE_2C64
              break;
            case 3:
              type = nvcv::TYPE_3C64
              break;
            case 4:
              type = nvcv::TYPE_4C64
              break;
            default:
              throw std::runtime_error(
                fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                             dtype.code,
                             dtype.bits,
                             channels));
          }
          break;
        case 128:
          switch (channels) {
            case 1:
              type = nvcv::TYPE_C128
              break;
            case 2:
              type = nvcv::TYPE_2C128
              break;
            default:
              throw std::runtime_error(
                fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                             dtype.code,
                             dtype.bits,
                             channels));
          }
          break;
        default:
          throw std::runtime_error(
            fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                         dtype.code,
                         dtype.bits,
                         channels));
      }
      break;
    default:
      throw std::runtime_error(
        fmt::format("Unsupported DLPack data type (code: {}, bits: {}, channels: {})",
                     dtype.code,
                     dtype.bits,
                     channels));
  }
}



  // nvcv::Tensor::Requirements in_reqs =
  //     nvcv::Tensor::CalcRequirements(cv_tensor_shape, cv_dtype);

  // // Create a tensor buffer to store the data pointer and pitch bytes for each plane
  // nvcv::TensorDataStridedCuda in_data(
  //     nvcv::TensorShape{in_reqs.shape, in_reqs.rank, in_reqs.layout},
  //     nvcv::DataType{in_reqs.dtype},
  //     in_buffer);

