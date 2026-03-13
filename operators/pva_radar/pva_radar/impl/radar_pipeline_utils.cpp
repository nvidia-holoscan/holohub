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

#include <ErrorCheckMacros.h>
#include <cupva_host.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <nvcv/TensorShape.hpp>
#include <stdexcept>

/*
 * Some helper functions are implemented in pva-solutions reference code and required by
 * radar_pipeline_tensors.cpp.
 *
 * We copy them locally because we don't need all of the reference code (standard C implementation
 * of PVA algorithms) to use radar pipeline, so we can avoid building all the reference
 * implementation. Only minor modifications have been made to the original code in adapting it for
 * this project.
 *
 * The original declarations of these functions are used directly from pva-solutions source code in
 * the following header files:
 */
#include "doa_target_processing_ref.h"
#include "doppler_fft_ref.h"
#include "range_fft_ref.h"

/*
 * Some hard-coded constants are duplicated in range_fft and doppler_fft device code header files.
 * We have consolidated them here rather than making copies of the entire header files.
 * See
 * pva-solutions/src/operator/radar_processing/[range_fft|doppler_fft]/device/[range|doppler]_fft_param.h
 */
#define DATA_QBITS (20)
#define WINDOW_QBITS (31)
#define TWIDDLE_QBITS (30)

#define DATA_SCALE (1UL << DATA_QBITS)
#define WINDOW_SCALE (1UL << WINDOW_QBITS)
#define TWIDDLE_SCALE (1UL << TWIDDLE_QBITS)

/**
 * @brief Populates the GP structure with the default values.
 *
 * @param GP Pointer to the GP structure to populate.
 * @param calibVectorQbits Number of Q bits for the calib vector.
 * @param fftQbits Number of Q bits for the FFT tensor.
 */
void populateGP(PVARadarGP* GP, int32_t calibVectorQbits, int32_t fftQbits) {
  assert(GP != nullptr);

  // pre-computed values.
  GP->nofAFFT = 256;
  GP->nofSnap = 16;
  GP->dis = 0.5f;
  GP->fs = 25000000.0f;
  GP->nofSamples = 512;
  GP->PRI = 2.798000059556216e-05;
  GP->nofRamps = 512;
  GP->deltaV = 0.47714347;
  GP->disE = 0.75f;
  GP->contR = 2.0580426e-05;
  GP->contVFast = -9.3106401e-06;
  GP->contVSlow = -0.0019080447;
  GP->nofAperture = 8;
  GP->repeatFold = 8;
  GP->nr = 512;
  GP->calibVectorQbits = calibVectorQbits;
  GP->fftQbits = fftQbits;

  int32_t ap_upper_orig[] = {0, 3, 4, 7, 8, 11, 12, 15};
  int32_t ap_lower_orig[] = {1, 2, 5, 6, 9, 10, 13, 14};
  int32_t ap_upper_ind_orig[] = {0, 9, 4, 13, 8, 17, 12, 21};
  int32_t ap_lower_ind_orig[] = {3, 6, 7, 10, 11, 14, 15, 18};

  std::copy(ap_upper_orig, ap_upper_orig + 8, GP->apUpper);
  std::copy(ap_lower_orig, ap_lower_orig + 8, GP->apLower);
  std::copy(ap_upper_ind_orig, ap_upper_ind_orig + 8, GP->apUpperInd);
  std::copy(ap_lower_ind_orig, ap_lower_ind_orig + 8, GP->apLowerInd);
}

/**
 * @brief Converts a floating-point value to a Q-bits fixed-point integer representation.
 *
 * @param val The floating-point value to convert.
 * @param qbits The number of fractional bits (Q-bits) for the fixed-point representation.
 * @return The Q-bits fixed-point representation as a 32-bit integer.
 */
static inline int32_t float_to_int32(float val, int qbits) {
  assert(qbits <= 30);
  return static_cast<int32_t>(std::round(val * (1 << qbits)));
}

/**
 * @brief Populates the calib vector with the default values.
 *
 * @param calibVector Pointer to the calib vector to populate.
 * @param calibVectorQbits Number of Q bits for the calib vector.
 */
void populateCalibVector(int32_t* calibVector, int32_t qbits) {
  const int32_t calibVector_orig[] = {
      // Real part
      float_to_int32(0.95419002f, qbits),
      float_to_int32(-0.34347999f, qbits),
      float_to_int32(-0.083509997f, qbits),
      float_to_int32(0.89093000f, qbits),
      float_to_int32(0.62287998f, qbits),
      float_to_int32(0.42886999f, qbits),
      float_to_int32(0.59582001f, qbits),
      float_to_int32(0.14289001f, qbits),

      // Imaginary part
      float_to_int32(-0.15831999f, qbits),
      float_to_int32(-0.82799000f, qbits),
      float_to_int32(-0.77504998f, qbits),
      float_to_int32(0.39897999f, qbits),
      float_to_int32(0.65314001f, qbits),
      float_to_int32(-0.73414999f, qbits),
      float_to_int32(-0.49675000f, qbits),
      float_to_int32(0.94439000f, qbits),

      // Real part
      float_to_int32(-0.70082003f, qbits),
      float_to_int32(0.76574999f, qbits),
      float_to_int32(0.42387000f, qbits),
      float_to_int32(-0.97081000f, qbits),
      float_to_int32(0.21018000f, qbits),
      float_to_int32(-1.0f, qbits),
      float_to_int32(-0.78645998f, qbits),
      float_to_int32(0.77932000f, qbits),

      // Imaginary part
      float_to_int32(0.57045001f, qbits),
      float_to_int32(0.52064002f, qbits),
      float_to_int32(0.64955002f, qbits),
      float_to_int32(-0.0039901999f, qbits),
      float_to_int32(-0.98920000f, qbits),
      float_to_int32(-0.029625000f, qbits),
      float_to_int32(-0.27674001f, qbits),
      float_to_int32(-0.60835999f, qbits),
  };

  memcpy(calibVector, calibVector_orig, sizeof(calibVector_orig));
}

/**
 * @brief Loads the calibration vector data into a tensor.
 *
 * @param[in] calibTensorHandle The handle to the calibration vector tensor.
 * @param[in] calibVector The calibration vector data.
 * @param[in] numElements The number of complex elements in the calibration vector.
 * @return Returns 0 on success, -1 otherwise.
 */
int32_t loadCalibVectorToTensorFixedPoint(NVCVTensorHandle calibTensorHandle,
                                          const int32_t calibVector[], int32_t numElements) {
  NVCVTensorData tensorData;
  NVCV_CHECK_ERROR(nvcvTensorExportData(calibTensorHandle, &tensorData));

  void* tensorPtr = nullptr;
  CUPVA_CHECK_ERROR(
      CupvaMemGetHostPointer((void**)&tensorPtr, (void*)tensorData.buffer.strided.basePtr));

  if (tensorPtr == nullptr) {
    throw std::runtime_error("Failed to get host pointer for calibration vector tensor");
  }

  size_t dataSize = numElements * sizeof(int32_t);
  memcpy(tensorPtr, calibVector, dataSize);

  return 0;
}

template <typename T>
void generateFFTWindowImpl(NVCVTensorHandle tensorHandle, PVABatchFFTWindowType windowType) {
  NVCVTensorData tensorData;
  NVCV_CHECK_ERROR(nvcvTensorExportData(tensorHandle, &tensorData));
  T* tensorPtr = NULL;
  CUPVA_CHECK_ERROR(
      CupvaMemGetHostPointer((void**)&tensorPtr, (void*)tensorData.buffer.strided.basePtr));
  if (tensorPtr == nullptr) {
    throw std::runtime_error("Failed to get host pointer for FFT window tensor");
  }

  int64_t tensorShapeNCHW[4];
  nvcvTensorShapePermute(tensorData.layout, tensorData.shape, NVCV_TENSOR_NCHW, tensorShapeNCHW);

  const int32_t width = tensorShapeNCHW[3];
  if (width <= 1) {
    throw std::runtime_error("FFT window size is invalid");
  }

  for (int32_t w = 0; w < width; w++) {
    switch (windowType) {
      case PVA_BATCH_FFT_WINDOW_HANNING: {
        tensorPtr[w] =
            static_cast<T>(WINDOW_SCALE * 0.5f * (1.0f - cosf(2.0f * M_PI * w / (width - 1))));
      } break;
      case PVA_BATCH_FFT_WINDOW_USER_DEFINED: {
        tensorPtr[w] = static_cast<T>(WINDOW_SCALE * 1.0f);
      } break;
      default:
        break;
    }
  }
}

/**
 * @brief Generates a window function to supply to an FFT operator.
 *
 * @param[in] tensorHandle The handle to the tensor to generate the window for.
 * @param[in] windowType The type of window to generate: HANNING or USER_DEFINED (all flat, intended
 * to be used as scale for user's normalized window with custom shape).
 */
void generateFFTWindow(NVCVTensorHandle tensorHandle, PVABatchFFTWindowType windowType) {
  NVCVTensorData tensorData;
  NVCV_CHECK_ERROR(nvcvTensorExportData(tensorHandle, &tensorData));

  switch (tensorData.dtype) {
    case NVCV_DATA_TYPE_S32:
      generateFFTWindowImpl<int32_t>(tensorHandle, windowType);
      break;
    default:
      throw std::runtime_error("Unsupported data type for FFT window tensor");
      break;
  }
}

// The original implementations in pva-solutions are identical, so we consolidated above.
void generateRangeFFTWindow(NVCVTensorHandle tensorHandle, PVABatchFFTWindowType windowType) {
  generateFFTWindow(tensorHandle, windowType);
}
void generateDopplerFFTWindow(NVCVTensorHandle tensorHandle, PVABatchFFTWindowType windowType) {
  generateFFTWindow(tensorHandle, windowType);
}
