# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging
import sys
from enum import Enum
from typing import Dict, Union

import cupy as cp
import numpy as np
from holoscan import as_tensor
from holoscan.core import DLDeviceType
from holoscan.gxf import Entity

from holohub.grpc_operators import holoscan_pb2 as holoscan_proto

logger = logging.getLogger(__name__)

current_module = sys.modules[__name__]


class ProtoTensorCudaArrayInterface:
    def __init__(self, proto_tensor):
        data = np.frombuffer(proto_tensor.data, dtype=TensorProto._get_numpy_type(proto_tensor))
        data = cp.asarray(data)
        self.__cuda_array_interface__ = {
            "version": 2,
            "shape": tuple(proto_tensor.dimensions),
            "data": (data.data.ptr, False),
            "typestr": proto_tensor.type_str,
            "strides": tuple(proto_tensor.strides),
        }


class DLDataTypeCode(Enum):
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLOpaqueHandle = 3
    kDLComplex = 5


class TensorProto:
    NULL_KEY_NAME = "__NULL__"

    STORAGE_TYPES = {
        holoscan_proto.Tensor.MemoryStorageType.kHost: DLDeviceType.DLCPU,
        holoscan_proto.Tensor.MemoryStorageType.kDevice: DLDeviceType.DLCUDA,
        holoscan_proto.Tensor.MemoryStorageType.kSystem: DLDeviceType.DLCUDAHOST,
    }

    PRIMITIVE_TYPES = {
        holoscan_proto.Tensor.PrimitiveType.kUnsigned8: np.uint8,
        holoscan_proto.Tensor.PrimitiveType.kUnsigned16: np.uint16,
        holoscan_proto.Tensor.PrimitiveType.kFloat32: np.float32,
    }

    @staticmethod
    def add_tensor_to_proto(gxf_entity, response: holoscan_proto.EntityResponse):
        TensorProto._gxf_tensors_to_proto_tensors(gxf_entity, response.tensors)

    @staticmethod
    def tensor_to_proto(gxf_entity: Dict) -> holoscan_proto.EntityRequest:
        entity_request = holoscan_proto.EntityRequest()
        TensorProto._gxf_tensors_to_proto_tensors(gxf_entity, entity_request.tensors)
        return entity_request

    @staticmethod
    def proto_to_tensor(
        entity_request: Union[holoscan_proto.EntityRequest, holoscan_proto.EntityResponse], context
    ) -> Entity:
        gxf_entity = Entity(context)
        TensorProto._proto_tensors_to_gxf_tensors(entity_request, gxf_entity)
        return gxf_entity

    @staticmethod
    def _gxf_tensors_to_proto_tensors(gxf_entity, tensors):
        for key, gxf_tensor in gxf_entity.items():
            try:
                tensor = tensors[key if key else TensorProto.NULL_KEY_NAME]

                for dim in gxf_tensor.shape:
                    tensor.dimensions.append(dim)

                for stride in gxf_tensor.strides:
                    tensor.strides.append(stride)

                tensor.primitive_type = TensorProto._get_gxf_type(gxf_tensor)
                tensor.type_str = TensorProto._get_numpy_type_str(gxf_tensor)
                tensor.memory_storage_type = TensorProto._get_proto_storage_type(gxf_tensor)

                if (
                    tensor.memory_storage_type == holoscan_proto.Tensor.MemoryStorageType.kHost
                    or tensor.memory_storage_type == holoscan_proto.Tensor.MemoryStorageType.kSystem
                ):
                    np_array = np.asarray(
                        gxf_tensor, dtype=TensorProto.PRIMITIVE_TYPES[tensor.primitive_type]
                    )
                    tensor.data = np_array.tobytes()
                elif tensor.memory_storage_type == holoscan_proto.Tensor.MemoryStorageType.kDevice:
                    cp_array = cp.asarray(
                        gxf_tensor, dtype=TensorProto.PRIMITIVE_TYPES[tensor.primitive_type]
                    )
                    tensor.data = cp.asnumpy(cp_array).tobytes()

            except Exception as e:
                logger.error(f"Failed to convert tensor with key {key}: {str(e)}")
                raise

    @staticmethod
    def _proto_tensors_to_gxf_tensors(entity_request, gxf_entity):
        for key, proto_tensor in entity_request.tensors.items():
            try:
                dtype = TensorProto._get_numpy_type(proto_tensor)

                if (
                    proto_tensor.memory_storage_type
                    == holoscan_proto.Tensor.MemoryStorageType.kHost
                    or proto_tensor.memory_storage_type
                    == holoscan_proto.Tensor.MemoryStorageType.kSystem
                ):
                    tensor = as_tensor(
                        np.frombuffer(proto_tensor.data, dtype=dtype)
                        .reshape(proto_tensor.dimensions)
                        .copy()
                    )
                elif (
                    proto_tensor.memory_storage_type
                    == holoscan_proto.Tensor.MemoryStorageType.kDevice
                ):
                    tensor = as_tensor(cp.asarray(ProtoTensorCudaArrayInterface(proto_tensor)))

                gxf_entity.add(tensor, key if key != TensorProto.NULL_KEY_NAME else "")
            except Exception as e:
                logger.error(f"Failed to convert tensor with key {key}: {str(e)}")
                raise

    @staticmethod
    def _get_gxf_storage_type(proto_tensor):
        if proto_tensor.memory_storage_type in TensorProto.STORAGE_TYPES:
            return TensorProto.STORAGE_TYPES[proto_tensor.memory_storage_type]
        raise ValueError(f"Unsupported memory storage type: {proto_tensor.memory_storage_type}")

    @staticmethod
    def _get_numpy_type(proto_tensor):

        if proto_tensor.primitive_type in TensorProto.PRIMITIVE_TYPES:
            return TensorProto.PRIMITIVE_TYPES[proto_tensor.primitive_type]

        raise ValueError(f"Unsupported primitive type: {proto_tensor.primitive_type}")

    @staticmethod
    def _get_proto_storage_type(gxf_tensor):
        if gxf_tensor.device.device_type in TensorProto.STORAGE_TYPES.values():
            return next(
                key
                for key, value in TensorProto.STORAGE_TYPES.items()
                if value == gxf_tensor.device.device_type
            )

        raise ValueError(f"Unsupported device type: {gxf_tensor.device.device_type}")

    @staticmethod
    def _get_gxf_type(value):
        if value.dtype.code == DLDataTypeCode.kDLUInt.value:
            if value.dtype.bits == 8:
                return holoscan_proto.Tensor.PrimitiveType.kUnsigned8
            if value.dtype.bits == 16:
                return holoscan_proto.Tensor.PrimitiveType.kUnsigned16
            if value.dtype.bits == 32:
                return holoscan_proto.Tensor.PrimitiveType.kUnsigned32
            if value.dtype.bits == 64:
                return holoscan_proto.Tensor.PrimitiveType.kUnsigned64
        elif value.dtype.code == DLDataTypeCode.kDLFloat.value:
            if value.dtype.bits == 16:
                return holoscan_proto.Tensor.PrimitiveType.kFloat16
            elif value.dtype.bits == 32:
                return holoscan_proto.Tensor.PrimitiveType.kFloat32
            elif value.dtype.bits == 64:
                return holoscan_proto.Tensor.PrimitiveType.kFloat64

        raise ValueError(f"Unsupported primitive type: {value.dtype}")

    @staticmethod
    def _get_numpy_type_str(gxf_tensor):
        if gxf_tensor.dtype.lanes != 1:
            raise ValueError(
                "typestring conversion only support DLDataType with one lane, "
                f"but found dtype.lanes: ({gxf_tensor.dtype.lanes})."
            )

        if gxf_tensor.dtype.code == DLDataTypeCode.kDLUInt.value:
            if gxf_tensor.dtype.bits == 8:
                return "|u1"
            if gxf_tensor.dtype.bits == 16:
                return "<u2"
            if gxf_tensor.dtype.bits == 32:
                return "<u4"
            if gxf_tensor.dtype.bits == 64:
                return "<u8"
        elif gxf_tensor.dtype.code == DLDataTypeCode.kDLFloat.value:
            if gxf_tensor.dtype.bits == 16:
                return "<f2"
            elif gxf_tensor.dtype.bits == 32:
                return "<f4"
            elif gxf_tensor.dtype.bits == 64:
                return "<f8"

        raise ValueError(f"Data type code ({gxf_tensor.dtype.code}) is not supported!")
