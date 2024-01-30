"""
 SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""  # noqa: E501

import os
import struct
import time
from enum import Enum
from io import BufferedIOBase, BytesIO
from typing import Any, Generator, List, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, List[float]]
ReadOnlyBuffer = bytes
WriteableBuffer = Union[bytearray, memoryview]
ReadableBuffer = Union[ReadOnlyBuffer, WriteableBuffer]


class EntityIndex:
    """Serializer/deserializer for the EntityIndex.

    ```
        struct EntityIndex {
            uint64_t log_time;     // Time when data was logged
            uint64_t data_size;    // Size of data block
            uint64_t data_offset;  // Location of data block
        };
    ```
    """

    HEADER_STRUCT = struct.Struct(
        "="  # no padding
        "Q"  # log_time => uint64_t
        "Q"  # data_size => uint64_t
        "Q"  # data_offset => uint64_t
    )
    HEADER_SIZE = HEADER_STRUCT.size

    def __init__(
        self,
        *,
        data: Tuple[int, int, int] = None,
        buffer: Sequence = None,
        reader: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ):
        self.read(data=data, buffer=buffer, reader=reader, offset=offset, whence=whence)

    @property
    def log_time(self) -> int:
        return self._log_time

    @property
    def data_size(self) -> int:
        return self._data_size

    @property
    def data_offset(self) -> int:
        return self._data_offset

    @property
    def size_in_bytes(self) -> int:
        return self.HEADER_SIZE

    def __repr__(self) -> str:
        return (
            f"EntityIndex(log_time={self.log_time}, data_size={self.data_size}, "
            f"data_offset={self.data_offset})"
        )

    def read(
        self,
        *,
        data: Tuple[int, int, int] = None,
        buffer: ReadableBuffer = None,
        reader: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ) -> None:
        if not data and not buffer and not reader:
            raise ValueError("Either data, buffer or reader must be provided")

        if buffer:
            reader = BytesIO(buffer)

        if data:
            log_time, data_size, data_offset = data
        else:
            reader.seek(offset, whence)
            buffer = reader.read(self.HEADER_SIZE)
            header_data = self.HEADER_STRUCT.unpack(buffer)

            log_time = header_data[0]
            data_size = header_data[1]
            data_offset = header_data[2]

        self._log_time = log_time
        self._data_size = data_size
        self._data_offset = data_offset

    def write(
        self,
        *,
        buffer: WriteableBuffer = None,
        writer: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ) -> Any:
        if not buffer and not writer:
            raise ValueError("Either buffer or writer must be provided")
        if buffer:
            writer = BytesIO(buffer)

        writer.seek(offset, whence)
        orig_offset = writer.tell()
        curr_offset = orig_offset

        serialized_buffer = self.HEADER_STRUCT.pack(
            self._log_time, self._data_size, self._data_offset
        )
        writer.write(serialized_buffer)

        curr_offset = writer.tell()

        return curr_offset - orig_offset


class EntityHeader:
    """Serializer/deserializer for the EntityHeader.

    ```
    struct EntityHeader {
        uint64_t serialized_size;  // Size of the serialized entity in bytes
        uint32_t checksum;         // Checksum to verify the integrity of the message
        uint64_t sequence_number;  // Sequence number of the message
        uint32_t flags;            // Flags to specify delivery options
        uint64_t component_count;  // Number of components in the entity
        uint64_t reserved;         // Bytes reserved for future use
    };
    ```
    """

    HEADER_STRUCT = struct.Struct(
        "="  # no padding
        "Q"  # serialized_size => uint64_t
        "I"  # checksum => uint32_t
        "Q"  # sequence_number => uint64_t
        "I"  # flags => uint32_t
        "Q"  # component_count => uint64_t
        "Q"  # reserved => uint64_t
    )
    HEADER_SIZE = HEADER_STRUCT.size

    def __init__(
        self,
        *,
        data: Tuple[int, int, int, int, int, int] = None,
        buffer: Sequence = None,
        reader: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ):
        self.deserialize(data=data, buffer=buffer, reader=reader, offset=offset, whence=whence)

    @property
    def serialized_size(self) -> int:
        return self._serialized_size

    @property
    def checksum(self) -> int:
        return self._checksum

    @property
    def sequence_number(self) -> int:
        return self._sequence_number

    @property
    def flags(self) -> int:
        return self._flags

    @property
    def component_count(self) -> int:
        return self._component_count

    @property
    def reserved(self) -> int:
        return self._reserved

    def __repr__(self) -> str:
        return f"EntityHeader(serialized_size={self.serialized_size}, checksum={self.checksum}, sequence_number={self.sequence_number}, flags={self.flags}, component_count={self.component_count}, reserved={self.reserved})"  # noqa

    def deserialize(
        self,
        *,
        data: Tuple[int, int, int, int, int, int] = None,
        buffer: ReadableBuffer = None,
        reader: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ) -> None:
        if not data and not buffer and not reader:
            raise ValueError("Either data, buffer or reader must be provided")

        if buffer:
            reader = BytesIO(buffer)

        if data:
            (
                serialized_size,
                checksum,
                sequence_number,
                flags,
                component_count,
                reserved,
            ) = data
        elif reader:
            reader.seek(offset, whence)
            buffer = reader.read(self.HEADER_SIZE)
            header_data = self.HEADER_STRUCT.unpack(buffer)

            serialized_size = header_data[0]
            checksum = header_data[1]
            sequence_number = header_data[2]
            flags = header_data[3]
            component_count = header_data[4]
            reserved = header_data[5]

        self._serialized_size = serialized_size
        self._checksum = checksum
        self._sequence_number = sequence_number
        self._flags = flags
        self._component_count = component_count
        self._reserved = reserved

        return self

    def serialize(
        self,
        *,
        buffer: WriteableBuffer = None,
        writer: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ) -> Any:
        if not buffer and not writer:
            raise ValueError("Either buffer or writer reader must be provided")

        if buffer:
            writer = BytesIO(buffer)

        writer.seek(offset, whence)
        serialized_buffer = self.HEADER_STRUCT.pack(
            self._serialized_size,
            self._checksum,
            self._sequence_number,
            self._flags,
            self._component_count,
            self._reserved,
        )
        writer.write(serialized_buffer)
        return serialized_buffer


class TID:
    def __init__(self, hash1: int, hash2: int):
        self.hash1 = hash1
        self.hash2 = hash2

    def __repr__(self):
        return f"TID(hash1={self.hash1}, hash2={self.hash2})"


class ComponentHeader:
    """Serializer/deserializer for the ComponentHeader.

    ```
    struct ComponentHeader {
        uint64_t serialized_size;  // Size of the serialized component in bytes
        gxf_tid_t tid;             // Type ID of the component
        uint64_t name_size;        // Size of the component name in bytes
    };
    ```
    """

    HEADER_STRUCT = struct.Struct(
        "="  # no padding
        "Q"  # serialized_size => uint64_t
        "Q"  # tid.hash1 => uint64_t
        "Q"  # tid.hash2 => uint64_t
        "Q"  # name_size => uint64_t
    )
    HEADER_SIZE = HEADER_STRUCT.size

    def __init__(
        self,
        *,
        data: Tuple[int, int, int, int] = None,
        buffer: Sequence = None,
        reader: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ):
        self.deserialize(data=data, buffer=buffer, reader=reader, offset=offset, whence=whence)

    @property
    def serialized_size(self) -> int:
        return self._serialized_size

    @property
    def tid(self) -> TID:
        return self._tid

    @property
    def name_size(self) -> int:
        return self._name_size

    def __repr__(self) -> str:
        return f"ComponentHeader(serialized_size={self.serialized_size}, tid={self.tid}, name_size={self.name_size})"  # noqa

    def deserialize(
        self,
        *,
        data: Tuple[int, int, int, int] = None,
        buffer: ReadableBuffer = None,
        reader: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ) -> None:
        if not data and not buffer and not reader:
            raise ValueError("Either data, buffer or reader must be provided")

        if buffer:
            reader = BytesIO(buffer)

        if data:
            serialized_size, tid_hash1, tid_hash2, name_size = data
        elif reader:
            reader.seek(offset, whence)
            buffer = reader.read(self.HEADER_SIZE)
            header_data = self.HEADER_STRUCT.unpack(buffer)

            serialized_size = header_data[0]
            tid_hash1 = header_data[1]
            tid_hash2 = header_data[2]
            name_size = header_data[3]

        self._serialized_size = serialized_size
        self._tid = TID(tid_hash1, tid_hash2)
        self._name_size = name_size

        return self

    def serialize(
        self,
        *,
        buffer: WriteableBuffer = None,
        writer: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ) -> Any:
        if not buffer and not writer:
            raise ValueError("Either buffer or writer reader must be provided")

        if buffer:
            writer = BytesIO(buffer)

        writer.seek(offset, whence)
        serialized_buffer = self.HEADER_STRUCT.pack(
            self._serialized_size, self._tid.hash1, self._tid.hash2, self._name_size
        )
        writer.write(serialized_buffer)
        return serialized_buffer


class MemoryStorageType(Enum):
    """
    ```
        enum struct MemoryStorageType { kHost = 0, kDevice = 1, kSystem = 2 };
    ```
    """

    kHost = 0
    kDevice = 1
    kSystem = 2


class PrimitiveType(Enum):
    kCustom = 0
    kInt8 = 1
    kUnsigned8 = 2
    kInt16 = 3
    kUnsigned16 = 4
    kInt32 = 5
    kUnsigned32 = 6
    kInt64 = 7
    kUnsigned64 = 8
    kFloat32 = 9
    kFloat64 = 10


PrimitiveType2DType = {
    PrimitiveType.kCustom: None,
    PrimitiveType.kInt8: np.int8,
    PrimitiveType.kUnsigned8: np.uint8,
    PrimitiveType.kInt16: np.int16,
    PrimitiveType.kUnsigned16: np.uint16,
    PrimitiveType.kInt32: np.int32,
    PrimitiveType.kUnsigned32: np.uint32,
    PrimitiveType.kInt64: np.int64,
    PrimitiveType.kUnsigned64: np.uint64,
    PrimitiveType.kFloat32: np.float32,
    PrimitiveType.kFloat64: np.float64,
}


class Shape:
    kMaxRank = 8


class TensorHeader:
    """Serializer/deserializer for the TensorHeader.

    ```
        struct TensorHeader {
            MemoryStorageType storage_type;     // CPU or GPU tensor
            PrimitiveType element_type;         // Tensor element type
            uint64_t bytes_per_element;         // Bytes per tensor element
            uint32_t rank;                      // Tensor rank
            int32_t dims[Shape::kMaxRank];      // Tensor dimensions
            uint64_t strides[Shape::kMaxRank];  // Tensor strides
        };
    ```
    """

    HEADER_STRUCT = struct.Struct(
        "="  # no padding
        "i"  # storage_type => int32_t
        "i"  # element_type => int32_t
        "Q"  # bytes_per_element => uint64_t
        "I"  # rank => uint32_t
        f"{int(Shape.kMaxRank)}i"  # dims => int32_t[Shape.kMaxRank]
        f"{int(Shape.kMaxRank)}Q"  # strides => uint64_t[Shape.kMaxRank]
    )
    HEADER_SIZE = HEADER_STRUCT.size

    def __init__(
        self,
        *,
        data: Tuple[int, int, int, int, Tuple[int, ...], Tuple[int, ...]] = None,
        buffer: Sequence = None,
        reader: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ):
        self.deserialize(data=data, buffer=buffer, reader=reader, offset=offset, whence=whence)

    @property
    def storage_type(self) -> MemoryStorageType:
        return self._storage_type

    @property
    def element_type(self) -> PrimitiveType:
        return self._element_type

    @property
    def bytes_per_element(self) -> int:
        return self._bytes_per_element

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def dims(self) -> Tuple[int, ...]:
        return self._dims

    @property
    def strides(self) -> Tuple[int, ...]:
        return self._strides

    @property
    def dtype(self) -> np.dtype:
        return PrimitiveType2DType[self.element_type]

    def __repr__(self) -> str:
        return f"TensorHeader(storage_type={self.storage_type}, element_type={self.element_type}, bytes_per_element={self.bytes_per_element}, rank={self.rank}, dims={self.dims}, strides={self.strides})"  # noqa

    def deserialize(
        self,
        *,
        data: Tuple[int, int, int, int, Tuple[int, ...], Tuple[int, ...]] = None,
        buffer: ReadableBuffer = None,
        reader: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ) -> None:
        if not data and not buffer and not reader:
            raise ValueError("Either data, buffer or reader must be provided")

        if buffer:
            reader = BytesIO(buffer)

        if data:
            storage_type, element_type, bytes_per_element, rank, dims, strides = data
        elif reader:
            reader.seek(offset, whence)
            buffer = reader.read(self.HEADER_SIZE)
            header_data = self.HEADER_STRUCT.unpack(buffer)

            storage_type = MemoryStorageType(header_data[0])
            element_type = PrimitiveType(header_data[1])
            bytes_per_element = header_data[2]
            rank = header_data[3]
            dims = header_data[4 : 4 + rank]
            strides = header_data[4 + Shape.kMaxRank : 4 + Shape.kMaxRank + rank]

        self._storage_type = storage_type
        self._element_type = element_type
        self._bytes_per_element = bytes_per_element
        self._rank = rank
        self._dims = dims
        self._strides = strides

        return self

    def serialize(
        self,
        *,
        buffer: WriteableBuffer = None,
        writer: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ) -> Any:
        if not buffer and not writer:
            raise ValueError("Either buffer or writer reader must be provided")

        if buffer:
            writer = BytesIO(buffer)

        writer.seek(offset, whence)
        dim = list(self._dims) + [1] * (Shape.kMaxRank - self._rank)
        stride = list(self._strides) + [0] * (Shape.kMaxRank - self._rank)
        serialized_buffer = self.HEADER_STRUCT.pack(
            self._storage_type.value,
            self._element_type.value,
            self._bytes_per_element,
            self._rank,
            *dim,
            *stride,
        )
        writer.write(serialized_buffer)
        return serialized_buffer


TensorType = (3996102265592038524, 11968035723744066232)


class Tensor:
    """Serializer/deserializer for the Tensor.

    ```
    struct Tensor {
        TensorHeader tensor_header        // Tensor header
        uint8_t* data;                    // Tensor data
    };
    ```
    """

    def __init__(
        self,
        *,
        data: Tuple[TensorHeader, ArrayLike] = None,
        buffer: Sequence = None,
        reader: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ):
        self.read(data=data, buffer=buffer, reader=reader, offset=offset, whence=whence)

    @property
    def header(self) -> TensorHeader:
        return self._header

    @property
    def array(self) -> ArrayLike:
        return self._array

    @property
    def size_in_bytes(self) -> int:
        return self._header.HEADER_SIZE + self._array.nbytes

    def __repr__(self) -> str:
        return f"Tensor(header={self.header}, array={self.array})"

    def read(
        self,
        *,
        data: Tuple[TensorHeader, ArrayLike] = None,
        buffer: ReadableBuffer = None,
        reader: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ) -> None:
        if not data and not buffer and not reader:
            raise ValueError("Either data, buffer or reader must be provided")
        if buffer:
            reader = BytesIO(buffer)

        if data:
            header, array = data
        else:
            curr_offset = offset
            header = TensorHeader(buffer=buffer, reader=reader, offset=curr_offset, whence=whence)
            data_size_in_bytes = header.dims[0] * header.strides[0] * header.bytes_per_element
            array_data = reader.read(data_size_in_bytes)

            array = np.ndarray(
                header.dims[: header.rank],
                dtype=header.dtype,
                strides=header.strides[: header.rank],
                buffer=array_data,
            )

        self._header = header
        self._array = array

    def write(
        self,
        *,
        buffer: WriteableBuffer = None,
        writer: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ) -> Any:
        if not buffer and not writer:
            raise ValueError("Either buffer or writer must be provided")
        if buffer:
            writer = BytesIO(buffer)

        writer.seek(offset, whence)
        orig_offset = writer.tell()
        curr_offset = orig_offset
        self._header.serialize(buffer=buffer, writer=writer, offset=curr_offset)
        array_data = self._array.tobytes()
        writer.write(array_data)
        curr_offset = writer.tell()

        return curr_offset - orig_offset


class Entity:
    """Serializer/deserializer for the Entity.

    ```
    struct Entity {
        EntityHeader entity_header  // Entity header
        Component components[];     // Entity components
    };
    ```
    """

    def __init__(
        self,
        *,
        data: Tuple[EntityHeader, List["Component"]] = None,
        buffer: Sequence = None,
        reader: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ):
        self.read(data=data, buffer=buffer, reader=reader, offset=offset, whence=whence)

    @property
    def header(self) -> EntityHeader:
        return self._header

    @property
    def components(self) -> List["Component"]:
        return self._components

    @property
    def size_in_bytes(self) -> int:
        size = self._header.HEADER_SIZE
        for component in self._components:
            size += component.size_in_bytes
        return size

    def __repr__(self) -> str:
        return f"Entity(EntityHeader={self.header}, components={self.components})"

    @staticmethod
    def create(sequence_number: int, array: ArrayLike) -> None:
        entity_header = EntityHeader(data=(0, 0, sequence_number, 0, 1, 0))
        component_header = ComponentHeader(data=(0, 3996102265592038524, 11968035723744066232, 0))
        tensor_header = TensorHeader(
            data=(
                MemoryStorageType.kDevice,
                PrimitiveType.kUnsigned8,
                1,
                3,
                (480, 640, 3),
                (1920, 3, 1),
            )
        )
        tensor = Tensor(data=(tensor_header, array))
        component = Component(data=(component_header, "", tensor))
        entity = Entity(data=(entity_header, [component]))
        return entity

    def read(
        self,
        *,
        data: Tuple[EntityHeader, List["Component"]] = None,
        buffer: ReadableBuffer = None,
        reader: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ) -> None:
        if not data and not buffer and not reader:
            raise ValueError("Either data, buffer or reader must be provided")
        if buffer:
            reader = BytesIO(buffer)

        if data:
            header, components = data
        else:
            curr_offset = offset
            header = EntityHeader(buffer=buffer, reader=reader, offset=curr_offset, whence=whence)
            curr_offset = reader.tell()
            components = []
            for _ in range(header.component_count):
                component = Component(buffer=buffer, reader=reader, offset=curr_offset)
                curr_offset += component.size_in_bytes
                components.append(component)

        self._header = header
        self._components = components

    def write(
        self,
        *,
        buffer: WriteableBuffer = None,
        writer: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ) -> Any:
        if not buffer and not writer:
            raise ValueError("Either buffer or writer must be provided")
        if buffer:
            writer = BytesIO(buffer)

        writer.seek(offset, whence)
        orig_offset = writer.tell()
        curr_offset = orig_offset
        self._header.serialize(buffer=buffer, writer=writer, offset=curr_offset)
        curr_offset = writer.tell()

        for component in self._components:
            component.write(buffer=buffer, writer=writer, offset=curr_offset)
            curr_offset += component.size_in_bytes

        return curr_offset - orig_offset


class Component:
    """Serializer/deserializer for the Component.

    ```
    struct Component {
        ComponentHeader header  // Component header
        uint8_t* name;          // Component name
        uint8_t* data;          // Component data
    };
    ```
    """

    def __init__(
        self,
        *,
        data: Tuple[ComponentHeader, str, Tensor] = None,
        buffer: Sequence = None,
        reader: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ):
        self.read(data=data, buffer=buffer, reader=reader, offset=offset, whence=whence)

    @property
    def size_in_bytes(self) -> int:
        return self._header.HEADER_SIZE + self._header.name_size + self._tensor.size_in_bytes

    @property
    def header(self) -> ComponentHeader:
        return self._header

    @property
    def name(self) -> str:
        return self._name

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    def __repr__(self) -> str:
        return f"Component(ComponentHeader={self.header}, name={self.name}, tensor={self.tensor})"

    def read(
        self,
        *,
        data: Tuple[ComponentHeader, str, Tensor] = None,
        buffer: ReadableBuffer = None,
        reader: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ) -> None:
        if not data and not buffer and not reader:
            raise ValueError("Either data, buffer or reader must be provided")
        if buffer:
            reader = BytesIO(buffer)

        if data:
            header, name, tensor = data
            header._name_size = len(name)
            assert header.name_size == len(name)
        else:
            curr_offset = offset
            header = ComponentHeader(
                buffer=buffer, reader=reader, offset=curr_offset, whence=whence
            )
            curr_offset = reader.tell()
            name = str(reader.read(header.name_size), "utf-8")
            curr_offset += header.name_size
            tensor = Tensor(buffer=buffer, reader=reader, offset=curr_offset)

        self._header = header
        self._name = name
        self._tensor = tensor

    def write(
        self,
        *,
        buffer: WriteableBuffer = None,
        writer: BufferedIOBase = None,
        offset: int = 0,
        whence: int = os.SEEK_SET,
    ) -> Any:
        if not buffer and not writer:
            raise ValueError("Either buffer or writer must be provided")
        if buffer:
            writer = BytesIO(buffer)

        writer.seek(offset, whence)
        orig_offset = writer.tell()
        curr_offset = orig_offset
        self._header.serialize(buffer=buffer, writer=writer, offset=curr_offset)
        curr_offset = writer.tell()
        writer.write(struct.pack(f"={len(self._name)}s", self._name.encode("utf-8")))
        curr_offset = writer.tell()
        self._tensor.write(buffer=buffer, writer=writer, offset=curr_offset)
        curr_offset = writer.tell()

        return curr_offset - orig_offset


def get_file_size(path_or_reader: Union[os.PathLike, BufferedIOBase]) -> int:
    if isinstance(path_or_reader, os.PathLike):
        return os.stat(path_or_reader).st_size

    old_pos = path_or_reader.tell()
    file_size = path_or_reader.seek(0, os.SEEK_END)
    path_or_reader.seek(old_pos)
    return file_size


class EntityReader:
    """Read from the GXF recording format that EntityReplayer is using."""

    def __init__(self, directory: os.PathLike = "./", basename: str = "tensor") -> None:
        """Initialize the reader.

        Args:
            directory: Directory to read the recording from.
            basename: Base name of the recording.
        """
        self._directory = directory
        self._basename = basename
        self._index_path = os.path.join(self._directory, f"{self._basename}.gxf_index")
        self._entities_path = os.path.join(self._directory, f"{self._basename}.gxf_entities")
        self._index_file = None
        self._entities_file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()

    def open(self) -> None:  # noqa: A003
        """Open the recording."""
        self._index_file = open(self._index_path, "rb")  # noqa: SIM115
        self._entities_file = open(self._entities_path, "rb")  # noqa: SIM115

    def close(self) -> None:
        """Close the recording."""
        self._index_file.close()
        self._entities_file.close()

    def get_entity_index(self, index: int) -> EntityIndex:
        """Get an entity index from the recording.

        Args:
            index: Index of the entity index to get.

        Returns:
            The entity index.
        """
        offset = index * EntityIndex.HEADER_SIZE
        return EntityIndex(reader=self._index_file, offset=offset)

    def get_entity(self, index: int) -> Entity:
        """Get an entity from the recording.

        Args:
            index: Index of the entity to get.

        Returns:
            The entity.
        """
        offset = index * EntityIndex.HEADER_SIZE
        entity_index = EntityIndex(reader=self._index_file, offset=offset)
        return Entity(reader=self._entities_file, offset=entity_index.data_offset)

    @property
    def num_entities(self) -> int:
        """Get the number of entities in the recording.

        Returns:
            The number of entities.
        """
        return get_file_size(self._index_file) // EntityIndex.HEADER_SIZE

    def get_entities(self) -> Generator[Entity, None, None]:
        """Get all entities from the recording.

        Returns:
            An iterator over all entities.
        """
        for index in range(self.num_entities):
            yield self.get_entity(index)

    def get_framerate(self) -> float:
        """Get the framerate of the recording.

        It is a guess based on the first and last timestamp in the EntityIndex file.

        Returns:
            The framerate.
        """

        num_entities = self.num_entities

        if num_entities < 2:
            raise ValueError("Not enough entities to guess framerate")

        first_entity_index = self.get_entity_index(0)
        first_timestamp = first_entity_index.log_time
        last_entity_index = self.get_entity_index(num_entities - 1)
        last_timestamp = last_entity_index.log_time
        duration = last_timestamp - first_timestamp
        return num_entities * 10**9 / duration

    def get_frame(self, index: int) -> np.ndarray:
        """Get a frame from the recording.

        Args:
            index: Index of the frame to get.

        Returns:
            The frame in the form of a numpy array.
        """
        entity = self.get_entity(index)
        return entity.components[0].tensor.array


class EntityWriter:
    """Write to the GXF recording format that EntityRecorder is using."""

    def __init__(
        self,
        directory: os.PathLike = "./",
        basename: str = "tensor",
        *,
        framerate: Union[int, float] = 30,
    ) -> None:
        """Initialize the writer.

        Args:
            directory: Directory to write the recording to.
            basename: Base name of the recording.
            framerate: Framerate of the recording.
        """
        self._directory = directory
        self._basename = basename
        self._framerate = framerate
        self._index_path = os.path.join(self._directory, f"{self._basename}.gxf_index")
        self._entities_path = os.path.join(self._directory, f"{self._basename}.gxf_entities")
        self._index_file = None
        self._entities_file = None
        self._initialize()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()

    def _initialize(self):
        self._index = 0
        self._start_timestemp = int(time.time() * 10**9)

    def open(self):  # noqa: A003
        self.close()
        self._index_file = open(self._index_path, "wb")  # noqa: SIM115
        self._entities_file = open(self._entities_path, "wb")  # noqa: SIM115
        self._initialize()

    def close(self):
        if self._index_file:
            self._index_file.close()
            self._index_file = None
        if self._entities_file:
            self._entities_file.close()
            self._entities_file = None

    def add(self, array: ArrayLike, *, name: str = "") -> Entity:
        """Add a new entity to the writer.

        Args:
            array: The array to be added to the writer.
            name: The name of the entity.

        Returns:
            The entity that was added.
        """
        if not self._index_file:
            raise ValueError("Recorder is not open")
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be an array-like type")
        if array.dtype != np.uint8:
            raise TypeError("array must be a uint8 array")
        if array.ndim != 3:
            raise TypeError("array must be a RGB array(height, width, channel)")
        if array.shape[0] <= 0:
            raise TypeError("array must have more than 0 rows (height)")
        if array.shape[1] <= 0:
            raise TypeError("array must have more than 0 rows (width)")
        if array.shape[2] not in [1, 3]:
            raise TypeError("Array must have 1 or 3 channels, but got {}", array.shape[2])
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        entity_header = EntityHeader(data=(0, 0, self._index, 0, 1, 0))
        component_header = ComponentHeader(data=(0, *TensorType, 0))
        tensor_header = TensorHeader(
            data=(
                MemoryStorageType.kDevice,
                PrimitiveType.kUnsigned8,
                array.dtype.itemsize,
                array.ndim,
                array.shape,
                array.strides,
            )
        )
        tensor = Tensor(data=(tensor_header, array))
        component = Component(data=(component_header, name, tensor))
        entity = Entity(data=(entity_header, [component]))

        timestamp = self._start_timestemp + int(self._index * 10**9 / self._framerate)
        offset = self._entities_file.tell()
        entity_index = EntityIndex(data=(timestamp, entity.size_in_bytes, offset))

        entity_index.write(writer=self._index_file, whence=os.SEEK_CUR)
        entity.write(writer=self._entities_file, whence=os.SEEK_CUR)

        self._index += 1

        return entity
