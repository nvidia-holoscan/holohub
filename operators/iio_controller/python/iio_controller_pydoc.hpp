// Copyright 2025 Analog Devices, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "macros.hpp"

namespace holoscan::doc {
namespace IIOAttributeRead {
PYDOC(IIOAttributeRead_python, R"doc(
      Operator for reading attributes from IIO capable devices.

      **==Named Outputs==**

      value : string
      	Contains the value of the attribute read

      Parameters
      ----------
      fragment : Fragment
      	The fragment that the operator belongs to.
      ctx : string, mandatory
      	The URI for the IIO context.
      attr_name : string, mandatory
      	The attribute that needs to be read, regardless of the type (Context, Device, Channel)
      dev : string, optional
      	The name of the device from where to read the attribute. If this is not set, the attribute
      	will be considered a context attribute.
      chan : string, optional
      	The name of the channel from which to read the attribute. If this parameter is used, the
      	"dev" and "channel_is_output" parameters should also be set.
      channel_is_output : bool, optional
      	Boolean value that represents weather the channel is input or output, necessary when working
      	with channel attributes because 2 channels can have the same name if they are of different
      	type (1 output and 1 input)
      name : str, optional
      	The name of the operator.
)doc");

PYDOC(initialize, R"doc(
      Initialize the operator

      This method is called only once when the operator is created for the first time,
      and searches for the iio structures based on the names provided as arguments.
)doc");
}  // namespace IIOAttributeRead

namespace IIOAttributeWrite {
PYDOC(IIOAttributeWrite_python, R"doc(
      Operator for writing attributes to IIO capable devices.

      **==Named Inputs==**

      value : string
      	Contains the value of the attribute to be written

      Parameters
      ----------
      fragment : Fragment
      	The fragment that the operator belongs to.
      ctx : string, mandatory
      	The URI for the IIO context.
      attr_name : string, mandatory
      	The attribute that needs to be written, regardless of the type (Context, Device, Channel)
      dev : string, optional
      	The name of the device where to write the attribute. If this is not set, the attribute
      	will be considered a context attribute.
      chan : string, optional
      	The name of the channel where to write the attribute. If this parameter is used, the
      	"dev" and "channel_is_output" parameters should also be set.
      channel_is_output : bool, optional
      	Boolean value that represents weather the channel is input or output, necessary when working
      	with channel attributes because 2 channels can have the same name if they are of different
      	type (1 output and 1 input)
      name : str, optional
      	The name of the operator.
)doc");

PYDOC(initialize, R"doc(
      Initialize the operator

      This method is called only once when the operator is created for the first time,
      and searches for the iio structures based on the names provided as arguments.
)doc");
}  // namespace IIOAttributeWrite

namespace IIOBufferWrite {
PYDOC(IIOBufferWrite_python, R"doc(
      Operator for writing IIO buffers to IIO capable devices.

      **==Named Inputs==**

      buffer : iio_buffer_info_t
      	Contains samples_count (size_t) and the buffer (void*) to be written to the device.

      Parameters
      ----------
      fragment : Fragment
      	The fragment that the operator belongs to.
      ctx : string, mandatory
      	The URI for the IIO context.
      dev : string, mandatory
      	The name of the device where to write the buffer.
      is_cyclic : bool, mandatory
      	Boolean value that represents whether the buffer is cyclic or not.
      enabled_channel_names : list of strings, mandatory
      	List of names of channels that are enabled. In order to create a buffer, at
	least one channel must be enabled.
      enabled_channel_output : list of bools, mandatory
      	List of booleans that represent whether each channel is output or not. Should
	be the same size as enabled_channel_names.
      name : str, optional
	The name of the operator.
)doc");

PYDOC(initialize, R"doc(
      Initialize the operator

      This method is called only once when the operator is created for the first time,
      and searches for the iio structures based on the names provided as arguments. It
      also enables all provided channels.
)doc");
}  // namespace IIOBufferWrite

namespace IIOBufferRead {
PYDOC(IIOBufferRead_python, R"doc(
      Operator for reading IIO buffers from IIO capable devices.

      **==Named Outputs==**

      buffer : iio_buffer_info_t
      	Contains samples_count (size_t) and the buffer (void*) read from the device.

      Parameters
      ----------
      fragment : Fragment
      	The fragment that the operator belongs to.
      ctx : string, mandatory
      	The URI for the IIO context.
      dev : string, mandatory
      	The name of the device from where to read the buffer.
      samples_count : size_t, mandatory
      	The number of samples to read from the device.
      is_cyclic : bool, mandatory
      	Boolean value that represents whether the buffer is cyclic or not.
      enabled_channel_names : list of strings, mandatory
      	List of names of channels that are enabled. In order to create a buffer, at
	least one channel must be enabled.
      enabled_channel_output : list of bools, mandatory
      	List of booleans that represent whether each channel is output or not. Should
	be the same size as enabled_channel_names.
      name : str, optional
	The name of the operator.
)doc");

PYDOC(initialize, R"doc(
      Initialize the operator

      This method is called only once when the operator is created for the first time,
      and searches for the iio structures based on the names provided as arguments. It
      also enables all provided channels.
)doc");
}  // namespace IIOBufferRead

namespace IIOConfigurator {
PYDOC(IIOConfigurator_python, R"doc(
      Operator for configuring IIO devices based on a provided path of a YAML file.

      Parameters
      ----------

      cfg : string
      	Contains the configuration file path to be applied to the device.
)doc");
}  // namespace IIOConfigurator
}  // namespace holoscan::doc
