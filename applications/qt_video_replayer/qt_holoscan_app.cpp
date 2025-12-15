/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "qt_holoscan_app.hpp"

#include <holoscan/logger/logger.hpp>

void QtHoloscanApp::set(const QString& op_name, const QString& param_name, const QVariant& value) {
  // get the parameter
  auto param_wrapper = findParam(op_name.toStdString(), param_name.toStdString());
  if (!param_wrapper) {
    return;
  }

  switch (param_wrapper->arg_type().element_type()) {
    case holoscan::ArgElementType::kBoolean: {
      auto arg = holoscan::Arg(param_name.toStdString(), value.toBool());
      holoscan::ArgumentSetter::set_param(*param_wrapper, arg);
      break;
    }
    case holoscan::ArgElementType::kFloat32: {
      auto arg = holoscan::Arg(param_name.toStdString(), value.toFloat());
      holoscan::ArgumentSetter::set_param(*param_wrapper, arg);
      break;
    }
    case holoscan::ArgElementType::kFloat64: {
      auto arg = holoscan::Arg(param_name.toStdString(), value.toDouble());
      holoscan::ArgumentSetter::set_param(*param_wrapper, arg);
      break;
    }
    case holoscan::ArgElementType::kInt8:
    case holoscan::ArgElementType::kInt16:
    case holoscan::ArgElementType::kInt32: {
      auto arg = holoscan::Arg(param_name.toStdString(), value.toInt());
      holoscan::ArgumentSetter::set_param(*param_wrapper, arg);
      break;
    }
    case holoscan::ArgElementType::kUnsigned8:
    case holoscan::ArgElementType::kUnsigned16:
    case holoscan::ArgElementType::kUnsigned32: {
      auto arg = holoscan::Arg(param_name.toStdString(), value.toUInt());
      holoscan::ArgumentSetter::set_param(*param_wrapper, arg);
      break;
    }
    case holoscan::ArgElementType::kInt64: {
      auto arg = holoscan::Arg(param_name.toStdString(), value.toLongLong());
      holoscan::ArgumentSetter::set_param(*param_wrapper, arg);
      break;
    }
    case holoscan::ArgElementType::kUnsigned64: {
      auto arg = holoscan::Arg(param_name.toStdString(), value.toULongLong());
      holoscan::ArgumentSetter::set_param(*param_wrapper, arg);
      break;
    }
    case holoscan::ArgElementType::kString: {
      auto arg = holoscan::Arg(param_name.toStdString(), value.toString().toStdString());
      holoscan::ArgumentSetter::set_param(*param_wrapper, arg);
      break;
    }
    default:
      HOLOSCAN_LOG_ERROR("Operator '{}' parameter `{}` has unhandled type",
                         op_name.toStdString(),
                         param_name.toStdString());
      break;
  }
}

QVariant QtHoloscanApp::get(const QString& op_name, const QString& param_name) {
  // get the parameter
  auto param_wrapper = findParam(op_name.toStdString(), param_name.toStdString());
  if (!param_wrapper) {
    return QVariant();
  }

  switch (param_wrapper->arg_type().element_type()) {
    case holoscan::ArgElementType::kBoolean:
      return QVariant(std::any_cast<holoscan::Parameter<bool>*>(param_wrapper->value())->get());
    case holoscan::ArgElementType::kFloat32:
      return QVariant(std::any_cast<holoscan::Parameter<float>*>(param_wrapper->value())->get());
    case holoscan::ArgElementType::kFloat64:
      return QVariant(std::any_cast<holoscan::Parameter<double>*>(param_wrapper->value())->get());
    case holoscan::ArgElementType::kInt8:
      return QVariant(std::any_cast<holoscan::Parameter<int8_t>*>(param_wrapper->value())->get());
    case holoscan::ArgElementType::kInt16:
      return QVariant(std::any_cast<holoscan::Parameter<int16_t>*>(param_wrapper->value())->get());
    case holoscan::ArgElementType::kInt32:
      return QVariant(std::any_cast<holoscan::Parameter<int32_t>*>(param_wrapper->value())->get());
    case holoscan::ArgElementType::kInt64:
      return QVariant(
          std::any_cast<holoscan::Parameter<qlonglong>*>(param_wrapper->value())->get());
    case holoscan::ArgElementType::kUnsigned8:
      return QVariant(std::any_cast<holoscan::Parameter<uint8_t>*>(param_wrapper->value())->get());
    case holoscan::ArgElementType::kUnsigned16:
      return QVariant(std::any_cast<holoscan::Parameter<uint16_t>*>(param_wrapper->value())->get());
    case holoscan::ArgElementType::kUnsigned32:
      return QVariant(std::any_cast<holoscan::Parameter<uint32_t>*>(param_wrapper->value())->get());
    case holoscan::ArgElementType::kUnsigned64:
      return QVariant(
          std::any_cast<holoscan::Parameter<qulonglong>*>(param_wrapper->value())->get());
    case holoscan::ArgElementType::kString:
      return QVariant(QString::fromStdString(
          std::any_cast<holoscan::Parameter<std::string>*>(param_wrapper->value())->get()));
    default:
      HOLOSCAN_LOG_ERROR("Operator '{}' parameter `{}` has unhandled type",
                         op_name.toStdString(),
                         param_name.toStdString());
      break;
  }
  return QVariant();
}

holoscan::ParameterWrapper* QtHoloscanApp::findParam(const std::string& op_name,
                                                     const std::string& param_name) {
  // get the operator
  holoscan::Operator* op = nullptr;
  auto& app_graph = graph();
  if (!app_graph.is_empty()) {
    op = app_graph.find_node(op_name).get();
  }
  if (!op) {
    HOLOSCAN_LOG_ERROR("Operator '{}' is not defined in app {}", op_name, name());
    return nullptr;
  }

  // get the parameter
  auto& params = op->spec()->params();
  auto it = params.find(param_name);
  if (it == params.end()) {
    HOLOSCAN_LOG_ERROR("Parameter '{}' is not defined by operator {}", param_name, op_name);
    return nullptr;
  }

  return &it->second;
}
