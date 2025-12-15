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

#ifndef APPLICATIONS_QT_QT_HOLOSCAN_APP
#define APPLICATIONS_QT_QT_HOLOSCAN_APP

#include <type_traits>

#include <holoscan/core/application.hpp>

#include <QObject>
#include <QVariant>

/**
 * @brief Qt Holoscan app base class
 *
 * This class is used to
 * - expose Holoscan operator parameters as Qt properties to be used in QML
 * - set and get Holoscan operator parameters from QML (Note this does not provide the automatic
 *   event/signal handling of Qt properties)
 *
 * To expose a Holoscan operator parameter as a Qt property define a list of parameters as a macro
 *
 * @code{.cpp}
 * #define HOLOSCAN_PARAMETERS(F)  \
 *   F("first_operator", first_parameter, bool) \
 *   F("first_operator", second_parameter, float) \
 *   F("second_operator", first_parameter, float)
 * @endcode
 *
 * The first parameter is the name of the operator as a string, the second parameter is the name
 * of the operator parameter (this is not a string), and the last parameter is the type of the
 * operator parameter (for std::string parameters use QString).
 *
 * Note that there is no synchronization between the thread which executes the Holoscan application
 * and the Qt GUI thread.
 *
 * Then in your class place the HOLOSCAN_PROPERTIES_DEF() macro to define the Qt properties and the
 * HOLOSCAN_PROPERTIES_INIT() macro to initialize the Qt properties with the defaults.
 *
 * @code{.cpp}
 * class MyApp : public QtHoloscanApp {
 * private:
 *   // needs to be a Q_OBJECT so that Qt can use the properties
 *   Q_OBJECT
 *   HOLOSCAN_PROPERTIES_DEF(HOLOSCAN_PARAMETERS)
 * public:
 *   void compose() override {
 *     // create the operators
 *     ...
 *     // This initializes the Qt properties with the defaults of the Holoscan operator parameters
 *     HOLOSCAN_PROPERTIES_INIT(HOLOSCAN_PARAMETERS);
 *   }
 * };
 * @endcode
 */
class QtHoloscanApp : public QObject, public holoscan::Application {
  Q_OBJECT

 public:
  /**
   * @brief Construct a new Qt Holoscan App object
   *
   * @param parent parent object
   */
  explicit QtHoloscanApp(QObject* parent = nullptr) : QObject(parent) {}

  /**
   * @brief Set a parameter of an Holoscan operator
   *
   * Note this does not provide the automatic event/signal handling of Qt properties.
   *
   * @param op_name operator name
   * @param param_name parameter name
   * @param value value to set
   */
  Q_INVOKABLE void set(const QString& op_name, const QString& param_name, const QVariant& value);

  /**
   * @brief Get a parameter of an Holoscan operator
   *
   * @param op_name operator name
   * @param param_name parameter name
   *
   * @returns parameter value
   */
  Q_INVOKABLE QVariant get(const QString& op_name, const QString& param_name);

 protected:
  /**
   * @brief Find a parameter of an operator
   *
   * @param op_name operator name
   * @param param_name parameter name
   * @return holoscan::ParameterWrapper* parameter wrapper
   */
  holoscan::ParameterWrapper* findParam(const std::string& op_name, const std::string& param_name);

  /**
   * @brief Template function to set a parameter of an Holoscan operator
   *
   * @tparam T Type of parameter
   * @param op_name operator name
   * @param param_name parameter name
   * @param value value to set
   */
  template <typename T>
  void setParam(const std::string& op_name, const std::string& param_name, T const& value) {
    // get the parameter
    auto param_wrapper = findParam(op_name, param_name);
    if (!param_wrapper) {
      return;
    }

    // set the parameter
    if constexpr (std::is_same_v<T, QString>) {
      // special handling of std::string to QString conversion
      auto arg = holoscan::Arg(param_name, value.toStdString());
      holoscan::ArgumentSetter::set_param(*param_wrapper, arg);
    } else {
      auto arg = holoscan::Arg(param_name, value);
      holoscan::ArgumentSetter::set_param(*param_wrapper, arg);
    }
  }

  /**
   * @brief Template function to get a parameter of an Holoscan operator
   *
   * @tparam T Type of parameter
   * @param op_name operator name
   * @param param_name parameter name
   * @return T parameter value
   */
  template <typename T>
  T getParam(const std::string& op_name, const std::string& param_name) {
    // get the parameter
    auto param_wrapper = findParam(op_name, param_name);
    if (!param_wrapper) {
      return T();
    }

    if constexpr (std::is_same_v<T, QString>) {
      // special handling of std::string to QString conversion
      auto& p = *std::any_cast<holoscan::Parameter<std::string>*>(param_wrapper->value());
      return QString::fromStdString(p.has_value() ? p.get() : p.default_value());
    } else {
      auto& p = *std::any_cast<holoscan::Parameter<T>*>(param_wrapper->value());
      return p.has_value() ? p.get() : p.default_value();
    }
  }
};

/**
 * @brief Define a parameter if an Holoscan operator as a Qt property
 *
 * @param OP_NAME operator name (string)
 * @param PARAM_NAME parameter/property name
 * @param TYPE parameter/property type
 */
#define HOLOSCAN_PROPERTY_DEF(OP_NAME, PARAM_NAME, TYPE)                                       \
 private:                                                                                      \
  Q_PROPERTY(TYPE PARAM_NAME READ PARAM_NAME WRITE set##PARAM_NAME NOTIFY PARAM_NAME##Changed) \
 public:                                                                                       \
  void set##PARAM_NAME(const TYPE& new_value) {                                                \
    if (PARAM_NAME##_ == new_value) {                                                \
      return;                                                                          \
    }                                                \
                                                                                               \
    PARAM_NAME##_ = new_value;                                                                 \
                                                                                               \
    setParam(OP_NAME, #PARAM_NAME, new_value);                                                 \
                                                                                               \
    emit PARAM_NAME##Changed();                                                                \
  }                                                                                            \
  TYPE PARAM_NAME() const {                                                                    \
    return PARAM_NAME##_;                                                                      \
  }                                                                                            \
 Q_SIGNALS: /* NOLINT */                                                                       \
  void PARAM_NAME##Changed();                                                                  \
                                                                                               \
 private:                                                                                      \
  TYPE PARAM_NAME##_;

/**
 * @brief Define the parameters exposed as Qt properties
 *
 * @param PROPERTIES name of macros with list of HOLOSCAN_PROPERTY_DEF()
 */
#define HOLOSCAN_PROPERTIES_DEF(PROPERTIES) PROPERTIES(HOLOSCAN_PROPERTY_DEF)

/**
 * @brief Init a Qt property with the Holoscan parameter defaults.
 *
 * @param OP_NAME operator name (string)
 * @param PARAM_NAME parameter/property name
 * @param TYPE parameter/property type
 */
#define HOLOSCAN_PROPERTY_INIT(OP_NAME, PARAM_NAME, TYPE)            \
  {                                                                  \
    const TYPE default_value = getParam<TYPE>(OP_NAME, #PARAM_NAME); \
    if (PARAM_NAME##_ != default_value) {                            \
      PARAM_NAME##_ = default_value;                                 \
      emit PARAM_NAME##Changed();                                    \
    }                                                                \
  }

/**
 * @brief Initialize the Qt properties with the Holoscan parameter defaults.
 *
 * @param PROPERTIES name of macros with list of HOLOSCAN_PROPERTY_DEF()
 */
#define HOLOSCAN_PROPERTIES_INIT(PROPERTIES) PROPERTIES(HOLOSCAN_PROPERTY_INIT);

#endif /* APPLICATIONS_QT_QT_HOLOSCAN_APP */
