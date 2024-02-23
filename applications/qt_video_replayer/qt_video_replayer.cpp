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

#include <algorithm>
#include <thread>

#include <holoscan/core/resources/gxf/unbounded_allocator.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

#include "npp_filter.hpp"

#include <QCommandLineParser>
#include <QGuiApplication>
#include <QQmlContext>
#include <QQmlEngine>
#include <QtQuick/QQuickView>

#include "qt_holoscan_app.hpp"
#include "qt_holoscan_video.hpp"
#include "qt_video_op.hpp"

#include <npp.h>

/**
 * @brief Expose Holoscan operator parameters as Qt properties.
 *
 * For each operator property add a line with the operator name as a string, the property name and
 * the property type.
 */
#define HOLOSCAN_PARAMETERS(F)  \
  F("replayer", realtime, bool) \
  F("filter", filter, QString)  \
  F("filter", mask_size, uint32_t)

/**
 * @brief Holoscan application displaying a video in a Qt QML window
 *
 * Inherits from QtHoloscanApp to be able to expose Holoscan operator parameters as Qt properties.
 * If you don't want to expose properties inherit from holoscan::Application as usual.
 */
class QtVideoApp : public QtHoloscanApp {
 private:
  /// Needs to be a Q_OBJECT so that Qt can use the properties
  Q_OBJECT
  /// Define the parameters exposed as Qt properties
  HOLOSCAN_PROPERTIES_DEF(HOLOSCAN_PARAMETERS)

 public:
  /**
   * @brief Construct a new Qt Video App object
   *
   * @param view
   * @param parent
   */
  explicit QtVideoApp(QQuickView* view, int count, std::string datadir,
                      std::string basename, QObject* parent = nullptr)
      : QtHoloscanApp(parent), view_(view), count_(count), datadir_(datadir),
                      basename_(basename) {
    // Install the event filter. This is used to check for the close event and stop the
    // Holoscan pipeline in that case.
    view_->installEventFilter(this);
    // The Holoscan application is embedded as a context property so that QML can access the
    // operator parameters we expose as Qt properties. Note that this has to be called before
    // `setSource()` below.
    view_->engine()->rootContext()->setContextProperty("holoscanApp", this);
  }

  /**
   * @brief Build the graph
   *
   */
  void compose() override {
    // Create resources
    const auto allocator = make_resource<holoscan::UnboundedAllocator>("allocator");
    const auto cuda_stream_pool =
        make_resource<holoscan::CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    // Create the operators
    const auto replayer = make_operator<holoscan::ops::VideoStreamReplayerOp>(
        "replayer",
        holoscan::Arg("directory", datadir_),
        holoscan::Arg("basename", basename_),
        holoscan::Arg("frame_rate", 0.f),
        holoscan::Arg("repeat", true),
        holoscan::Arg("realtime", true),
        holoscan::Arg("count", size_t(count_)));
    const auto converter = make_operator<holoscan::ops::FormatConverterOp>(
        "converter",
        holoscan::Arg("out_dtype", std::string("rgba8888")),
        holoscan::Arg("pool", allocator),
        holoscan::Arg("cuda_stream_pool", cuda_stream_pool));
    const auto filter = make_operator<holoscan::ops::NppFilterOp>(
        "filter",
        holoscan::Arg("filter", std::string("SobelHoriz")),
        holoscan::Arg("allocator", allocator),
        holoscan::Arg("cuda_stream_pool", cuda_stream_pool));

    // Find the Qt Holoscan video object in the QML view, the QtVideoOp need this
    // to pass the video buffer to render to the video object
    QtHoloscanVideo* const qt_holoscan_video =
        static_cast<QtHoloscanVideo*>(view_->rootObject()->findChild<QObject*>("video"));
    if (!qt_holoscan_video) {
      throw std::runtime_error("Could not find 'video' element in Qt scene");
    }

    qtvideo_ = make_operator<holoscan::ops::QtVideoOp>(
        "qtvideo", holoscan::Arg("qt_holoscan_video", qt_holoscan_video));

    // Define the workflow
    add_flow(replayer, converter);
    add_flow(converter, filter);
    add_flow(filter, qtvideo_);

    // This initializes the Qt properties with the defaults of the Holoscan operator parameters
    HOLOSCAN_PROPERTIES_INIT(HOLOSCAN_PARAMETERS);
  }

  /**
   * @brief Check for the window close event and stop the QtVideoOp and with that the pipeline.
   */
  bool eventFilter(QObject* obj, QEvent* event) {
    if (event->type() == QEvent::Close) {
      auto boolean_condition = qtvideo_->condition<holoscan::BooleanCondition>("stop_condition");
      boolean_condition->disable_tick();
    }

    return QObject::eventFilter(obj, event);
  }

 private:
  QQuickView* view_ = nullptr;
  int count_ = -1;
  std::string datadir_ = "../data/racerx";
  std::string basename_ = "racerx";
  std::shared_ptr<holoscan::ops::QtVideoOp> qtvideo_;
};

int main(int argc, char** argv) {
  QGuiApplication qt_app(argc, argv);

  // Command line argument parsing
  QCommandLineParser parser;
  parser.addHelpOption();

  QCommandLineOption disableVsyncOption(
      "disable_vsync", QCoreApplication::translate("main", "Disable vertical sync"));
  parser.addOption(disableVsyncOption);
  QCommandLineOption countOption(QStringList() << "n"
                                               << "count",
                                 "Set number of frames to be played back to <count>",
                                 "count",
                                 "-1");
  parser.addOption(countOption);
  QCommandLineOption dataOption(QStringList() << "d"
                                               << "data",
                                 "Set the directory for the datasets",
                                 "data",
                                 "../data/racerx");
  parser.addOption(dataOption);
  QCommandLineOption basenameOption(QStringList() << "b"
                                               << "basename",
                                 "Basename for the dataset",
                                 "basename",
                                 "racerx");
  parser.addOption(basenameOption);
  parser.process(qt_app);

  if (parser.isSet(disableVsyncOption)) {
    QSurfaceFormat format;
    format.setSwapInterval(0);
    QSurfaceFormat::setDefaultFormat(format);
  }

  // QQuickView is a convenience class of QQuickWindow which automatically loads and displays a QML
  // scene
  QQuickView view;
  // Force OpenGL, needed by the QtHoloscanVideo object
  view.setGraphicsApi(QSGRendererInterface::OpenGL);

  // Create the Holoscan application, this needs to be done before calling `setSource()` below
  // to make the Holoscan operator parameters visible in QML.
  auto holoscan_app =
      holoscan::make_application<QtVideoApp>(&view, parser.value(countOption).toInt(),
                                                    parser.value(dataOption).toStdString(),
                                                    parser.value(basenameOption).toStdString());

  view.setSource(QUrl("qrc:///scenegraph/qt_video_replayer/qt_video_replayer.qml"));
  view.show();

  // Start the Holoscan app in a thread
  std::thread thread([&holoscan_app, &view]() {
    holoscan_app->run();
    // close the view in case the Holoscan app stops before the window closes, happens when passing
    // the number of frames to play back
    view.close();
  });

  // Execute the Qt app
  const int ret = qt_app.exec();

  // Wait for the Holoscan app to finish
  thread.join();

  return ret;
}

#include "qt_video_replayer.moc"
