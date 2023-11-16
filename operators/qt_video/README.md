### Qt Video Operator

The `qt_video` operator is used to display a video in a [QtQuick](https://doc.qt.io/qt-6/qtquick-index.html) application.

For more information on how to use this operator in an application see [Qt video replayer example](../../applications/qt_video_replayer/README.md).

#### `holoscan::ops::QtVideoOp`

Operator class.

##### Parameters

- **`QtHoloscanVideo`**: Instance of QtHoloscanVideo to be used
      - type: `QtHoloscanVideo

##### Inputs

- **`input`**: Input frame data
  - type: `nvidia::gxf::Tensor` or `nvidia::gxf::VideoBuffer`
