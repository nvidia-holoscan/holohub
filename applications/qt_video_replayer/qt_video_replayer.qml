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

import QtQuick
import QtHoloscanVideo
import QtQuick.Controls
import QtQuick.Layouts

Item {

    width: 840
    height: 480

    QtHoloscanVideo {
        objectName: "video"
    }

    Rectangle {
        color: Qt.rgba(1, 1, 1, 0.7)
        radius: 10
        border.width: 1
        border.color: "white"
        anchors.fill: controls
        anchors.margins: -10
    }

    ColumnLayout {
        id: controls
        anchors.right: parent.right
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.margins: 20

        RowLayout {
            Text {
                text: "Filter"
                Layout.alignment: Layout.Center
            }
            ComboBox {
                id: filter
                model: ["SobelHoriz", "SobelVert", "Gauss"]
                Component.onCompleted: currentIndex = indexOfValue("SobelHoriz")
                onActivated: {
                    holoscanApp.filter = currentValue
                }
            }
        }
        RowLayout {
            visible: (filter.currentValue == "Gauss")
            Text {
                text: "Gaussian Blur Radius [px]"
                Layout.alignment: Layout.Center
            }
            Slider {
                id: maskSlider
                enabled: (filter.currentValue == "Gauss")

                property var maskSizeTable: [3, 5, 7, 9, 11, 13];
                readonly property int maskSize: (() => maskSizeTable[value])();

                from: 0; to: maskSizeTable.length - 1; stepSize: 1

                onValueChanged: {
                    holoscanApp.mask_size = maskSize
                }
            }
            Text {
                text: maskSlider.maskSize
                Layout.alignment: Layout.Center
            }
        }

        CheckBox {
            id: realtime
            text: "Use Video Framerate"
            checked: holoscanApp.realtime
            onCheckedChanged: {
                holoscanApp.realtime = checked;
            }
        }
    }
}
