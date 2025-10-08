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

import os
import sys

from florence2_app import FlorenceApp  # Import the FlorenceApp class
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Global variable to hold the Holoscan application instance
gApp = None


# Worker class to run the Holoscan application in a separate thread
class FlorenceWorker(QObject):
    finished = Signal()  # Signal to indicate the worker has finished
    progress = Signal(int)  # Signal to indicate progress (if needed)

    def run(self):
        """Run the Holoscan application."""
        config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
        global gApp
        gApp = app = FlorenceApp()
        app.config(config_file)
        app.run()


# Main window class for the PySide2 UI
class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()  # Setup the UI
        self.runHoloscanApp()  # Run the Holoscan application

    def setupUi(self):
        """Setup the UI components."""
        self.setWindowTitle("Florence-2")
        self.resize(400, 150)
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        layout = QVBoxLayout()

        # Create and add dropdown for task selection
        self.dropdown = QComboBox()
        self.dropdown.addItems(
            [
                "Object Detection",
                "Caption",
                "Detailed Caption",
                "More Detailed Caption",
                "Dense Region Caption",
                "Region Proposal",
                "Caption to Phrase Grounding",
                "Referring Expression Segmentation",
                "Open Vocabulary Detection",
                "OCR",
                "OCR with Region",
            ]
        )
        layout.addWidget(QLabel("Select an option:"))
        layout.addWidget(self.dropdown)

        # Create and add text input for prompt
        self.text_input = QLineEdit()
        layout.addWidget(QLabel("Enter text:"))
        layout.addWidget(self.text_input)

        # Create and add submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.on_submit)
        layout.addWidget(self.submit_button)

        self.centralWidget.setLayout(layout)

    def on_submit(self):
        """Handle the submit button click event."""
        selected_option = self.dropdown.currentText()
        entered_text = self.text_input.text()

        # Set parameters in the Holoscan application
        global gApp
        if gApp:
            gApp.set_parameters(selected_option, entered_text)

    def runHoloscanApp(self):
        """Run the Holoscan application in a separate thread."""
        self.thread = QThread()
        self.worker = FlorenceWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Escape:
            self.close()


def main():
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
