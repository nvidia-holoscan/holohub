# Adding a GUI to Holoscan Python Applications

When developing Holoscan applications, incorporating a graphical user interface (GUI) can enhance usability and allow modification of the application's behavior at runtime.

This tutorial demonstrates how GUI controls were integrated into the [Florence-2](https://github.com/nvidia-holoscan/holohub/tree/main/applications/florence-2-vision) Python application using [PySide6](https://doc.qt.io/qtforpython-6/). This addition enables users to dynamically change the vision task performed by the application.

<p align="center">
  <img src="./demo.gif" alt="Holoscan VILA Live">
</p>

## Table of Contents

1. [Overview](#overview)
    1. [Dockerfile](#dockerfile)
    1. [Application Code](#application-code)
    1. [GUI Code](#gui-code)
2. [Creating the GUI Widgets and Layout](#creating-the-gui-widgets-and-layout)
3. [Starting the Holoscan Application Thread](#starting-the-holoscan-application-thread)
4. [Adding a GUI to Your Own Application](#adding-a-gui-to-your-own-application)


## Overview

The Florence-2 application includes the typical components of a Holohub application, with the addition of a GUI component. 
The main components are:

- <b>Dockerfile</b>: For installing additional dependencies.
- <b>Application Code</b>: Defines the Holoscan application and its operators.
- <b>GUI Code</b>: Utilizes PySide6 to add UI controls.


### Dockerfile

The Dockerfile is used in Holohub when the application requires additional dependencies. For GUI functionality, this application's 
[Dockerfile](https://github.com/nvidia-holoscan/holohub/blob/main/applications/florence-2-vision/Dockerfile) installs:

* `qt6-base-dev` for Qt6 framework
* `PySide6` for Python bindings for Qt6 (as specified in [requirements.txt](https://github.com/nvidia-holoscan/holohub/blob/main/applications/florence-2-vision/requirements.txt))

### Application Code

The Florence-2 application code is organized across several files:

* `florence2_app.py`: Main application code.
* `florence2_op.py`: Florence-2 model inference code.
* `florence2_postprocessor_op.py`: Post-processing code to send overlays (e.g., bounding boxes, labels, segmentation masks) to Holoviz.
* `config.yaml`: Default application parameters.

The Florence-2 application can be run independently of the GUI code. E.g., the application can be run with `python application/florence-2-vision/florence2_app.py` inside the Florence-2 Docker container. This will run the application without the GUI controls.  The only code needed for GUI integration in the application code is the `set_parameters()` method in the `FlorenceApp` class. This method updates two fields in the Florence-2 operator:

```python
class FlorenceApp(Application):
    def set_parameters(self, task, prompt):
        """Set parameters for the Florence2Operator."""
        if self.florence_op:
            self.florence_op.task = task
            self.florence_op.prompt = prompt
```

These updated parameters are passed to the model during the next `compute()` method execution of the Florence-2 operator.

### GUI Code

The GUI code resides in [qt_app.py](https://github.com/nvidia-holoscan/holohub/blob/main/applications/florence-2-vision/qt_app.py).
The code in this file defines a class for the main window which calls `setupUi()` and `runHoloscanApp()` when the instance is initialized.

```python
class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()  # Setup the UI
        self.runHoloscanApp()  # Run the Holoscan application
```

At a high level, this is all we need to launch a Python Holoscan application with a GUI.
The `setupUi()` method defines the GUI widgets and layout, while `runHoloscanApp()` runs the Florence-2 application in a separate thread within the process.
Details of these methods are explored in the following sections.

## Creating the GUI Widgets and Layout

The `setupUi()` method creates the GUI with a few simple widgets using PySide6 APIs. 
For those unfamiliar with PySide6, this [tutorial](https://www.pythonguis.com/pyside6-tutorial/) provides an introduction.

```python
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
```

This code creates the following widgets:

* <b>Drop-down Menu</b>: Lists the vision tasks supported by Florence-2.
* <b>Text input Widget</b>: Allows text input for tasks such as Open Vocabulary Detection.
* <b>Submit Button</b>: Triggers the `on_submit()` method when clicked.

When the application is running, the user selects a vision task, enters text (if 
needed), and clicks "Submit" to change the task performed by the model.
The `on_submit()` method is then invoked, calling the `set_parameters()` method 
in the `FlorenceApp` class to update the operator's parameters.

```python
    def on_submit(self):
        """Handle the submit button click event."""
        selected_option = self.dropdown.currentText()
        entered_text = self.text_input.text()

        # Set parameters in the Holoscan application
        global gApp
        if gApp:
            gApp.set_parameters(selected_option, entered_text)
```

## Starting the Holoscan Application Thread

The `runHoloscanApp()` method starts the Florence-2 application by creating an instance of `FlorenceWorker` 
and running it in a thread.

```python
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
```

When the thread is started, it calls the `FlorenceWorker` class's `run()` method
which creates and runs the Holoscan application.

```python
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
```

This covers the essential steps for creating a GUI to control your Python Holoscan applications. 
To try out the application, follow the instructions provided [here](https://github.com/nvidia-holoscan/holohub/tree/main/applications/florence-2-vision#-build-and-run-instructions).

## Adding a GUI to Your Own Application

To integrate a GUI into your Python application using PySide6, follow these steps:

1. Ensure Qt and PySide6 dependencies are included in your Dockerfile. Verify that Qt and PySide6 package licenses meet your project requirements.
2. Copy the `qt_app.py` file to your application directory.  Rename and modify the `FlorenceWorker` class to create an instance of your application.
Update the import statement `from florence2_app import FlorenceApp` as necessary.
3. Customize the `setupUi()` method to include the controls relevant to your application.
4. Update `set_parameters()` methoed to reflect the parameters your application needs to update.

