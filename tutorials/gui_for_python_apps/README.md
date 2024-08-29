# GUI for Holoscan Python Applications

When developing Holoscan applications you may find it useful to add a graphical user interface (GUI) to your application.
A GUI usually helps make an application more user friendly and allows modifying an application's behavior during runtime.

In this tutorial, we look at how GUI controls were added to the [Florence 2](https://github.com/nvidia-holoscan/holohub/tree/main/applications/florence-2-vision) python application using [PySide6](https://doc.qt.io/qtforpython-6/) to allow the user to change the vision task performed by the application.

<p align="center">
  <img src="./demo.gif" alt="Holoscan VILA Live">
</p>

## Overview

The Florence 2 application is composed of the typical components of a Holohub application, with the addition of
a GUI component.  The components are:

- <b>Dockerfile</b> to install additional dependencies
- <b>Application Code</b> that defines the Holoscan application and it's operators
- <b>GUI Code</b> that uses PySide6 to add UI controls


### Dockerfile

A Dockerfile is used for Holohub applications that require the installation of additional dependencies. For GUI dependencies, 
this application's [Dockerfile](https://github.com/nvidia-holoscan/holohub/blob/main/applications/florence-2-vision/Dockerfile) installs:

* `qt6-base-dev` for the Qt6 framework 
* `PySide6` for the python library bindings to Qt6 (see [requirements.txt](https://github.com/nvidia-holoscan/holohub/blob/main/applications/florence-2-vision/requirements.txt))

### Application Code

The Florence 2 application code is contained in the following files:

* `florence2_app.py`: Contains the main application code
* `florence2_op.py`: Contains the florence 2 model inference code
* `florence2_postprocessor_op.py`: Contains the post-processing code to send overlays such as bounding boxes or labels to Holoviz
* `config.yaml`: Contains the default application parameters

Note that the Florence 2 application is independent of the GUI code and can be run with `python application/florence-2-vision/florence2_app.py` within the 
Florence 2 docker container.  The only logic that has been added to the application code for the purposes of the GUI is
the `set_parameters()` method in the `FlorenceApp` class.  This method is called by the GUI code, and as shown in the snippet below,
 it simply updates two fields in the Florence 2 operator.

```python
class FlorenceApp(Application):
    def set_parameters(self, task, prompt):
        """Set parameters for the Florence2Operator."""
        if self.florence_op:
            self.florence_op.task = task
            self.florence_op.prompt = prompt
```

When the next `compute()` method of the Florence 2 operator is run, it sees the new values and they will be passed to the model as input.


### GUI Code

The GUI code for the Florence 2 application is in [qt_app.py](https://github.com/nvidia-holoscan/holohub/blob/main/applications/florence-2-vision/qt_app.py).
The code in this file defines a class for the main window which calls `setupUi()` and `runHoloscanApp()` when the instance is initialized.

```python
class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()  # Setup the UI
        self.runHoloscanApp()  # Run the Holoscan application
```

At a high level, this is all we need to launch a python Holoscan application with a GUI.
The `setupUi()` method defines the GUI widgets and their layout in the main window while the 
`runHoloscanApp()` method runs the Florence 2 Holoscan Application in a separate thread.

We'll take a closer look at the details of these two methods in the next section.

## Creating the GUI Widgets and Layout

The `setupUi()` method creates a few simple widgets in our GUI window using PySide6 APIs.
If you are not familiar with PySide6, you can take a look at this [tutorial](https://www.pythonguis.com/pyside6-tutorial/).

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

In the code snippet above, we see that it creates the following widgets:

* <b>a drop-down menu</b> populated with the different vision tasks that are supported by Florence 2
* <b>a text input widget</b> which can be used as input for tasks such as the Open Vocabulary Detection
* <b>a submit button</b> which calls the `on_submit()` method when clicked

When the app is running, the user can select the desired vision task from the drop-down menu, enter text if 
needed, and then click on the submit button to change the vision task performed by the model.  Upon clicking
the submit button, the `on_submit()` method is called which in turn call's the FlorenceApp's `set_parameters()`
method to update the operator's parameters.

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

The call to `runHoloscanApp()` starts up the Florence 2 application by creating an instance of `FlorenceWorker` 
and running that in a thread.

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
which creates the Holoscan application and runs the application.

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

This covers the main details of what is needed to create a GUI for controlling your Python Holoscan Application.
To try out the application, follow the instructions [here](https://github.com/nvidia-holoscan/holohub/tree/main/applications/florence-2-vision#-build-and-run-instructions).

## Adding a GUI to Your Own Application

To add a GUI to your own Python application with PySide6, you'll generally need to:

1. Make sure the proper dependencies for Qt and PySide6 are available by adding them into your application's Dockerfile
2. Copy the `qt_app.py` code to your application directory and rename and modify the `FlorenceWorker` class so that it creates an instance of your application.
There is also an import statement `from florence2_app import FlorenceApp` at the top of this file that you'll need to modify.
3. Change the widgets created in `setupUi()` based on what controls you need for your application
4. Update the behavior of `set_parameters()` to reflect the parameters you need to update

