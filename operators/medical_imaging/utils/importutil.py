# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import inspect
import runpy
import sys
import warnings
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

import pkg_resources

if TYPE_CHECKING:
    from holoscan.core import Application


def get_docstring(cls: Type) -> str:
    """Get docstring of a class.

    Tries to get docstring from class itself, from its __doc__.
    It trims the preceding whitespace from docstring.
    If __doc__ is not available, it returns empty string.

    Args:
        cls (Type): class to get docstring from.

    Returns:
        A docstring of the class.
    """
    doc = cls.__doc__
    if doc is None:
        return ""
    # Trim white-space for each line in the string
    return "\n".join([line.strip() for line in doc.split("\n")])


def is_subclass(cls: Type, class_or_tuple: Union[str, Tuple[str]]) -> bool:
    """Check if the given type is a subclass of a MONAI Deploy App SDK class.

    Args:
        cls (Type): A class to check.
        class_or_tuple (Union[str, Tuple[str]]): A class name or a tuple of class names.

    Returns:
        True if the given class is a subclass of the given class or one of the classes in the tuple.
    """
    if type(class_or_tuple) is str:
        class_or_tuple = (class_or_tuple,)

    if hasattr(cls, "_class_id") and cls._class_id in class_or_tuple:
        if (
            inspect.isclass(cls)
            and hasattr(cls, "__abstractmethods__")
            and len(cls.__abstractmethods__) != 0
        ):
            return False
        return True
    return False


def get_application(path: Union[str, Path]) -> Optional["Application"]:
    """Get application object from path."""
    from holoscan.core import Application

    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    # Setup PYTHONPATH if the target path is a file
    if path.is_file() and sys.path[0] != str(path.parent):
        sys.path.insert(0, str(path.parent))

    # Execute the module with runpy (`run_name` would be '<run_path>' by default.)
    vars = runpy.run_path(str(path))

    # Get the Application class from the module and return an instance of it
    for var in vars.keys():
        if not var.startswith("_"):  # skip private variables
            app_cls: Type[Application] = vars[var]

            if is_subclass(app_cls, Application._class_id):
                if path.is_file():
                    app_path = path
                else:
                    app_path = path / f"{app_cls.__module__}.py"

                # Create Application object with the application path
                app_obj = app_cls(do_run=False, path=app_path)
                return app_obj
    return None


def get_class_file_path(cls: Type) -> Path:
    """Get the file path of a class.

    If the file path is not available, it tries to see each frame information
    in the stack to check whether the file name ends with "interactiveshell.py"
    and the function name is "run_code".
    If so, it returns Path("ipython") to notify that the class is defined
    inside IPython.

    Args:
        cls (Type): A class to get file path from.

    Returns:
        A file path of the class.
    """

    try:
        return Path(inspect.getfile(cls))
    except (TypeError, OSError):
        # If in IPython shell, use inspect.stack() to get the caller's file path
        stack = inspect.stack()
        for frame in stack:
            if frame.filename.endswith("interactiveshell.py") and frame.function == "run_code":
                return Path("ipython")
        # If not in IPython shell, re-raise the error
        raise


######################################################################################
# The following implementations are borrowed from `monai.utils.module` of MONAI Core.
######################################################################################

OPTIONAL_IMPORT_MSG_FMT = "{}"


class OptionalImportError(ImportError):
    """Raises when an optional dependency could not be imported."""


def min_version(the_module, min_version_str: str = "") -> bool:
    """
    Convert version strings into tuples of int and compare them.

    Returns True if the module's version is greater or equal to the 'min_version'.
    When min_version_str is not provided, it always returns True.
    """
    if not min_version_str or not hasattr(the_module, "__version__"):
        return True  # always valid version

    mod_version = tuple(int(x) for x in the_module.__version__.split(".")[:2])
    required = tuple(int(x) for x in min_version_str.split(".")[:2])
    return mod_version >= required


def exact_version(the_module, version_str: str = "") -> bool:
    """
    Returns True if the module's __version__ matches version_str
    """
    if not hasattr(the_module, "__version__"):
        warnings.warn(
            f"{the_module} has no attribute __version__ in exact_version check.", stacklevel=2
        )
        return False
    return bool(the_module.__version__ == version_str)


def optional_import(
    module: str,
    version: str = "",
    version_checker: Callable[..., bool] = min_version,
    name: str = "",
    descriptor: str = OPTIONAL_IMPORT_MSG_FMT,
    version_args: Any = None,
    allow_namespace_pkg: bool = False,
    as_type: str = "default",
) -> Tuple[Any, bool]:
    """
    Imports an optional module specified by `module` string.
    Any importing related exceptions will be stored, and exceptions raise lazily
    when attempting to use the failed-to-import module.

    Args:
        module: name of the module to be imported.
        version: version string used by the version_checker.
        version_checker: a callable to check the module version, Defaults to monai.utils.min_version.
        name: a non-module attribute (such as method/class) to import from the imported module.
        descriptor: a format string for the final error message when using a not imported module.
        version_args: additional parameters to the version checker.
        allow_namespace_pkg: whether importing a namespace package is allowed. Defaults to False.
        as_type: there are cases where the optionally imported object is used as
            a base class, or a decorator, the exceptions should raise accordingly. The current supported values
            are "default" (call once to raise), "decorator" (call the constructor and the second call to raise),
            and anything else will return a lazy class that can be used as a base class (call the constructor to raise).

    Returns:
        The imported module and a boolean flag indicating whether the import is successful.

    Examples::

        >>> torch, flag = optional_import('torch', '1.1')
        >>> print(torch, flag)
        <module 'torch' from 'python/lib/python3.6/site-packages/torch/__init__.py'> True

        >>> the_module, flag = optional_import('unknown_module')
        >>> print(flag)
        False
        >>> the_module.method  # trying to access a module which is not imported
        OptionalImportError: import unknown_module (No module named 'unknown_module').

        >>> torch, flag = optional_import('torch', '42', exact_version)
        >>> torch.nn  # trying to access a module for which there isn't a proper version imported
        OptionalImportError: import torch (requires version '42' by 'exact_version').

        >>> conv, flag = optional_import('torch.nn.functional', '1.0', name='conv1d')
        >>> print(conv)
        <built-in method conv1d of type object at 0x11a49eac0>

        >>> conv, flag = optional_import('torch.nn.functional', '42', name='conv1d')
        >>> conv()  # trying to use a function from the not successfully imported module (due to unmatched version)
        OptionalImportError: from torch.nn.functional import conv1d (requires version '42' by 'min_version').
    """

    tb = None
    exception_str = ""
    if name:
        actual_cmd = f"from {module} import {name}"
    else:
        actual_cmd = f"import {module}"
    try:
        pkg = __import__(module)  # top level module
        the_module = import_module(module)
        if not allow_namespace_pkg:
            is_namespace = getattr(the_module, "__file__", None) is None and hasattr(
                the_module, "__path__"
            )
            if is_namespace:
                raise AssertionError
        if name:  # user specified to load class/function/... from the module
            the_module = getattr(the_module, name)
    except Exception as import_exception:  # any exceptions during import
        tb = import_exception.__traceback__
        exception_str = f"{import_exception}"
    else:  # found the module
        if version_args and version_checker(pkg, f"{version}", version_args):
            return the_module, True
        if not version_args and version_checker(pkg, f"{version}"):
            return the_module, True

    # preparing lazy error message
    msg = descriptor.format(actual_cmd)
    if version and tb is None:  # a pure version issue
        msg += f" (requires '{module} {version}' by {version_checker.__name__!r})"
    if exception_str:
        msg += f" ({exception_str})"

    class _LazyRaise:
        def __init__(self, *_args, **_kwargs):
            _default_msg = (
                f"{msg}."
                + "\n\nFor details about installing the optional dependencies, please visit:"
                + "\n    https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies"
            )
            if tb is None:
                self._exception = OptionalImportError(_default_msg)
            else:
                self._exception = OptionalImportError(_default_msg).with_traceback(tb)

        def __getattr__(self, name):
            """
            Raises:
                OptionalImportError: When you call this method.
            """
            raise self._exception

        def __call__(self, *_args, **_kwargs):
            """
            Raises:
                OptionalImportError: When you call this method.
            """
            raise self._exception

        def __getitem__(self, item):
            raise self._exception

        def __iter__(self):
            raise self._exception

    if as_type == "default":
        return _LazyRaise(), False

    class _LazyCls(_LazyRaise):
        def __init__(self, *_args, **kwargs):
            super().__init__()
            if not as_type.startswith("decorator"):
                raise self._exception

    return _LazyRaise(), False


######################################################################################


def is_dist_editable(project_name: str) -> bool:
    distributions: Dict = {v.key: v for v in pkg_resources.working_set}
    dist: Any = distributions.get(project_name)
    if not hasattr(dist, "egg_info"):
        return False
    egg_info = Path(dist.egg_info)
    if egg_info.is_dir():
        if egg_info.suffix == ".egg-info":
            return True
        elif egg_info.suffix == ".dist-info":
            if (egg_info / "direct_url.json").exists():
                import json

                # Check direct_url.json for "editable": true
                # (https://packaging.python.org/en/latest/specifications/direct-url/)
                with open(egg_info / "direct_url.json", "r") as f:
                    data = json.load(f)
                    try:
                        if data["dir_info"]["editable"]:
                            return True
                    except KeyError:
                        pass
    return False


def dist_module_path(project_name: str) -> str:
    distributions: Dict = {v.key: v for v in pkg_resources.working_set}
    dist: Any = distributions.get(project_name)
    if hasattr(dist, "egg_info"):
        egg_info = Path(dist.egg_info)
        if egg_info.is_dir() and egg_info.suffix == ".dist-info":
            if (egg_info / "direct_url.json").exists():
                import json

                # Check direct_url.json for "url"
                # (https://packaging.python.org/en/latest/specifications/direct-url/)
                with open(egg_info / "direct_url.json", "r") as f:
                    data = json.load(f)
                    try:
                        file_url = data["url"]
                        if file_url.startswith("file://"):
                            return str(file_url[7:])
                    except KeyError:
                        pass

    if hasattr(dist, "module_path"):
        return str(dist.module_path)
    return ""


def is_module_installed(project_name: str) -> bool:
    distributions: Dict = {v.key: v for v in pkg_resources.working_set}
    dist: Any = distributions.get(project_name)
    if dist:
        return True
    else:
        return False


def dist_requires(project_name: str) -> List[str]:
    distributions: Dict = {v.key: v for v in pkg_resources.working_set}
    dist: Any = distributions.get(project_name)
    if hasattr(dist, "requires"):
        return [str(req) for req in dist.requires()]
    return []


holoscan_init_content_txt = """
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# We import core and gxf to make sure they're available before other modules that rely on them
from . import core, gxf

as_tensor = core.Tensor.as_tensor
__all__ = ["as_tensor", "core", "gxf"]

# Other modules are exposed to the public API but will only be lazily loaded
_EXTRA_MODULES = [
    "conditions",
    "executors",
    "graphs",
    "logger",
    "operators",
    "resources",
]
__all__.extend(_EXTRA_MODULES)


# Autocomplete
def __dir__():
    return __all__


# Lazily load extra modules
def __getattr__(name):
    import importlib
    import sys

    if name in _EXTRA_MODULES:
        module_name = f"{__name__}.{name}"
        module = importlib.import_module(module_name)  # import
        sys.modules[module_name] = module  # cache
        return module
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")

"""


def fix_holoscan_import():
    """Fix holoscan __init__ to enable lazy load for avoiding failure on loading low level libs."""

    try:
        project_name = "holoscan"
        holoscan_init_path = Path(dist_module_path(project_name)) / project_name / "__init__.py"

        with open(str(holoscan_init_path), "w") as f_w:
            f_w.write(holoscan_init_content_txt)
        return str(holoscan_init_path)
    except Exception as ex:
        return ex


if __name__ == "__main__":
    """Utility functions that can be used in the command line."""

    argv = sys.argv
    if len(argv) == 2 and argv[1] == "fix_holoscan_import":
        file_path = fix_holoscan_import()
        if file_path:
            print(file_path)
            sys.exit(0)
        else:
            sys.exit(1)
    if len(argv) == 3 and argv[1] == "is_dist_editable":
        if is_dist_editable(argv[2]):
            sys.exit(0)
        else:
            sys.exit(1)
    if len(argv) == 3 and argv[1] == "dist_module_path":
        module_path = dist_module_path(argv[2])
        if module_path:
            print(module_path)
            sys.exit(0)
        else:
            sys.exit(1)
    if len(argv) == 3 and argv[1] == "is_module_installed":
        is_installed = is_module_installed(argv[2])
        if is_installed:
            sys.exit(0)
        else:
            sys.exit(1)
    if len(argv) == 3 and argv[1] == "dist_requires":
        requires = dist_requires(argv[2])
        if requires:
            print("\n".join(requires))
            sys.exit(0)
        else:
            sys.exit(1)
    if len(argv) >= 3 and argv[1] == "get_package_info":
        import json

        app = get_application(argv[2])
        if app:
            print(json.dumps(app.get_package_info(argv[3] if len(argv) > 3 else ""), indent=2))
