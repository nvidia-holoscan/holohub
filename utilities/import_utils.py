import importlib
from typing import Any, TypeVar

T = TypeVar("T")


class LazyRaise:
    """A proxy object that raises the stored exception when accessed."""

    def __init__(self, exception: Exception):
        self._exception = exception

    def __getattr__(self, name: str) -> None:
        """Raises the stored exception when any attribute is accessed."""
        raise self._exception

    def __call__(self, *args, **kwargs) -> None:
        """Raises the stored exception when the object is called."""
        raise self._exception

    def __getitem__(self, item) -> None:
        """Raises the stored exception when indexed."""
        raise self._exception

    def __iter__(self) -> None:
        """Raises the stored exception when iterated."""
        raise self._exception

    def __repr__(self) -> str:
        """Returns a string representation of the stored exception."""
        return f"LazyRaise({self._exception})"


def lazy_import(module_name: str) -> Any:
    """
    Import an optional module specified by module_name.

    If the module cannot be imported, returns a proxy object that raises
    the ImportError when accessed.

    Args:
        module_name: name of the module to be imported.

    Returns:
        The imported module or a proxy object that raises ImportError when accessed
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as import_error:
        return LazyRaise(import_error)
