"""dormbench - compartment attribution of dormancy signatures in ER+ breast cancer."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version

try:                                   # single source of truth: pyproject.toml
    __version__ = _installed_version("dormbench")
except PackageNotFoundError:           # running straight from a source tree
    __version__ = "0.3.1"

from .config import Config, load_config  # noqa: E402,F401

__all__ = ["Config", "load_config", "__version__"]
