# gdeltngrams/__init__.py
from ._version import __version__
from .ingestion import *
from .multiprocess import *

__all__ = ['ingestion','multiprocess']