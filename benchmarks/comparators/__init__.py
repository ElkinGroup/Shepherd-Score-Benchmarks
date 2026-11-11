"""Comparator software: install.sh builds every tool into one prefix, paths.py locates them, and
drivers.py builds every command line and parses every output."""
from . import paths, drivers          # noqa: F401

__all__ = ["paths", "drivers"]
