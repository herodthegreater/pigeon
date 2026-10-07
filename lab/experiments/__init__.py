"""Importing this package registers every experiment in ``common.REGISTRY``."""

from . import distill, shrink, small, specialist, teacher, transfer  # noqa: F401
