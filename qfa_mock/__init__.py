"""Host-only, single-invocation QFA metadata prototype."""

from .model import MockError
from .provider import MetadataProvider, MetadataResult

__all__ = ["MetadataProvider", "MetadataResult", "MockError"]
