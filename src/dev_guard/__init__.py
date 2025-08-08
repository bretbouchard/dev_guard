"""dev_guard: Developer-focused safety and policy utilities.

Bootstrap API surface; implementation details are subject to change.
"""

from __future__ import annotations

from .api.swarm import DevGuardSwarm
from .api.config import Config

__all__ = ["DevGuardSwarm", "Config", "__version__"]

# Keep in sync with pyproject.toml
__version__ = "0.0.0"
