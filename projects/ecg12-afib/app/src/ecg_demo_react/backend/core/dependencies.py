from __future__ import annotations

from typing import TypeAlias
from ._defaults import ConfigDependency, ClientDependency


class Dependencies:
    """FastAPI dependency injection shorthand for route handler parameters."""

    Client: TypeAlias = ClientDependency
    """Databricks WorkspaceClient using app-level service principal credentials.
    Recommended usage: `ws: Dependencies.Client`"""

    Config: TypeAlias = ConfigDependency
    """Application configuration loaded from environment variables.
    Recommended usage: `config: Dependencies.Config`"""
