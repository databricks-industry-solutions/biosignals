from __future__ import annotations
from typing import Annotated, AsyncGenerator, TypeAlias
from contextlib import asynccontextmanager

from databricks.sdk import WorkspaceClient
from fastapi import FastAPI, Request

from ._base import LifespanDependency
from ._config import AppConfig, logger


class _ConfigDependency(LifespanDependency):
    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.config = AppConfig()
        logger.info(f"Starting app with configuration:\n{app.state.config}")
        yield

    @staticmethod
    def __call__(request: Request) -> AppConfig:
        return request.app.state.config


class _WorkspaceClientDependency(LifespanDependency):
    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.workspace_client = WorkspaceClient()
        yield

    @staticmethod
    def __call__(request: Request) -> WorkspaceClient:
        return request.app.state.workspace_client


ConfigDependency: TypeAlias = Annotated[AppConfig, _ConfigDependency.depends()]

ClientDependency: TypeAlias = Annotated[
    WorkspaceClient, _WorkspaceClientDependency.depends()
]
