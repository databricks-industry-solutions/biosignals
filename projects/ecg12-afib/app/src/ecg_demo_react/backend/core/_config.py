from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..._metadata import app_name, app_slug
from ..common.config import SCHEMA

# --- Config ---

project_root = Path(__file__).parent.parent.parent.parent.parent
env_file = project_root / ".env"

if env_file.exists():
    load_dotenv(dotenv_path=env_file)


class AppConfig(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=env_file,
        env_prefix=f"{app_slug.upper()}_",
        extra="ignore",
        env_nested_delimiter="__",
    )
    app_name: str = Field(default=app_name)
    # Where this deployment's tables and models live, and which AI/BI dashboard the Batch tab
    # embeds. The bundle's `full` target sets them (databricks.yml, resources.apps.*.config.env)
    # from its `catalog` / `schema` variables and the id of the dashboard it deploys, as
    # ECG_DEMO_REACT_CATALOG, ECG_DEMO_REACT_UC_SCHEMA and ECG_DEMO_REACT_DASHBOARD_ID. The catalog
    # has no default, so the app fails at startup without it; no dashboard id = no embed.
    catalog: str
    uc_schema: str = Field(default=SCHEMA)
    dashboard_id: str | None = None

    @property
    def static_assets_path(self) -> Path:
        return Path(str(resources.files(app_slug))).joinpath("__dist__")

    def __hash__(self) -> int:
        return hash(self.app_name)


# --- Logger ---

logger = logging.getLogger(app_name)
