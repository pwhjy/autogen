import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from autogen_core.models import ChatCompletionClient


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        env_value = os.environ.get(name)
        if env_value:
            return env_value
        return default or ""

    return ENV_PATTERN.sub(replace, value)


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_string(value)
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def load_benchmark_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing benchmark config file: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        raw_config = yaml.safe_load(fh) or {}
    return expand_env(raw_config)


def load_model_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config = load_benchmark_config(path)
    model_config = config.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError(f"Missing model_config in {path}")
    return model_config


def load_model_client(path: str | Path = "config.yaml") -> ChatCompletionClient:
    return ChatCompletionClient.load_component(load_model_config(path))


def describe_model_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    model_config = deepcopy(load_model_config(path))
    config = model_config.get("config", {})
    if isinstance(config, dict) and "api_key" in config:
        config["api_key"] = "***"
    return model_config
