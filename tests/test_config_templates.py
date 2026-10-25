"""Hermetic coverage for the shipped per-adapter configuration templates."""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from pathlib import Path

import pytest

from rumen import config as config_module
from rumen.config import load_config

TEMPLATE_DIRECTORY = Path(__file__).parents[1] / "config" / "templates"
NO_CONFIG_RATIONALES: dict[str, str] = {}
_SPECIAL_CONFIGURED_TYPES = {"chrome_reading_list"}


def _registered_types(function: object) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {
        comparison.comparators[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        for comparison in [node]
        if len(comparison.ops) == 1
        and isinstance(comparison.ops[0], ast.Eq)
        and len(comparison.comparators) == 1
        and isinstance(comparison.comparators[0], ast.Constant)
        and isinstance(comparison.comparators[0].value, str)
    }


def _template_type(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    [source] = payload["sources"]
    return source["type"]


def test_given_adapter_registry_when_templates_are_checked_then_each_kind_is_covered() -> None:
    registered = _registered_types(config_module._build_adapter) | _SPECIAL_CONFIGURED_TYPES
    template_types = {_template_type(path) for path in TEMPLATE_DIRECTORY.glob("*.json")}

    assert all(reason.strip() for reason in NO_CONFIG_RATIONALES.values())
    assert registered == template_types | set(NO_CONFIG_RATIONALES)
    assert template_types.isdisjoint(NO_CONFIG_RATIONALES)


def test_given_each_shipped_template_when_loaded_then_it_builds_or_is_explicitly_disabled() -> None:
    for template in sorted(TEMPLATE_DIRECTORY.glob("*.json")):
        config = load_config(template)
        source = json.loads(template.read_text(encoding="utf-8"))["sources"][0]

        assert (
            source["type"]
            in _registered_types(config_module._build_adapter) | _SPECIAL_CONFIGURED_TYPES
        )
        assert config.adapters or source.get("enabled") is False


def test_given_sanitized_example_when_loaded_then_it_is_llm_free_and_uses_tuning_defaults() -> None:
    config = load_config(Path(__file__).parents[1] / "config.example.json")

    assert config.adapters == []
    assert config.database.dsn == "sqlite:///rumen.db"
    assert config.storage.backend == "sqlalchemy"
    assert config.llm_command is None
    assert config.digest_top_n == 20
    assert config.tau_class == 0.75


@pytest.mark.parametrize("backend", ["unsupported", "", None, 1, ["sqlalchemy"]])
def test_given_unsupported_storage_backend_when_config_loaded_then_rejected(
    tmp_path, backend
) -> None:
    payload = json.loads((TEMPLATE_DIRECTORY.parents[1] / "config.example.json").read_text())
    payload["storage"] = {"backend": backend}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="storage.backend must be 'sqlalchemy'"):
        load_config(path)
