"""Weak-supervision release deferral at the public API boundary."""

from __future__ import annotations

import builtins
import importlib
import tomllib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module", "function"),
    [
        ("snorkel_label_model", "apply_labeling_functions"),
        ("snorkel_label_model", "combine_with_label_model"),
        ("snorkel_label_model", "persist_snorkel_labels"),
        ("snorkel_label_model", "run_bootstrap"),
        ("labeling_functions", "build_topic_labeling_functions"),
    ],
)
def test_given_deferred_feature_when_called_then_explains_without_importing_dependency(
    monkeypatch, module, function
):
    original = builtins.__import__
    attempted = []

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"snorkel", "pandas"}:
            attempted.append(name)
            raise ImportError("optional dependency absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    api = importlib.import_module(f"rumen.taxonomy.bootstrap.{module}")
    with pytest.raises(RuntimeError, match="Weak supervision is unavailable.*security"):
        getattr(api, function)([])
    assert attempted == []


def test_given_release_metadata_when_resolved_then_vulnerable_feature_is_not_declared():
    project = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())["project"]
    assert "weak-supervision" not in project["optional-dependencies"]
    assert not any(
        "snorkel" in dependency.lower()
        for group in [project["dependencies"], *project["optional-dependencies"].values()]
        for dependency in group
    )
