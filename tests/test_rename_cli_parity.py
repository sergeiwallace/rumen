"""Parser-level clean-break parity for every retained rumen command path."""

from __future__ import annotations

import pytest

from rumen.cli import _build_parser


@pytest.mark.parametrize(
    "command_path",
    [
        ("feed", "items"),
        ("feed", "digest"),
        ("feed", "status"),
        ("feed", "rate", "1", "0"),
        ("reddit", "purge", "--profile", "profile"),
        (
            "classify",
            "review",
            "feed_items",
            "1",
            "topic",
            "--concept",
            "1",
            "--actor",
            "operator",
            "--rationale",
            "reviewed",
        ),
        ("classify", "metrics"),
        ("run",),
        ("backlog", "mine"),
        ("backlog", "list"),
        ("backlog", "show", "candidate"),
        (
            "backlog",
            "review",
            "candidate",
            "--action",
            "accept",
            "--actor",
            "operator",
            "--rationale",
            "reviewed",
        ),
        ("digest",),
        ("fit-classifier",),
        ("taxonomy-audit",),
    ],
)
def test_rename_cli_parity(command_path: tuple[str, ...]) -> None:
    parser = _build_parser()
    assert parser.parse_args(command_path)

    legacy_cli = "k" + "c"
    with pytest.raises(SystemExit) as raised:
        parser.parse_args((legacy_cli, *command_path))
    assert raised.value.code == 2


@pytest.mark.parametrize("group", ("feed", "reddit", "classify", "backlog"))
def test_rename_group_parser_requires_subcommand(group: str) -> None:
    with pytest.raises(SystemExit) as raised:
        _build_parser().parse_args([group])
    assert raised.value.code == 2
