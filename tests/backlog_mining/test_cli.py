from __future__ import annotations

from rumen.cli import _build_parser


def test_backlog_mine_when_parsed_then_exposes_standalone_command() -> None:
    args = _build_parser().parse_args(["backlog", "mine", "--config", "config.json"])
    assert args.group == "backlog"
    assert args.backlog_command == "mine"
