from datetime import UTC, datetime

import pytest

from rumen.paperpile.parsers import BibtexParseError, parse_bibtex


def test_bibtex_when_export_has_unicode_macros_and_identifiers_then_returns_plain_records() -> None:
    records = parse_bibtex(
        """@string{venue = "Journál"}
        @article{curated2026, title={Braced {Unicode} café}, author={Ada Lovelace and Grace Hopper},
        journal=venue, doi={10.1000/Example}, url={https://example.test/paper}, year=2026,
        abstract={A} # { B}}""",
        observed_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    assert records[0].title == "Braced Unicode café"
    assert records[0].authors == ["Ada Lovelace", "Grace Hopper"]
    assert records[0].identifiers == {"doi": "10.1000/example"}
    assert records[0].date_precision == "year"


@pytest.mark.parametrize(
    ("entry", "precision"),
    [
        ("date={2026-03-02}", "date"),
        ("year={2026}", "year"),
        ("", "observed"),
    ],
)
def test_bibtex_when_dates_have_different_precision_then_normalizes_deterministically(
    entry: str, precision: str
) -> None:
    observed = datetime(2026, 9, 1, tzinfo=UTC)
    record = parse_bibtex(f"@article{{x,title={{X}},{entry}}}", observed_at=observed)[0]
    assert record.date_precision == precision
    assert record.published_at == (
        datetime(2026, 3, 2, tzinfo=UTC)
        if precision == "date"
        else datetime(2026, 1, 1, tzinfo=UTC)
        if precision == "year"
        else observed
    )


@pytest.mark.parametrize(
    "text",
    ["@article{x,title={X}", "@article{x,title={X}} @article{x,title={Y}}"],
)
def test_bibtex_when_malformed_or_duplicate_then_rejects_without_records(text: str) -> None:
    with pytest.raises(BibtexParseError):
        parse_bibtex(text, observed_at=datetime(2026, 9, 1, tzinfo=UTC))
