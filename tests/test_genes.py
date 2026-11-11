"""Tests for gene symbol validation and date-corruption recovery."""
import pandas as pd
import pytest

from dormbench.genes import (assert_no_dates, looks_like_date, recover_symbol,
                             resolve_against, validate_and_recover)


def test_detects_date_strings_and_timestamps():
    assert looks_like_date("2021-09-09 00:00:00")
    assert looks_like_date(pd.Timestamp("2021-03-01"))
    assert not looks_like_date("TP53")
    assert not looks_like_date("SEPTIN9")


@pytest.mark.parametrize("value,current,legacy", [
    ("2021-09-09 00:00:00", "SEPTIN9", "SEPT9"),
    ("2023-09-09 00:00:00", "SEPTIN9", "SEPT9"),   # year is irrelevant
    ("2021-03-01 00:00:00", "MARCHF1", "MARCH1"),
    ("2021-12-02 00:00:00", "BHLHE41", "DEC2"),    # a dormancy gene
    ("2021-12-01 00:00:00", "BHLHE40", "DEC1"),
])
def test_recovery_is_deterministic(value, current, legacy):
    assert recover_symbol(value) == (current, legacy)


def test_unmappable_date_raises():
    df = pd.DataFrame({"gene": ["2021-07-04 00:00:00"], "signature": ["x"]})
    with pytest.raises(ValueError):
        validate_and_recover(df, strict=True)


def test_validate_repairs_and_records_original():
    df = pd.DataFrame({"signature": ["s"] * 3,
                       "gene": ["TP53", "2021-09-09 00:00:00", "nan"]})
    out = validate_and_recover(df, strict=True)
    assert list(out["gene"]) == ["TP53", "SEPTIN9"]      # junk dropped
    assert out.iloc[1]["gene_legacy"] == "SEPT9"
    assert out.iloc[1]["gene_original"].startswith("2021-09-09")


def test_assert_no_dates_guards_scoring():
    bad = pd.DataFrame({"signature": ["s"], "gene": ["2021-09-09 00:00:00"]})
    with pytest.raises(ValueError):
        assert_no_dates(bad)
    assert_no_dates(pd.DataFrame({"signature": ["s"], "gene": ["TP53"]}))


def test_legacy_alias_fallback():
    # dataset uses pre-HGNC symbols; recovered current symbols must still match
    assert resolve_against(["SEPTIN9", "MARCHF1"], {"SEPT9", "MARCH1"}) == \
        ["SEPT9", "MARCH1"]
    # and current symbols match when the dataset is up to date
    assert resolve_against(["SEPTIN9"], {"SEPTIN9"}) == ["SEPTIN9"]
    # genuinely absent genes are dropped
    assert resolve_against(["SEPTIN9"], {"TP53"}) == []
