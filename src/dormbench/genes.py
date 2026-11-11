"""Gene symbol validation and recovery.

Spreadsheet software silently converts a number of gene symbols to dates:
SEPT1-SEPT15 become 1-Sep and so on, MARCH1-MARCH11 become 1-Mar, and DEC1/DEC2
become 1-Dec. Once saved, the original symbol is gone from the cell and only the
serial date remains. The problem has been documented since 2016 and affects a
substantial fraction of published supplementary tables.

The corruption in this project originates in the published supplementary files,
not in our parsing: reading the source workbooks returns datetime objects in the
gene column. Recovery is therefore necessary rather than optional, and the
number recovered should be reported.

Recovery is deterministic for the month/day encoding:

    month 9  -> SEPT{day}   (current HGNC: SEPTIN{day})
    month 3  -> MARCH{day}  (current HGNC: MARCHF{day})
    month 12 -> DEC{day}    (DEC1 = BHLHE40, DEC2 = BHLHE41)

The year is irrelevant -- it reflects when the file was opened, which is why the
same gene appears as both 2021-09-09 and 2023-09-09 in the same panel.

DEC1 and DEC2 are handled explicitly because they are dormancy genes: DEC2 is
BHLHE41, which appears in this project's own marker panel. A silent loss there
would remove a gene central to the analysis.

Anything date-like that cannot be mapped raises rather than passing through, so
a malformed identifier cannot reach a score.
"""

from __future__ import annotations

import datetime
import re

import pandas as pd

DATE_RE = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[ T]00:00:00)?$")

# month -> (legacy prefix, current HGNC prefix)
MONTH_GENES = {
    3: ("MARCH", "MARCHF"),
    9: ("SEPT", "SEPTIN"),
}
# December symbols are not a numbered family
DEC_GENES = {1: ("DEC1", "BHLHE40"), 2: ("DEC2", "BHLHE41")}

# identifiers that are never gene symbols
JUNK = {"", "na", "nan", "none", "null", "gene", "symbol", "-", "."}


def looks_like_date(value) -> bool:
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return True
    return bool(DATE_RE.match(str(value).strip()))


def recover_symbol(value) -> tuple[str | None, str | None]:
    """Return (current_symbol, legacy_symbol) for a date-corrupted gene.

    Returns (None, None) when the value is date-like but does not correspond to
    a known month-gene family, so the caller can raise rather than guess.
    """
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        month, day = value.month, value.day
    else:
        m = DATE_RE.match(str(value).strip())
        if not m:
            return None, None
        month, day = int(m.group(2)), int(m.group(3))

    if month == 12:
        pair = DEC_GENES.get(day)
        return (pair[1], pair[0]) if pair else (None, None)
    if month in MONTH_GENES:
        legacy_p, current_p = MONTH_GENES[month]
        return f"{current_p}{day}", f"{legacy_p}{day}"
    return None, None


def clean_symbol(value) -> str | None:
    """Normalise a gene identifier; None if it is not usable."""
    s = str(value).strip()
    if s.lower() in JUNK:
        return None
    return s


def validate_and_recover(df: pd.DataFrame, gene_col: str = "gene",
                         log=None, strict: bool = True) -> pd.DataFrame:
    """Repair date-corrupted symbols in a signature table.

    Adds `gene_original` (as received) and `gene_legacy` (pre-HGNC symbol where
    applicable). `gene` is set to the current HGNC symbol. Rows whose identifier
    is junk are dropped; date-like values that cannot be mapped raise under
    strict mode.
    """
    def say(m):
        if log is not None:
            log.info(m)

    out = df.copy()
    out["gene_original"] = out[gene_col].astype(str)
    out["gene_legacy"] = pd.NA

    is_date = out[gene_col].map(looks_like_date)
    n_date = int(is_date.sum())

    if n_date:
        say(f"   [!] {n_date} date-corrupted gene identifiers detected")
        unmapped = []
        for i in out.index[is_date]:
            cur, leg = recover_symbol(out.at[i, gene_col])
            if cur is None:
                unmapped.append(str(out.at[i, gene_col]))
                continue
            out.at[i, gene_col] = cur
            out.at[i, "gene_legacy"] = leg
        if unmapped:
            msg = ("Date-like gene identifiers that could not be mapped to a "
                   f"known month-gene family: {sorted(set(unmapped))}")
            if strict:
                raise ValueError(msg)
            say(f"   [X] {msg}")
        rec = out.loc[is_date & out["gene_legacy"].notna()]
        if len(rec):
            pairs = (rec[["gene_legacy", gene_col]].drop_duplicates()
                     .sort_values(gene_col))
            say(f"   recovered {len(rec)} rows covering "
                f"{len(pairs)} distinct genes:")
            for _, r in pairs.iterrows():
                say(f"      {r['gene_legacy']:10s} -> {r[gene_col]}")

    # drop junk identifiers
    cleaned = out[gene_col].map(clean_symbol)
    n_junk = int(cleaned.isna().sum())
    if n_junk:
        say(f"   dropped {n_junk} unusable identifiers")
    out[gene_col] = cleaned
    out = out[out[gene_col].notna()].copy()

    return out


def assert_no_dates(df: pd.DataFrame, gene_col: str = "gene") -> None:
    """Raise if any date-like identifier survives. Call before scoring."""
    bad = df[df[gene_col].map(looks_like_date)]
    if len(bad):
        raise ValueError(
            "Date-like gene identifiers present in the signature panel:\n"
            + bad[[c for c in ("signature", gene_col) if c in bad.columns]]
            .to_string(index=False))


def resolve_against(genes, available) -> list[str]:
    """Match signature genes to a dataset, trying the legacy symbol as fallback.

    Recovered genes are stored under the current HGNC symbol, but reference
    datasets vary in whether they use SEPTIN9 or SEPT9. Trying both prevents a
    correctly recovered gene from being dropped for cosmetic reasons.
    """
    avail = set(available)
    out = []
    for g in genes:
        if g in avail:
            out.append(g)
            continue
        alt = None
        if g.startswith("SEPTIN"):
            alt = "SEPT" + g[6:]
        elif g.startswith("MARCHF"):
            alt = "MARCH" + g[6:]
        elif g == "BHLHE40":
            alt = "DEC1"
        elif g == "BHLHE41":
            alt = "DEC2"
        if alt and alt in avail:
            out.append(alt)
    return out
