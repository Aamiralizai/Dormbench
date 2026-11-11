"""Tests for the data access layer. Run: pytest"""
import gzip
import pandas as pd
import pytest

from dormbench import io as dio


@pytest.fixture
def scanb_dir(tmp_path):
    d = tmp_path / "GSE96058"
    d.mkdir()
    rows = [
        '!Sample_title\t"F1"\t"F2"',
        '!Sample_geo_accession\t"GSM1"\t"GSM2"',
        '!Sample_characteristics_ch1\t"er status: 1"\t"er status: 0"',
        '!Sample_characteristics_ch1\t"overall survival days: 2100"\t"overall survival days: 900"',
        "!series_matrix_table_begin",
    ]
    with gzip.open(d / "GSE96058_series_matrix.txt.gz", "wt") as f:
        f.write("\n".join(rows) + "\n")
    with gzip.open(d / "GSE96058_gene_expression_transformed.csv.gz", "wt") as f:
        f.write("gene,F1,F2\nESR1,9.1,2.2\n")
    return d


def test_series_matrix_pivots_characteristics(scanb_dir):
    out = dio.load_scanb(scanb_dir)
    cl = out["clinical"]
    assert len(cl) == 2
    assert "er status" in cl.columns
    assert cl.loc[0, "er status"] == "1"
    assert cl.loc[1, "overall survival days"] == "900"


def test_missing_directory_raises(tmp_path):
    with pytest.raises(dio.DataNotFound):
        dio._first_match(tmp_path / "nope", ["*.txt"], "thing")


def test_error_lists_present_files(tmp_path):
    (tmp_path / "actual_file.csv").write_text("x")
    with pytest.raises(dio.DataNotFound) as e:
        dio._first_match(tmp_path, ["*.txt"], "thing")
    assert "actual_file.csv" in str(e.value)


def test_tidy_metabric_collapses_duplicates():
    expr = pd.DataFrame({
        "Hugo_Symbol": ["TP53", "ESR1", "TP53"],
        "Entrez_Gene_Id": [7157, 2099, 7157],
        "S1": [5.1, 9.4, 5.9],
        "S2": [6.2, 8.8, 6.0],
    })
    tidy = dio.tidy_metabric_expression(expr)
    assert tidy.shape == (2, 2)
    assert tidy.loc["TP53", "S1"] == 5.9      # max of duplicates
    assert "Entrez_Gene_Id" not in tidy.columns
