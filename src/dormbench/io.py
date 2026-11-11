"""Data access layer.

One function per dataset. Each one *locates* its files rather than assuming
a filename, because GEO and cBioPortal both rename things between releases
(METABRIC's expression matrix has been called at least two different things).
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

import pandas as pd


class DataNotFound(FileNotFoundError):
    pass


def _first_match(directory: Path, patterns: list[str], what: str) -> Path:
    """Find the first file matching any glob pattern, searching recursively."""
    if not directory.exists():
        raise DataNotFound(f"Directory does not exist: {directory}")
    for pat in patterns:
        hits = sorted(directory.rglob(pat))
        if hits:
            return hits[0]
    listing = "\n  ".join(sorted(p.name for p in directory.rglob("*") if p.is_file())[:40])
    raise DataNotFound(
        f"Could not locate {what} in {directory}\n"
        f"Tried patterns: {patterns}\nFiles present:\n  {listing}"
    )


# ---------------------------------------------------------------- atlas ----

def find_atlas_h5ad(atlas_dir: Path) -> Path:
    """Largest .h5ad under the atlas directory."""
    hits = sorted(atlas_dir.rglob("*.h5ad"), key=lambda p: p.stat().st_size, reverse=True)
    if not hits:
        raise DataNotFound(
            f"No .h5ad under {atlas_dir}. Either download the CELLxGENE object "
            "or use find_atlas_mtx() for the GEO tarball."
        )
    return hits[0]


def find_atlas_mtx(atlas_dir: Path) -> dict[str, Path]:
    """Locate the GEO 10x-style triplet plus metadata (fallback reference)."""
    return {
        "matrix": _first_match(atlas_dir, ["*sparse.mtx", "*matrix.mtx*"], "count matrix"),
        "genes": _first_match(atlas_dir, ["*genes.tsv*", "*features.tsv*"], "gene list"),
        "barcodes": _first_match(atlas_dir, ["*barcodes.tsv*"], "barcode list"),
        "metadata": _first_match(atlas_dir, ["*metadata.csv*", "*meta*.csv*"], "cell metadata"),
    }


def load_atlas(atlas_dir: Path, backed: str | None = "r"):
    """Load the atlas as AnnData. backed='r' keeps the matrix on disk."""
    import anndata as ad

    return ad.read_h5ad(find_atlas_h5ad(atlas_dir), backed=backed)


# ------------------------------------------------------------- METABRIC ----

METABRIC_EXPR_PATTERNS = [
    "data_mrna_illumina_microarray.txt",
    "data_expression_median.txt",
    "data_mRNA_median_all_sample_Zscores.txt",
    "data_mrna*.txt",
]


def load_metabric(metabric_dir: Path) -> dict[str, pd.DataFrame]:
    """Return {'expr', 'patient', 'sample'} for METABRIC."""
    expr_path = _first_match(metabric_dir, METABRIC_EXPR_PATTERNS, "expression matrix")
    pat_path = _first_match(metabric_dir, ["data_clinical_patient.txt"], "patient clinical")
    smp_path = _first_match(metabric_dir, ["data_clinical_sample.txt"], "sample clinical")

    expr = pd.read_csv(expr_path, sep="\t", low_memory=False)
    # cBioPortal clinical files carry 4 leading '#' comment rows
    patient = pd.read_csv(pat_path, sep="\t", comment="#", low_memory=False)
    sample = pd.read_csv(smp_path, sep="\t", comment="#", low_memory=False)

    return {"expr": expr, "patient": patient, "sample": sample,
            "_paths": {"expr": expr_path, "patient": pat_path, "sample": smp_path}}


def tidy_metabric_expression(expr: pd.DataFrame) -> pd.DataFrame:
    """Genes x samples, indexed by HUGO symbol, duplicates collapsed by max."""
    df = expr.copy()
    sym = "Hugo_Symbol" if "Hugo_Symbol" in df.columns else df.columns[0]
    drop = [c for c in ("Entrez_Gene_Id", "Hugo_Symbol") if c in df.columns]
    mat = df.drop(columns=drop)
    mat.index = df[sym].astype(str).values
    mat = mat.apply(pd.to_numeric, errors="coerce")
    mat = mat[~mat.index.isin(["nan", "NA", ""])]
    return mat.groupby(level=0).max()


# ---------------------------------------------------------------- SCAN-B ---

def load_scanb(scanb_dir: Path) -> dict:
    """Return {'expr_path', 'clinical'} for GSE96058.

    The expression matrix is large; the path is returned rather than the
    matrix so the caller decides when to pay for it.
    """
    expr_path = _first_match(
        scanb_dir,
        ["*gene_expression*transformed.csv*", "*gene_expression*.csv*", "*.csv.gz"],
        "SCAN-B expression matrix",
    )
    series = None
    try:
        series = _first_match(scanb_dir, ["*series_matrix.txt*"], "series matrix")
    except DataNotFound:
        pass

    clinical = parse_series_matrix(series) if series else None
    return {"expr_path": expr_path, "clinical": clinical, "series_matrix": series}


def parse_series_matrix(path: Path) -> pd.DataFrame:
    """Extract per-sample characteristics from a GEO series matrix file.

    GEO encodes clinical variables as repeated '!Sample_characteristics_ch1'
    rows of 'key: value' strings. This pivots them into a tidy frame.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    titles: list[str] = []
    geo_acc: list[str] = []
    char_rows: list[list[str]] = []

    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")]
            key, values = parts[0], parts[1:]
            if key == "!Sample_title":
                titles = values
            elif key == "!Sample_geo_accession":
                geo_acc = values
            elif key == "!Sample_characteristics_ch1":
                char_rows.append(values)

    n = len(geo_acc) or len(titles)
    if n == 0:
        raise ValueError(f"No samples parsed from {path}")

    out = pd.DataFrame(index=range(n))
    if geo_acc:
        out["geo_accession"] = geo_acc
    if titles:
        out["title"] = titles

    for row in char_rows:
        row = list(row) + [""] * (n - len(row))
        keys = [v.split(":", 1)[0].strip() for v in row if ":" in v]
        if not keys:
            continue
        colname = pd.Series(keys).mode().iat[0]
        out[colname] = [
            v.split(":", 1)[1].strip() if ":" in v else None for v in row
        ]

    return out


# ------------------------------------------------------------------ TCGA ---

TCGA_EXPR_PATTERNS = [
    "data_mrna_seq_v2_rsem.txt",
    "data_RNA_Seq_v2_expression_median.txt",
    "data_mrna_*.txt",
    "HiSeqV2*",
    "TCGA-BRCA.htseq_fpkm*.tsv*",
    "*expression*.tsv*",
    "*.tsv.gz",
]

# files that live in a TCGA folder but are never the expression matrix
TCGA_NOT_EXPR = ("clinical", "sample_sheet", "sample_types", "manifest",
                 "metadata", "degs", "limma")


def assemble_gdc(tcga_dir: Path, value_col: str = "tpm_unstranded",
                 log=None) -> pd.DataFrame:
    """Assemble a GDC per-sample download into one genes x samples matrix.

    The GDC delivers one folder per file, each holding a STAR gene-counts TSV
    with columns gene_id / gene_name / gene_type / unstranded / tpm_unstranded /
    fpkm_unstranded, preceded by four summary rows (N_unmapped and friends)
    that must be dropped. Sample identity lives in gdc_sample_sheet.tsv, not in
    the files themselves; without it the folder UUID is used, which is stable
    but useless for joining clinical data.

    The assembled matrix is cached as tcga_expression_matrix.parquet, so this
    cost is paid once.
    """
    def _say(m):
        if log is not None:
            log.info(m)

    cache = tcga_dir / "tcga_expression_matrix.parquet"
    if cache.exists():
        _say(f"   using cached {cache.name}")
        return pd.read_parquet(cache)

    files = sorted(tcga_dir.rglob("*star_gene_counts.tsv"))
    if not files:
        files = sorted(f for f in tcga_dir.rglob("*.tsv")
                       if "sample_sheet" not in f.name.lower()
                       and "manifest" not in f.name.lower())
    if not files:
        raise DataNotFound(
            f"No per-sample TSVs under {tcga_dir}. Expected a GDC download "
            "containing *.rna_seq.augmented_star_gene_counts.tsv")
    _say(f"   assembling {len(files)} per-sample files (first run only)")

    # UUID -> TCGA barcode, and UUID -> sample type
    barcode, stype = {}, {}
    sheets = list(tcga_dir.rglob("*sample_sheet*.tsv"))
    if sheets:
        sh = pd.read_csv(sheets[0], sep="\t")
        fid = next((c for c in sh.columns if c.lower().startswith("file id")), None)
        fnm = next((c for c in sh.columns if c.lower().startswith("file name")), None)
        sid = next((c for c in sh.columns if "sample id" in c.lower()), None)
        # GDC has used several names for this over the years: older exports
        # say "Sample Type" (Primary Tumor / Solid Tissue Normal), current ones
        # say "Tissue Type" (Tumor / Normal). Check both, plus the fallbacks.
        sty = next((c for c in sh.columns
                    if c.lower().strip() in ("sample type", "tissue type",
                                             "specimen type", "tumor descriptor")),
                   None)
        if sid and (fid or fnm):
            for _, r in sh.iterrows():
                t = str(r[sty]) if sty else "Unknown"
                if fid:
                    barcode[str(r[fid])] = str(r[sid])
                    stype[str(r[fid])] = t
                if fnm:
                    barcode[str(r[fnm])] = str(r[sid])
                    stype[str(r[fnm])] = t
        _say(f"   sample sheet: {sheets[0].name} — {len(sh)} rows, "
             f"{len(barcode)} key mappings")
        if sty:
            for k, v in sh[sty].value_counts().items():
                _say(f"      {k}: {v}")
        else:
            _say("   [!] no sample/tissue type column in the sheet. Columns:")
            _say(f"       {list(sh.columns)}")
            _say("       Normals cannot be identified. Supply them manually as")
            _say("       tcga_sample_types.csv with columns sample,sample_type,")
            _say("       or derive them from the barcode: -01A is primary")
            _say("       tumour, -11A is solid tissue normal.")
        if len(sheets) > 1:
            _say(f"   [!] {len(sheets)} sample sheets found; using "
                 f"{sheets[0].name}")
    else:
        _say("   [!] no gdc_sample_sheet.tsv found; columns will be file UUIDs")
        _say("       and cannot be joined to clinical data")

    cols, names, types = [], [], []
    for f in files:
        try:
            d = pd.read_csv(f, sep="\t", comment="#", low_memory=False)
        except Exception:  # noqa: BLE001
            continue
        if "gene_name" not in d.columns:
            continue
        vcol = value_col if value_col in d.columns else next(
            (c for c in ("tpm_unstranded", "fpkm_unstranded", "unstranded")
             if c in d.columns), None)
        if vcol is None:
            continue
        d = d[~d["gene_name"].astype(str).str.startswith("N_")]
        ser = pd.to_numeric(d[vcol], errors="coerce")
        ser.index = d["gene_name"].astype(str).values
        ser = ser.groupby(level=0).max()
        cols.append(ser)
        key = f.name if f.name in barcode else f.parent.name
        names.append(barcode.get(key, f.parent.name))
        types.append(stype.get(key, "Unknown"))

    if not cols:
        raise DataNotFound(
            f"Found {len(files)} TSVs under {tcga_dir} but none had a gene_name "
            "column with usable values.")

    mat = pd.concat(cols, axis=1)
    mat.columns = names
    mat = mat.loc[:, ~mat.columns.duplicated()]
    n_mapped = sum(1 for n in names if not str(n).startswith(("uuid", "u0"))
                   and "-" in str(n))
    if barcode and n_mapped < 0.9 * len(names):
        _say(f"   [!] only {n_mapped}/{len(names)} files matched the sample "
             "sheet.")
        _say("       The sheet covers fewer files than are present. Unmatched")
        _say("       columns keep their folder UUID and cannot be typed or")
        _say("       joined to clinical data.")
    mat = np.log2(mat + 1) if float(mat.max().max()) > 100 else mat
    _say(f"   assembled {mat.shape[0]} genes x {mat.shape[1]} samples")

    # sample types travel in a sidecar so they survive the parquet cache
    st = pd.DataFrame({"sample": names, "sample_type": types})
    st = st.drop_duplicates(subset="sample")
    st.to_csv(tcga_dir / "tcga_sample_types.csv", index=False)
    if st["sample_type"].nunique() > 1:
        _say("   sample types recorded:")
        for k, v in st["sample_type"].value_counts().items():
            _say(f"      {k}: {v}")
    try:
        mat.to_parquet(cache)
        _say(f"   cached to {cache.name}")
    except Exception as e:  # noqa: BLE001
        _say(f"   [!] could not cache: {e}")
    return mat


def load_tcga(tcga_dir: Path, log=None) -> dict:
    """Load TCGA-BRCA expression, whatever format it arrived in.

    Handles the two common shapes: cBioPortal study folders (Hugo_Symbol +
    Entrez_Gene_Id + one column per sample) and UCSC Xena matrices (gene rows,
    sample columns, no Entrez column). Returns genes x samples indexed by
    symbol, plus whatever clinical table is present.
    """
    try:
        expr_path = _first_match(tcga_dir, TCGA_EXPR_PATTERNS, "TCGA expression")
        if any(k in expr_path.name.lower() for k in TCGA_NOT_EXPR):
            raise DataNotFound(f"{expr_path.name} is not an expression matrix")
        sep = "," if expr_path.suffix.lower() in (".csv",) else "\t"
        raw = pd.read_csv(expr_path, sep=sep, low_memory=False)
        if raw.shape[1] < 20:      # a single-sample file, not a matrix
            raise DataNotFound("looks like one sample, not a matrix")
    except DataNotFound:
        # fall back to assembling a GDC per-sample download
        mat = assemble_gdc(tcga_dir, log=log)
        clinical = None
        for pat in ("data_clinical_patient.txt", "*clinical*.tsv*",
                    "*clinical*.txt"):
            try:
                cp = _first_match(tcga_dir, [pat], "clinical")
                clinical = pd.read_csv(cp, sep="\t", comment="#",
                                       low_memory=False)
                break
            except DataNotFound:
                continue
        stp = tcga_dir / "tcga_sample_types.csv"
        types = (pd.read_csv(stp).set_index("sample")["sample_type"]
                 if stp.exists() else None)
        return {"expr": mat, "clinical": clinical, "sample_types": types,
                "expr_path": tcga_dir / "tcga_expression_matrix.parquet"}

    first = raw.columns[0]
    if "Hugo_Symbol" in raw.columns:          # cBioPortal
        sym = raw["Hugo_Symbol"].astype(str)
        drop = [c for c in ("Hugo_Symbol", "Entrez_Gene_Id") if c in raw.columns]
        mat = raw.drop(columns=drop)
    else:                                     # Xena or generic
        sym = raw[first].astype(str)
        mat = raw.drop(columns=[first])
    mat.index = sym.values
    mat = mat.apply(pd.to_numeric, errors="coerce")
    mat = mat[~mat.index.isin(["nan", "NA", "", "None"])]
    mat = mat.groupby(level=0).max()

    clinical = None
    for pat in ("data_clinical_patient.txt", "data_clinical_sample.txt",
                "*clinical*.tsv*", "*clinical*.csv*"):
        try:
            cp = _first_match(tcga_dir, [pat], "clinical")
            csep = "," if cp.suffix.lower() == ".csv" else "\t"
            clinical = pd.read_csv(cp, sep=csep, comment="#", low_memory=False)
            break
        except DataNotFound:
            continue

    stp = tcga_dir / "tcga_sample_types.csv"
    types = (pd.read_csv(stp).set_index("sample")["sample_type"]
             if stp.exists() else None)
    return {"expr": mat, "clinical": clinical, "sample_types": types,
            "expr_path": expr_path}


# ---------------------------------------------------------------- Rosano ---

STRICT_NS = {
    "m": "http://purl.oclc.org/ooxml/spreadsheetml/main",
    "r": "http://purl.oclc.org/ooxml/officeDocument/relationships",
}


def read_strict_xlsx(path: Path) -> dict[str, pd.DataFrame]:
    """Read a 'strict OOXML' workbook that openpyxl reports as having no sheets.

    Rosano Supplementary Table 1 is saved in the ISO strict variant, which uses
    the purl.oclc.org namespace instead of schemas.openxmlformats.org. openpyxl
    silently returns zero sheets rather than raising, so this is a manual parser.
    Returns {sheet_name: DataFrame} with the first row used as the header.
    """
    import zipfile
    from xml.etree import ElementTree as ET

    z = zipfile.ZipFile(path)
    m = STRICT_NS["m"]

    try:
        sst = [
            "".join(t.text or "" for t in si.iter(f"{{{m}}}t"))
            for si in ET.fromstring(z.read("xl/sharedStrings.xml"))
        ]
    except KeyError:
        sst = []

    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = {
        r.get("Id"): r.get("Target")
        for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    }

    out: dict[str, pd.DataFrame] = {}
    for sh in wb.iter(f"{{{m}}}sheet"):
        target = rels[sh.get(f"{{{STRICT_NS['r']}}}id")]
        member = "xl/" + target.lstrip("/").replace("xl/", "", 1)
        rows = []
        for row in ET.fromstring(z.read(member)).iter(f"{{{m}}}row"):
            vals = []
            for c in row.iter(f"{{{m}}}c"):
                v = c.find(f"{{{m}}}v")
                txt = None if v is None else v.text
                if c.get("t") == "s" and txt is not None:
                    txt = sst[int(txt)]
                vals.append(txt)
            rows.append(vals)
        if not rows:
            out[sh.get("name")] = pd.DataFrame()
            continue
        width = max(len(r) for r in rows)
        rows = [r + [None] * (width - len(r)) for r in rows]
        out[sh.get("name")] = pd.DataFrame(rows[1:], columns=rows[0])
    return out


def load_rosano_table(rosano_dir: Path, number: int, sheet=0) -> pd.DataFrame:
    path = _first_match(
        rosano_dir,
        [f"*supplementary_table_{number}_*.xlsx", f"*table_{number}*.xlsx"],
        f"Rosano supplementary table {number}",
    )
    try:
        df = pd.read_excel(path, sheet_name=sheet)
        if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
            return df
    except (ValueError, KeyError, IndexError):
        pass
    # Fall back to the strict-OOXML reader (Table 1 needs this)
    sheets = read_strict_xlsx(path)
    if isinstance(sheet, int):
        return list(sheets.values())[sheet]
    return sheets[sheet]
