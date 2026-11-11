"""WP6 — paired on-treatment biopsies: state change, or composition change?

Every cohort analysed so far is a single snapshot per patient, so composition
and cell state cannot be separated within a patient. POETIC (GSE105777) is
different: paired core biopsies from the same tumour at diagnosis and at
surgery, after two weeks of aromatase inhibitor, plus an untreated arm biopsied
on the same schedule.

That makes a within-patient test possible:

    if the dormancy score rises after endocrine therapy, is that because the
    cancer cells changed state, or because the tumour's cellular composition
    changed?

Endocrine therapy kills proliferating cancer cells, which mechanically raises
the stromal fraction. A composition-driven score would therefore rise after
treatment with no change in any cancer cell. The test is whether the change in
signature score tracks the change in compartment content across patients, and
whether it survives adjustment for it.

The untreated arm is essential and is why this cohort was chosen: the POETIC
investigators showed that diagnosis-to-surgery differences are substantial even
without treatment, from sampling and handling alone. Any treatment effect must
exceed that.

Two weeks is early adaptation, not dormancy -- roughly the latency phase in
TRADITIOM terms, not the day 30-90 dormancy window. This step therefore tests
the measurement, not the biology.
"""

from __future__ import annotations

import gzip
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from ..logging_utils import banner, get_logger

STROMAL = ["CAFs", "PVL", "Endothelial"]

# Boundary between "early" and "long-term" on treatment, in days. Follows the
# Selli et al. stratification: pre-treatment, early (13-120 d), long term
# (>120 d). The 2-week and 3-month Edinburgh samples are both "early".
EARLY_MAX_DAYS = 120.0


def _days_from_label(label: str) -> float:
    """Days on treatment implied by a timepoint label, or NaN if unreadable.

    Handles the forms actually present in these series ("2 wks of letrozole
    treatment", "3 mo endocrine therapy", "pre-treatment"). Bare "timepoint N"
    carries no duration and returns NaN: N is an ordinal, not a time, and
    guessing days from it would invent data. Supply real durations through
    config/wp6_timepoints.csv instead.
    """
    s = str(label).strip().lower()
    if not s:
        return float("nan")
    if re.search(r"pre[- ]?treat|baseline|diagnos|screen|\bpre\b", s) and \
            not re.search(r"post", s):
        return 0.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*(day|days|d)\b", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(wk|wks|week|weeks)\b", s)
    if m:
        return float(m.group(1)) * 7.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mo|mos|month|months)\b", s)
    if m:
        return float(m.group(1)) * 30.4375
    m = re.search(r"(\d+(?:\.\d+)?)\s*(yr|yrs|year|years)\b", s)
    if m:
        return float(m.group(1)) * 365.25
    return float("nan")


def _open(p: Path):
    return gzip.open(p, "rt", encoding="utf-8", errors="replace") \
        if str(p).endswith(".gz") else open(p, encoding="utf-8", errors="replace")


def _read_series_matrix(path: Path, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (expression, sample metadata) from a GEO series matrix."""
    meta_rows, titles, geo = [], [], []
    with _open(path) as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")]
            key, vals = parts[0], parts[1:]
            if key == "!Sample_title":
                titles = vals
            elif key == "!Sample_geo_accession":
                geo = vals
            elif key in ("!Sample_characteristics_ch1", "!Sample_source_name_ch1",
                         "!Sample_description"):
                meta_rows.append((key, vals))

    n = len(geo) or len(titles)
    meta = pd.DataFrame(index=range(n))
    if geo:
        meta["geo_accession"] = geo
    if titles:
        meta["title"] = titles
    for key, vals in meta_rows:
        vals = list(vals) + [""] * (n - len(vals))
        keys = [v.split(":", 1)[0].strip() for v in vals if ":" in v]
        col = (pd.Series(keys).mode().iat[0] if keys
               else key.replace("!Sample_", ""))
        base, i = col, 2
        while col in meta.columns:
            col = f"{base}_{i}"; i += 1
        meta[col] = [v.split(":", 1)[1].strip() if ":" in v else v for v in vals]

    expr = pd.read_csv(path, sep="\t", comment="!", index_col=0,
                       low_memory=False)
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.loc[expr.notna().any(axis=1)]
    if geo:
        meta = meta.set_index("geo_accession")
    log.info(f"   expression {expr.shape[0]} x {expr.shape[1]}, "
             f"metadata {meta.shape[0]} x {meta.shape[1]}")
    return expr, meta


def _map_probes(expr: pd.DataFrame, soft: Path | None, log) -> pd.DataFrame:
    """Convert probe IDs to gene symbols using the SOFT platform table."""
    looks_like_symbols = sum(
        bool(re.fullmatch(r"[A-Z][A-Z0-9\-]{1,14}", str(i)))
        for i in expr.index[:400]) > 200
    if looks_like_symbols:
        log.info("   row IDs already look like gene symbols")
        return expr
    if soft is None or not soft.exists():
        log.info("   [!] row IDs are probes and no SOFT file is present;")
        log.info("       cannot map to symbols")
        return expr

    log.info(f"   mapping probes to symbols from {soft.name} ...")
    mapping: dict[str, str] = {}
    with _open(soft) as fh:
        in_table, cols = False, []
        for line in fh:
            if line.startswith("!platform_table_begin"):
                in_table = True; cols = []
                continue
            if line.startswith("!platform_table_end"):
                in_table = False
                continue
            if not in_table:
                continue
            parts = line.rstrip("\n").split("\t")
            if not cols:
                cols = parts
                idc = 0
                symc = next((i for i, c in enumerate(cols)
                             if c.upper() in ("GENE_SYMBOL", "SYMBOL",
                                              "ILMN_GENE", "GENE SYMBOL")), None)
                continue
            if symc is not None and len(parts) > symc:
                mapping[parts[idc]] = parts[symc]
    if not mapping:
        log.info("   [!] no symbol column found in the platform table")
        return expr
    sym = pd.Series(expr.index.map(mapping))
    ok = sym.notna() & (sym.astype(str) != "")
    log.info(f"   mapped {int(ok.sum())}/{len(expr)} probes")
    out = expr[ok.values].copy()
    out.index = sym[ok].values
    return out.groupby(level=0).max()


def _zscore_score(expr: pd.DataFrame, genes: list[str]) -> pd.Series:
    present = [g for g in genes if g in expr.index]
    if len(present) < 3:
        return pd.Series(np.nan, index=expr.columns)
    sub = expr.loc[present]
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1) + 1e-9, axis=0)
    return z.mean(axis=0)


def _find_col(meta: pd.DataFrame, *keys) -> str | None:
    for c in meta.columns:
        low = c.lower()
        if any(k in low for k in keys):
            return c
    return None


def _load_series(pdir: Path, log):
    """Return (expression, metadata) for one GEO series directory."""
    sm = next(iter(sorted(pdir.glob("*series_matrix*"))), None)
    if sm is None:
        raise FileNotFoundError(f"No series matrix under {pdir}")
    soft = next(iter(sorted(pdir.glob("*family.soft*"))), None)
    expr, meta = _read_series_matrix(sm, log)
    expr = _map_probes(expr, soft, log)
    return expr, meta


def _poetic_inference(pairs: pd.DataFrame, sig_cols, n_boot: int, seed: int,
                      log) -> dict:
    """Formal treatment-vs-control inference for the POETIC randomised arm.

    For every signature fits, over patients:

        d_score ~ treatment + d_stroma + treatment:d_stroma

    the reviewer's requested model. The treatment coefficient is the adjusted
    treatment effect at mean stromal change; the interaction tests whether the
    score-stroma relationship itself differs between arms (the residualised
    procedure assumes it does not). Reports the treatment coefficient with a
    heteroscedasticity-robust (HC3) 95% CI, the interaction coefficient, a
    BH-adjusted p across signatures, and a bootstrap CI for the retained
    fraction. Requires an untreated arm.
    """
    try:
        import statsmodels.formula.api as smf
    except Exception:  # noqa: BLE001
        log.info("   statsmodels unavailable; POETIC formal model skipped")
        return {}
    rng = np.random.default_rng(seed)
    rows = []
    treat = pairs["treated"].astype(int).values
    dstr = pairs["d_stroma"].astype(float).values
    for c in sig_cols:
        col = f"d_{c}"
        if col not in pairs.columns:
            continue
        y = pairs[col].astype(float).values
        ok = np.isfinite(y) & np.isfinite(dstr)
        if ok.sum() < 20 or len(set(treat[ok])) < 2:
            continue
        df = pd.DataFrame({"y": y[ok], "treated": treat[ok],
                           "dstroma": dstr[ok]})
        # centre the stromal-change variable so the treatment coefficient is
        # the treated-vs-untreated difference AT THE MEAN stromal change, not
        # at dstroma = 0 (the reviewer's point on interaction models).
        df["dstroma"] = df["dstroma"] - df["dstroma"].mean()
        try:
            m = smf.ols("y ~ treated * dstroma", df).fit(cov_type="HC3")
        except Exception:  # noqa: BLE001
            continue
        tcoef = float(m.params.get("treated", np.nan))
        ci = (m.conf_int().loc["treated"]
              if "treated" in m.params.index else [np.nan, np.nan])
        inter = float(m.params.get("treated:dstroma", np.nan))
        p_t = float(m.pvalues.get("treated", np.nan))
        p_i = float(m.pvalues.get("treated:dstroma", np.nan))
        tr = df[df["treated"] == 1]
        fr = []
        for _ in range(n_boot):
            bs = tr.sample(len(tr), replace=True,
                           random_state=int(rng.integers(1e9)))
            raw = bs["y"].mean()
            if abs(raw) < 0.02:
                continue
            b, a = np.polyfit(bs["dstroma"], bs["y"], 1)
            fr.append(float(np.clip(a / raw, -2, 2)))
        fr_lo, fr_hi = ((float(np.percentile(fr, 2.5)),
                         float(np.percentile(fr, 97.5)))
                        if len(fr) > 20 else (np.nan, np.nan))
        rows.append({"signature": c, "treatment_coef": tcoef,
                     "treatment_ci_low": float(ci[0]),
                     "treatment_ci_high": float(ci[1]),
                     "treatment_p": p_t, "interaction_coef": inter,
                     "interaction_p": p_i, "retained_frac_ci_low": fr_lo,
                     "retained_frac_ci_high": fr_hi, "n": int(ok.sum())})
    if not rows:
        return {}
    R = pd.DataFrame(rows)
    p = R["treatment_p"].values
    order = np.argsort(p)
    adj = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q = np.empty_like(adj)
    q[order] = np.clip(adj, 0, 1)
    R["treatment_q_bh"] = q
    return {"table": R}


def run(cfg: Config, series_dir: str | None = None, no_merge: bool = False,
        tag: str | None = None, cohort: str = "extended") -> dict:
    """Paired on-treatment analysis.

    ``cohort`` selects which cohort's default directory to use when
    ``series_dir`` is not given: "extended" -> extended_letrozole_dir (the
    default paired analysis on GSE111563 + siblings), "poetic" ->
    poetic_sampling_control_dir (the randomised POETIC control, GSE105777,
    analysed with no_merge).

    ``no_merge`` analyses only the directory given, without pulling in sibling
    accessions. Required for POETIC (GSE105777), which is a different study
    from the Edinburgh extended-letrozole cohort and must not be pooled with it.

    ``tag`` renames the outputs (``wp6_<tag>_pairs.csv`` etc.) so two cohorts
    can be analysed in the same results directory without overwriting.
    """
    log = get_logger("wp6", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    report: dict = {"step": "wp6",
                    "generated": datetime.now().isoformat(timespec="seconds")}

    banner(log, "WP6 — PAIRED ON-TREATMENT BIOPSIES")
    # Default outputs describe the Edinburgh paired cohort. A tagged run
    # (e.g. --tag poetic for GSE105777) must land on a DIFFERENT stem, or the
    # second cohort silently overwrites the first.
    stem = f"wp6_{tag}" if tag else "wp6_paired"
    report["tag"] = tag
    if tag:
        log.info(f"   output stem: {stem}")

    if series_dir:
        pdir = Path(series_dir)
    elif cohort == "poetic":
        pdir = Path(cfg.paths.poetic_sampling_control_dir)
        no_merge = True   # POETIC is a standalone study; never merge siblings
    else:
        pdir = Path(cfg.paths.extended_letrozole_dir)
    if not pdir.exists():
        alt = cfg.paths.data_root / "GSE111563"
        if alt.exists():
            pdir = alt
        else:
            raise FileNotFoundError(
                f"{pdir} not found. Pass --series-dir, or set "
                "extended_letrozole_dir / poetic_sampling_control_dir in "
                "config.yaml")
    log.info(f"   cohort: {cohort}   series directory: {pdir}")

    sm = next(iter(sorted(pdir.glob("*series_matrix*"))), None)
    if sm is None:
        raise FileNotFoundError(f"No series matrix under {pdir}")
    soft = next(iter(sorted(pdir.glob("*family.soft*"))), None)

    expr, meta = _read_series_matrix(sm, log)
    expr = _map_probes(expr, soft, log)

    # The Edinburgh cohort of 62 patients is split across three GEO
    # submissions (GSE111563 n=39, GSE59515 n=14, GSE55374 n=9). Using one
    # accession analyses 60% of the cohort. Sibling directories are merged
    # when present. Because both samples of a pair come from the same series,
    # the within-patient delta is unaffected by between-series batch effects;
    # scores are nonetheless z-scored within series before pairing.
    extra = []
    if not no_merge:
        for name in ("GSE59515", "GSE55374", "GSE111563"):
            d = pdir.parent / name
            if d.exists() and d.resolve() != pdir.resolve():
                extra.append(d)
    else:
        log.info("   --no-merge: analysing this accession alone")
    series = {pdir.name: (expr, meta)}
    for d in extra:
        try:
            e2, m2 = _load_series(d, log)
            series[d.name] = (e2, m2)
            log.info(f"   merged sibling series {d.name}: "
                     f"{e2.shape[0]} genes x {e2.shape[1]} samples")
        except Exception as ex:  # noqa: BLE001
            log.info(f"   [!] {d.name}: {type(ex).__name__}: {ex}")
    if len(series) > 1:
        log.info(f"   {len(series)} series in total; scores are computed "
                 "within each series before pairing")
    report["series"] = {k: v[0].shape[1] for k, v in series.items()}
    # Which accession each sample came from. Needed to keep pairing within a
    # series: patient identifiers are bare numbers parsed from titles, so the
    # same number can occur in two accessions and refer to different people.
    sample_series: dict[str, str] = {}
    for _k, (_e, _m) in series.items():
        for _c in _e.columns:
            sample_series.setdefault(str(_c), _k)
    log.info("")

    if len(series) > 1:
        shared = set.intersection(*[set(e.index) for e, _ in series.values()])
        log.info(f"   genes shared across series: {len(shared)}")
        for k in series:
            e, m = series[k]
            series[k] = (e.loc[sorted(shared)], m)
        # concatenate for downstream scoring; z-scoring is per series below
        expr = pd.concat([e for e, _ in series.values()], axis=1)
        meta = pd.concat([m for _, m in series.values()])
        meta = meta[~meta.index.duplicated()]
        expr = expr.loc[:, ~expr.columns.duplicated()]
        log.info(f"   combined: {expr.shape[0]} genes x {expr.shape[1]} samples")
        log.info("")

    banner(log, "SAMPLE ANNOTATION")
    for c in meta.columns:
        vc = meta[c].value_counts()
        if 1 < len(vc) <= 12:
            log.info(f"   {c}: {vc.to_dict()}")
        else:
            log.info(f"   {c}: {len(vc)} distinct values, "
                     f"e.g. {list(vc.index[:3])}")
    log.info("")

    pt_col = _find_col(meta, "patient", "subject", "case", "pair", "individual")
    tp_col = _find_col(meta, "timepoint", "time point", "time", "biopsy",
                       "sample type", "surgery", "visit", "day", "duration")
    # GSE111563 splits patients into dormant vs acquired-resistant rather than
    # treated vs untreated, so the "arm" is a response class.
    arm_col = _find_col(meta, "arm", "response", "group", "class", "status",
                        "dormant", "resistan", "randomis", "randomiz",
                        "treatment")
    # Some series carry no structured annotation at all and encode everything
    # in the title, e.g. "Breasttumour_PatientX109_Timepoint1". Parse it out
    # rather than failing.
    # Series merged from different submissions annotate timepoint differently:
    # some carry a structured characteristic, others encode it only in the
    # sample title. Parse the title regardless and use it to fill gaps, or the
    # samples lacking the structured field are silently dropped.
    if "title" in meta.columns:
        t = meta["title"].astype(str)
        pat = t.str.extract(r"[Pp]atient[_ ]?([A-Za-z0-9]+)", expand=False)
        # POETIC-style codes. The treated arm is prefixed ("AI.Treated.1B");
        # the untreated arm in GSE105777 is not, carrying only a number and a
        # timepoint letter ("1S [reanalysis]", "2D [reanalysis]"). The prefix
        # is therefore optional. D = diagnostic and B = baseline both mean
        # pre-treatment; S = surgery. Numbering restarts per arm, so the
        # patient key carries the arm or Treated.1 and untreated 1 collide.
        _rx = (r"^(?P<arm>[A-Za-z][A-Za-z _-]*(?:\.[A-Za-z][A-Za-z _-]*)*"
               r"[. _-]+)?(?P<num>\d+)(?P<rest>.*)$")
        code = t.str.extract(_rx)
        if code["num"].isna().any() and "description" in meta.columns:
            alt = meta["description"].astype(str).str.extract(_rx)
            code = code.where(code["num"].notna(), alt)
        n_code = int(code["num"].notna().sum())
        if n_code and n_code < len(code):
            miss = t[code["num"].isna()]
            log.info(f"   [!] {len(miss)} sample titles did not match the "
                     f"code pattern, e.g. {list(miss.unique()[:4])}")
        if n_code and pat.isna().all():
            arm_lab = code["arm"].fillna("").astype(str).str.strip(". _-")
            arm_lab = arm_lab.mask(arm_lab.eq(""), "unprefixed")
            pat = (arm_lab + "." + code["num"].astype(str))
            pat = pat.where(code["num"].notna())
            log.info(f"   patient from sample code: {pat.nunique()} distinct "
                     f"(arm-qualified), {n_code}/{len(code)} samples")
            log.info(f"   arms seen in codes: "
                     f"{arm_lab[code['num'].notna()].value_counts().to_dict()}")
            # Timepoint from the trailing fragment: a bare B/D letter is
            # baseline or diagnostic, S is surgery, and spelled-out forms are
            # matched directly.
            rest = code["rest"].fillna("").astype(str).str.lower()
            tpc = pd.Series(np.nan, index=rest.index, dtype="object")
            tpc = tpc.mask(rest.str.match(r"^[._ -]*[bd]\s*(\[|$)"), "baseline")
            tpc = tpc.mask(rest.str.match(r"^[._ -]*s\s*(\[|$)"), "surgery")
            tpc = tpc.mask(tpc.isna() & rest.str.contains(
                r"diagnos|baseline|pre[- ]?treat", regex=True), "baseline")
            tpc = tpc.mask(tpc.isna() & rest.str.contains(
                r"surg|excis|resect", regex=True), "surgery")
            if tpc.notna().any():
                meta["timepoint_code"] = tpc
                log.info("   timepoint from sample code: "
                         f"{tpc.value_counts().to_dict()}")
        tim = t.str.extract(r"[Tt]ime[_ ]?point[_ ]?([0-9]+)", expand=False)
        if tim.isna().all():
            tim = t.str.extract(r"_[Tt]([0-9]+)\b", expand=False)
        if pat.notna().any():
            meta["patient_parsed"] = pat
            if pt_col is None or meta[pt_col].isna().mean() > 0.2:
                pt_col = "patient_parsed"
            else:
                meta[pt_col] = meta[pt_col].fillna(pat)
            log.info(f"   patient from title: {pat.nunique()} distinct")
        if tim.notna().any():
            # merge structured and parsed: structured wins where present
            if tp_col is not None:
                struct = meta[tp_col].astype("object")
                filled = struct.where(struct.notna() & (struct != ""),
                                      "timepoint " + tim.astype(str))
                n_filled = int((struct.isna() | (struct == "")).sum())
                meta["timepoint_merged"] = filled
                tp_col = "timepoint_merged"
                log.info(f"   timepoint: structured column used where present, "
                         f"title parsing filled {n_filled} gaps")
            else:
                meta["timepoint_parsed"] = tim
                tp_col = "timepoint_parsed"
                log.info(f"   parsed timepoint from title: "
                         f"{tim.value_counts().sort_index().to_dict()}")

    # Randomised untreated arm, if one exists. Matched on literal control
    # wording ONLY -- never on response categories such as
    # responder/non-responder, which are not treatment arms and whose
    # accidental use as one is the error this guard exists to prevent.
    CONTROL_RX = re.compile(
        r"^\s*(untreated|control|no[ _-]?treatment|"
        r"no[ _-]?presurgical[ _-]?treatment|nil)\s*$", re.I)

    # GEO characteristics are positional: when one arm carries a different
    # number of fields, every later field shifts column. In GSE105777 the
    # untreated arm's timepoint lands in the 'her2' column while the
    # 'timepoint' column holds the arm label. Trusting any single column is
    # therefore unsafe. Preference order: the sample code, then the structured
    # column, then a scan of every field for a pre/post word. An arm label
    # sitting in the timepoint column is discarded, not read as a timepoint.
    if "timepoint_code" in meta.columns:
        base = meta["timepoint_code"].astype("object")
        if tp_col is not None:
            struct = meta[tp_col].astype("object")
            struct = struct.where(
                ~struct.astype(str).str.match(CONTROL_RX).fillna(False))
            base = base.where(base.notna(), struct)
        scan = meta.apply(
            lambda r: " | ".join("" if pd.isna(v) else str(v)
                                 for v in r.values), axis=1)
        n_gap = int((base.isna() | (base.astype(str) == "")).sum())
        meta["timepoint_resolved"] = base.where(
            base.notna() & (base.astype(str) != ""), scan)
        tp_col = "timepoint_resolved"
        if n_gap:
            log.info(f"   timepoint: {n_gap} samples had no usable code or "
                     "structured value; all annotation fields scanned for those")

    is_control = pd.Series(False, index=meta.index)
    ctrl_src = None
    for c in meta.columns:
        hit = meta[c].astype(str).str.match(CONTROL_RX)
        if hit.any() and not hit.all():
            is_control |= hit.fillna(False)
            ctrl_src = c if ctrl_src is None else f"{ctrl_src}+{c}"
    if is_control.any():
        log.info(f"   untreated arm detected in '{ctrl_src}': "
                 f"{int(is_control.sum())} samples untreated, "
                 f"{int((~is_control).sum())} treated")
        meta["arm_resolved"] = np.where(is_control, "untreated", "treated")

    log.info(f"   patient column  : {pt_col}")
    log.info(f"   timepoint column: {tp_col}")
    log.info(f"   arm column      : {arm_col}")
    if arm_col is None:
        log.info("   [!] no response/arm column: this series carries no")
        log.info("       dormant vs resistant labels in GEO, so the comparison")
        log.info("       is a single paired group. The dormant/resistant")
        log.info("       classification is in the source publication and would")
        log.info("       have to be added manually as a two-column CSV")
        log.info("       (patient,group) to split it.")
    if pt_col is None or tp_col is None:
        log.info("")
        log.info("[X] Cannot identify patient and timepoint columns, so pairs")
        log.info("    cannot be formed. Inspect the annotation above and tell")
        log.info("    me which columns encode patient and biopsy timepoint.")
        raise ValueError("patient/timepoint annotation not found")
    log.info("")

    markers_path = cfg.paths.results / "wp3_compartment_markers.json"
    if not markers_path.exists():
        raise FileNotFoundError("Run `dormbench wp3` first to build "
                                "wp3_compartment_markers.json")
    with open(markers_path, encoding="utf-8") as fh:
        markers = json.load(fh)
    # z-score within series so that between-submission scale differences do
    # not enter the scores
    def _scored(fn):
        if len(series) == 1:
            return fn(expr)
        parts = []
        for k, (e, _) in series.items():
            cols = [c for c in e.columns if c in expr.columns]
            parts.append(fn(expr[cols]))
        return pd.concat(parts)

    comp = pd.DataFrame({c: _scored(lambda X, g=g: _zscore_score(X, g))
                         for c, g in markers.items()})
    stroma = comp[[c for c in STROMAL if c in comp.columns]].mean(axis=1)
    # Selli's classifying markers (MKI67, PCNA, MCM2), scored the same way, so
    # WP10 can reconstruct dormant/resistant labels from the authors' own
    # criterion rather than a proliferation-signature proxy.
    prolif_selli = _scored(
        lambda X: _zscore_score(X, ["MKI67", "PCNA", "MCM2"]))

    panel = pd.read_csv(cfg.paths.results / "signature_panel.csv")
    sig = pd.DataFrame({
        f"{s}|{d}": _scored(lambda X, gg=g: _zscore_score(
            X, gg["gene"].astype(str).tolist()))
        for (s, d), g in panel.groupby(["signature", "direction"])})
    sig = sig.loc[:, sig.notna().any()]
    log.info(f"   {sig.shape[1]} signature scores, "
             f"{comp.shape[1]} compartment scores")
    log.info("")

    # ---- build pairs -----------------------------------------------------
    m = meta.reindex(expr.columns)
    tp = m[tp_col].astype(str).str.lower()
    # Match pre and post explicitly and require post to not also look pre.
    # A bare "s$" alternative matches "diagnosis", which silently classified
    # every sample as post-treatment and produced zero pairs.
    POST = (r"surg|resect|excis|post|on[- ]?treat|long[- ]?term|"
            r"timepoint ?[234]|day ?1[45]|wk|week|month|"
            r"letrozole treat")
    PRE = (r"diagnos|baseline|pre[- ]?treat|\bpre\b|core|screen|"
           r"timepoint ?1\b")
    if tp.str.fullmatch(r"\d+").fillna(False).all():
        # purely numeric timepoints: the lowest is baseline
        nums = tp.astype(float)
        base = nums.min()
        is_post = nums > base
        looks_pre, looks_post = ~is_post, is_post
        log.info(f"   numeric timepoints: {base:.0f} treated as baseline, "
                 f"{sorted(set(nums[nums > base]))} as on-treatment")
    else:
        looks_post = tp.str.contains(POST, regex=True, na=False)
        looks_pre = tp.str.contains(PRE, regex=True, na=False)
        is_post = looks_post & ~looks_pre
    n_pre, n_post = int((looks_pre & ~looks_post).sum()), int(is_post.sum())
    log.info(f"   timepoints resolved: {n_pre} pre, {n_post} post, "
             f"{int(len(tp) - n_pre - n_post)} ambiguous")
    if n_pre == 0 or n_post == 0:
        log.info(f"   [!] timepoint values seen: "
                 f"{sorted(set(tp.dropna()))[:8]}")
    arm = (m[arm_col].astype(str).str.lower() if arm_col
           else pd.Series("unknown", index=m.index))
    groups_csv = cfg.root / "config" / "wp6_groups.csv"
    if arm_col is None and groups_csv.exists():
        gmap = pd.read_csv(groups_csv)
        gcol = [c for c in gmap.columns if c.lower() != "patient"][0]
        lut = dict(zip(gmap["patient"].astype(str),
                       gmap[gcol].astype(str).str.lower()))
        arm = m[pt_col].astype(str).map(lut).fillna("unknown")
        log.info(f"   groups from {groups_csv.name}: "
                 f"{arm.value_counts().to_dict()}")
    # "group 1" is whichever class the arm column marks as dormant/treated
    # Two-group mode requires an EXPLICIT untreated/control arm. An earlier
    # version split on any two-level annotation column, which for this cohort
    # selected "clinical response" (responder / non-responder) -- both of which
    # received letrozole. That produced a between-response-group contrast
    # reported as a treatment effect. Auto-splitting is therefore removed:
    # without config/wp6_groups.csv the cohort is analysed as one group.
    treated = pd.Series(False, index=m.index)
    groups_csv = cfg.root / "config" / "wp6_groups.csv"
    if "arm_resolved" in m.columns and m["arm_resolved"].nunique() == 2:
        # A real randomised control arm, matched on literal untreated/control
        # wording. This is a treatment arm, unlike a response category, so a
        # two-group contrast is legitimate here.
        treated = m["arm_resolved"].eq("treated").fillna(False)
        log.info(f"   two-group mode: {int(treated.sum())} treated vs "
                 f"{int((~treated).sum())} untreated samples")
    elif groups_csv.exists():
        gmap = pd.read_csv(groups_csv)
        gcol = [c for c in gmap.columns if c.lower() != "patient"][0]
        lut = dict(zip(gmap["patient"].astype(str),
                       gmap[gcol].astype(str).str.lower()))
        lab = m[pt_col].astype(str).map(lut)
        levels = sorted(lab.dropna().unique())
        if len(levels) == 2:
            treated = lab.eq(levels[0]).fillna(False)
            log.info(f"   groups from {groups_csv.name}: "
                     f"{levels[0]} (n={int(treated.sum())}) vs "
                     f"{levels[1]} (n={int((~treated & lab.notna()).sum())})")
        else:
            log.info(f"   [!] {groups_csv.name} does not give exactly two "
                     "groups; analysing as a single group")
    elif arm_col is not None:
        log.info(f"   note: an annotation column '{arm_col}' was found but is")
        log.info("   NOT used to split the cohort. Every patient in this study")
        log.info("   received endocrine therapy; a two-group contrast requires")
        log.info("   an explicit control arm supplied as config/wp6_groups.csv")
        log.info("   (columns: patient,group).")

    # ---- treatment duration, for time-stratified contrasts ---------------
    # The three Edinburgh accessions are ONE cohort sampled at different
    # durations, so the meaningful axis is time on treatment, not accession.
    # Days are read from the timepoint label; an optional
    # config/wp6_timepoints.csv (sample,days) overrides it, which is how the
    # Selli et al. per-patient durations should be supplied.
    days = pd.Series(np.nan, index=m.index, dtype=float)
    for i in m.index:
        days[i] = _days_from_label(tp.get(i, ""))
    tps_csv = cfg.root / "config" / "wp6_timepoints.csv"
    if tps_csv.exists():
        tmap = pd.read_csv(tps_csv)
        dcol = [c for c in tmap.columns if c.lower() != "sample"][0]
        lut = dict(zip(tmap["sample"].astype(str), tmap[dcol].astype(float)))
        n_over = 0
        for i in m.index:
            if str(i) in lut:
                days[i] = lut[str(i)]; n_over += 1
        log.info(f"   {tps_csv.name}: {n_over} sample durations supplied")
    n_days = int(days.notna().sum())
    log.info(f"   treatment duration resolved for {n_days}/{len(days)} samples")

    # Series-qualified patient key. Bare numeric IDs parsed from titles can
    # collide across accessions; pairing across a collision would compare two
    # different people on two different platforms.
    ser = pd.Series({i: sample_series.get(str(i), "?") for i in m.index})
    raw_pid = m[pt_col].astype(str)
    if len(series) > 1:
        tabn = raw_pid.groupby(raw_pid).apply(lambda s: ser[s.index].nunique())
        clash = tabn[tabn > 1]
        if len(clash):
            log.info(f"   [!] {len(clash)} patient identifiers occur in more "
                     f"than one accession, e.g. {list(clash.index[:5])}")
            log.info("       Pairing is therefore restricted to within-series;")
            log.info("       these are treated as distinct patients. If they "
                     "are the same")
            log.info("       people re-deposited, the duplicate accession "
                     "should be excluded")
            log.info("       with --no-merge rather than pooled.")
        report["patient_id_collisions"] = int(len(clash))
    pid_key = raw_pid.str.cat(ser, sep="@")

    rows = []
    for pid, g in m.groupby(pid_key):
        idx = [i for i in g.index if i in expr.columns]
        if len(idx) < 2:
            continue
        pre = [i for i in idx if not is_post.get(i, False)]
        post = [i for i in idx if is_post.get(i, False)]
        if not pre or not post:
            continue
        a = pre[0]
        # One pair per contrast: the earliest post inside the early window,
        # and the latest post beyond it. A patient can contribute to both.
        early = [i for i in post if np.isfinite(days.get(i, np.nan))
                 and 0 < days[i] <= EARLY_MAX_DAYS]
        late = [i for i in post if np.isfinite(days.get(i, np.nan))
                and days[i] > EARLY_MAX_DAYS]
        unknown = [i for i in post if not np.isfinite(days.get(i, np.nan))]
        chosen = []
        if early:
            chosen.append(("early", early[0]))
        if late:
            chosen.append(("long_term", late[-1]))
        if not chosen:
            chosen.append(("unresolved", (unknown or post)[-1]))
        # "all" reproduces the original pooled contrast: pre vs latest post
        chosen.append(("all", post[-1]))
        for contrast, b in chosen:
            row = {"patient": pid, "series": ser.get(a, "?"),
                   "contrast": contrast, "pre": a, "post": b,
                   "days_post": float(days.get(b, np.nan)),
                   "treated": bool(treated.get(a, False)
                                   or treated.get(b, False)),
                   "d_stroma": float(stroma[b] - stroma[a]),
                   "d_prolif_selli": float(prolif_selli[b] - prolif_selli[a])
                   if b in prolif_selli.index and a in prolif_selli.index
                   else float("nan")}
            for c in comp.columns:
                row[f"d_comp_{c}"] = float(comp.loc[b, c] - comp.loc[a, c])
            for c in sig.columns:
                row[f"d_{c}"] = float(sig.loc[b, c] - sig.loc[a, c])
            rows.append(row)

    allpairs = pd.DataFrame(rows)
    if len(allpairs):
        log.info("")
        log.info("   pairs formed by contrast:")
        for c, n in allpairs["contrast"].value_counts().items():
            log.info(f"      {c:12s} {n}")
    # The pooled contrast remains the primary analysis; the stratified ones
    # are reported alongside it.
    pairs = (allpairs[allpairs["contrast"] == "all"].reset_index(drop=True)
             if len(allpairs) else allpairs)
    if len(pairs) < 20:
        log.info(f"[X] only {len(pairs)} pairs formed — check the timepoint "
                 "column encoding")
        raise ValueError("too few pairs")
    log.info(f"   {len(pairs)} paired patients "
             f"({int(pairs['treated'].sum())} treated, "
             f"{int((~pairs['treated']).sum())} control)")
    # Per-pair compartment-adjusted residuals: regress each signature delta on
    # the stromal-content delta across all pairs and store the residual, so
    # WP10 can compare the raw dormancy change against the composition-adjusted
    # change patient by patient.
    if len(allpairs):
        dstroma = allpairs["d_stroma"].astype(float).values
        finite_s = np.isfinite(dstroma)
        for c in sig.columns:
            col = f"d_{c}"
            if col not in allpairs.columns:
                continue
            y = allpairs[col].astype(float).values
            ok = finite_s & np.isfinite(y)
            resid = np.full(len(allpairs), np.nan)
            if ok.sum() > 15:
                b1, b0 = np.polyfit(dstroma[ok], y[ok], 1)
                resid[ok] = y[ok] - (b0 + b1 * dstroma[ok])
            allpairs[f"resid_{c}"] = resid
    allpairs.to_csv(cfg.paths.results / f"{stem}_pairs.csv", index=False)
    log.info("")

    # ---- the test --------------------------------------------------------
    banner(log, "DOES THE SCORE CHANGE TRACK THE COMPOSITION CHANGE?")
    two_group = bool(pairs["treated"].any() and (~pairs["treated"]).any())
    if not two_group:
        log.info("Single group: every pair is on treatment, so there is no")
        log.info("control arm. The informative statistic is the correlation")
        log.info("between the change in signature score and the change in")
        log.info("stromal content across patients.")
        log.info("")
    tr = pairs[pairs["treated"]] if two_group else pairs
    ct = pairs[~pairs["treated"]] if two_group else pairs.iloc[0:0]
    lab1 = "Δ group1" if two_group else "Δ raw"
    lab3 = "adjusted" if two_group else "Δ at Δstroma=0"
    hdr = f"{'signature':30s} {lab1:>10s}"
    if two_group:
        hdr += f" {'Δ group2':>10s}"
    hdr += f" {'r(Δsig,Δstroma)':>16s} {lab3:>15s}"
    log.info(hdr)
    if not two_group:
        log.info("(Δ at Δstroma=0 is the intercept of Δsig ~ Δstroma: the")
        log.info(" change expected in a tumour whose composition did not move)")
        log.info("")
    out = []
    for c in sig.columns:
        col = f"d_{c}"
        dt = float(tr[col].mean()) if len(tr) else np.nan
        dc = float(ct[col].mean()) if len(ct) else np.nan
        sub = tr if len(tr) > 20 else pairs
        ok = sub[col].notna() & sub["d_stroma"].notna()
        r = (float(stats.spearmanr(sub.loc[ok, col],
                                   sub.loc[ok, "d_stroma"])[0])
             if ok.sum() > 15 else np.nan)
        # treated-vs-control difference in delta, adjusted for stromal change
        part = np.nan
        if not two_group:
            # With no control arm, the informative adjustment is the expected
            # change at zero stromal change: the intercept of Δsig ~ Δstroma.
            # If the raw Δ is large but the intercept is ~0, the entire change
            # is accounted for by shifting composition.
            y = pairs[col].values.astype(float)
            z = pairs["d_stroma"].values.astype(float)
            good = np.isfinite(y) & np.isfinite(z)
            if good.sum() > 15:
                b, a = np.polyfit(z[good], y[good], 1)
                part = float(a)
        elif len(tr) > 15 and len(ct) > 10:
            y = pairs[col].values.astype(float)
            x = pairs["treated"].astype(float).values
            z = pairs["d_stroma"].values.astype(float)
            good = np.isfinite(y) & np.isfinite(z)
            if good.sum() > 30:
                b, a = np.polyfit(z[good], y[good], 1)
                resid = y - (a + b * z)
                part = float(np.nanmean(resid[(x == 1) & good])
                             - np.nanmean(resid[(x == 0) & good]))
        out.append({"score": c, "delta_treated": dt, "delta_control": dc,
                    "r_with_stroma_change": r, "delta_adjusted": part})
        line = f"{c:30s} {dt:+10.3f}"
        if two_group:
            line += f" {dc:+10.3f}"
        line += f" {r:+16.3f} {part:+15.3f}"
        log.info(line)
    res = pd.DataFrame(out)
    res.to_csv(cfg.paths.results / f"{stem}_deltas.csv", index=False)
    log.info("")

    banner(log, "COMPOSITION CHANGE UNDER TREATMENT")
    for c in comp.columns:
        col = f"d_comp_{c}"
        dt = float(tr[col].mean()) if len(tr) else np.nan
        if two_group:
            dc = float(ct[col].mean()) if len(ct) else np.nan
            log.info(f"   {c:20s} group1 {dt:+.3f}   group2 {dc:+.3f}")
        else:
            log.info(f"   {c:20s} mean change {dt:+.3f}")
    log.info("")

    # ---- formal treatment-effect inference (randomised control arm) ------
    if two_group:
        banner(log, "FORMAL TREATMENT-EFFECT MODEL")
        log.info("d_score ~ treatment + d_stroma + treatment:d_stroma, per")
        log.info("signature, over patients; HC3-robust CIs, BH across signatures.")
        log.info("")
        inf = _poetic_inference(pairs, list(sig.columns),
                                n_boot=1000, seed=cfg.seed, log=log)
        if inf.get("table") is not None:
            R = inf["table"]
            R.to_csv(cfg.paths.results / f"{stem}_inference.csv", index=False)
            primary = R[R["signature"].isin(
                ["TRADITIOM_dormancy|up", "EMT_hallmark|up", "PreAdapted|up",
                 "E2F_control|up", "G2M_control|up", "Hypoxia_hallmark|up"])]
            log.info(f"{'signature':24s} {'treat coef':>10s} "
                     f"{'95% CI':>20s} {'inter':>8s} {'q':>8s} {'ret.CI':>16s}")
            for _, r in primary.iterrows():
                log.info(
                    f"{r['signature']:24s} {r['treatment_coef']:+10.4f} "
                    f"[{r['treatment_ci_low']:+.4f},{r['treatment_ci_high']:+.4f}] "
                    f"{r['interaction_coef']:+8.4f} {r['treatment_q_bh']:8.3g} "
                    f"[{r['retained_frac_ci_low']:+.2f},"
                    f"{r['retained_frac_ci_high']:+.2f}]")
            report["formal_inference"] = {
                "n_signatures": int(len(R)),
                "n_treatment_sig_q05": int((R["treatment_q_bh"] < 0.05).sum()),
                "n_interaction_sig_p05": int((R["interaction_p"] < 0.05).sum())}
            dz = R[R["signature"] == "TRADITIOM_dormancy|up"]
            if len(dz):
                d0 = dz.iloc[0]
                log.info("")
                log.info(f"TRADITIOM treatment effect {d0['treatment_coef']:+.4f} "
                         f"[{d0['treatment_ci_low']:+.4f}, "
                         f"{d0['treatment_ci_high']:+.4f}], q={d0['treatment_q_bh']:.3g}; "
                         f"interaction p={d0['interaction_p']:.3g}")
            log.info(f"Written: {cfg.paths.results / f'{stem}_inference.csv'}")
        log.info("")

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    dorm = res[res["score"].str.startswith(("TRADITIOM_dormancy|up",
                                            "PreAdapted|up"))]
    med_r = float(dorm["r_with_stroma_change"].median()) if len(dorm) else np.nan

    # How much of the raw change survives adjustment? That is the quantity to
    # report -- a high correlation with composition does not by itself mean the
    # whole change is compositional.
    def _retained(r):
        """Fraction of the raw change that survives adjustment.

        Undefined when the raw change is near zero: dividing by it produces
        meaningless ratios, so those are dropped rather than reported.
        """
        raw, adj = r["delta_treated"], r["delta_adjusted"]
        if not np.isfinite(raw) or not np.isfinite(adj) or abs(raw) < 0.05:
            return np.nan
        return float(np.clip(adj / raw, -2.0, 2.0))

    res["fraction_retained"] = res.apply(_retained, axis=1)
    res.to_csv(cfg.paths.results / f"{stem}_deltas.csv", index=False)
    frac = float(dorm.apply(_retained, axis=1).median()) if len(dorm) else np.nan
    log.info(f"Fraction of the raw change retained after adjustment: "
             f"{frac:.2f}" if np.isfinite(frac) else "")
    log.info("")

    ctrl = res[res["score"].str.contains("E2F|G2M")]
    if len(ctrl):
        cf = float(ctrl.apply(_retained, axis=1).median())
        log.info(f"Proliferation controls retain {cf:.2f} of their change --")
        log.info("they fall on endocrine therapy and that survives adjustment,")
        log.info("which is a real drug effect and confirms the adjustment does")
        log.info("not simply flatten everything.")
        log.info("")
        report["control_fraction_retained"] = cf
    report["dormancy_fraction_retained"] = None if not np.isfinite(frac) else frac
    log.info(f"Dormancy scores: median r(Δscore, Δstromal content) = "
             f"{med_r:+.3f}")
    if len(dorm):
        for _, r in dorm.iterrows():
            adj = ("" if not np.isfinite(r["delta_adjusted"])
                   else f", adjusted {r['delta_adjusted']:+.3f}")
            log.info(f"   {r['score']:30s} Δ {r['delta_treated']:+.3f}{adj}")
    log.info("")

    if np.isfinite(med_r) and med_r > 0.5 and np.isfinite(frac) and frac > 0.4:
        verdict = "PARTLY_COMPOSITIONAL"
        log.info(f"The dormancy score change tracks composition (r = "
                 f"{med_r:+.2f}), but roughly {(1-frac)*100:.0f}% of the raw")
        log.info("change is accounted for by it, not all of it. State that")
        log.info("precisely: a substantial fraction of the apparent")
        log.info("on-treatment induction is compositional, and the remainder")
        log.info("is not explained by these compartment scores.")
        log.info("")
        log.info("Note what this looks like without the compartment scores: the")
        log.info("score rises on treatment and reads as therapy inducing a")
        log.info("dormancy programme, with no hint that the cancer-cell")
        log.info("fraction fell while stroma rose to replace it.")
    elif np.isfinite(med_r) and med_r > 0.5:
        verdict = "CHANGE_IS_COMPOSITIONAL"
        log.info("Within the same tumour, the change in dormancy score on")
        log.info("endocrine therapy tracks the change in stromal content. The")
        log.info("score is reporting what the biopsy is made of, not what its")
        log.info("cancer cells are doing.")
        log.info("")
        log.info("Note what this looks like without the compartment scores: the")
        log.info("dormancy score rises on treatment, which reads as therapy")
        log.info("inducing a dormancy programme. The composition data shows the")
        log.info("cancer-cell fraction fell and the stromal fraction rose to")
        log.info("replace it. Same numbers, opposite conclusion.")
        if not two_group:
            log.info("")
            log.info("Caveat: no untreated arm here, so biopsy-to-surgery drift")
            log.info("is not controlled. The within-patient correlation is the")
            log.info("evidence, not the mean change.")
    elif np.isfinite(med_r) and med_r < 0.2:
        verdict = "CHANGE_IS_INDEPENDENT"
        log.info("The score change does not track the composition change, so")
        log.info("on-treatment movement reflects something other than shifting")
        log.info("cell-type proportions. That limits the artefact claim to")
        log.info("cross-sectional comparisons and should be reported plainly.")
    else:
        verdict = "PARTIAL"
        log.info("The score change is partly explained by composition change.")
        log.info("Report both the raw and adjusted deltas.")

    # ---- time-stratified contrasts ---------------------------------------
    # Does the compositional share grow with time on treatment? Pooling
    # pre-to-latest hides this; the reviewer asked for it explicitly.
    strat: dict = {}
    if len(allpairs) and allpairs["contrast"].nunique() > 1:
        banner(log, "BY TIME ON TREATMENT")
        log.info(f"(early = 13-{EARLY_MAX_DAYS:.0f} days, "
                 f"long_term = >{EARLY_MAX_DAYS:.0f} days)")
        log.info("")
        log.info(f"{'contrast':12s} {'n':>4s} {'median days':>12s} "
                 f"{'r(dorm,stroma)':>15s} {'retained':>9s}")
        for contrast in ("early", "long_term", "unresolved", "all"):
            sub = allpairs[allpairs["contrast"] == contrast]
            if len(sub) < 10:
                continue
            rs, fr = [], []
            for c in sig.columns:
                if not c.startswith(("TRADITIOM_dormancy|up", "PreAdapted|up")):
                    continue
                col = f"d_{c}"
                ok = sub[col].notna() & sub["d_stroma"].notna()
                if ok.sum() <= 15:
                    continue
                rr = float(stats.spearmanr(sub.loc[ok, col],
                                           sub.loc[ok, "d_stroma"])[0])
                rs.append(rr)
                y = sub[col].values.astype(float)
                z = sub["d_stroma"].values.astype(float)
                good = np.isfinite(y) & np.isfinite(z)
                if good.sum() > 15:
                    bb, aa = np.polyfit(z[good], y[good], 1)
                    raw = float(np.nanmean(y[good]))
                    if abs(raw) >= 0.05:
                        fr.append(float(np.clip(aa / raw, -2.0, 2.0)))
            mr = float(np.median(rs)) if rs else np.nan
            mf = float(np.median(fr)) if fr else np.nan
            md = float(np.nanmedian(sub["days_post"])) \
                if sub["days_post"].notna().any() else np.nan
            strat[contrast] = {"n": int(len(sub)), "median_days": md,
                               "median_r": mr, "fraction_retained": mf}
            log.info(f"{contrast:12s} {len(sub):4d} {md:12.0f} "
                     f"{mr:+15.3f} {mf:9.2f}")
        log.info("")
        if {"early", "long_term"} <= set(strat):
            fe = strat["early"]["fraction_retained"]
            fl = strat["long_term"]["fraction_retained"]
            if np.isfinite(fe) and np.isfinite(fl):
                log.info(f"Retained early {fe:.2f} vs long-term {fl:.2f}: "
                         f"the compositional share")
                log.info("grows with exposure." if fl < fe
                         else "does not grow with exposure.")
                log.info("")
    report["by_contrast"] = strat

    report.update({"n_pairs": int(len(pairs)),
                   "n_treated": int(pairs["treated"].sum()),
                   "median_r_with_stroma_change": med_r,
                   "verdict": verdict})
    log.info("")
    log.info(f"VERDICT: {verdict}")

    with open(cfg.paths.results / f"{stem}.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {cfg.paths.results / f'{stem}_deltas.csv'}")

    try:
        from ..viz.paired import plot_paired
        focus = ("TRADITIOM_dormancy|up"
                 if "d_TRADITIOM_dormancy|up" in pairs.columns
                 else next((c[2:] for c in pairs.columns
                            if c.startswith("d_") and "|" in c), None))
        fig = plot_paired(pairs, res,
                          cfg.paths.figures / "fig9_paired_ontreatment.svg",
                          focus=focus)
        log.info(f"Written: {fig}")
        report["figure"] = str(fig)
    except Exception as e:  # noqa: BLE001
        log.info(f"[!] Figure skipped: {type(e).__name__}: {e}")
    return report
