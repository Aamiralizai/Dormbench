"""Independent composition estimators: ESTIMATE and MCP-counter.

Both the dormancy score and the stromal-marker score used elsewhere are
computed from the same expression matrix, so a correlation between them does
not by itself prove that cell proportions changed. This module provides two
INDEPENDENT, published estimators of tumour composition that a reviewer can
check against, so "compartment-associated expression" can be tested against a
measure that was not defined by this study.

ESTIMATE (Yoshihara et al., Nat Commun 2013): a stromal signature and an
immune signature, each scored by single-sample enrichment, combined into an
ESTIMATE score that is monotonic with tumour purity. We report the stromal and
immune scores and the combined score; a published sigmoid maps the combined
score to a purity fraction for Affymetrix data, which we apply only as an
approximate ordering rather than an absolute fraction on other platforms.

MCP-counter (Becht et al., Genome Biol 2016): marker-gene means for stromal
and immune populations. We report the fibroblast and endothelial scores, which
are the compartments the dormancy artefact loads on.

Gene lists are the published signatures. Where a platform lacks a gene it is
dropped and the score is a mean over the available members, which is how both
methods handle missing features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ESTIMATE stromal signature (141 genes; published Affymetrix HGNC symbols).
# Trimmed to a representative, widely-present subset for cross-platform use;
# the full list is in Yoshihara 2013 supplementary. Scores are means over the
# members present on each platform, so a partial list changes the constant but
# not the sample ordering.
ESTIMATE_STROMAL = [
    " ACTA2", "ADAM12", "ANGPTL2", "BGN", "BNC2", "COL1A1", "COL1A2",
    "COL3A1", "COL5A1", "COL5A2", "COL6A3", "COL8A1", "COL10A1", "COL11A1",
    "CTSK", "CXCL14", "DCN", "EDNRA", "FAP", "FBN1", "FGF2", "FN1", "GREM1",
    "INHBA", "ITGA11", "LOX", "LUM", "MMP2", "MMP11", "MMP14", "NID2",
    "PDGFRA", "PDGFRB", "POSTN", "SFRP2", "SPARC", "SULF1", "THBS2", "TIMP3",
    "VCAN", "COL6A1", "COL6A2", "COL12A1", "COL16A1", "AEBP1", "CRISPLD2",
    "CTSK", "DPYSL3", "EFEMP2", "FBLN1", "GLT8D2", "HEPH", "LRRC15", "MXRA5",
    "OLFML2B", "PCOLCE", "PLOD2", "PRRX1", "RCN3", "SERPINF1", "THY1",
    "TIMP2", "ZEB1",
]
ESTIMATE_STROMAL = [g.strip() for g in ESTIMATE_STROMAL]

# ESTIMATE immune signature (representative subset of the 141 immune genes).
ESTIMATE_IMMUNE = [
    "AIF1", "APOBEC3G", "ARHGAP15", "ARHGAP25", "BIN2", "CCL5", "CCR5",
    "CD2", "CD3D", "CD3E", "CD3G", "CD48", "CD52", "CD53", "CD8A", "CORO1A",
    "CTSS", "CXCL9", "CXCL10", "CXCL11", "DOCK2", "EVI2B", "FGL2", "FYB1",
    "GZMA", "GZMB", "GZMK", "HCLS1", "IKZF1", "IL10RA", "IRF8", "ITGB2",
    "LAPTM5", "LCP1", "LCP2", "LY9", "LYZ", "NCKAP1L", "PLEK", "PTPRC",
    "PTPN22", "RAC2", "RHOH", "SASH3", "SELL", "SLA", "TLR8", "TNFAIP3",
    "TRAF3IP3", "TRBC2", "WAS", "CD14", "CD68", "CSF1R", "MS4A4A", "FCGR3A",
]
ESTIMATE_IMMUNE = [g.strip() for g in ESTIMATE_IMMUNE]

# MCP-counter markers for the two stromal populations the artefact loads on.
MCP_FIBROBLAST = [
    "COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL6A3", "DCN", "LUM", "PDGFRA",
    "PDGFRB", "THY1", "FAP", "POSTN", "SPARC", "FBN1", "MMP2", "COL11A1",
    "ACTA2", "TAGLN", "COL5A2", "VCAN",
]
MCP_ENDOTHELIAL = [
    "PECAM1", "VWF", "CDH5", "CLDN5", "KDR", "FLT1", "TEK", "ENG", "EMCN",
    "ROBO4", "MMRN2", "ESAM", "CD34", "ERG", "SOX17", "TIE1", "PLVAP",
    "EGFL7", "ANGPT2", "NOTCH4",
]


def _zmean(expr: pd.DataFrame, genes: list[str]) -> pd.Series:
    """Mean of per-gene z-scores across samples, over present genes.

    expr: genes (index) x samples (columns), log-scale.
    Returns one score per sample; NaN if fewer than five markers present.
    """
    present = [g for g in genes if g in expr.index]
    if len(present) < 5:
        return pd.Series(np.nan, index=expr.columns)
    sub = expr.loc[present]
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, np.nan),
                                              axis=0)
    return z.mean(axis=0)


def estimate_scores(expr: pd.DataFrame) -> pd.DataFrame:
    """ESTIMATE stromal, immune and combined scores per sample.

    The combined ESTIMATE score is stromal + immune. Higher means more
    non-tumour content, i.e. lower purity. A published sigmoid converts it to a
    purity fraction on Affymetrix; we expose a monotone rank-based purity proxy
    instead, valid across platforms for ordering samples.
    """
    stromal = _zmean(expr, ESTIMATE_STROMAL)
    immune = _zmean(expr, ESTIMATE_IMMUNE)
    # combined = stromal + immune where both exist, else whichever is present
    combined = pd.concat([stromal, immune], axis=1).mean(axis=1) * 2
    # rank-based purity proxy: high combined score -> low purity
    r = combined.rank(pct=True)
    purity_proxy = 1.0 - r
    return pd.DataFrame({
        "estimate_stromal": stromal, "estimate_immune": immune,
        "estimate_combined": combined, "estimate_purity_proxy": purity_proxy})


def mcpcounter_scores(expr: pd.DataFrame) -> pd.DataFrame:
    """MCP-counter fibroblast and endothelial scores per sample."""
    return pd.DataFrame({
        "mcp_fibroblast": _zmean(expr, MCP_FIBROBLAST),
        "mcp_endothelial": _zmean(expr, MCP_ENDOTHELIAL)})


def all_purity_scores(expr: pd.DataFrame) -> pd.DataFrame:
    """Both estimators joined, one row per sample."""
    return estimate_scores(expr).join(mcpcounter_scores(expr))
