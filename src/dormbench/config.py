"""Configuration loading and validation.

All paths live in config/config.yaml. Nothing in this package hard-codes a
path; if a script needs a file it asks the config for it. That is what makes
the pipeline portable between your laptop and a cluster.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PKG_ROOT = Path(__file__).resolve().parents[2]  # .../dormbench
DEFAULT_CONFIG = PKG_ROOT / "config" / "config.yaml"


class ConfigError(RuntimeError):
    pass


@dataclass
class Paths:
    data_root: Path
    atlas_dir: Path
    metabric_dir: Path
    scanb_dir: Path
    rosano_dir: Path
    depmap_dir: Path
    tcga_dir: Path
    poetic_sampling_control_dir: Path
    extended_letrozole_dir: Path
    results: Path
    figures: Path
    logs: Path
    processed: Path

    def check(self, required: list[str] | None = None) -> list[str]:
        """Return a list of human-readable problems; empty means fine."""
        problems = []
        for name in required or ["data_root"]:
            p = getattr(self, name)
            if not p.exists():
                problems.append(f"{name} does not exist: {p}")
        return problems

    def ensure_outputs(self) -> None:
        for p in (self.results, self.figures, self.logs, self.processed):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    paths: Paths
    er_positive_labels: list[str] = field(default_factory=list)
    seed: int = 42
    raw: dict = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return PKG_ROOT


def _resolve(value: str | None, base: Path) -> Path:
    """Resolve a config path, expanding ~ and $VARS, relative to base."""
    if value is None:
        return base
    v = os.path.expandvars(os.path.expanduser(str(value)))
    p = Path(v)
    return p if p.is_absolute() else (base / p)


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise ConfigError(
            f"Config not found: {cfg_path}\n"
            "Copy config/config.example.yaml to config/config.yaml and edit it."
        )

    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    pblock = raw.get("paths", {}) or {}
    data_root = _resolve(pblock.get("data_root"), PKG_ROOT)

    paths = Paths(
        data_root=data_root,
        atlas_dir=_resolve(pblock.get("atlas_dir", "GSE176078"), data_root),
        metabric_dir=_resolve(pblock.get("metabric_dir", "METABRIC"), data_root),
        scanb_dir=_resolve(pblock.get("scanb_dir", "GSE96058"), data_root),
        rosano_dir=_resolve(pblock.get("rosano_dir", "rosano"), data_root),
        depmap_dir=_resolve(pblock.get("depmap_dir", "depmap"), data_root),
        tcga_dir=_resolve(pblock.get("tcga_dir", "TCGA"), data_root),
        poetic_sampling_control_dir=_resolve(
            pblock.get("poetic_sampling_control_dir",
                       pblock.get("poetic_dir", "GSE105777")), data_root),
        extended_letrozole_dir=_resolve(
            pblock.get("extended_letrozole_dir",
                       pblock.get("poetic_dir", "GSE111563")), data_root),
        results=_resolve(pblock.get("results", "results"), PKG_ROOT),
        figures=_resolve(pblock.get("figures", "figures"), PKG_ROOT),
        logs=_resolve(pblock.get("logs", "logs"), PKG_ROOT),
        processed=_resolve(pblock.get("processed", "data_processed"), PKG_ROOT),
    )

    return Config(
        paths=paths,
        er_positive_labels=raw.get("er_positive_labels", []),
        seed=int((raw.get("analysis", {}) or {}).get("random_seed", 42)),
        raw=raw,
    )
