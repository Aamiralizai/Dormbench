# dormbench

A reproducible pipeline for benchmarking transcriptional **dormancy signatures**
in ER-positive breast cancer.

`dormbench` scores direction-resolved gene signatures, calibrates them against
matched nulls, attributes them to single-cell compartments with donor-level
paired inference, and evaluates them against clinical baselines. Every step is a
subcommand that writes a machine-readable summary, a timestamped log and a
verdict string, so a run is auditable without reading the code.

```bash
pip install -e .
dormbench doctor          # check that data paths and dependencies resolve
dormbench list            # every registered step, with a one-line description
dormbench wp1             # run one step
dormbench all             # run the whole pipeline in order
```

## Install

```bash
conda create -n dormbench python=3.11 -y
conda activate dormbench
git clone https://github.com/Aamiralizai/Dormbench.git
cd Dormbench
pip install -e ".[scoring,dev]"
```

`[scoring]` adds `decoupler` for the AUCell and ssGSEA scoring steps; `[dev]`
adds pytest. An editable install means changes to `src/` take effect without
reinstalling.

Python 3.10+ is supported; 3.11 is what CI runs.

## Configure

Data lives **outside** the repository. Copy the example and point it at your
copy of the public datasets:

```bash
cp config/config.example.yaml config/config.yaml
```

```yaml
paths:
  data_root: "/path/to/your/data"
  atlas_dir: "GSE176078"          # relative to data_root
  metabric_dir: "METABRIC"
  scanb_dir: "GSE96058"
  tcga_dir: "TCGA-BRCA"
```

`config/config.yaml` is gitignored. Nothing else should need editing: every step
resolves its inputs from `data_root` plus these subdirectories.

```bash
dormbench doctor            # verifies each path exists and reports what is missing
```

## Use

| Command | What it does |
|---|---|
| `dormbench list` | catalogue of registered steps |
| `dormbench doctor` | validate configuration and inputs |
| `dormbench <step>` | run one step |
| `dormbench all` | run every step in pipeline order |
| `dormbench report` | assemble the per-step summaries into one report |
| `dormbench compare <tag>` | compare two runs |

Each step accepts its own options; `dormbench <step> --help` documents them.

## Steps

Thirty-four steps are registered. The core screens the signature panel and
attributes it to compartments; the later ones stress-test the result.

| Step | Analysis |
|---|---|
| `wp0` | histone-modification reference analysis |
| `wp1` | reference construction and panel QC |
| `wp2a`–`wp2i` | signature assembly, per-cell scoring, deconfounding, direction resolution, matched nulls, drivers, in-vitro and CCLE comparison |
| `wp2m` | cell-cycle filtering multiverse |
| `wp2n` | coherent Hallmark-module comparator |
| `wp2r` | rare malignant-cell upper-tail analysis |
| `wp3`, `wp3r`, `wp3u` | bulk association, independent-atlas replication, treatment response |
| `wp4`, `wp4s` | bulk prognosis; late-recurrence landmark against CTS5 |
| `wp5` | purity and composition controls |
| `wp6`, `wp6-poetic` | treatment-response analyses including the POETIC randomised control |
| `wp7`, `wp7n` | compartment-invariance analysis and its fully nested redesign |
| `wp8`, `wp8p` | donor-level primary inference; donor x compartment pseudobulk |
| `wp9`, `wp9b`, `wp10` | sensitivity, calibration and response modelling |
| `tcgaer`, `tcgamap` | TCGA ER status retrieval and gene mapping |

## Outputs

```
results/     machine-readable summaries (JSON) and tables (CSV)
figures/     figures (SVG)
logs/        timestamped per-step logs
```

All three are gitignored: they are products of a run, not part of the package.

## Verdicts

A step reports a **verdict**, not just a p-value, so that a null or unstable
result is stated as such rather than left to prose. Typical vocabulary:

| Verdict | Meaning |
|---|---|
| `REPLICATED` | reproduced in independent data |
| `ATTENUATED_*` | present but weakened by a specific control |
| `ADDS_NOTHING` | no incremental information over the baseline |
| `STATISTIC_DEPENDENT` | direction depends on the scoring statistic |
| `NOT_*` / `INDISTINGUISHABLE_*` | the expected signal was not found |
| `INSPECTED` / `BLOCKED` | stopped deliberately, with the reason logged |

Steps that cannot proceed log why and return a `BLOCKED` verdict rather than
guessing — for example `wp3r` refuses to auto-detect cell-type columns in an
unfamiliar atlas.

## Tests

```bash
python -m pytest -q
```

Tests use small synthetic `AnnData` objects where the answer is known by
construction, so they run without any of the real datasets.

CI (`.github/workflows/ci.yml`) runs on Linux and Windows: unit tests, a CLI
smoke test, and `scripts/check_repo_hygiene.py`, which fails the build on
absolute paths, tracked data files or credential-looking strings.

## Data

No data are redistributed. The pipeline reads, depending on the step:

| Resource | Access |
|---|---|
| Wu et al. single-cell atlas | GEO `GSE176078` |
| METABRIC | cBioPortal |
| SCAN-B | GEO `GSE96058` |
| TCGA-BRCA | NCI GDC, plus `ER_STATUS_BY_IHC` from the cBioPortal API |
| POETIC sampling control | GEO `GSE105777` |
| Visium spatial sections | Zenodo `4739739` |
| Replication atlas | CELLxGENE dataset `de5416ef-bbfa-41f3-99d4-01a3554173f5` |

## Citation

See `CITATION.cff`. Cite the accompanying manuscript and this repository.

## License

MIT — see `LICENSE`.
