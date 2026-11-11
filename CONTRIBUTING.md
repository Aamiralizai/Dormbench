# Contributing

Thanks for looking. This is a research codebase accompanying a manuscript, so
the priorities are reproducibility and honesty of the reported numbers, not
feature breadth.

## Setup

```bash
git clone https://github.com/Aamiralizai/Dormbench.git
cd Dormbench
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest -q
```

Data is **never** committed. Copy `config/config.example.yaml` to
`config/config.yaml` and point `data_root` at your own copy of the public
datasets; that file is gitignored.

## Before opening a pull request

Run the same gates CI runs:

```bash
python -m pytest -q
python scripts/check_repo_hygiene.py      # absolute paths, data files, secrets
```

Both must pass. `check_repo_hygiene.py` fails on absolute paths in source:
a fresh clone must not inherit anyone's drive layout, so pass paths as CLI
arguments or environment variables.

## Adding an analysis step

1. Create `src/dormbench/steps/wpNN_name.py` exposing `run(cfg, ...) -> dict`.
2. Return a report dict containing a `verdict` string. Keep the verdict
   vocabulary consistent with the existing steps: a step should say what it
   found, including when it found nothing.
3. Register the name in `REGISTRY` in `steps/__init__.py`.
4. Add the subparser and the report entry in `cli.py`.
5. Add a test. Tests must not read the real atlases; build a small synthetic
   `AnnData` where the answer is known by construction, as
   `tests/test_review_steps.py` does.
6. Log what was excluded and why. Silent exclusion of cells or donors is the
   most common way a pipeline like this produces a wrong answer.

## Style

Plain Python, type hints where they help, no notebooks in the pipeline path.
Long analyses print a banner and a summary line so a log is readable without the
code.
