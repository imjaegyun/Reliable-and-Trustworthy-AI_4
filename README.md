# Assignment 4 - alpha-beta-CROWN experiment

This is my Assignment 4 repository for the Reliable and Trustworthy AI course. I used a small external PyTorch model instead of one of the built-in alpha-beta-CROWN examples, then prepared the config and scripts needed to run a local robustness check.

## What is included

- `docs/problem1_model_directory_review.md`: notes from checking the official alpha-beta-CROWN model/config examples
- `scripts/build_toy_external_model.py`: generates the toy dataset and trains the PyTorch model
- `custom/toy_model_data.py`: model and data loader definitions
- `configs/toy_linf_robustness.yaml`: verifier configuration
- `test.py`: runs `abcrown.py` through the selected alpha-beta-CROWN checkout and records the result
- `run_all.sh`: one-command setup and execution script
- `report.pdf`: final Korean report

## One-command run

Run:

```bash
chmod +x run_all.sh
./run_all.sh
```

The script does the following steps:

1. creates or activates the assignment conda environment
2. installs the Python dependencies in `requirements.txt`
3. checks the Python files with `py_compile`
4. creates the toy model and dataset if they are missing
5. clones alpha-beta-CROWN into `external/alpha-beta-CROWN` if it is missing
6. creates the official `alpha-beta-crown` conda environment from `complete_verifier/environment.yaml`
7. runs `test.py`, which calls `abcrown.py` and writes `results/verification_result.json`

The first run can take a long time because alpha-beta-CROWN and its conda environment are large.

## Useful options

Use a different alpha-beta-CROWN checkout:

```bash
ABCROWN_ROOT=/path/to/alpha-beta-CROWN ./run_all.sh
```

Only prepare the assignment side and record the command without running the verifier:

```bash
./run_all.sh --dry-run
```

Use an existing alpha-beta-CROWN environment/check-out without reinstalling it:

```bash
ABCROWN_ROOT=/path/to/alpha-beta-CROWN ./run_all.sh --skip-abcrown-install
```

## Result file

The script writes the latest run result to:

```text
results/verification_result.json
```
