# Assignment 4 - alpha-beta-CROWN experiment

This repository is set up so that the assignment experiment can be run from a fresh checkout with one command. The script does not assume that alpha-beta-CROWN is already installed.

## One-command execution

Run:

```bash
chmod +x run_all.sh
./run_all.sh
```

The script automatically performs the full workflow:

1. creates the assignment conda environment `assignment4-abcrown`
2. installs the Python packages in `requirements.txt`
3. compiles the assignment Python files as a quick syntax check
4. creates the toy PyTorch model and dataset if they are missing
5. clones alpha-beta-CROWN into `external/alpha-beta-CROWN` if it is missing
6. initializes alpha-beta-CROWN submodules
7. creates the official alpha-beta-CROWN conda environment `alpha-beta-crown`
8. runs `abcrown.py` on `configs/toy_linf_robustness.yaml`
9. writes the result to `results/verification_result.json`

The first run is slow because it downloads alpha-beta-CROWN and creates two conda environments. Later runs reuse the existing checkout and environments.

## Useful options

Use a different alpha-beta-CROWN checkout:

```bash
ABCROWN_ROOT=/path/to/alpha-beta-CROWN ./run_all.sh
```

Run setup and command generation without executing the verifier:

```bash
./run_all.sh --dry-run
```

Reuse an existing alpha-beta-CROWN checkout and conda environment without reinstalling:

```bash
ABCROWN_ROOT=/path/to/alpha-beta-CROWN ./run_all.sh --skip-abcrown-install
```

Set a verifier timeout:

```bash
./run_all.sh --timeout 60
```

## Experiment files

- `scripts/build_toy_external_model.py`: creates the synthetic dataset and trains the toy model
- `custom/toy_model_data.py`: custom model/data functions used by alpha-beta-CROWN
- `configs/toy_linf_robustness.yaml`: verification configuration
- `test.py`: calls `abcrown.py` and records structured JSON output
- `report.pdf`: final Korean report

## Current expected result

For the included toy model and reference point, the expected result is `verified`. The result JSON records the exact command, return code, runtime, stdout tail, stderr tail, and parsed status.
