# Problem 2: External model + dataset execution log

## Setup

- Model: `models/toy_binary_mlp.pt`
- Model type: PyTorch MLP, `Linear(2, 16) -> ReLU -> Linear(16, 2)`
- Dataset: `data/toy_dataset.csv`
- Custom model/data loader: `custom/toy_model_data.py`
- alpha-beta-CROWN config: `configs/toy_linf_robustness.yaml`
- Runner: `run_all.sh` -> `test.py` -> `abcrown.py`

## One-command run

The experiment can be started with:

```bash
./run_all.sh
```

The script prepares the assignment environment, clones alpha-beta-CROWN into
`external/alpha-beta-CROWN` when needed, creates the official `alpha-beta-crown`
conda environment from `complete_verifier/environment.yaml`, and then runs
`abcrown.py` through `test.py`.

## Result from local run

The full script was executed successfully after aligning the YAML file with the
current alpha-beta-CROWN config format.

- Status: `verified`
- Return code: `0`
- Runtime reported by `test.py`: about `4.71` seconds
- alpha-beta-CROWN verification time for the single instance: about `0.048` seconds
- Summary: 1 problem instance, 1 verified safe, 0 falsified, 0 timeout

The detailed JSON output is stored in `results/verification_result.json`.

## Notes

The first run takes much longer because alpha-beta-CROWN and its official conda
environment are downloaded and installed. Later runs reuse `external/alpha-beta-CROWN`
and the `alpha-beta-crown` conda environment.
