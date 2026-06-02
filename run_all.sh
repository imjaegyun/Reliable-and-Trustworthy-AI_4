#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ASSIGNMENT_ENV_NAME="${ASSIGNMENT4_CONDA_ENV:-assignment4-abcrown}"
ABCROWN_ENV_NAME="${ABCROWN_CONDA_ENV:-alpha-beta-crown}"
ABCROWN_ROOT="${ABCROWN_ROOT:-$ROOT_DIR/external/alpha-beta-CROWN}"
ABCROWN_REPO="${ABCROWN_REPO:-https://github.com/Verified-Intelligence/alpha-beta-CROWN.git}"
RUN_VERIFIER=1
SKIP_ABCROWN_INSTALL=0
TEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      RUN_VERIFIER=0
      shift
      ;;
    --skip-abcrown-install)
      SKIP_ABCROWN_INSTALL=1
      shift
      ;;
    --abcrown-root)
      ABCROWN_ROOT="$2"
      shift 2
      ;;
    --timeout)
      TEST_ARGS+=(--timeout "$2")
      shift 2
      ;;
    *)
      TEST_ARGS+=("$1")
      shift
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  for candidate in "$HOME/miniconda3/bin/conda" "/data/home/ijg2603/miniconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      export PATH="$(dirname "$candidate"):$PATH"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda/Anaconda or add conda to PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ASSIGNMENT_ENV_NAME"; then
  if [[ "$ASSIGNMENT_ENV_NAME" == "assignment4-abcrown" ]]; then
    conda env create -f environment.yml
  else
    conda create -y -n "$ASSIGNMENT_ENV_NAME" python=3.10 pip
  fi
fi

conda activate "$ASSIGNMENT_ENV_NAME"
python -m pip install -r requirements.txt
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/assignment4_pycache"

mapfile -t python_files < <(find . -maxdepth 2 -name "*.py" -not -path "./.git/*" -not -path "./.venv/*" -not -path "./external/*" | sort)
if [[ "${#python_files[@]}" -gt 0 ]]; then
  python -m py_compile "${python_files[@]}"
fi

if [[ ! -f models/toy_binary_mlp.pt || ! -f data/toy_dataset.csv ]]; then
  python scripts/build_toy_external_model.py --seed 42
fi

if [[ "$SKIP_ABCROWN_INSTALL" -eq 0 ]]; then
  if [[ ! -d "$ABCROWN_ROOT/.git" ]]; then
    mkdir -p "$(dirname "$ABCROWN_ROOT")"
    git clone --recursive "$ABCROWN_REPO" "$ABCROWN_ROOT"
  fi

  if ! conda env list | awk '{print $1}' | grep -qx "$ABCROWN_ENV_NAME"; then
    conda env create -f "$ABCROWN_ROOT/complete_verifier/environment.yaml" --name "$ABCROWN_ENV_NAME"
  fi
fi

ABCROWN_PYTHON="$(conda run -n "$ABCROWN_ENV_NAME" python -c 'import sys; print(sys.executable)')"

cmd=(python test.py --config configs/toy_linf_robustness.yaml --abcrown-root "$ABCROWN_ROOT" --python "$ABCROWN_PYTHON")
if [[ "$RUN_VERIFIER" -eq 1 ]]; then
  cmd+=(--run)
fi
cmd+=("${TEST_ARGS[@]}")

"${cmd[@]}"
