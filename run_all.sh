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

log() {
  printf '\n== %s ==\n' "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$1"
}

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
      [[ $# -ge 2 ]] || die "--abcrown-root requires a path"
      ABCROWN_ROOT="$2"
      shift 2
      ;;
    --abcrown-repo)
      [[ $# -ge 2 ]] || die "--abcrown-repo requires a URL"
      ABCROWN_REPO="$2"
      shift 2
      ;;
    --timeout)
      [[ $# -ge 2 ]] || die "--timeout requires seconds"
      TEST_ARGS+=(--timeout "$2")
      shift 2
      ;;
    *)
      TEST_ARGS+=("$1")
      shift
      ;;
  esac
done

case "$ABCROWN_ROOT" in
  /*) ;;
  *) ABCROWN_ROOT="$ROOT_DIR/$ABCROWN_ROOT" ;;
esac

log "Locate conda"
if ! command -v conda >/dev/null 2>&1; then
  for candidate in "$HOME/miniconda3/bin/conda" "/data/home/ijg2603/miniconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      export PATH="$(dirname "$candidate"):$PATH"
      break
    fi
  done
fi
command -v conda >/dev/null 2>&1 || die "conda was not found. Install Miniconda/Anaconda or add conda to PATH."
command -v git >/dev/null 2>&1 || die "git was not found. Install git or add it to PATH."
eval "$(conda shell.bash hook)"

log "Prepare assignment environment: $ASSIGNMENT_ENV_NAME"
if ! env_exists "$ASSIGNMENT_ENV_NAME"; then
  conda env create -f environment.yml --name "$ASSIGNMENT_ENV_NAME"
fi
conda activate "$ASSIGNMENT_ENV_NAME"
python -m pip install -r requirements.txt
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/assignment4_pycache"
export ASSIGNMENT4_ROOT="$ROOT_DIR"

log "Compile assignment Python files"
mapfile -t python_files < <(find . -maxdepth 2 -name "*.py" -not -path "./.git/*" -not -path "./.venv/*" -not -path "./external/*" | sort)
if [[ "${#python_files[@]}" -gt 0 ]]; then
  python -m py_compile "${python_files[@]}"
fi

log "Prepare toy model and data"
if [[ ! -f models/toy_binary_mlp.pt || ! -f data/toy_dataset.csv || ! -f data/toy_reference_point.json ]]; then
  python scripts/build_toy_external_model.py --seed 42
else
  printf 'Toy artifacts already exist.\n'
fi

log "Prepare alpha-beta-CROWN checkout"
if [[ "$SKIP_ABCROWN_INSTALL" -eq 0 ]]; then
  if [[ -e "$ABCROWN_ROOT" && ! -d "$ABCROWN_ROOT/.git" ]]; then
    die "$ABCROWN_ROOT exists but is not a git checkout. Remove it or pass --abcrown-root to another path."
  fi
  if [[ ! -d "$ABCROWN_ROOT/.git" ]]; then
    mkdir -p "$(dirname "$ABCROWN_ROOT")"
    git clone --recursive "$ABCROWN_REPO" "$ABCROWN_ROOT"
  else
    git -C "$ABCROWN_ROOT" submodule update --init --recursive
  fi
fi
ABCROWN_ENTRY="$ABCROWN_ROOT/complete_verifier/abcrown.py"
[[ -f "$ABCROWN_ENTRY" ]] || die "abcrown.py was not found at $ABCROWN_ENTRY"

log "Prepare alpha-beta-CROWN conda environment: $ABCROWN_ENV_NAME"
ABCROWN_ENV_FILE="$ABCROWN_ROOT/complete_verifier/environment.yaml"
[[ -f "$ABCROWN_ENV_FILE" ]] || die "alpha-beta-CROWN environment file not found: $ABCROWN_ENV_FILE"
if ! env_exists "$ABCROWN_ENV_NAME"; then
  conda env create -f "$ABCROWN_ENV_FILE" --name "$ABCROWN_ENV_NAME"
fi
ABCROWN_PYTHON="$(conda run -n "$ABCROWN_ENV_NAME" python -c 'import sys; print(sys.executable)' | sed '/^[[:space:]]*$/d' | tail -n 1)"
[[ -x "$ABCROWN_PYTHON" ]] || die "Could not find Python executable for $ABCROWN_ENV_NAME"

log "Run verifier"
cmd=(python test.py --config configs/toy_linf_robustness.yaml --abcrown-root "$ABCROWN_ROOT" --python "$ABCROWN_PYTHON")
if [[ "$RUN_VERIFIER" -eq 1 ]]; then
  cmd+=(--run)
fi
cmd+=("${TEST_ARGS[@]}")
printf 'Command: '
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

log "Done"
printf 'Result file: %s\n' "$ROOT_DIR/results/verification_result.json"
