#!/bin/bash
# Orchestrate dorsal + ventral runs of notebooks 02 and 03.
# Each mode: patch a TEMP copy of the notebook (originals untouched), then
# execute it with nbconvert under the active env (tick_env).
#
# Skips "both" mode — previous untagged run dirs already cover it
# (the new "both" math is algebraically identical to the old code).

set -u
set -o pipefail

REPO="/fs/ess/PAS2136/jjivandas/projects/tick_classification_project"
LOG_DIR="${REPO}/logs/view_mode_runs/$(date +%y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

NB02="${REPO}/notebooks/02_bioclip_inference.ipynb"
NB03="${REPO}/notebooks/03_finetuning_svm_new.ipynb"

# Cell IDs to patch (set during the VIEW_MODE refactor)
CELL_ID_NB02="0df1e41a"   # data-load cell with VIEW_MODE
CELL_ID_NB03="d30e6522"   # config cell with VIEW_MODE

patch_view_mode() {
  # $1 = source notebook path
  # $2 = dest notebook path
  # $3 = cell id to patch
  # $4 = new VIEW_MODE value
  python3 - "$1" "$2" "$3" "$4" <<'PYEOF'
import json, sys, re
src, dst, cell_id, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(src) as f:
    nb = json.load(f)
patched = False
for c in nb["cells"]:
    if c.get("id") != cell_id:
        continue
    body = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    new_body, n = re.subn(r'VIEW_MODE\s*=\s*"[^"]+"', f'VIEW_MODE = "{mode}"', body, count=1)
    if n != 1:
        sys.exit(f"Failed to patch VIEW_MODE in cell {cell_id} of {src}")
    c["source"] = new_body.splitlines(keepends=True)
    c["outputs"] = []
    c["execution_count"] = None
    patched = True
    break
if not patched:
    sys.exit(f"Cell {cell_id} not found in {src}")
with open(dst, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Patched VIEW_MODE=\"{mode}\" in cell {cell_id} -> {dst}")
PYEOF
}

run_notebook() {
  # $1 = notebook path to execute (the temp patched copy)
  # $2 = log file
  jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=7200 \
    --ExecutePreprocessor.kernel_name=python3 \
    "$1" > "$2" 2>&1
}

# Activate env (must already be loaded once via module spider earlier;
# this re-asserts the env in case the script is called fresh).
source /apps/cm/lmod/lmod/init/bash 2>/dev/null || true
module load miniconda3/24.1.2-py310
source activate tick_env

cd "${REPO}"

MODES=("dorsal" "ventral")

NB_DIR="${REPO}/notebooks"

for MODE in "${MODES[@]}"; do
  echo ""
  echo "================================================================"
  echo "  MODE: ${MODE}   ($(date))"
  echo "================================================================"

  # IMPORTANT: notebooks compute repo_root = Path.cwd().parent, so the patched
  # copy must live inside notebooks/ for the src/ import to resolve. We use a
  # dotfile-prefixed name and move it to LOG_DIR after execution so the
  # final artifact + executed notebook are preserved alongside the logs.
  TMP_NB02="${NB_DIR}/.tmp_nb02_${MODE}.ipynb"
  TMP_NB03="${NB_DIR}/.tmp_nb03_${MODE}.ipynb"

  # --- Notebook 02: BioCLIP zero-shot (slow, GPU-bound) ---
  echo "[${MODE}] Patching notebook 02..."
  patch_view_mode "${NB02}" "${TMP_NB02}" "${CELL_ID_NB02}" "${MODE}" \
    || { echo "[${MODE}] PATCH FAILED for notebook 02"; continue; }

  echo "[${MODE}] Executing notebook 02 ($(date))..."
  if run_notebook "${TMP_NB02}" "${LOG_DIR}/nb02_${MODE}.log"; then
    echo "[${MODE}] notebook 02 OK"
    mv "${TMP_NB02}" "${LOG_DIR}/nb02_${MODE}.ipynb"
  else
    echo "[${MODE}] notebook 02 FAILED — see ${LOG_DIR}/nb02_${MODE}.log"
    mv "${TMP_NB02}" "${LOG_DIR}/nb02_${MODE}.ipynb" 2>/dev/null || true
    continue
  fi

  # --- Notebook 03: SVM (fast, cache-only) ---
  echo "[${MODE}] Patching notebook 03..."
  patch_view_mode "${NB03}" "${TMP_NB03}" "${CELL_ID_NB03}" "${MODE}" \
    || { echo "[${MODE}] PATCH FAILED for notebook 03"; continue; }

  echo "[${MODE}] Executing notebook 03 ($(date))..."
  if run_notebook "${TMP_NB03}" "${LOG_DIR}/nb03_${MODE}.log"; then
    echo "[${MODE}] notebook 03 OK"
    mv "${TMP_NB03}" "${LOG_DIR}/nb03_${MODE}.ipynb"
  else
    echo "[${MODE}] notebook 03 FAILED — see ${LOG_DIR}/nb03_${MODE}.log"
    mv "${TMP_NB03}" "${LOG_DIR}/nb03_${MODE}.ipynb" 2>/dev/null || true
  fi
done

echo ""
echo "================================================================"
echo "  DONE  $(date)"
echo "================================================================"
echo "Logs:   ${LOG_DIR}"
echo "Outputs:"
echo "  results/bioclip_zeroshot/runs/<ts>_dorsal/  (and _ventral)"
echo "  results/svm/runs/<ts>_dorsal/               (and _ventral)"
