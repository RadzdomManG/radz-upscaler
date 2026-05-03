#!/usr/bin/env bash
set -euo pipefail

find_comfy_dir() {
  local candidates=(
    "/workspace/ComfyUI"
    "/workspace/runpod-slim/ComfyUI"
    "/comfyui"
    "/app/ComfyUI"
  )

  for path in "${candidates[@]}"; do
    if [[ -d "$path" && -f "$path/main.py" ]]; then
      echo "$path"
      return 0
    fi
  done

  return 1
}

COMFY_DIR="${COMFY_DIR:-$(find_comfy_dir || true)}"
if [[ -z "${COMFY_DIR}" ]]; then
  echo "Could not detect ComfyUI directory. Set COMFY_DIR and rerun."
  exit 1
fi

RADZ_DIR="${COMFY_DIR}/custom_nodes/Radz Human Skin details"
if [[ ! -d "${RADZ_DIR}" ]]; then
  echo "Radz Human Skin details not found at: ${RADZ_DIR}"
  echo "Copy this folder into ${COMFY_DIR}/custom_nodes first."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "${COMFY_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${COMFY_DIR}/.venv/bin/python"
fi

echo "Using ComfyUI directory: ${COMFY_DIR}"
echo "Using Python: ${PYTHON_BIN}"

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r "${RADZ_DIR}/requirements.runpod-cuda13.txt"

mkdir -p "${RADZ_DIR}/bundled_models/clip_vision"
mkdir -p "${RADZ_DIR}/bundled_models/ipadapter"
mkdir -p "${RADZ_DIR}/bundled_models/loras"
mkdir -p "${RADZ_DIR}/bundled_models/insightface"

cat > "${COMFY_DIR}/extra_model_paths.yaml" <<'YAML'
radz_nodes:
  base_path: custom_nodes/Radz Human Skin details/bundled_models
  is_default: false
  clip_vision: clip_vision
  ipadapter: ipadapter
  loras: loras
YAML

echo ""
echo "Radz Human Skin details Runpod CUDA13 install complete."
echo "Next steps:"
echo "  1. Place model files into ${RADZ_DIR}/bundled_models"
echo "  2. Restart ComfyUI"
echo "  3. Use Radz Insight Face loaders after models are present"
