#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${DOJO_PYTHON:-}" ]]; then
  PYTHON_BIN="$DOJO_PYTHON"
elif python -c 'import torch' >/dev/null 2>&1; then
  PYTHON_BIN=python
elif [[ -x /home/zsj/anaconda3/envs/medsam/bin/python ]]; then
  PYTHON_BIN=/home/zsj/anaconda3/envs/medsam/bin/python
else
  echo '请设置 DOJO_PYTHON 为安装了 PyTorch 的 Python 路径。'
  exit 1
fi
"$PYTHON_BIN" -c 'import torch' || { echo '所选 Python 未安装 PyTorch'; exit 1; }
if [[ ! -d node_modules ]]; then npm ci --no-audit --no-fund; fi
npm run build
exec "$PYTHON_BIN" backend/server.py --port "${DOJO_PORT:-8765}"
