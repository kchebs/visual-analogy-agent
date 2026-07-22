#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Prefer this repo's .venv over an ambient VIRTUAL_ENV from another project
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PY="${VIRTUAL_ENV}/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi

run_py() {
  if command -v arch >/dev/null 2>&1 && arch -arm64 /usr/bin/true >/dev/null 2>&1; then
    arch -arm64 "$PY" "$@"
  else
    "$PY" "$@"
  fi
}

echo "== RPM smoke (interpreter: $PY) =="
run_py -m pytest -q
echo "SMOKE PASS"
