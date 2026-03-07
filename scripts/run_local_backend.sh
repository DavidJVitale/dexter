#!/usr/bin/env bash
set -euo pipefail

dirname="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${dirname%/scripts}"

if [[ ! -d "${root}/.venv" ]]; then
  echo "Missing ${root}/.venv. Create it first (python3 -m venv .venv)." >&2
  exit 1
fi

source "${root}/.venv/bin/activate"
cd "$root"

export HF_HOME="${HF_HOME:-/Users/davidjvitale/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/Users/davidjvitale/.cache/huggingface/hub}"

exec python -m api.main
