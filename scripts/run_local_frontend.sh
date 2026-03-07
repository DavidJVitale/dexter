#!/usr/bin/env bash
set -euo pipefail

dirname="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${dirname%/scripts}"
cd "$root/frontend"

exec python3 -m http.server 5173
