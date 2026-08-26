#!/bin/bash
# Run normalize_paths.py with repo environment configured.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/setup_env.sh"

python "${SCRIPT_DIR}/normalize_paths.py" "$@"
