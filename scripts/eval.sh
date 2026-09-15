#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m app.rag.eval_harness "$@"
