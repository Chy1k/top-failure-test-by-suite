#!/usr/bin/env bash
set -Eeuo pipefail

# Defaults (override by env if needed)
MESSAGE_COLS=${MESSAGE_COLS:-"FAILURE MESSAGE 1,FAILURE MESSAGE 2"}
SUITE_COL=${SUITE_COL:-"TEST_SUITE"}
STATUS_COL=${STATUS_COL:-"EXECUTION RESULT"}
STATUS_INCLUDE=${STATUS_INCLUDE:-"FAILED,UNSTABLE"}
TOPN=${TOPN:-5}
GROUP_BY=${GROUP_BY:-norm}
SEP=${SEP:-","}
ENCODING=${ENCODING:-"utf-8"}
EXTRA_FLAGS=${EXTRA_FLAGS:-"--format both --pretty --format xlsx --no-colors"}   # no truncate flag

usage(){ echo "Usage: $0 <logs.csv> [out_dir]"; exit 1; }
[[ $# -lt 1 ]] && usage

CSV="$1"
OUT="${2:-out_$(basename "$CSV" .csv)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/suite_error_summary.py"
[[ -f "$PY_SCRIPT" ]] || { echo "Not found: $PY_SCRIPT" >&2; exit 2; }
[[ -f "$CSV" ]] || { echo "CSV not found: $CSV" >&2; exit 2; }

# Resolve Python (prefer local venv; create if missing)
if [[ -x "$SCRIPT_DIR/.venv/Scripts/python.exe" ]]; then PY="$SCRIPT_DIR/.venv/Scripts/python.exe"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then PY="$SCRIPT_DIR/.venv/bin/python"
else
  if command -v py >/dev/null 2>&1; then py -3 -m venv "$SCRIPT_DIR/.venv"; PY="$SCRIPT_DIR/.venv/Scripts/python.exe"
  elif command -v python3 >/dev/null 2>&1; then python3 -m venv "$SCRIPT_DIR/.venv"; PY="$SCRIPT_DIR/.venv/bin/python"
  else python -m venv "$SCRIPT_DIR/.venv"; PY="$SCRIPT_DIR/.venv/bin/python"; fi
  "$PY" -m pip install --upgrade pip -q
  if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then "$PY" -m pip install -r "$SCRIPT_DIR/requirements.txt" -q
  else "$PY" -m pip install pandas -q; fi
fi

# Convert paths on Git Bash/Windows
CSV_ARG="$CSV"; OUT_ARG="$OUT"
case "$OSTYPE" in msys*|cygwin*) command -v cygpath >/dev/null 2>&1 && {
  CSV_ARG="$(cygpath -w "$CSV")"; OUT_ARG="$(cygpath -w "$OUT")"; };; esac

mkdir -p "$OUT"
RUNLOG="$OUT/run.log"
set -x
"$PY" "$PY_SCRIPT" \
  --input "$CSV_ARG" \
  --output-dir "$OUT_ARG" \
  --message-cols "$MESSAGE_COLS" \
  --suite-col "$SUITE_COL" \
  --status-col "$STATUS_COL" \
  --status-include "$STATUS_INCLUDE" \
  --group-by "$GROUP_BY" \
  --top-n "$TOPN" \
  --hash-signature \
  --sep "$SEP" \
  --encoding "$ENCODING" \
  $EXTRA_FLAGS \
  2>&1 | tee "$RUNLOG"
set +x

Done "✔ Output: $OUT/suite_error_summary.csv"
