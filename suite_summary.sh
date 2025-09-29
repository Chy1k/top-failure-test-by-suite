#!/usr/bin/env bash
set -Eeuo pipefail

# Defaults (override by env if needed)
MESSAGE_COLS=${MESSAGE_COLS:-"FAILURE MESSAGE 1,FAILURE MESSAGE 2"}
SUITE_COL=${SUITE_COL:-"TEST_SUITE"}
STATUS_COL=${STATUS_COL:-"EXECUTION RESULT"}
TOPN=${TOPN:-5}
GROUP_BY=${GROUP_BY:-norm}
SEP=${SEP:-","}
ENCODING=${ENCODING:-"utf-8"}
EXTRA_FLAGS=${EXTRA_FLAGS:-"--format both --pretty --format xlsx --no-colors"}

usage(){ echo "Usage: $0 <logs.csv> [out_dir]"; exit 1; }
[[ $# -lt 1 ]] && usage

CSV="$1"
OUT="${2:-out_$(basename "$CSV" .csv)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/suite_error_summary.py"
[[ -f "$PY_SCRIPT" ]] || { echo "Not found: $PY_SCRIPT" >&2; exit 2; }
[[ -f "$CSV" ]] || { echo "CSV not found: $CSV" >&2; exit 2; }

# Use system Python (users should install requirements.txt dependencies)
if command -v python >/dev/null 2>&1; then
    PY="python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
elif command -v py >/dev/null 2>&1; then
    PY="py"
else
    echo "Error: Python not found. Please install Python 3." >&2
    echo "Try: python --version, python3 --version, or py --version" >&2
    exit 2
fi

# Convert paths on Git Bash/Windows
CSV_ARG="$CSV"; OUT_ARG="$OUT"
case "$OSTYPE" in msys*|cygwin*) command -v cygpath >/dev/null 2>&1 && {
  CSV_ARG="$(cygpath -w "$CSV")"; OUT_ARG="$(cygpath -w "$OUT")"; };; esac

mkdir -p "$OUT"
RUNLOG="$OUT/run.log"

# Build command with smart defaults
CMD_ARGS=(
  --input "$CSV_ARG"
  --output-dir "$OUT_ARG"
  --message-cols "$MESSAGE_COLS"
  --group-by "$GROUP_BY"
  --top-n "$TOPN"
  --sep "$SEP"
  --encoding "$ENCODING"
)

# Only add suite/status columns if they differ from defaults
[[ "$SUITE_COL" != "TEST_SUITE" ]] && CMD_ARGS+=(--suite-col "$SUITE_COL")
[[ "$STATUS_COL" != "EXECUTION RESULT" ]] && CMD_ARGS+=(--status-col "$STATUS_COL")

set -x
"$PY" "$PY_SCRIPT" "${CMD_ARGS[@]}" $EXTRA_FLAGS 2>&1 | tee "$RUNLOG"
set +x

echo "✔ Success! Generated test failure analysis reports:"
echo "   📊 CSV Report: $OUT/suite_error_summary.csv"
echo "   📈 Excel Report: $OUT/suite_error_summary.xlsx"
echo "   📝 Execution Log: $OUT/run.log"
