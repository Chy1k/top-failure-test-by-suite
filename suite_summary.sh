#!/usr/bin/env bash
set -Eeuo pipefail

# Defaults (override by env if needed)
# Simplified configuration - core options only
ERROR_MESSAGE_COLUMNS=${ERROR_MESSAGE_COLUMNS:-"FAILURE MESSAGE 1,FAILURE MESSAGE 2"}
TEST_SUITE_COLUMN=${TEST_SUITE_COLUMN:-"TEST_SUITE"}
TEST_STATUS_COLUMN=${TEST_STATUS_COLUMN:-"EXECUTION RESULT"}
TOP_ERRORS_COUNT=${TOP_ERRORS_COUNT:-5}

# Backwards compatibility with legacy variable names
ERROR_MESSAGE_COLUMNS=${MESSAGE_COLS:-$ERROR_MESSAGE_COLUMNS}
TEST_SUITE_COLUMN=${SUITE_COL:-$TEST_SUITE_COLUMN}
TEST_STATUS_COLUMN=${STATUS_COL:-$TEST_STATUS_COLUMN}
TOP_ERRORS_COUNT=${TOPN:-$TOP_ERRORS_COUNT}

EXTRA_FLAGS=${EXTRA_FLAGS:-"--output-format both"}

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

# Build command with simplified defaults
CMD_ARGS=(
  --input-file "$CSV_ARG"
  --output-directory "$OUT_ARG"
  --error-message-columns "$ERROR_MESSAGE_COLUMNS"
  --top-errors-count "$TOP_ERRORS_COUNT"
)

# Only add suite/status columns if they differ from defaults
[[ "$TEST_SUITE_COLUMN" != "TEST_SUITE" ]] && CMD_ARGS+=(--test-suite-column "$TEST_SUITE_COLUMN")
[[ "$TEST_STATUS_COLUMN" != "EXECUTION RESULT" ]] && CMD_ARGS+=(--test-status-column "$TEST_STATUS_COLUMN")

set -x
"$PY" "$PY_SCRIPT" "${CMD_ARGS[@]}" $EXTRA_FLAGS 2>&1 | tee "$RUNLOG"
set +x

echo "✔ Success! Generated test failure analysis reports:"
echo "   📊 CSV Report: $OUT/suite_error_summary.csv"
echo "   📈 Excel Report: $OUT/suite_error_summary.xlsx"
echo "   📝 Execution Log: $OUT/run.log"
