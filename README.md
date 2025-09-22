# Suite Error Summary

Bash wrapper that sets up a local Python virtual environment and runs `suite_error_summary.py` on a CSV of test runs to produce a per-suite **Top-N error summary** (CSV and/or XLSX).

## Prerequisites

* **Bash** (macOS/Linux; on Windows use **Git Bash** or **WSL**)
* **Python 3** on PATH (`python3`, `python`, or Windows `py`)
* Internet on first run (to create `.venv` and install deps)
* Install xlsxwriter (if you want the XLSX File)

> The script auto-creates and uses a local `.venv` next to itself—no global installs needed.

## Files

```
.
├─ suite_summary.sh            # the bash wrapper
├─ suite_error_summary.py      # the Python script it calls
└─ logs.csv                    # your input data (example)
```

## Quick Start

```bash
chmod +x suite_summary.sh
./suite_summary.sh path/to/logs.csv
# or choose an output dir:
./suite_summary.sh path/to/logs.csv my_reports
```

### Recommended example

Generate **both CSV and XLSX**, with pretty headers and no colors:

```bash
EXTRA_FLAGS="--format both --pretty --no-colors" ./suite_summary.sh ./logs.csv
```

## Customization (env vars)

Override defaults by prefixing the command:

```bash
MESSAGE_COLS="FAILURE MESSAGE 1,FAILURE MESSAGE 2" \
SUITE_COL="TEST_SUITE" \
STATUS_COL="EXECUTION RESULT" \
STATUS_INCLUDE="FAILED,UNSTABLE" \
TOPN=10 \
GROUP_BY="norm" \
SEP="," \
ENCODING="utf-8" \
EXTRA_FLAGS="--format both --pretty --no-colors" \
./suite_summary.sh ./logs.csv
```

**Common flags passed via `EXTRA_FLAGS` to the Python script:**

* `--format csv|xlsx|both`
* `--pretty`
* `--truncate-len N`
* `--no-colors` (for XLSX)

## Output

```
out_<csv_basename>/
  ├─ suite_error_summary.csv
  ├─ suite_error_summary.xlsx     # if format includes xlsx and xlsxwriter is installed
  └─ run.log                      # full command output
```
