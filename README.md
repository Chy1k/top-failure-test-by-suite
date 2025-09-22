Suite Error Summary

Runs suite_error_summary.py on a CSV of test runs and produces a per-suite Top-N error summary (CSV/XLSX).

Requirements

Bash (macOS/Linux; on Windows use Git Bash or WSL)

Python 3 on PATH (python3, python, or Windows py)

First run needs internet to install deps (pandas, optional xlsxwriter)

The script auto-creates and uses a local .venv next to it—no global setup needed.

Quick Start
chmod +x script.sh
./script.sh path/to/logs.csv           # output → out_<csvname>/
# or pick an output dir:
./script.sh path/to/logs.csv my_reports

Customize (env vars)

You can override defaults when calling the script, e.g.:

STATUS_INCLUDE="FAILED,UNSTABLE" TOPN=10 \
MESSAGE_COLS="FAILURE MESSAGE 1,FAILURE MESSAGE 2" \
SUITE_COL="TEST_SUITE" \
./script.sh logs.csv


Other useful vars: SEP (, or ;), ENCODING (utf-8, latin-1), GROUP_BY (norm|raw), EXTRA_FLAGS (passed to the Python script).

Output

out_<csv_basename>/suite_error_summary.csv

(optional) suite_error_summary.xlsx if xlsxwriter is available or --format xlsx/both

out_<...>/run.log (full command output)

Note: The script runs with your user permissions; ensure you can read the CSV and write the output directory.
