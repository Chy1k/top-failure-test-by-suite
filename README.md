Suite Error Summary

Tiny wrapper that sets up a local Python venv and runs suite_error_summary.py on a CSV of test logs to produce per-suite error summaries.

Prereqs

Bash (Linux/macOS; on Windows use Git Bash or WSL)

Python 3 on PATH (python3, python, or Windows py)

Internet on first run (to create .venv and install deps)

Quick Start
chmod +x script.sh              # first time
./script.sh path/to/logs.csv    # writes to out_<csvname>/
# or choose an output dir:
./script.sh path/to/logs.csv my_reports

What it does

Ensures ./.venv exists (creates it if needed)

Installs deps (requirements.txt if present, otherwise pandas)

Runs suite_error_summary.py with sensible defaults

Saves logs to out_.../run.log

Customize (env vars)

Override any of these when calling the script:

MESSAGE_COLS (default: FAILURE MESSAGE 1,FAILURE MESSAGE 2)

SUITE_COL (default: TEST_SUITE)

STATUS_COL (default: EXECUTION RESULT)

STATUS_INCLUDE (default: FAILED,UNSTABLE)

TOPN (default: 5)

GROUP_BY (default: norm)

SEP (default: ,)

ENCODING (default: utf-8)

EXTRA_FLAGS (passed straight to the Python script)

Example:

STATUS_INCLUDE="FAILED" TOPN=10 ./script.sh logs.csv

Output

out_<csv_basename>/suite_error_summary.csv

(optionally) .xlsx if enabled via EXTRA_FLAGS

out_<...>/run.log with full command output

Windows note

Works in Git Bash/Cygwin; paths are auto-converted with cygpath.
