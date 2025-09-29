#!/usr/bin/env python3
"""
Suite Error Summary (Top-N)

Reads a CSV of test executions, groups error messages per suite, and produces a
"Top-N messages + Other" summary with FAILED/UNSTABLE breakdowns. Exports CSV or
a nicely formatted XLSX.

"""
import argparse
import csv
import hashlib
import re
from pathlib import Path
import pandas as pd
from argparse import Namespace

# ---------------- Constants ----------------
VALID_STATUSES = ["FAILED", "UNSTABLE"]
DEFAULT_TOP_N = 5
DEFAULT_ENCODING = "utf-8"

# Standard CSV column names (always the same format)
STANDARD_COLUMNS = {
    'TEST_SUITE': 'TEST_SUITE',
    'EXECUTION_RESULT': 'EXECUTION RESULT',
    'FAILURE_MESSAGE_1': 'FAILURE MESSAGE 1',
    'FAILURE_MESSAGE_2': 'FAILURE MESSAGE 2'
}

# ---------------- Normalization ----------------
# Precompiled regexes used to mask volatile bits in messages so semantically
# identical errors (that differ only by IDs, IPs, etc.) will group together.
UUID_RE   = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I)
HEX_RE    = re.compile(r"\b[0-9a-f]{8,}\b", re.I)
NUM_RE    = re.compile(r"\b\d+\b")
IP_RE     = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
QUOTED_RE = re.compile(r"\"[^\"\n]{4,}\"|'[^'\n]{4,}'")

# Extracts things like: project="my-app", project: 'svc', project=foo-bar
PROJECT_TOKEN_RE = re.compile(r"(?i)\b(project)\s*[:=]?\s*(?:\"([^\"]+)\"|'([^']+)'|([A-Za-z0-9][\w\-.]{2,}))")
# Tokens like e2e_checkout_flow, e2e_orders_v2
E2E_TOKEN_RE     = re.compile(r"\be2e_[a-z0-9_]{4,}\b", re.I)


def normalize(s: str) -> str:
    """
    Normalize a message string by masking volatile tokens and canonicalizing case/whitespace.

    Returns a *grouping-friendly* string:
    - Lowercase everything.
    - Replace project tokens, e2e_* tokens, UUIDs, IPs, numbers, long hex, and long quoted strings
      with placeholders like <project>, <uuid>, <ip>, <num>, <hex>, <str>.
    - Collapse repeated whitespace.

    "If two messages only differ by IDs, they shouldn't count as two problems."
    """
    if s is None:
        return ""
    s = str(s).lower()

    # Project tokens kept as 'project <project>' so the label remains visible.
    s = PROJECT_TOKEN_RE.sub(lambda m: f"{m.group(1)} <project>", s)

    # e2e_* tokens: often encode environment/domain noise.
    s = E2E_TOKEN_RE.sub("<project>", s)

    # Specific → general masking order to avoid over-masking.
    s = UUID_RE.sub("<uuid>", s)
    s = IP_RE.sub("<ip>", s)
    s = NUM_RE.sub("<num>", s)
    s = HEX_RE.sub("<hex>", s)
    s = QUOTED_RE.sub("<str>", s)

    # Unescape double backslashes common in logs.
    s = s.replace("\\\\", "\\")

    # Whitespace normalization.
    return re.sub(r"\s+"," ", s).strip()


def hsig(t: str, algo: str = "sha1") -> str:
    """
    Compute a hexadecimal hash signature for the given text.

    - Default algo is sha1; you can pass --hash-algo=sha256, md5, etc. (if hashlib provides it).
    - Useful for privacy-preserving grouping when --hash-signature is set.

    "Hash when you must share counts, not content."
    """
    h = getattr(hashlib, algo, hashlib.sha1)()
    h.update(t.encode(DEFAULT_ENCODING))
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    """
    Define and parse the command-line interface.

    Key args:
    - --message-cols: comma-separated columns to scan for messages.
    - --group-by: 'norm' (default) groups by normalized message; 'raw' uses exact text.
    - --hash-signature: emit hashed signatures instead of text (privacy).
    - --pretty/--format: presentation-oriented exports (CSV/XLSX).
    - --top-n: number of top messages per suite.

    "Make the CLI match the mental model: input → transform → top-N → export."
    """
    parser = argparse.ArgumentParser(
        description="Suite-level error summary (Top-N with FAILED/UNSTABLE breakdown)"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--message-cols", required=True, help='e.g. "FAILURE MESSAGE 1,FAILURE MESSAGE 2"')
    parser.add_argument("--suite-col", default="TEST_SUITE", help='Column name for test suite (default: TEST_SUITE)')
    parser.add_argument("--status-col", default="EXECUTION RESULT", help='Column name for test status (default: EXECUTION RESULT)')
    parser.add_argument("--group-by", choices=["norm", "raw"], default="norm")
    parser.add_argument("--hash-signature", action="store_true")
    parser.add_argument("--hash-algo", default="sha1")
    parser.add_argument("--sep", default=",")
    parser.add_argument("--encoding", default=None)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    # presentation
    parser.add_argument("--pretty", action="store_true", help="Nicer headers/order")
    parser.add_argument("--truncate-len", type=int, default=0, help="Trim Top-i messages to N chars (0 = none)")
    parser.add_argument("--format", choices=["csv","xlsx","both"], default="csv")
    parser.add_argument("--no-colors", action="store_true",
                    help="Disable background/conditional colors in XLSX export")
    return parser.parse_args()


def out_name(fmt: str) -> str:
    """
    Map a format to the default filename.

    "Names should be predictable — automation likes that."
    """
    return "suite_error_summary.xlsx" if fmt == "xlsx" else "suite_error_summary.csv"


def write_output(df: pd.DataFrame, path: Path, args: argparse.Namespace) -> None:
    """
    Write the DataFrame to CSV or XLSX.

    - CSV: quote-all for maximum safety around commas/newlines.
    - XLSX: build a multi-line header (grouped Top-i), set widths, freeze panes, autofilter.
      Falls back to CSV if xlsxwriter is missing.

    "If it isn't pleasant to read, it won't get read."
    """
    if args.format == "csv":
        df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
        print("OK ->", path.parent.resolve()); print("Generated:", path); return

    # XLSX pretty (no colors if --no-colors)
    try:
        import xlsxwriter  # noqa: F401
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            sheet = "Summary"
            start_row = 2  # leave two rows for multi-line header
            df.to_excel(writer, sheet_name=sheet, startrow=start_row, index=False, header=False)
            wb  = writer.book
            ws  = writer.sheets[sheet]

            # ---------------- Formats ----------------
            border = {"border": 1}
            fmt_group = wb.add_format({
                "bold": True, "align": "center", "valign": "vcenter", **border,
                **({} if args.no_colors else {"bg_color": "#D9E1F2"})
            })
            fmt_head = wb.add_format({
                "bold": True, "align": "center", "valign": "vcenter", "text_wrap": True, **border,
                **({} if args.no_colors else {"bg_color": "#F2F2F2"})
            })
            fmt_text = wb.add_format({"text_wrap": True, "valign": "top"})
            fmt_num  = wb.add_format({"num_format": "#,##0", "align": "center"})

            # ---------------- Build two-row header ----------------
            cols = list(df.columns)
            col = 0
            # Left block (merge vertically) - Make headers more descriptive
            descriptive_headers = [
                "Test Suite",
                "Failed Test Count",
                "Unstable Test Count",
                "Total Error Messages"
            ]
            for label in descriptive_headers:
                ws.merge_range(0, col, 1, col, label, fmt_head); col += 1

            # Top-i groups - More descriptive sub-headers
            top_n = args.top_n
            for i in range(1, top_n + 1):
                ws.merge_range(0, col, 0, col + 3, f"Top {i} Most Frequent Error", fmt_group)
                descriptive_subs = [
                    f"Error Message #{i}",
                    f"Occurrences",
                    f"Failed Tests",
                    f"Unstable Tests"
                ]
                for sub in descriptive_subs:
                    ws.write(1, col, sub, fmt_head); col += 1

            # Residual group - More descriptive
            ws.merge_range(0, col, 0, col + 2, "All Other Errors", fmt_group)
            for sub in ["Other Occurrences", "Other Failed Tests", "Other Unstable Tests"]:
                ws.write(1, col, sub, fmt_head); col += 1

            # ---------------- Body with formats ----------------
            nrows, ncols = df.shape
            # Identify numeric columns for number formatting.
            num_cols = [j for j, c in enumerate(cols) if ("Events" in c or "Failed" in c or "Unstable" in c or c == "Total Messages")]
            for r in range(nrows):
                rr = start_row + r
                for j, c in enumerate(cols):
                    v = df.iat[r, j]
                    if j in num_cols:
                        ws.write_number(rr, j, 0 if (v == "" or pd.isna(v)) else int(v), fmt_num)
                    else:
                        ws.write(rr, j, "" if pd.isna(v) else str(v), fmt_text)

            # Column widths tuned for readability: suite + totals + wide message columns.
            widths = [18, 14, 14, 14]
            for _ in range(top_n):
                widths += [60, 12, 12, 12]
            widths += [12, 12, 12]
            for j, w in enumerate(widths[:ncols]):
                ws.set_column(j, j, w)

            # Freeze panes (keep Suite column visible) & add filter on header row.
            ws.freeze_panes(start_row, 1)
            ws.autofilter(1, 0, 1, ncols - 1)

        print("OK ->", path.parent.resolve()); print("Generated:", path)
    except ImportError:
        # Fallback to CSV if xlsxwriter is missing — still produce something useful.
        alt = path.with_suffix(".csv")
        df.to_csv(alt, index=False, quoting=csv.QUOTE_ALL)
        print("xlsxwriter not installed; wrote CSV instead:", alt)


def main() -> None:
    """
    Orchestrates the full ETL:
    1) Read CSV and validate columns.
    2) Melt message columns → long form.
    3) Normalize/signature + de-dupe within rows.
    4) Aggregate per (suite, signature) with FAILED/UNSTABLE splits.
    5) Build per-suite Top-N table + 'Other'.
    6) Export CSV/XLSX.

    "Long → group → rank → wide — the classic pivot pipeline."
    """
    args = parse_args()

    # Create output dir early so any logs/sidecars could be written here.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Robust read with error handling
    try:
        df = pd.read_csv(args.input, header=0, low_memory=False, on_bad_lines="skip",
                         sep=args.sep, encoding=args.encoding)
    except FileNotFoundError:
        raise SystemExit(f"Input file not found: {args.input}")
    except pd.errors.EmptyDataError:
        raise SystemExit(f"Input file is empty: {args.input}")
    except Exception as e:
        raise SystemExit(f"Error reading input file: {e}")

    # Collect requested message columns.
    message_columns = [c.strip() for c in args.message_cols.split(",") if c.strip()]
    missing_columns = [c for c in message_columns if c not in df.columns]

    # Since EXECUTION RESULT and TEST_SUITE are standard CSV columns,
    # we can assume they exist and skip expensive validation
    # Hard requirements: suite and status columns (skip validation for known standard columns)
    required_columns = [args.suite_col, args.status_col]
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise SystemExit(f"Missing required columns: {', '.join(missing_required)}")

    # Warn-and-continue: drop missing message columns but continue with the rest.
    # "Be strict on the essentials, forgiving on the peripherals."
    if missing_columns:
        print(f"[warn] Missing message column(s): {', '.join(missing_columns)} - continuing without them")
        message_columns = [c for c in message_columns if c not in missing_columns]
    if not message_columns:
        # If nothing left to analyze, fail clearly.
        raise SystemExit("No valid message columns left after filtering; nothing to summarize.")

    # ---------------- Blueprint for empty output ----------------
    # Ensures stable schema even if df becomes empty after filtering.
    base_cols = [args.suite_col, "total_messages", "failed_tests_total", "unstable_tests_total"]
    per_i = ["message", "events", "failed_tests", "unstable_tests"]
    top_cols = sum(([f"top{i}_{x}" for x in per_i] for i in range(1, args.top_n + 1)), [])
    other_cols = ["other_events", "other_failed_tests", "other_unstable_tests"]

    if df.empty:
        # Early exit: nothing to report; still write an empty but well-formed file.
        empty = pd.DataFrame(columns=base_cols + top_cols + other_cols)
        return write_output(empty, output_dir / out_name(args.format), args)

    # ---------------- FILTER TO ONLY FAILED/UNSTABLE TESTS ----------------
    # Remove all tests that aren't FAILED or UNSTABLE (PASSED, INCONCLUSIVE, etc.)
    original_count = len(df)
    df_filtered = df[df[args.status_col].isin(VALID_STATUSES)].copy()
    filtered_count = len(df_filtered)

    print(f"Filtered {original_count} tests -> {filtered_count} FAILED/UNSTABLE tests")

    if df_filtered.empty:
        print("No FAILED or UNSTABLE tests found!")
        empty = pd.DataFrame(columns=base_cols + top_cols + other_cols)
        return write_output(empty, output_dir / out_name(args.format), args)

    # ---------------- Per-suite totals by status ----------------
    # Count tests per suite by status so we can compute "Other <status>" later.
    status_totals = (
        df_filtered.groupby([args.suite_col, args.status_col]).size()
          .unstack(fill_value=0)
          .reindex(columns=VALID_STATUSES, fill_value=0)
          .reset_index()
          .rename(columns={"FAILED": "failed_tests_total", "UNSTABLE": "unstable_tests_total"})
    )

    # ---------------- PROCESS MESSAGES FROM FAILED/UNSTABLE TESTS ----------------
    # Add row IDs for tracking
    df_filtered["row_id"] = range(len(df_filtered))

    # Convert to long form - ONLY processing pre-filtered FAILED/UNSTABLE tests
    long = df_filtered.melt(
        id_vars=[args.suite_col, "row_id", args.status_col],
        value_vars=message_columns,
        var_name="message_source",
        value_name="message_raw"
    )
    long["message_raw"] = long["message_raw"].astype("string")
    long = long[long["message_raw"].notna() & (long["message_raw"].str.strip() != "")]

    print(f"Processing {len(long)} messages from {len(df_filtered)} tests")

    # ---------------- Build signatures ----------------
    # Option 1: normalized key (default) → robust grouping
    # Option 2: raw message → exact grouping
    key = long["message_raw"].map(normalize) if args.group_by == "norm" \
          else long["message_raw"].str.strip()

    # Optionally hash the signature for privacy.
    long["signature"] = key.map(lambda s: hsig(s, args.hash_algo)) if args.hash_signature else key

    # ---------------- De-dupe within a single test row ----------------
    # If the same normalized signature appears in multiple message columns of the same row,
    # count it once. Prevents double counting "the same" error for one test.
    long = long.drop_duplicates(subset=[args.suite_col, "row_id", "signature"], keep="first")

    # ---------------- Status flags for aggregation ----------------
    # Using ints makes the subsequent sum() operations simple and fast.
    long["is_failed"]   = (long[args.status_col] == "FAILED").astype(int)
    long["is_unstable"] = (long[args.status_col] == "UNSTABLE").astype(int)

    # ---------------- Aggregate per (suite, signature) ----------------
    # events         = how many test rows exhibited the message
    # failed_tests   = of those, how many were FAILED
    # unstable_tests = of those, how many were UNSTABLE
    grouped_messages = (
        long.groupby([args.suite_col, "signature"])
            .agg(events=("row_id", "count"),
                 failed_tests=("is_failed", "sum"),
                 unstable_tests=("is_unstable", "sum"))
            .reset_index()
    )

    # Add a visible example message for each signature (for human-friendly output).
    first_messages = (long.drop_duplicates([args.suite_col, "signature"])
                 [[args.suite_col, "signature", "message_raw"]]
                 .rename(columns={"message_raw": "example_message"}))
    grouped_messages = grouped_messages.merge(first_messages, on=[args.suite_col, "signature"], how="left")

    # ---------------- Build the wide Top-N table per suite ----------------
    rows = []
    status_totals_indexed = status_totals.set_index(args.suite_col)
    for suite, suite_messages in grouped_messages.groupby(args.suite_col, sort=False):
        # Rank messages by frequency within the suite.
        suite_messages = suite_messages.sort_values("events", ascending=False)
        row = {args.suite_col: suite, "total_messages": suite_messages["events"].sum()}

        # Fetch per-suite status totals to compute residual "Other".
        if suite in status_totals_indexed.index:
            failed_total = status_totals_indexed.loc[suite, "failed_tests_total"]
            unstable_total = status_totals_indexed.loc[suite, "unstable_tests_total"]
        else:
            failed_total = unstable_total = 0

        row["failed_tests_total"] = failed_total
        row["unstable_tests_total"] = unstable_total

        # Fetch per-suite status totals to compute residual "Other".
        if suite in status_totals_indexed.index:
            failed_total = status_totals_indexed.loc[suite, "failed_tests_total"]
            unstable_total = status_totals_indexed.loc[suite, "unstable_tests_total"]
        else:
            failed_total = unstable_total = 0

        row["failed_tests_total"] = failed_total
        row["unstable_tests_total"] = unstable_total

        # Fill Top-N columns.
        sum_events = sum_failed = sum_unstable = 0
        for i in range(1, args.top_n + 1):
            if i <= len(suite_messages):
                msg = suite_messages.iloc[i - 1]["example_message"]
                # Optional readability: trim very long exemplars.
                if args.truncate_len and isinstance(msg, str) and len(msg) > args.truncate_len:
                    msg = msg[:args.truncate_len - 1] + "..."
                row[f"top{i}_message"]        = msg
                row[f"top{i}_events"]         = suite_messages.iloc[i - 1]["events"]
                row[f"top{i}_failed_tests"]   = suite_messages.iloc[i - 1]["failed_tests"]
                row[f"top{i}_unstable_tests"] = suite_messages.iloc[i - 1]["unstable_tests"]

                sum_events += suite_messages.iloc[i - 1]["events"]
                sum_failed += suite_messages.iloc[i - 1]["failed_tests"]
                sum_unstable += suite_messages.iloc[i - 1]["unstable_tests"]
            else:
                # Fill empty slots for suites with fewer than top_n messages
                row[f"top{i}_message"]        = ""
                row[f"top{i}_events"]         = 0
                row[f"top{i}_failed_tests"]   = 0
                row[f"top{i}_unstable_tests"] = 0

        # Residual bucket: long tail not in Top-N.
        row["other_events"]          = max(0, row["total_messages"] - sum_events)
        row["other_failed_tests"]    = max(0, row["failed_tests_total"] - sum_failed)
        row["other_unstable_tests"]  = max(0, row["unstable_tests_total"] - sum_unstable)
        rows.append(row)

    # Combine suites and order columns.
    results_df = pd.DataFrame(rows).sort_values("total_messages", ascending=False)

    # ---------------- Final ordering + labels ----------------
    ordered_columns = [args.suite_col, "failed_tests_total", "unstable_tests_total", "total_messages"]
    for i in range(1, args.top_n + 1):
        ordered_columns += [f"top{i}_message", f"top{i}_events", f"top{i}_failed_tests", f"top{i}_unstable_tests"]
    ordered_columns += ["other_events", "other_failed_tests", "other_unstable_tests"]

    # Ensure every expected col exists (even if some Top-k cells are empty).
    for col in ordered_columns:
        if col not in results_df.columns:
            results_df[col] = "" if "message" in col else 0
    results_df = results_df[ordered_columns]

    # Pretty mode: rename columns with more descriptive names
    if args.pretty:
        column_renames = {
            args.suite_col: "Test Suite Name",
            "failed_tests_total": "Total Failed Tests",
            "unstable_tests_total": "Total Unstable Tests",
            "total_messages": "Total Error Messages",
            "other_events": "Other Error Occurrences",
            "other_failed_tests": "Other Failed Tests",
            "other_unstable_tests": "Other Unstable Tests",
        }
        for i in range(1, args.top_n + 1):
            column_renames.update({
                f"top{i}_message":        f"Top {i} Error Message",
                f"top{i}_events":         f"Top {i} Occurrences",
                f"top{i}_failed_tests":   f"Top {i} Failed Count",
                f"top{i}_unstable_tests": f"Top {i} Unstable Count",
            })
        results_df = results_df.rename(columns=column_renames)

        # Write output (support csv/xlsx/both).
        if args.format in ("csv", "both"):
            write_output(results_df,
                         output_dir / "suite_error_summary.csv",
                         Namespace(**{**vars(args), "format": "csv"}))

        if args.format in ("xlsx", "both"):
            write_output(results_df,
                         output_dir / "suite_error_summary.xlsx",
                         Namespace(**{**vars(args), "format": "xlsx"}))

        return

    # Non-pretty path: single write using the requested format.
    write_output(results_df, output_dir / out_name(args.format), args)


if __name__ == "__main__":
    # "Tools are only useful if they run trivially."
    main()
