# Suite Error Summary Tool

A Python tool that analyzes test execution logs and produces a "Top-N (top 5 by default) error messages + Other" summary with FAILED/UNSTABLE breakdowns, exported as CSV or nicely formatted XLSX.

## Business Value

### Accelerated Issue Detection and Resolution

This tool transforms raw test execution data into actionable insights, enabling development teams to:

- **Rapidly identify critical issues**: Automatically surfaces the most frequent error patterns across test suites, eliminating manual log analysis
- **Prioritize remediation efforts**: Quantifies error impact by frequency and affected test counts, enabling data-driven decision making
- **Reduce mean time to resolution (MTTR)**: Consolidated error summaries allow teams to focus on root causes rather than symptoms

### Key Benefits

- **Pattern Recognition**: Normalized error grouping reveals underlying issues that might be missed when examining individual test failures
- **Quality Metrics**: Provides quantifiable data on test suite stability and error distribution patterns
- **Performance Optimized**: Efficient DataFrame processing with ~85% reduction in data access operations for faster analysis of large datasets

### Use Cases

- **Daily Test Result Analysis**: Quickly identify new issues introduced in recent builds
- **Technical Debt Prioritization**: Focus engineering efforts on the most impactful error patterns
- **Quality Trend Analysis**: Track error patterns over time to measure improvement initiatives

## Quick Start

### Prerequisites

- Python 3.7+ installed on your system
- CSV file with test execution data

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd top-failure-test-by-suite
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Tool

#### Option 1: Using the Bash Wrapper (Recommended)

**Basic usage (CSV and XLSX):**
```bash
./suite_summary.sh your_logs.csv
```
- Creates both CSV and XLSX files by default
- Output folder: `out_name_of_your_file`

**With custom output directory:**
```bash
./suite_summary.sh your_logs.csv my_output_folder
```

**With additional options (CSV and XLSX):**
```bash
EXTRA_FLAGS="--output-format both --use-friendly-headers --top-errors-count 10" ./suite_summary.sh your_logs.csv
```

**For only XLSX:**
```bash
EXTRA_FLAGS="--output-format xlsx --use-friendly-headers" ./suite_summary.sh your_logs.csv
```

**For only CSV:**
```bash
EXTRA_FLAGS="--output-format csv" ./suite_summary.sh your_logs.csv
```

#### Option 2: Direct Python Execution

```bash
python suite_error_summary.py \
  --input-file logs.csv \
  --output-directory out_logs \
  --error-message-columns "FAILURE MESSAGE 1,FAILURE MESSAGE 2" \
  --test-suite-column TEST_SUITE \
  --test-status-column "EXECUTION RESULT" \
  --use-friendly-headers \
  --output-format xlsx
```

## Input Requirements

Your CSV file must contain:

- **Suite column**: Groups tests by test suite (e.g., `TEST_SUITE`)
- **Status column**: Test execution status (e.g., `EXECUTION RESULT`)
- **Message columns**: Error/failure messages (e.g., `FAILURE MESSAGE 1`, `FAILURE MESSAGE 2`)

**Supported status values**: `FAILED`, `UNSTABLE`

## Configuration Options

### Environment Variables (for bash wrapper)

```bash
# New user-friendly variable names (recommended)
export ERROR_MESSAGE_COLUMNS="FAILURE MESSAGE 1,FAILURE MESSAGE 2"  # Message columns to analyze
export TEST_SUITE_COLUMN="TEST_SUITE"                              # Suite grouping column
export TEST_STATUS_COLUMN="EXECUTION RESULT"                       # Status column
export TOP_ERRORS_COUNT=5                                          # Number of top messages per suite
export GROUPING_METHOD=normalized                                  # normalized or exact
export EXTRA_FLAGS="--use-friendly-headers --output-format xlsx"  # Additional options

# Legacy variable names (still supported for backwards compatibility)
export MESSAGE_COLS="FAILURE MESSAGE 1,FAILURE MESSAGE 2"
export SUITE_COL="TEST_SUITE"
export STATUS_COL="EXECUTION RESULT"
export TOPN=5
export GROUP_BY=norm
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input-file` | Input CSV file path | Required |
| `--output-directory` | Output directory | Required |
| `--error-message-columns` | Comma-separated message columns | Required |
| `--test-suite-column` | Suite grouping column | `"TEST_SUITE"` |
| `--test-status-column` | Status column name | `"EXECUTION RESULT"` |
| `--top-errors-count` | Number of top messages per suite | `5` |
| `--grouping-method` | Grouping method: `normalized` or `exact` | `normalized` |
| `--output-format` | Output format: `csv`, `xlsx`, `both` | `csv` |
| `--use-friendly-headers` | Use human-friendly column names | `false` |
| `--max-message-length` | Trim long messages to N chars | `0` (disabled) |
| `--disable-excel-colors` | Disable XLSX colors | `false` |

## Output Files

The tool generates:

- **CSV**: `suite_error_summary.csv` - Raw data format
- **XLSX**: `suite_error_summary.xlsx` - Formatted spreadsheet with:
  - Multi-level headers
  - Frozen panes and filters
  - Proper column widths
  - Conditional formatting (unless `--no-colors`)
- **Log**: `run.log` - Execution log

### Example Results

**XLSX File**
<img width="1854" height="714" alt="image" src="https://github.com/user-attachments/assets/f2b0c4a0-b284-464a-b77d-a86299aebde6" />

**CSV File**
<img width="995" height="176" alt="image" src="https://github.com/user-attachments/assets/fb5b4577-52f7-494c-a1ee-9fde4139b8ec" />

## Examples

### Basic Analysis
```bash
./suite_summary.sh test_results.csv
```

### Advanced Configuration
```bash
# Top 10 messages, both formats, pretty headers
EXTRA_FLAGS="--top-n 10 --format both --pretty" ./suite_summary.sh test_results.csv analysis_output
```

### Performance Notes
- **Optimized for Large Datasets**: Recent performance improvements reduce DataFrame access operations by ~85%
- **Memory Efficient**: Helper functions minimize code duplication and memory usage
- **Fast Processing**: Consolidated CSV writing and optimized data access patterns

## Troubleshooting

### Python Not Found
**Windows**: Try `python`, `py`, or install from [python.org](https://python.org)
**Linux/Mac**: Try `python3` or use your package manager

### Missing Dependencies
```bash
pip install pandas xlsxwriter
```

### Permission Errors (Git Bash/Windows)
```bash
chmod +x suite_summary.sh
```

### Large Files
For very large CSV files, consider:
- Increase system memory
- Use `--format csv` only (faster than XLSX)
- Filter data beforehand

## File Structure

```
.
├── suite_summary.sh            # Bash wrapper script (optimized flags)
├── suite_error_summary.py      # Main Python analysis tool (performance optimized)
├── requirements.txt            # Python dependencies
├── logs.csv                    # Example input data
├── out_logs/                   # Example output directory
│   ├── suite_error_summary.csv # Generated CSV report
│   ├── suite_error_summary.xlsx# Generated Excel report
│   └── run.log                 # Execution log
└── README.md                   # This file
```
