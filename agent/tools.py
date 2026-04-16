# ============================================================
# agent/tools.py
# ============================================================
# This file defines all the TOOLS the agent can use.
#
# WHAT IS A TOOL?
#   A tool is a Python function wrapped so the LLM can call it.
#   The LLM cannot run Python, open files, or draw charts on
#   its own — it can only generate text. Tools are the bridge
#   between the LLM's text output and real-world actions.
#
#   Think of it like this:
#     LLM Brain  →  decides WHAT to do
#     Tools      →  actually DO it
#
# TOOLS IN THIS FILE:
#   1. load_file_tool       → reads CSV / Excel into pandas
#   2. python_repl_tool     → executes Python code live
#   3. generate_chart_tool  → runs chart code, saves PNG
#   4. save_report_tool     → writes text report to disk
#
# HOW LANGCHAIN TOOLS WORK:
#   We use @tool decorator from langchain. It:
#     - Reads the function's docstring as the tool description
#     - The LLM uses this description to decide WHEN to use it
#     - Wraps the function so LangGraph nodes can call it
# ============================================================

import os
import traceback
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # IMPORTANT: use non-GUI backend
                                # Without this, matplotlib tries to
                                # open a display window and crashes
                                # on servers / headless environments
import matplotlib.pyplot as plt
import seaborn as sns

from langchain.tools import tool
from langchain_experimental.tools import PythonREPLTool
from datetime import datetime


# ============================================================
# SHARED HELPER — used internally by multiple tools
# ============================================================

def _ensure_output_dir() -> str:
    """
    Makes sure the outputs/ folder exists before saving files.
    Returns the path to the outputs directory.

    We call this inside every tool that saves a file so we
    never get a 'directory not found' error.
    """
    output_dir = os.getenv('OUTPUT_FOLDER', 'outputs')
    os.makedirs(output_dir, exist_ok=True)   # exist_ok=True means
                                              # no error if it already exists
    return output_dir


# ============================================================
# TOOL 1 — load_file_tool
# ============================================================

@tool
def load_file_tool(file_path: str) -> str:
    """
    Load a CSV or Excel file from disk and return a detailed
    summary of its structure for the AI to understand.

    The summary includes:
    - Column names and their data types
    - Shape (rows x columns)
    - First 5 rows as a table
    - Basic statistics (min, max, mean for numeric columns)
    - Missing value counts per column

    Use this tool FIRST before writing any analysis code.
    The summary tells you what columns exist and what type
    of data they contain, so you can write correct pandas code.

    Args:
        file_path: Path to the CSV or Excel file.
                   Examples: 'data/bank_statement.csv'
                             'data/stocks.xlsx'

    Returns:
        A multi-line string summarising the dataset, OR
        an error message if the file cannot be loaded.
    """
    try:
        # -------------------------------------------------------
        # STEP 1: Detect file type and load accordingly
        # -------------------------------------------------------
        file_path = file_path.strip()           # remove accidental spaces
        extension = os.path.splitext(file_path)[1].lower()

        if extension == '.csv':
            # Try UTF-8 first, fall back to latin-1 encoding
            # (bank statements often use latin-1 for £ symbols)
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')

        elif extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)       # openpyxl handles this

        else:
            return (f"ERROR: Unsupported file type '{extension}'.\n"
                    f"Supported formats: .csv, .xlsx, .xls")

        # -------------------------------------------------------
        # STEP 2: Build a rich text summary of the dataframe
        # This is what the LLM will read to understand the data.
        # -------------------------------------------------------
        lines = []

        lines.append("=" * 50)
        lines.append(f"FILE LOADED: {file_path}")
        lines.append("=" * 50)

        # Basic shape info
        lines.append(f"\nSHAPE: {df.shape[0]} rows × {df.shape[1]} columns")

        # Column names and their data types
        lines.append("\nCOLUMNS AND DATA TYPES:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_info = f"  ({null_count} nulls)" if null_count > 0 else ""
            lines.append(f"  • {col}  [{dtype}]{null_info}")

        # First 5 rows — gives the LLM a concrete example of the data
        lines.append("\nFIRST 5 ROWS:")
        lines.append(df.head().to_string(index=True))

        # Numeric column statistics — helps LLM understand ranges
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            lines.append("\nNUMERIC COLUMN STATISTICS:")
            lines.append(df[numeric_cols].describe().round(2).to_string())

        # Unique value counts for non-numeric columns (categorical data)
        # This tells the LLM what categories exist to group by
        categorical_cols = df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()

        if categorical_cols:
            lines.append("\nCATEGORICAL COLUMNS — UNIQUE VALUES:")
            for col in categorical_cols[:5]:        # limit to first 5 cols
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 15:          # show all if few unique
                    lines.append(f"  {col}: {list(unique_vals)}")
                else:                               # otherwise show count
                    lines.append(f"  {col}: {len(unique_vals)} unique values")

        lines.append("\n" + "=" * 50)
        lines.append("READY FOR ANALYSIS")
        lines.append("=" * 50)

        return "\n".join(lines)

    except FileNotFoundError:
        return (f"ERROR: File not found at path '{file_path}'.\n"
                f"Make sure the file exists in the data/ folder.\n"
                f"Available files in data/: "
                f"{os.listdir('data') if os.path.exists('data') else 'data/ folder missing'}")

    except Exception as e:
        # Catch any other error (permission denied, corrupt file, etc.)
        return f"ERROR loading file: {type(e).__name__}: {str(e)}"


# ============================================================
# TOOL 2 — python_repl_tool
# ============================================================
# We use LangChain's built-in PythonREPLTool here.
# It creates a live Python interpreter session that persists
# across multiple calls — so if you import pandas in one call,
# it's still imported in the next call. This is exactly what
# we need for multi-step data analysis.
#
# HOW IT WORKS INTERNALLY:
#   PythonREPLTool uses Python's exec() function to run code.
#   It captures everything printed to stdout and returns it.
#   If the code raises an exception, it returns the traceback.
#
# SAFETY NOTE:
#   This tool executes REAL Python on your machine.
#   Only use it with code generated by YOUR agent — never
#   expose it to untrusted user input directly.
# ============================================================

python_repl_tool = PythonREPLTool()

# Give it a clearer name so the LLM knows when to use it
python_repl_tool.name = "python_repl"
python_repl_tool.description = (
    "Execute Python code and return the printed output. "
    "Use this to run pandas operations, compute statistics, "
    "filter data, group by categories, calculate totals, "
    "and any other data analysis that requires real computation. "
    "Always use print() to output results — return values are not captured. "
    "pandas (pd), numpy (np) are available. "
    "The session persists, so imports from previous calls still work."
)


# ============================================================
# TOOL 3 — generate_chart_tool
# ============================================================

@tool
def generate_chart_tool(chart_code: str) -> str:
    """
    Execute Python matplotlib/seaborn chart code and save
    the resulting chart as a PNG image to the outputs/ folder.

    IMPORTANT RULES for the chart_code you provide:
    1. Always import what you need (pandas, matplotlib, seaborn)
    2. Always load the data inside the code (read_csv / read_excel)
    3. Always call plt.tight_layout() before saving
    4. Do NOT call plt.show() — the tool saves it automatically
    5. Do NOT call plt.savefig() — the tool handles saving

    Example chart_code:
        import pandas as pd
        import matplotlib.pyplot as plt
        df = pd.read_csv('data/bank_statement.csv')
        totals = df.groupby('Category')['Amount'].sum()
        plt.figure(figsize=(8, 6))
        totals.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Spending by Category')
        plt.tight_layout()

    Args:
        chart_code: Complete Python code to generate the chart.
                    Must include imports and data loading.

    Returns:
        Success message with the saved file path, OR
        an error message with the traceback if it fails.
    """
    try:
        # -------------------------------------------------------
        # STEP 1: Close any previously open matplotlib figures
        # This prevents charts from stacking on top of each other
        # -------------------------------------------------------
        plt.close('all')

        # -------------------------------------------------------
        # STEP 2: Build a safe execution environment
        # We give the code access to pandas, matplotlib, seaborn,
        # and numpy — the standard data analysis stack.
        # -------------------------------------------------------
        import numpy as np

        exec_globals = {
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'np': np,
            '__builtins__': __builtins__,   # allow print(), len(), etc.
        }

        # -------------------------------------------------------
        # STEP 3: Execute the chart code
        # exec() runs the string as Python code in exec_globals
        # -------------------------------------------------------
        exec(chart_code, exec_globals)

        # -------------------------------------------------------
        # STEP 4: Save the chart to outputs/ with a timestamp
        # Timestamped filenames prevent overwriting previous charts
        # -------------------------------------------------------
        output_dir = _ensure_output_dir()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        chart_filename = f"chart_{timestamp}.png"
        chart_path = os.path.join(output_dir, chart_filename)

        # Save with high DPI for crisp images
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close('all')    # always close after saving to free memory

        return (f"SUCCESS: Chart saved to '{chart_path}'\n"
                f"The chart has been generated and saved as a PNG image.")

    except Exception as e:
        plt.close('all')    # close even on error to avoid memory leaks
        error_details = traceback.format_exc()
        return (f"ERROR generating chart: {type(e).__name__}: {str(e)}\n"
                f"Full traceback:\n{error_details}")


# ============================================================
# TOOL 4 — save_report_tool
# ============================================================

@tool
def save_report_tool(report_content: str) -> str:
    """
    Save the final analysis report as a text file to the
    outputs/ folder. Call this as the LAST step after all
    analysis and chart generation is complete.

    The report should include:
    - A clear answer to the user's question
    - Key numbers and findings from the analysis
    - Reference to any charts that were generated
    - Any important observations or patterns noticed

    Args:
        report_content: The full text of the analysis report.
                        Write it in plain English that a non-technical
                        person can understand.

    Returns:
        Success message with the saved file path, OR
        an error message if saving fails.
    """
    try:
        output_dir = _ensure_output_dir()

        # Timestamp in the filename so reports don't overwrite each other
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"report_{timestamp}.txt"
        report_path = os.path.join(output_dir, report_filename)

        # Add a header to every report automatically
        header = (
            f"DATA ANALYSIS REPORT\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'=' * 50}\n\n"
        )

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(header + report_content)

        return (f"SUCCESS: Report saved to '{report_path}'\n"
                f"The analysis report has been written to disk.")

    except Exception as e:
        return f"ERROR saving report: {type(e).__name__}: {str(e)}"


# ============================================================
# TOOLS LIST — exported for use in nodes.py and graph.py
# ============================================================
# We collect all tools into one list so we can pass them
# to the LangGraph agent in a single line:
#   agent = create_react_agent(llm, tools=ALL_TOOLS)
#
# The order here doesn't matter — the LLM picks which tool
# to use based on each tool's name and description.
# ============================================================

ALL_TOOLS = [
    load_file_tool,
    python_repl_tool,
    generate_chart_tool,
    save_report_tool,
]




# """
# Executable tools available to the agent nodes.
# """

# import io
# import sys
# import traceback
# from pathlib import Path

# import matplotlib
# matplotlib.use("Agg")  # non-interactive backend
# import matplotlib.pyplot as plt


# # ---------------------------------------------------------------------------
# # Python REPL
# # ---------------------------------------------------------------------------

# def run_python(code: str, local_vars: dict | None = None) -> tuple[str, dict]:
#     """
#     Execute *code* and return (stdout_output, updated_locals).
#     Captures stdout; surfaces exceptions as returned strings (not raises).
#     """
#     buffer = io.StringIO()
#     local_vars = local_vars or {}
#     try:
#         sys.stdout = buffer
#         exec(code, {}, local_vars)  # noqa: S102
#     except Exception:
#         local_vars["__error__"] = traceback.format_exc()
#     finally:
#         sys.stdout = sys.__stdout__
#     return buffer.getvalue(), local_vars


# # ---------------------------------------------------------------------------
# # Chart generator
# # ---------------------------------------------------------------------------

# def save_current_chart(output_path: str = "outputs/chart.png") -> str:
#     """
#     Save whatever matplotlib figure is currently active to *output_path*.
#     Returns the absolute path of the saved file.
#     """
#     path = Path(output_path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(path, bbox_inches="tight", dpi=150)
#     plt.close()
#     return str(path.resolve())


# # ---------------------------------------------------------------------------
# # File saver
# # ---------------------------------------------------------------------------

# def save_text(content: str, output_path: str = "outputs/report.txt") -> str:
#     """
#     Write *content* to *output_path* and return the absolute path.
#     """
#     path = Path(output_path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     path.write_text(content, encoding="utf-8")
#     return str(path.resolve())
