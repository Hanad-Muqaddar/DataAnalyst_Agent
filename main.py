# ============================================================
# main.py
# ============================================================
# This is the ENTRY POINT of the Data Analyst Agent.
# Run this file from your terminal to start the agent:
#
#   python main.py
#
# WHAT THIS FILE DOES:
#   1. Loads environment variables from .env
#   2. Validates your setup (API key, folders, packages)
#   3. Shows a welcome banner and lists available data files
#   4. Asks you which file to analyse and what question to ask
#   5. Builds and runs the LangGraph agent
#   6. Streams live progress as each node executes
#   7. Displays the final answer and output file paths
#   8. Asks if you want to run another analysis (loop)
#
# ARCHITECTURE REMINDER:
#   main.py
#     └── build_graph()         ← agent/graph.py
#           ├── load_file_node  ← agent/nodes.py
#           ├── llm_reason_node ← agent/nodes.py  (calls OpenAI)
#           ├── run_code_node   ← agent/nodes.py  (calls tools.py)
#           ├── generate_chart  ← agent/nodes.py  (calls tools.py)
#           ├── save_report     ← agent/nodes.py  (calls tools.py)
#           └── handle_error    ← agent/nodes.py
# ============================================================

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env FIRST before any other imports that need env vars ──
load_dotenv()

from agent.graph import build_graph, build_initial_state


# ============================================================
# TERMINAL COLOURS
# ============================================================
# These ANSI escape codes add colour to terminal output.
# They make it much easier to follow what the agent is doing
# as it prints progress updates step by step.
#
# Format: \033[CODEm  where CODE is the colour number
# \033[0m resets back to default terminal colour.
# ============================================================

class Colour:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    PURPLE  = "\033[95m"
    TEAL    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    GRAY    = "\033[90m"
    WHITE   = "\033[97m"

def c(text: str, colour: str) -> str:
    """Wrap text in a terminal colour code."""
    return f"{colour}{text}{Colour.RESET}"


# ============================================================
# VALIDATION
# ============================================================

def validate_setup() -> bool:
    """
    Checks that everything is in place before running the agent.
    Prints helpful error messages if something is missing.

    Checks:
      1. OPENAI_API_KEY is set in .env
      2. data/ folder exists and has at least one file
      3. outputs/ folder exists (creates it if not)
      4. Required packages are importable

    Returns:
        True if all checks pass, False if any check fails.
    """
    print(c("\n Validating setup...", Colour.GRAY))
    all_ok = True

    # ── Check 1: OpenAI API key ─────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "your-openai-api-key-here":
        print(c("  ✗ OPENAI_API_KEY not set in .env", Colour.RED))
        print(c("    → Open .env and paste your key from platform.openai.com/api-keys", Colour.YELLOW))
        all_ok = False
    else:
        masked = api_key[:8] + "..." + api_key[-4:]
        print(c(f"  ✓ OPENAI_API_KEY found ({masked})", Colour.GREEN))

    # ── Check 2: data/ folder ───────────────────────────────
    data_folder = os.getenv("DATA_FOLDER", "data")
    if not os.path.exists(data_folder):
        print(c(f"  ✗ '{data_folder}/' folder not found", Colour.RED))
        print(c(f"    → Create the folder and add a CSV or Excel file", Colour.YELLOW))
        all_ok = False
    else:
        # Find all CSV and Excel files in the data folder
        data_files = _get_data_files(data_folder)
        if not data_files:
            print(c(f"  ✗ No CSV or Excel files in '{data_folder}/'", Colour.RED))
            print(c(f"    → Add a .csv or .xlsx file to the data/ folder", Colour.YELLOW))
            all_ok = False
        else:
            print(c(f"  ✓ data/ folder found ({len(data_files)} file(s))", Colour.GREEN))

    # ── Check 3: outputs/ folder ────────────────────────────
    output_folder = os.getenv("OUTPUT_FOLDER", "outputs")
    os.makedirs(output_folder, exist_ok=True)
    print(c(f"  ✓ outputs/ folder ready", Colour.GREEN))

    # ── Check 4: Required packages ──────────────────────────
    required = {
        "langgraph":   "langgraph",
        "langchain":   "langchain",
        "openai":      "langchain_openai",
        "pandas":      "pandas",
        "matplotlib":  "matplotlib",
    }
    missing = []
    for display_name, import_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(display_name)

    if missing:
        print(c(f"  ✗ Missing packages: {', '.join(missing)}", Colour.RED))
        print(c(f"    → Run: pip install -r requirements.txt", Colour.YELLOW))
        all_ok = False
    else:
        print(c(f"  ✓ All required packages installed", Colour.GREEN))

    return all_ok


def _get_data_files(folder: str) -> list:
    """
    Returns a list of CSV and Excel file paths in the given folder.

    Args:
        folder: Path to the data directory

    Returns:
        List of file path strings relative to the project root.
    """
    supported = {".csv", ".xlsx", ".xls"}
    files = []
    for f in sorted(os.listdir(folder)):
        if Path(f).suffix.lower() in supported:
            files.append(os.path.join(folder, f))
    return files


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_banner():
    """
    Prints a welcome banner to the terminal when the agent starts.
    This is purely cosmetic — it makes the startup feel polished.
    """
    banner = f"""
{c('╔══════════════════════════════════════════════════════╗', Colour.PURPLE)}
{c('║', Colour.PURPLE)}       {c('DATA ANALYST AI AGENT', Colour.WHITE + Colour.BOLD)}                         {c('║', Colour.PURPLE)}
{c('║', Colour.PURPLE)}       {c('Powered by LangGraph + GPT-4o', Colour.GRAY)}                {c('║', Colour.PURPLE)}
{c('╚══════════════════════════════════════════════════════╝', Colour.PURPLE)}
"""
    print(banner)


def print_available_files(data_folder: str):
    """
    Prints a numbered list of available data files so the user
    can pick one by number or type the path manually.

    Args:
        data_folder: Path to the data directory
    """
    files = _get_data_files(data_folder)

    print(c("\n Available data files:", Colour.TEAL))
    print(c(" " + "─" * 45, Colour.GRAY))

    for i, filepath in enumerate(files, 1):
        filename   = os.path.basename(filepath)
        size_bytes = os.path.getsize(filepath)
        size_kb    = round(size_bytes / 1024, 1)
        ext        = Path(filename).suffix.upper()

        # Colour-code by file type
        ext_colour = Colour.GREEN if ext == ".CSV" else Colour.YELLOW
        print(
            f"  {c(str(i), Colour.WHITE)}. "
            f"{c(filename, Colour.WHITE)}  "
            f"{c(ext, ext_colour)}  "
            f"{c(f'{size_kb} KB', Colour.GRAY)}"
        )

    print(c(" " + "─" * 45, Colour.GRAY))
    return files


def print_section(title: str):
    """Prints a formatted section header during agent execution."""
    print(f"\n{c('▶', Colour.PURPLE)} {c(title, Colour.WHITE)}")
    print(c("  " + "─" * 50, Colour.GRAY))


def print_final_answer(final_answer: str):
    """
    Displays the agent's final answer in a formatted box
    so it stands out clearly from the progress logs.

    Args:
        final_answer: The complete answer string from state
    """
    print(f"\n{c('╔══════════════════════════════════════════════════════╗', Colour.GREEN)}")
    print(f"{c('║', Colour.GREEN)}  {c('ANALYSIS COMPLETE', Colour.WHITE + Colour.BOLD)}                                   {c('║', Colour.GREEN)}")
    print(f"{c('╚══════════════════════════════════════════════════════╝', Colour.GREEN)}")
    print()

    # Print each line of the final answer with light indentation
    for line in final_answer.split("\n"):
        if line.startswith("ANSWER:"):
            print(c(f"  {line}", Colour.GREEN + Colour.BOLD))
        elif line.startswith("KEY FINDINGS:") or line.startswith("OUTPUTS SAVED:"):
            print(c(f"  {line}", Colour.TEAL))
        elif line.startswith("•"):
            print(c(f"  {line}", Colour.WHITE))
        else:
            print(f"  {line}")


def print_output_files(result_state: dict):
    """
    Lists all output files the agent produced during the run.
    Checks the outputs/ folder and prints size and path for each.

    Args:
        result_state: The final state dict returned by app.invoke()
    """
    output_folder = os.getenv("OUTPUT_FOLDER", "outputs")

    print(f"\n{c('  Output files created:', Colour.TEAL)}")

    # Show chart path from state
    chart_path = result_state.get("chart_path")
    if chart_path and os.path.exists(chart_path):
        size_kb = round(os.path.getsize(chart_path) / 1024, 1)
        print(f"  {c('Chart :', Colour.YELLOW)}  {chart_path}  {c(f'({size_kb} KB)', Colour.GRAY)}")

    # Show all recently created files in outputs/ folder
    if os.path.exists(output_folder):
        for f in sorted(os.listdir(output_folder)):
            full_path = os.path.join(output_folder, f)
            # Skip the chart (already shown above)
            if chart_path and full_path == chart_path:
                continue
            if f.endswith(".txt"):
                size_kb = round(os.path.getsize(full_path) / 1024, 1)
                print(f"  {c('Report:', Colour.YELLOW)}  {full_path}  {c(f'({size_kb} KB)', Colour.GRAY)}")


# ============================================================
# USER INPUT HELPERS
# ============================================================

def get_file_choice(data_folder: str) -> str:
    """
    Interactively asks the user to choose a data file.
    Accepts either a number (from the listed files) or a
    full file path typed manually.

    Args:
        data_folder: Path to the folder containing data files

    Returns:
        The selected file path as a string.
    """
    files = _get_data_files(data_folder)

    while True:
        print()
        user_input = input(
            c("  Enter file number or full path: ", Colour.TEAL)
        ).strip()

        # User entered a number → pick from the list
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(files):
                chosen = files[idx]
                print(c(f"  Selected: {chosen}", Colour.GREEN))
                return chosen
            else:
                print(c(f"  Invalid number. Choose 1–{len(files)}.", Colour.RED))

        # User typed a path directly
        elif os.path.exists(user_input):
            print(c(f"  Selected: {user_input}", Colour.GREEN))
            return user_input

        else:
            print(c(f"  File not found: '{user_input}'", Colour.RED))
            print(c(f"  Try again or enter a number from the list above.", Colour.YELLOW))


def get_question() -> str:
    """
    Asks the user to type their analysis question.
    Shows example questions to help them get started.

    Returns:
        The user's question as a string.
    """
    print(c("\n  Example questions you can ask:", Colour.GRAY))
    examples = [
        "What category did I spend the most on?",
        "Show me a breakdown of all spending by category",
        "What was my total spending this month?",
        "Which day had the highest transaction?",
        "Show me a trend of my spending over time",
    ]
    for ex in examples:
        print(c(f"    → {ex}", Colour.GRAY))

    print()
    while True:
        question = input(
            c("  Your question: ", Colour.TEAL)
        ).strip()

        if len(question) < 5:
            print(c("  Question too short. Please be more specific.", Colour.RED))
        else:
            return question


# ============================================================
# MAIN RUN FUNCTION
# ============================================================

def run_agent(file_path: str, question: str) -> dict:
    """
    Builds the LangGraph agent and runs it with the given
    file and question. Streams live node-by-node progress
    to the terminal as the agent works.

    Args:
        file_path: Path to the CSV or Excel file
        question:  The user's analysis question

    Returns:
        The final state dictionary after the graph completes.
        Contains: final_answer, chart_path, code_output, etc.
    """
    print_section("Building agent graph")
    app = build_graph()

    # Build the initial state — all fields initialised
    initial_state = build_initial_state(
        file_path=file_path,
        user_question=question,
    )

    print_section("Running analysis")
    print(c(f"  File     : {file_path}", Colour.WHITE))
    print(c(f"  Question : {question}", Colour.WHITE))
    print()

    # -------------------------------------------------------
    # STREAM THE GRAPH EXECUTION
    # -------------------------------------------------------
    # app.stream() runs the graph and yields a dict after
    # each node completes. The dict has the node name as key
    # and the state updates that node returned as value.
    #
    # This lets us print live progress:
    #   "✓ load_file completed"
    #   "✓ llm_reason completed"
    #   etc.
    #
    # We collect all updates and merge them into final_state.
    # -------------------------------------------------------
    final_state = {}
    start_time  = time.time()

    # Node display names — maps internal node name to
    # a human-readable description shown in the terminal
    node_labels = {
        "load_file":       "Loading data file",
        "llm_reason":      "LLM reasoning (GPT-4o thinking...)",
        "run_code":        "Executing Python code",
        "generate_chart":  "Generating chart",
        "save_report":     "Writing analysis report",
        "handle_error":    "Handling error — preparing retry",
    }

    try:
        # stream() yields {node_name: state_updates} dicts
        for step in app.stream(initial_state):

            for node_name, state_updates in step.items():

                # Skip LangGraph internal nodes
                if node_name.startswith("__"):
                    continue

                # Get human-readable label for this node
                label = node_labels.get(node_name, node_name)
                elapsed = round(time.time() - start_time, 1)

                # Print node completion with elapsed time
                print(
                    c(f"  ✓ {label}", Colour.GREEN) +
                    c(f"  ({elapsed}s)", Colour.GRAY)
                )

                # Show brief status info for key nodes
                if node_name == "load_file" and state_updates.get("dataframe_summary"):
                    summary = state_updates["dataframe_summary"]
                    # Extract just the SHAPE line for a quick preview
                    for line in summary.split("\n"):
                        if "SHAPE:" in line:
                            print(c(f"    {line.strip()}", Colour.GRAY))
                            break

                elif node_name == "run_code" and state_updates.get("code_output"):
                    output = state_updates["code_output"].strip()
                    # Show first 2 lines of code output as preview
                    preview_lines = output.split("\n")[:2]
                    for line in preview_lines:
                        if line.strip():
                            print(c(f"    {line.strip()[:70]}", Colour.GRAY))

                elif node_name == "generate_chart" and state_updates.get("chart_path"):
                    print(c(f"    Saved: {state_updates['chart_path']}", Colour.GRAY))

                elif node_name == "handle_error":
                    retry = state_updates.get("retry_count", 0)
                    max_r = int(os.getenv("MAX_RETRIES", "3"))
                    print(c(f"    Retry {retry} of {max_r}", Colour.YELLOW))

                # Merge this node's updates into the running final state
                final_state.update(state_updates)

    except KeyboardInterrupt:
        print(c("\n\n  Agent interrupted by user.", Colour.YELLOW))
        sys.exit(0)

    except Exception as e:
        print(c(f"\n  Agent encountered an unexpected error:", Colour.RED))
        print(c(f"  {type(e).__name__}: {str(e)}", Colour.RED))
        print(c(f"\n  Check your OPENAI_API_KEY and try again.", Colour.YELLOW))
        raise

    total_time = round(time.time() - start_time, 1)
    print(c(f"\n  Total time: {total_time}s", Colour.GRAY))

    return final_state


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    """
    Main loop — runs the agent repeatedly until the user
    chooses to exit. Each iteration:
      1. Asks user to pick a file
      2. Asks user for a question
      3. Runs the agent
      4. Shows results
      5. Asks if they want to run again
    """
    print_banner()

    # Validate everything is set up correctly before starting
    if not validate_setup():
        print(c("\n  Setup incomplete. Fix the issues above and try again.\n", Colour.RED))
        sys.exit(1)

    data_folder = os.getenv("DATA_FOLDER", "data")

    # ── Main interaction loop ───────────────────────────────
    while True:

        # STEP 1: Show available files and get user's choice
        print_section("Select your data file")
        available_files = print_available_files(data_folder)

        if not available_files:
            print(c("  No data files found. Add a CSV or Excel file to data/", Colour.RED))
            break

        file_path = get_file_choice(data_folder)

        # STEP 2: Get the user's analysis question
        print_section("What would you like to know?")
        question = get_question()

        # STEP 3: Run the agent
        print()
        try:
            result_state = run_agent(file_path, question)
        except Exception:
            # Error already printed inside run_agent()
            # Ask if user wants to try again
            again = input(c("\n  Try again? (y/n): ", Colour.TEAL)).strip().lower()
            if again != 'y':
                break
            continue

        # STEP 4: Display the final answer
        final_answer = result_state.get("final_answer", "")
        if final_answer:
            print_final_answer(final_answer)
        else:
            print(c("\n  No final answer was generated.", Colour.YELLOW))

        # STEP 5: Show output files that were created
        print_output_files(result_state)

        # STEP 6: Ask if user wants to run another analysis
        print()
        again = input(
            c("  Run another analysis? (y/n): ", Colour.TEAL)
        ).strip().lower()

        if again != 'y':
            print(c("\n  Thank you for using the Data Analyst Agent!\n", Colour.PURPLE))
            break

        print("\n" + c("─" * 56, Colour.GRAY) + "\n")


# ============================================================
# SCRIPT GUARD
# ============================================================
# This block ensures main() only runs when you execute this
# file directly:
#     python main.py          ← runs main()
#
# It does NOT run if this file is imported by another module:
#     from main import something   ← does NOT run main()
#
# This is a Python best practice for all entry point files.
# ============================================================

if __name__ == "__main__":
    main()


# In any Python file, after building the graph:
# from agent.graph import build_graph

# app = build_graph()

# Get Mermaid code as a string
# mermaid_code = app.get_graph().draw_mermaid()
# print(mermaid_code)

# Or save it directly as a PNG image (requires extra deps)
# app.get_graph().draw_mermaid_png(output_file_path="graph.png")