# ============================================================
# agent/state.py
# ============================================================
# This file defines the "State" of our LangGraph agent.
#
# WHAT IS STATE?
#   Think of State like a shared whiteboard that every node
#   (step) in our agent can read from and write to.
#
#   When the agent starts, it writes the file path and question
#   onto the whiteboard. Each node reads what it needs, does
#   its job, and writes the result back. The next node picks
#   up from where the last one left off.
#
#   This is the #1 concept that makes LangGraph different from
#   plain LangChain — everything is connected through state.
# ============================================================

from typing import TypedDict, Optional
# TypedDict   → lets us define a dictionary with fixed keys and types
# Optional    → means a field can be None (not yet filled in)


class AnalystState(TypedDict):
    """
    The complete state of our Data Analyst Agent.

    This TypedDict acts as a contract — it defines exactly what
    data can exist at any point during the agent's execution.

    Flow of data through state:
        main.py sets:         file_path, user_question
        load_file node sets:  dataframe_summary, raw_data_info
        llm_reason node sets: code_to_run, next_action
        run_code node sets:   code_output, error_message
        gen_chart node sets:  chart_path
        save_report node sets: final_answer
    """

    # ----------------------------------------------------------
    # INPUT FIELDS (set by the user at the start)
    # ----------------------------------------------------------

    file_path: str
    # The full path to the uploaded CSV or Excel file.
    # Example: "data/bank_statement.csv"

    user_question: str
    # The natural language question the user wants answered.
    # Example: "What category did I spend the most on?"

    # ----------------------------------------------------------
    # DATA LOADING FIELDS (set by the load_file node)
    # ----------------------------------------------------------

    dataframe_summary: Optional[str]
    # A text preview of the loaded data — first 5 rows,
    # column names, and data types. The LLM reads this
    # to understand the structure of the data before
    # writing any Pandas code.
    # Example:
    #   "Columns: Date, Description, Amount, Category
    #    Shape: (120, 4)
    #    First rows: ..."

    raw_data_info: Optional[str]
    # Additional info about the dataset — null counts,
    # unique values, basic statistics. Helps the LLM
    # write more accurate code.

    # ----------------------------------------------------------
    # LLM REASONING FIELDS (set by the llm_reason node)
    # ----------------------------------------------------------

    code_to_run: Optional[str]
    # The Python code the LLM wrote to answer the question.
    # This is actual runnable code using pandas/matplotlib.
    # Example:
    #   "import pandas as pd
    #    df = pd.read_csv('data/bank_statement.csv')
    #    result = df.groupby('Category')['Amount'].sum()
    #    print(result)"

    next_action: Optional[str]
    # Tells the graph WHAT to do next after LLM reasoning.
    # Possible values:
    #   "run_code"    → execute the code in Python REPL
    #   "gen_chart"   → generate a chart/visualization
    #   "save_report" → write final answer to a file
    #   "end"         → task complete, stop the graph
    # LangGraph uses this field in conditional edges to decide
    # which node to route to next.

    # ----------------------------------------------------------
    # CODE EXECUTION FIELDS (set by the run_code node)
    # ----------------------------------------------------------

    code_output: Optional[str]
    # The printed output from running code_to_run in the REPL.
    # If code runs successfully, this has the result.
    # Example: "Category\nFood      450.00\nRent     1200.00"

    error_message: Optional[str]
    # If the code throws an exception, the error is stored here.
    # The LLM reads this in the next reasoning step to
    # understand what went wrong and fix the code.
    # Example: "KeyError: 'Category' — column not found"

    # ----------------------------------------------------------
    # OUTPUT FIELDS (set by later nodes)
    # ----------------------------------------------------------

    chart_path: Optional[str]
    # The file path where the generated chart PNG was saved.
    # Example: "outputs/spending_by_category.png"

    final_answer: Optional[str]
    # The agent's natural language conclusion / summary.
    # This is what gets printed to the user at the end.
    # Example: "You spent the most on Rent ($1,200) followed
    #           by Food ($450). A chart has been saved to outputs/"

    # ----------------------------------------------------------
    # CONTROL FIELDS (used internally by the graph)
    # ----------------------------------------------------------

    retry_count: Optional[int]
    # Tracks how many times the agent has retried after
    # a code error. When this reaches MAX_RETRIES (from .env),
    # the graph stops retrying and ends with an error message.
    # Starts at 0, increments in the handle_error node.

    messages: Optional[list]
    # Stores the full conversation history between the user
    # and the LLM. LangGraph uses this to give the LLM
    # memory of what it already tried, so it doesn't repeat
    # the same mistake when retrying after an error.