# ============================================================
# agent/prompts.py
# ============================================================
# This file contains ALL prompt text for the agent.
#
# WHY A SEPARATE FILE FOR PROMPTS?
#   Prompts are not code — they are instructions written in
#   natural language. Keeping them separate from Python logic
#   means you can improve the agent's behaviour just by editing
#   this file, without touching any other code.
#
#   Think of it like this:
#     prompts.py   = the job description you give the employee
#     nodes.py     = the employee's actual daily workflow
#     tools.py     = the tools the employee has on their desk
#
# WHAT IS A SYSTEM PROMPT?
#   When you call an LLM, you send two types of messages:
#     1. system prompt  → sets the role, rules, and personality
#                         (read ONCE at the start, not repeated)
#     2. user messages  → the actual conversation / task
#
#   The system prompt is the most powerful way to control
#   how the LLM behaves. It's read before anything else.
#
# CONTENTS:
#   SYSTEM_PROMPT          → main instructions for the agent
#   build_analysis_prompt  → builds the first user message
#   build_error_prompt     → builds the retry message after errors
#   build_chart_prompt     → builds the chart generation message
# ============================================================


# ============================================================
# MAIN SYSTEM PROMPT
# ============================================================
# This is sent as the "system" role message at the start of
# every conversation. It tells the LLM:
#   - What it is and what its job is
#   - What tools it has and when to use each one
#   - How to write correct pandas code
#   - How to handle errors and retry
#   - What format to use for its final answer
# ============================================================

SYSTEM_PROMPT = """
You are an expert Data Analyst AI Agent. Your job is to help
users understand their data by analysing CSV and Excel files,
computing statistics, and generating clear visual charts.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TOOLS — use them in this order:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. load_file_tool
   → ALWAYS use this FIRST before writing any code.
   → It reads the file and gives you the column names,
     data types, and a preview of the data.
   → You CANNOT write correct pandas code without knowing
     the exact column names first.

2. python_repl
   → Use this to run pandas code for calculations.
   → Always use print() to show results — return values
     are NOT captured, only printed output is.
   → The session persists, so imports from one call
     carry over to the next call.

3. generate_chart_tool
   → Use this to create and save visual charts as PNG.
   → Your code must include imports and data loading.
   → Do NOT call plt.show() or plt.savefig() — the tool
     handles saving automatically.
   → Call this AFTER python_repl so you know what data
     to visualise.

4. save_report_tool
   → Use this LAST to write the final summary report.
   → Write in plain English a non-technical person can
     understand.
   → Include key numbers, chart reference, and insight.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES FOR WRITING PANDAS CODE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Always use the EXACT column names from load_file_tool output.
  Column names are case-sensitive. 'Amount' ≠ 'amount'.

- Always import pandas at the top of every code block:
    import pandas as pd

- Always load the file at the top of every code block:
    df = pd.read_csv('path/to/file.csv')
  or for Excel:
    df = pd.read_excel('path/to/file.xlsx')

- Always print() your results — never just compute them:
    WRONG:  df['Amount'].sum()
    RIGHT:  print(df['Amount'].sum())

- For currency/money columns, round to 2 decimal places:
    print(round(df['Amount'].sum(), 2))

- If a column has mixed types (e.g. '1,200' as a string),
  clean it before computing:
    df['Amount'] = pd.to_numeric(
        df['Amount'].astype(str).str.replace(',', ''),
        errors='coerce'
    )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES FOR GENERATING CHARTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Choose the right chart type for the question:
    Comparison between categories → bar chart
    Part of a whole / percentages → pie chart
    Trend over time               → line chart
    Distribution / spread         → histogram
    Two numeric columns           → scatter plot

- Always set a clear title:
    plt.title('Spending by Category — January 2024')

- Always call plt.tight_layout() before the code ends.

- Do NOT call plt.show() — it will block execution.
- Do NOT call plt.savefig() — the tool saves it for you.

- Make charts readable:
    - Rotate x-axis labels if they overlap: rotation=45
    - Add value labels on bars when helpful
    - Use figsize=(10, 6) or similar for clear sizing

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO HANDLE ERRORS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If a tool returns an error message:
  1. Read the error carefully — it tells you exactly what
     went wrong (wrong column name, wrong data type, etc.)
  2. Look at the load_file_tool output again to check the
     exact column names and data types
  3. Fix ONLY the part that caused the error
  4. Run the corrected code

Common errors and fixes:
  KeyError: 'Category'     → column name is wrong, check
                             the exact name from file summary
  ValueError: could not    → column has text mixed with
  convert string to float    numbers, clean it first
  FileNotFoundError        → wrong file path, check the
                             path given by the user

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKFLOW — follow this sequence every time:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  STEP 1 → load_file_tool      (understand the data)
  STEP 2 → python_repl         (compute the answer)
  STEP 3 → generate_chart_tool (visualise the answer)
  STEP 4 → save_report_tool    (write the summary)
  STEP 5 → reply to user       (explain findings clearly)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL ANSWER FORMAT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When you have completed the analysis, end with a clear
summary in this format:

  ANSWER: [Direct answer to the user's question]

  KEY FINDINGS:
  • [Finding 1 with specific number]
  • [Finding 2 with specific number]
  • [Finding 3 with specific number]

  OUTPUTS SAVED:
  • Chart: [path to PNG file]
  • Report: [path to report file]

Always be specific with numbers. "Rent was PKR 1,200"
is better than "Rent was the highest category".
"""


# ============================================================
# PROMPT BUILDER FUNCTIONS
# ============================================================
# These functions build the user-turn messages dynamically.
# We use functions (not hardcoded strings) because the content
# changes based on:
#   - Which file the user uploaded
#   - What question they asked
#   - What error occurred (for retries)
#   - What results were computed (for chart prompts)
#
# Each function returns a plain string — LangGraph passes
# these strings as HumanMessage content to the LLM.
# ============================================================


def build_analysis_prompt(file_path: str, user_question: str) -> str:
    """
    Builds the opening user message that starts the analysis.

    This is the very first message the LLM receives after the
    system prompt. It tells the agent:
      - Which file to analyse
      - What question to answer
      - That it should start with load_file_tool

    Args:
        file_path:     Path to the data file (e.g. 'data/bank.csv')
        user_question: The user's question in plain English

    Returns:
        A formatted string to send as the first user message.

    Example output:
        Please analyse the following data file and answer my question.

        FILE TO ANALYSE: data/bank_statement.csv
        MY QUESTION: What category did I spend the most on?

        Start by using load_file_tool to understand the data,
        then compute the answer, generate a chart, and save a report.
    """
    return f"""Please analyse the following data file and answer my question.

FILE TO ANALYSE: {file_path}
MY QUESTION: {user_question}

Start by using load_file_tool to read and understand the data structure.
Then use python_repl to compute the answer.
Then use generate_chart_tool to create a relevant visualisation.
Finally use save_report_tool to write a summary report.

Be thorough and specific — include exact numbers in your findings."""


def build_error_prompt(
    error_message: str,
    code_that_failed: str,
    retry_count: int,
    max_retries: int
) -> str:
    """
    Builds the retry message sent to the LLM after a code error.

    When the Python REPL throws an exception, we don't just
    give up — we tell the LLM exactly what went wrong and ask
    it to fix and retry. This is the message that enables the
    retry loop you saw in the state diagram.

    Args:
        error_message:    The exception message from the REPL
        code_that_failed: The exact code that caused the error
        retry_count:      How many retries have happened so far
        max_retries:      Maximum retries allowed (from .env)

    Returns:
        A formatted string explaining the error and asking for a fix.

    Example output:
        Your code produced an error. Please fix it and try again.
        Attempt 2 of 3.

        ERROR: KeyError: 'Category'

        CODE THAT FAILED:
        df.groupby('Category')['Amount'].sum()

        Please look at the column names from load_file_tool output
        and correct the column name. Then run the fixed code.
    """
    attempts_left = max_retries - retry_count

    return f"""Your code produced an error. Please analyse the error,
fix the code, and try again.

Attempt {retry_count + 1} of {max_retries}. Attempts remaining: {attempts_left}.

ERROR MESSAGE:
{error_message}

CODE THAT FAILED:
```python
{code_that_failed}
```

INSTRUCTIONS FOR FIXING:
1. Read the error message carefully — it points to the exact problem.
2. Check the column names from the load_file_tool output above.
   Column names are case-sensitive — 'Amount' is not 'amount'.
3. Fix ONLY the part that caused the error.
4. Run the corrected code using python_repl.

Do not apologise or explain — just fix and run the code."""


def build_chart_prompt(
    user_question: str,
    computation_result: str,
    file_path: str
) -> str:
    """
    Builds the message that instructs the LLM to generate a chart
    based on the computation results it already produced.

    This is called after python_repl has successfully computed
    the answer. Now we want the LLM to visualise that answer.

    Args:
        user_question:      The original user question
        computation_result: The printed output from python_repl
        file_path:          Path to the original data file

    Returns:
        A formatted string asking the LLM to create a chart.

    Example output:
        Now create a chart to visualise the analysis results.

        ORIGINAL QUESTION: What category did I spend the most on?
        COMPUTED RESULTS:
        Category
        Rent             1200.00
        Groceries         157.80
        ...

        Use generate_chart_tool with complete Python code that:
        - Imports pandas and matplotlib
        - Loads the file: data/bank_statement.csv
        - Creates an appropriate chart type
        - Sets a clear title and labels
    """
    return f"""Now create a chart to visualise the analysis results.

ORIGINAL QUESTION: {user_question}

COMPUTED RESULTS FROM ANALYSIS:
{computation_result}

Use generate_chart_tool with complete Python code that:
  1. Imports pandas, matplotlib.pyplot, and seaborn if needed
  2. Loads the data file: {file_path}
  3. Creates the most appropriate chart type for this question:
       - Category comparisons → bar chart (horizontal if many categories)
       - Proportions/percentages → pie chart
       - Trends over time → line chart
       - Distributions → histogram
  4. Sets a descriptive title that answers the question
  5. Labels axes clearly
  6. Calls plt.tight_layout() at the end
  7. Does NOT call plt.show() or plt.savefig()

Make the chart easy to read and visually informative."""


def build_report_prompt(
    user_question: str,
    computation_result: str,
    chart_path: str
) -> str:
    """
    Builds the message that instructs the LLM to write and save
    the final analysis report after all computation is done.

    This is the last prompt in the chain. The agent has already:
      1. Loaded the file
      2. Computed the answer
      3. Generated the chart

    Now it writes and saves the plain-English summary report.

    Args:
        user_question:      The original user question
        computation_result: All printed output from python_repl
        chart_path:         Path where the chart PNG was saved

    Returns:
        A formatted string asking the LLM to write and save a report.
    """
    return f"""Now write and save the final analysis report using save_report_tool.

ORIGINAL QUESTION: {user_question}

ANALYSIS RESULTS:
{computation_result}

CHART SAVED AT: {chart_path}

Write a clear, plain-English report that includes:
  1. A direct answer to the question (the most important finding first)
  2. The top 3-5 specific numbers or findings
  3. A reference to the chart file
  4. One practical insight or recommendation based on the data

Write for a non-technical audience. Avoid jargon.
Use specific numbers, not vague descriptions.
Keep it concise — around 150-200 words.

After saving the report, give the user your final ANSWER
in the format specified in your instructions."""


# SYSTEM_PROMPT = """
# You are an expert data analyst assistant. You help users understand their data by:
# - Loading and summarising datasets (CSV, Excel, JSON)
# - Writing and executing clean Python (pandas / matplotlib) code
# - Generating charts and saving them to disk
# - Producing concise, insight-driven reports

# Guidelines:
# - Always inspect the data before writing analysis code.
# - Prefer pandas for tabular manipulation and matplotlib/seaborn for visualisations.
# - When saving charts, write them to the outputs/ directory.
# - When writing reports, be concise: lead with the key insight, then supporting detail.
# - If the code raises an error, diagnose it and retry with a corrected version.
# """
