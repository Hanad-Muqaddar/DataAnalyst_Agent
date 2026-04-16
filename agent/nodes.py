# ============================================================
# agent/nodes.py
# ============================================================
# This file defines every NODE in our LangGraph state graph.
#
# WHAT IS A NODE?
#   A node is a Python function that:
#     1. Receives the full current STATE as its only argument
#     2. Does exactly ONE job (load file, run code, etc.)
#     3. Returns a DICTIONARY of only the state fields it changed
#
#   LangGraph automatically merges that return dictionary
#   back into the full state. Fields you don't return stay
#   exactly as they were — nodes only update what they touch.
#
# NODE SIGNATURE — every node looks like this:
#
#   def my_node(state: AnalystState) -> dict:
#       value = do_something(state["some_field"])
#       return {"updated_field": value}
#
# NODES IN THIS FILE (in execution order):
#   1. load_file_node    → reads CSV/Excel, builds data summary
#   2. llm_reason_node   → LLM thinks, writes code, picks action
#   3. run_code_node     → executes Python code in REPL
#   4. generate_chart_node → runs chart code, saves PNG
#   5. save_report_node  → writes final report to disk
#   6. handle_error_node → catches errors, manages retry loop
#
# HOW NODES CONNECT:
#   Nodes are connected by EDGES in graph.py.
#   Normal edges:   A always goes to B
#   Conditional:    A goes to B or C depending on state value
#
# ============================================================

import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This must happen before we read any os.getenv() values
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agent.state import AnalystState
from agent.tools import (
    load_file_tool,
    python_repl_tool,
    generate_chart_tool,
    save_report_tool,
    ALL_TOOLS,
)
from agent.prompts import (
    SYSTEM_PROMPT,
    build_analysis_prompt,
    build_error_prompt,
    build_chart_prompt,
    build_report_prompt,
)


# ============================================================
# LLM SETUP
# ============================================================
# We initialise the LLM once here, at module level.
# All nodes that need to call the LLM share this single
# instance — no need to create it inside each function.
#
# temperature=0  → deterministic output (best for code writing)
# model          → read from .env (default: gpt-4o)
# ============================================================

def _get_llm():
    """
    Creates and returns the LLM instance.
    Reading from environment at call time so .env is loaded first.
    """
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=float(os.getenv("TEMPERATURE", "0")),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ============================================================
# HELPER — extract text content from LLM response
# ============================================================

def _get_content(response) -> str:
    """
    Safely extracts the text content from an LLM response.

    LangChain LLM responses are AIMessage objects, not plain
    strings. This helper gets the .content attribute and
    ensures we always have a clean string to work with.

    Args:
        response: AIMessage object returned by llm.invoke()

    Returns:
        The text content as a plain Python string.
    """
    if hasattr(response, 'content'):
        return str(response.content)
    return str(response)


# ============================================================
# NODE 1 — load_file_node
# ============================================================

def load_file_node(state: AnalystState) -> dict:
    """
    NODE 1: Load the data file and build a summary for the LLM.

    READS FROM STATE:
        file_path       → path to the CSV or Excel file

    WRITES TO STATE:
        dataframe_summary → rich text description of the data
        raw_data_info     → confirmation message
        error_message     → set only if file loading fails
        messages          → initialises the conversation history

    WHAT IT DOES:
        Calls load_file_tool with the file path from state.
        The tool reads the file using pandas and returns a
        formatted text summary of columns, types, and sample rows.

        This summary is then stored in state so the LLM can
        read it in the next node (llm_reason_node) before
        writing any pandas code.

    WHY THIS NODE EXISTS:
        The LLM cannot directly open or read files. It can only
        process text. This node bridges that gap — it reads the
        file with Python and converts the structure into text
        the LLM can understand and reason about.
    """
    print("\n" + "="*55)
    print("NODE 1: load_file_node")
    print("="*55)
    print(f"  Loading file: {state['file_path']}")

    # Call the file loading tool
    # .invoke() is how LangChain tools are called programmatically
    file_summary = load_file_tool.invoke({
        "file_path": state["file_path"]
    })

    # Check if the tool returned an error
    if file_summary.startswith("ERROR"):
        print(f"  File load FAILED: {file_summary[:80]}...")
        return {
            "error_message": file_summary,
            "dataframe_summary": None,
            # Initialise messages even on error so later nodes
            # don't crash trying to append to None
            "messages": [SystemMessage(content=SYSTEM_PROMPT)]
        }

    print(f"  File loaded successfully.")
    print(f"  Summary length: {len(file_summary)} characters")

    # Build the initial conversation history.
    # We start with the system prompt as the first message,
    # then add the file summary as context for the LLM.
    # The LLM will read all messages in order when it reasons.
    initial_messages = [
        # System message: sets the agent's role and rules
        SystemMessage(content=SYSTEM_PROMPT),

        # Human message: the opening task with file summary embedded
        # We include the file summary here so the LLM sees the data
        # structure BEFORE we ask it to write any analysis code.
        HumanMessage(content=(
            f"{build_analysis_prompt(state['file_path'], state['user_question'])}"
            f"\n\nDATA FILE SUMMARY (from load_file_tool):\n{file_summary}"
        )),
    ]

    return {
        "dataframe_summary": file_summary,
        "raw_data_info": f"File loaded: {state['file_path']}",
        "error_message": None,          # clear any previous error
        "messages": initial_messages,
        "retry_count": 0,               # reset retry counter
    }


# ============================================================
# NODE 2 — llm_reason_node
# ============================================================

def llm_reason_node(state: AnalystState) -> dict:
    """
    NODE 2: The LLM brain — reasons about the data and decides
    what to do next.

    READS FROM STATE:
        messages        → full conversation history so far
        dataframe_summary → data structure (already in messages)
        user_question   → what the user wants to know
        error_message   → previous error if retrying
        code_to_run     → previous failed code if retrying
        retry_count     → how many retries have happened

    WRITES TO STATE:
        code_to_run   → Python code the LLM decided to run
        next_action   → routing decision: what to do next
        messages      → updated with LLM's response

    WHAT IT DOES:
        Sends the full message history to the LLM (GPT-4o).
        The LLM reads the data summary and question, then
        responds with Python code and a decision about what
        to do next.

        We parse the LLM's response to extract:
          1. The Python code it wants to run
          2. The next_action it wants to take

    HOW next_action ROUTING WORKS:
        The LLM's response text is scanned for keywords.
        Based on what keywords appear, we set next_action
        which tells the conditional edge in graph.py where
        to route next:

          "run_code"    → go to run_code_node
          "gen_chart"   → go to generate_chart_node
          "save_report" → go to save_report_node
          "end"         → finish the graph

    WHY LLM DECIDES THE ROUTE:
        This is what makes it an AGENT rather than a fixed
        pipeline. The LLM can decide to skip steps, repeat
        steps, or take a different path based on what it
        finds in the data. A fixed pipeline always runs the
        same steps in the same order.
    """
    print("\n" + "="*55)
    print("NODE 2: llm_reason_node")
    print("="*55)
    print(f"  Messages in history : {len(state.get('messages', []))}")
    print(f"  Retry count         : {state.get('retry_count', 0)}")

    llm = _get_llm()
    messages = state.get("messages", [])

    # If there was a previous error, add it to the conversation
    # so the LLM knows what went wrong and can fix the code
    if state.get("error_message") and state.get("retry_count", 0) > 0:
        print(f"  Adding error context to prompt for retry...")
        error_prompt = build_error_prompt(
            error_message=state["error_message"],
            code_that_failed=state.get("code_to_run", ""),
            retry_count=state.get("retry_count", 0),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
        )
        # Add the error as a human message so LLM sees it as
        # a new instruction in the conversation
        messages = messages + [HumanMessage(content=error_prompt)]

    # Call the LLM with the full message history
    print("  Calling LLM...")
    response = llm.invoke(messages)
    response_text = _get_content(response)

    print(f"  LLM responded ({len(response_text)} chars)")

    # -------------------------------------------------------
    # PARSE THE LLM RESPONSE
    # Extract Python code from the response.
    # LLMs wrap code in markdown code fences: ```python ... ```
    # We extract everything between those fences.
    # -------------------------------------------------------
    code_to_run = _extract_code(response_text)

    if code_to_run:
        print(f"  Code extracted: {len(code_to_run)} characters")
    else:
        print("  No code block found in response")

    # -------------------------------------------------------
    # DETERMINE next_action
    # We look for specific keywords in the LLM response to
    # decide where the graph should route next.
    # The LLM is instructed via the system prompt to include
    # these keywords in its response.
    # -------------------------------------------------------
    next_action = _determine_next_action(response_text, code_to_run)
    print(f"  next_action determined: '{next_action}'")

    # Add the LLM's response to the message history
    # so future nodes have the full context
    updated_messages = messages + [AIMessage(content=response_text)]

    return {
        "code_to_run": code_to_run,
        "next_action": next_action,
        "messages": updated_messages,
        "error_message": None,      # clear error since LLM has seen it
    }


def _extract_code(text: str) -> str:
    """
    Extract Python code from markdown code fences in LLM output.

    LLMs format code like this:
        ```python
        import pandas as pd
        df = pd.read_csv(...)
        ```

    This function pulls out just the code between the fences.
    Falls back to the full text if no code fences are found,
    in case the LLM forgot to format it.

    Args:
        text: Raw LLM response text

    Returns:
        Extracted Python code string, or empty string if none found.
    """
    # Look for ```python ... ``` block first (most common)
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()

    # Fall back to plain ``` ... ``` block
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()

    # No code fences found — return empty string
    # The node will handle this gracefully
    return ""


def _determine_next_action(response_text: str, code: str) -> str:
    """
    Determines the next_action routing value by scanning the
    LLM response for intent keywords.

    Priority order (most specific to least):
      1. "save_report" → LLM wants to write the final report
      2. "gen_chart"   → LLM wants to generate a chart
      3. "run_code"    → LLM wants to execute Python code
      4. "end"         → LLM says the task is complete
      5. "end"         → default if nothing is detected

    This function is intentionally simple — it uses keyword
    matching rather than another LLM call to keep it fast
    and deterministic.

    Args:
        response_text: Full LLM response text
        code:          Extracted code (if any)

    Returns:
        One of: "run_code", "gen_chart", "save_report", "end"
    """
    text_lower = response_text.lower()

    # Check for save_report intent FIRST
    # (it might also mention charts, so check report first)
    if any(kw in text_lower for kw in [
        "save_report_tool",
        "save report",
        "saving report",
        "final report",
        "write the report",
    ]):
        return "save_report"

    # Check for chart generation intent
    if any(kw in text_lower for kw in [
        "generate_chart_tool",
        "generate chart",
        "create a chart",
        "create chart",
        "visuali",          # catches visualise, visualize, visualization
        "plot",
        "chart_code",
        "matplotlib",
        "seaborn",
    ]):
        return "gen_chart"

    # Check for code execution intent
    # (code block present is a strong signal)
    if code or any(kw in text_lower for kw in [
        "python_repl",
        "execute",
        "run the code",
        "run this code",
        "import pandas",
        "pd.read_csv",
        "pd.read_excel",
    ]):
        return "run_code"

    # Check for explicit completion signals
    if any(kw in text_lower for kw in [
        "answer:",
        "key findings:",
        "outputs saved:",
        "analysis complete",
        "task complete",
        "analysis is complete",
    ]):
        return "end"

    # Default: if the LLM wrote code, run it
    # If no code and no clear signal, end gracefully
    return "run_code" if code else "end"


# ============================================================
# NODE 3 — run_code_node
# ============================================================

def run_code_node(state: AnalystState) -> dict:
    """
    NODE 3: Execute the Python code the LLM wrote using the
    Python REPL tool.

    READS FROM STATE:
        code_to_run → the Python code string from llm_reason_node

    WRITES TO STATE:
        code_output   → printed output from successful execution
        error_message → exception text if code crashes
        next_action   → set to "gen_chart" after success,
                        or "error" to trigger handle_error_node

    WHAT IT DOES:
        Passes the code string to python_repl_tool which runs
        it in a live Python interpreter and captures everything
        printed to stdout.

        On success: stores the printed output in code_output
                    and routes to generate_chart_node

        On failure: stores the error in error_message
                    and routes to handle_error_node

    THE REPL SESSION:
        python_repl_tool maintains a persistent Python session.
        This means if the LLM imported pandas in a previous call,
        it's still imported now. This is why we can break analysis
        into multiple steps without re-importing everything.
    """
    print("\n" + "="*55)
    print("NODE 3: run_code_node")
    print("="*55)

    code = state.get("code_to_run", "")

    if not code:
        print("  No code to execute — skipping")
        return {
            "code_output": "No code was provided to execute.",
            "next_action": "gen_chart",
        }

    # Show first 3 lines of code for debugging
    preview = "\n  ".join(code.split("\n")[:3])
    print(f"  Executing code:\n  {preview}")
    if code.count("\n") > 3:
        print(f"  ... ({code.count(chr(10))+1} lines total)")

    # Execute the code via the REPL tool
    # python_repl_tool.invoke() accepts the code as a plain string
    result = python_repl_tool.invoke(code)

    # -------------------------------------------------------
    # CHECK IF EXECUTION SUCCEEDED OR FAILED
    # The REPL tool returns error messages as plain text
    # starting with the exception type (e.g. "Traceback",
    # "SyntaxError", "KeyError", etc.)
    # -------------------------------------------------------
    is_error = _is_error_output(result)

    if is_error:
        print(f"  Execution FAILED:")
        print(f"  {result[:120]}...")
        return {
            "error_message": result,
            "next_action": "error",     # routes to handle_error_node
        }

    print(f"  Execution SUCCESS")
    print(f"  Output preview: {result[:100].strip()}...")

    return {
        "code_output": result,
        "error_message": None,          # clear any old error
        "next_action": "gen_chart",     # proceed to chart generation
    }


def _is_error_output(output: str) -> bool:
    """
    Detects whether the REPL output represents an error/exception.

    The PythonREPLTool returns error text (not raises exceptions),
    so we check the text content for common Python error patterns.

    Args:
        output: String returned by python_repl_tool.invoke()

    Returns:
        True if output looks like an error, False if it's normal output.
    """
    if not output:
        return False

    error_indicators = [
        "Traceback (most recent call last)",
        "SyntaxError",
        "KeyError",
        "ValueError",
        "TypeError",
        "AttributeError",
        "NameError",
        "IndexError",
        "FileNotFoundError",
        "ImportError",
        "ModuleNotFoundError",
        "ZeroDivisionError",
        "PermissionError",
        "RuntimeError",
        "Exception",
    ]

    output_start = output[:500]     # only check first 500 chars
    return any(indicator in output_start for indicator in error_indicators)


# ============================================================
# NODE 4 — generate_chart_node
# ============================================================

def generate_chart_node(state: AnalystState) -> dict:
    """
    NODE 4: Ask the LLM to write chart code, then execute it
    to produce and save a PNG chart.

    READS FROM STATE:
        messages        → conversation history (LLM context)
        user_question   → original question (for chart title)
        code_output     → computed results to visualise
        file_path       → needed to reload data in chart code

    WRITES TO STATE:
        chart_path    → path to the saved PNG file
        code_to_run   → the chart code that was generated
        next_action   → "save_report" to proceed, or "error"
        messages      → updated conversation history

    WHAT IT DOES:
        This node does TWO things:
          1. Calls the LLM again with build_chart_prompt() to
             generate matplotlib/seaborn code for the chart
          2. Executes that code using generate_chart_tool
             which saves the PNG and returns the file path

        This two-step approach (LLM writes code → tool runs it)
        is the core pattern of a code-generating agent. The LLM
        is smart enough to pick the right chart type, title,
        and styling based on the data and question.
    """
    print("\n" + "="*55)
    print("NODE 4: generate_chart_node")
    print("="*55)

    llm = _get_llm()
    messages = state.get("messages", [])

    # Build the chart generation prompt using results computed
    # by run_code_node (stored in code_output)
    chart_prompt = build_chart_prompt(
        user_question=state.get("user_question", ""),
        computation_result=state.get("code_output", "No results yet"),
        file_path=state.get("file_path", ""),
    )

    # Ask LLM to write the chart code
    print("  Asking LLM to write chart code...")
    chart_messages = messages + [HumanMessage(content=chart_prompt)]
    response = llm.invoke(chart_messages)
    response_text = _get_content(response)

    # Extract the chart code from LLM response
    chart_code = _extract_code(response_text)

    if not chart_code:
        print("  No chart code found in LLM response — skipping chart")
        return {
            "chart_path": "No chart generated",
            "next_action": "save_report",
            "messages": messages + [
                HumanMessage(content=chart_prompt),
                AIMessage(content=response_text),
            ],
        }

    print(f"  Chart code extracted: {len(chart_code)} characters")
    print("  Executing chart code...")

    # Execute the chart code using generate_chart_tool
    chart_result = generate_chart_tool.invoke({"chart_code": chart_code})

    # Parse the result to get the file path
    if "SUCCESS" in chart_result:
        # Extract path from: "SUCCESS: Chart saved to 'outputs/chart_xxx.png'"
        chart_path = _extract_path(chart_result)
        print(f"  Chart saved: {chart_path}")

        return {
            "chart_path": chart_path,
            "code_to_run": chart_code,
            "next_action": "save_report",
            "messages": messages + [
                HumanMessage(content=chart_prompt),
                AIMessage(content=response_text),
            ],
        }

    else:
        # Chart generation failed — treat as error
        print(f"  Chart generation FAILED: {chart_result[:80]}")
        return {
            "chart_path": "Chart generation failed",
            "error_message": chart_result,
            "next_action": "save_report",   # continue even if chart failed
            "messages": messages + [
                HumanMessage(content=chart_prompt),
                AIMessage(content=response_text),
            ],
        }


def _extract_path(result_text: str) -> str:
    """
    Extract the file path from a tool success message.

    Tool success messages look like:
        "SUCCESS: Chart saved to 'outputs/chart_20260414_002800.png'"

    We extract the path between single quotes.

    Args:
        result_text: Success message from generate_chart_tool

    Returns:
        The file path string, or the full result if parsing fails.
    """
    if "'" in result_text:
        start = result_text.find("'") + 1
        end = result_text.find("'", start)
        if end != -1:
            return result_text[start:end]
    return result_text


# ============================================================
# NODE 5 — save_report_node
# ============================================================

def save_report_node(state: AnalystState) -> dict:
    """
    NODE 5: Ask the LLM to write a final summary report, save
    it to disk, and produce the final answer for the user.

    READS FROM STATE:
        messages        → full conversation history
        user_question   → original question (echoed in report)
        code_output     → all computed results
        chart_path      → path to saved chart PNG

    WRITES TO STATE:
        final_answer  → the agent's complete response to display
        next_action   → "end" to finish the graph

    WHAT IT DOES:
        Calls the LLM one more time with build_report_prompt()
        asking it to write a plain-English summary.

        The LLM generates report text which is passed to
        save_report_tool to write to the outputs/ folder.

        The LLM's full response (including the ANSWER: section)
        becomes the final_answer — this is what gets printed
        to the user at the end of main.py.
    """
    print("\n" + "="*55)
    print("NODE 5: save_report_node")
    print("="*55)

    llm = _get_llm()
    messages = state.get("messages", [])

    # Build the report writing prompt
    report_prompt = build_report_prompt(
        user_question=state.get("user_question", ""),
        computation_result=state.get("code_output", "No results computed"),
        chart_path=state.get("chart_path", "No chart generated"),
    )

    # Ask LLM to write the report
    print("  Asking LLM to write final report...")
    report_messages = messages + [HumanMessage(content=report_prompt)]
    response = llm.invoke(report_messages)
    response_text = _get_content(response)

    print(f"  LLM report written ({len(response_text)} chars)")

    # Save the report using save_report_tool
    # We ask the LLM to produce the report content, but we
    # could also let the LLM call the tool itself — either
    # approach works. Here we call it directly for reliability.
    save_result = save_report_tool.invoke({
        "report_content": response_text
    })

    if "SUCCESS" in save_result:
        report_path = _extract_path(save_result)
        print(f"  Report saved: {report_path}")
    else:
        print(f"  Report save warning: {save_result[:60]}")
        report_path = "Report save failed"

    # The full LLM response IS the final answer shown to the user
    # It should contain the ANSWER:, KEY FINDINGS:, and OUTPUTS SAVED:
    # sections as instructed in the system prompt
    return {
        "final_answer": response_text,
        "next_action": "end",
    }


# ============================================================
# NODE 6 — handle_error_node
# ============================================================

def handle_error_node(state: AnalystState) -> dict:
    """
    NODE 6: Handle code execution errors and manage the retry loop.

    READS FROM STATE:
        error_message → the exception text from the failed code
        retry_count   → current number of retries attempted

    WRITES TO STATE:
        retry_count   → incremented by 1
        next_action   → "retry" if attempts remain, "end" if exhausted
        final_answer  → set only if max retries is reached

    WHAT IT DOES:
        This node is the control centre of the retry loop.

        It increments retry_count and checks it against MAX_RETRIES
        (loaded from .env, default 3).

        IF retries remain:
            → sets next_action = "retry"
            → routes back to llm_reason_node
            → LLM will read error_message and fix the code

        IF max retries reached:
            → sets next_action = "end"
            → sets final_answer to a graceful error message
            → the graph terminates without crashing

    WHY THIS NODE IS IMPORTANT:
        Without this node, a single bad line of generated code
        would crash the entire agent. With this node, the agent
        can fail up to MAX_RETRIES times and self-correct each
        time by reading its own error messages. This is what
        makes the agent robust in production.
    """
    print("\n" + "="*55)
    print("NODE 6: handle_error_node")
    print("="*55)

    # Read current retry state
    current_retries = state.get("retry_count", 0)
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    error_msg = state.get("error_message", "Unknown error")
    new_retry_count = current_retries + 1

    print(f"  Error         : {error_msg[:80]}...")
    print(f"  Retry attempt : {new_retry_count} of {max_retries}")

    # -------------------------------------------------------
    # DECISION: retry or give up?
    # -------------------------------------------------------
    if new_retry_count < max_retries:
        # Retries remaining → loop back to LLM reasoning
        print(f"  Decision: RETRY ({max_retries - new_retry_count} attempts left)")
        return {
            "retry_count": new_retry_count,
            "next_action": "retry",     # graph.py routes this back to llm_reason
        }

    else:
        # Max retries exhausted → end gracefully
        print(f"  Decision: GIVE UP (max retries reached)")

        failure_message = (
            f"ANALYSIS INCOMPLETE\n\n"
            f"The agent was unable to complete the analysis after "
            f"{max_retries} attempts.\n\n"
            f"Last error encountered:\n{error_msg}\n\n"
            f"SUGGESTION:\n"
            f"- Check that column names in your question match the file\n"
            f"- Verify the file is not corrupted or empty\n"
            f"- Try rephrasing your question with exact column names\n"
            f"  (you can find column names by opening the file in Excel)"
        )

        return {
            "retry_count": new_retry_count,
            "final_answer": failure_message,
            "next_action": "end",       # terminates the graph
        }


# ============================================================
# ROUTING FUNCTIONS — used by conditional edges in graph.py
# ============================================================
# These functions are NOT nodes. They are called by LangGraph's
# conditional edges to decide which node to go to next.
#
# A routing function:
#   - Receives the current state
#   - Returns a STRING key
#   - That key is looked up in the edge's routing map
#   - The matched node is the next destination
#
# Example in graph.py:
#   graph.add_conditional_edges(
#       "llm_reason",
#       route_after_llm,           ← this function
#       {
#           "run_code":    "run_code_node",
#           "gen_chart":   "generate_chart_node",
#           "save_report": "save_report_node",
#           "end":         END,
#       }
#   )
# ============================================================

def route_after_llm(state: AnalystState) -> str:
    """
    Routing function called after llm_reason_node.

    Reads next_action from state and returns the appropriate
    routing key for the conditional edge.

    Returns one of: "run_code", "gen_chart", "save_report", "end"
    """
    action = state.get("next_action", "end")
    print(f"\n  [ROUTER] After LLM → routing to: '{action}'")
    return action


def route_after_code(state: AnalystState) -> str:
    """
    Routing function called after run_code_node.

    If code ran successfully → "gen_chart"
    If code threw an error   → "error"

    Returns one of: "gen_chart", "error"
    """
    action = state.get("next_action", "gen_chart")
    print(f"\n  [ROUTER] After code → routing to: '{action}'")
    return action


def route_after_error(state: AnalystState) -> str:
    """
    Routing function called after handle_error_node.

    If retries remain → "retry" (back to llm_reason_node)
    If max retries    → "end"

    Returns one of: "retry", "end"
    """
    action = state.get("next_action", "end")
    print(f"\n  [ROUTER] After error → routing to: '{action}'")
    return action

# """
# Each function here is a node in the StateGraph.
# Nodes receive the full AnalystState and return a partial state dict.
# """

# from pathlib import Path

# import pandas as pd

# from .state import AnalystState
# from .tools import run_python, save_current_chart, save_text


# # ---------------------------------------------------------------------------
# # Node: load_file
# # ---------------------------------------------------------------------------

# def load_file(state: AnalystState) -> dict:
#     """Read the user-supplied file and produce a text summary for the LLM."""
#     path = Path(state["file_path"])
#     suffix = path.suffix.lower()

#     if suffix == ".csv":
#         df = pd.read_csv(path)
#     elif suffix in {".xlsx", ".xls"}:
#         df = pd.read_excel(path)
#     elif suffix == ".json":
#         df = pd.read_json(path)
#     else:
#         return {"error": f"Unsupported file type: {suffix}"}

#     summary_lines = [
#         f"File: {path.name}",
#         f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
#         f"Columns: {list(df.columns)}",
#         f"Dtypes:\n{df.dtypes.to_string()}",
#         f"Sample (first 5 rows):\n{df.head().to_string()}",
#         f"Describe:\n{df.describe(include='all').to_string()}",
#     ]
#     return {"df_summary": "\n\n".join(summary_lines), "error": None}


# # ---------------------------------------------------------------------------
# # Node: generate_code
# # ---------------------------------------------------------------------------

# def generate_code(state: AnalystState, llm) -> dict:
#     """Ask the LLM to write analysis code based on the data summary and query."""
#     from langchain_core.messages import HumanMessage, SystemMessage
#     from .prompts import SYSTEM_PROMPT

#     messages = [
#         SystemMessage(content=SYSTEM_PROMPT),
#         HumanMessage(content=(
#             f"Data summary:\n{state['df_summary']}\n\n"
#             f"User query: {state['user_query']}\n\n"
#             "Write Python code (pandas + matplotlib) that answers the query. "
#             "Return ONLY the code block, no explanation."
#         )),
#     ]
#     response = llm.invoke(messages)
#     # Strip markdown fences if present
#     code = response.content.strip().removeprefix("```python").removesuffix("```").strip()
#     return {"code": code, "messages": state.get("messages", []) + [response]}


# # ---------------------------------------------------------------------------
# # Node: run_code
# # ---------------------------------------------------------------------------

# def run_code(state: AnalystState) -> dict:
#     """Execute the generated code and capture the output."""
#     output, locals_ = run_python(state["code"])
#     if "__error__" in locals_:
#         return {"code_output": None, "error": locals_["__error__"]}
#     return {"code_output": output or "(no stdout)", "error": None}


# # ---------------------------------------------------------------------------
# # Node: save_chart
# # ---------------------------------------------------------------------------

# def save_chart(state: AnalystState) -> dict:
#     """Save the active matplotlib figure, if any code produced one."""
#     try:
#         chart_path = save_current_chart("outputs/chart.png")
#         return {"chart_path": chart_path}
#     except Exception as exc:
#         return {"chart_path": None, "error": str(exc)}


# # ---------------------------------------------------------------------------
# # Node: write_report
# # ---------------------------------------------------------------------------

# def write_report(state: AnalystState, llm) -> dict:
#     """Ask the LLM to synthesise a final report from the code output."""
#     from langchain_core.messages import HumanMessage, SystemMessage
#     from .prompts import SYSTEM_PROMPT

#     messages = [
#         SystemMessage(content=SYSTEM_PROMPT),
#         HumanMessage(content=(
#             f"User query: {state['user_query']}\n\n"
#             f"Code executed:\n{state['code']}\n\n"
#             f"Output:\n{state['code_output']}\n\n"
#             "Write a concise analyst report answering the user's query. "
#             "Use bullet points for key insights."
#         )),
#     ]
#     response = llm.invoke(messages)
#     report_text = response.content.strip()
#     path = save_text(report_text, "outputs/report.txt")
#     return {
#         "report": report_text,
#         "messages": state.get("messages", []) + [response],
#     }


