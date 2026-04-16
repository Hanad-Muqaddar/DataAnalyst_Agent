# ============================================================
# agent/graph.py
# ============================================================
# This file ASSEMBLES the complete LangGraph agent by:
#   1. Creating a StateGraph with our AnalystState schema
#   2. Adding every node (each node = one processing step)
#   3. Adding edges (connections between nodes)
#   4. Compiling the graph into a runnable application
#
# WHAT IS A STATEGRAPH?
#   StateGraph is LangGraph's core class. You build it like
#   this:
#     graph = StateGraph(YourStateClass)
#     graph.add_node("name", function)
#     graph.add_edge("from_node", "to_node")
#     app = graph.compile()
#     result = app.invoke(initial_state)
#
# TWO TYPES OF EDGES:
#
#   1. Normal edge → always goes from A to B
#      graph.add_edge("load_file", "llm_reason")
#      No decision, no branching. A always leads to B.
#
#   2. Conditional edge → goes to different nodes based on state
#      graph.add_conditional_edges(
#          "source_node",     ← where we're coming FROM
#          routing_function,  ← function that reads state and
#                               returns a string key
#          {                  ← mapping of key → destination
#              "run_code":    "run_code_node",
#              "gen_chart":   "generate_chart_node",
#              "end":         END,
#          }
#      )
#
# SPECIAL NODES:
#   START → built-in LangGraph node, the entry point
#   END   → built-in LangGraph node, terminates the graph
#
# GRAPH STRUCTURE (what we're building):
#
#   START
#     │
#     ▼
#   load_file ──────────────────────────────► END (on file error)
#     │
#     ▼
#   llm_reason ──┬──► run_code_node
#                ├──► generate_chart_node
#                ├──► save_report_node
#                └──► END
#                         │
#              run_code_node ──► generate_chart_node (success)
#                         └──► handle_error_node    (failure)
#                                      │
#                         handle_error ──► llm_reason (retry)
#                                     └──► END        (max retries)
#                         │
#              generate_chart_node ──► save_report_node
#                         │
#              save_report_node ──► END
#
# ============================================================

import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END

from agent.state import AnalystState
from agent.nodes import (
    # ── Node functions ───────────────────────────────────────
    load_file_node,
    llm_reason_node,
    run_code_node,
    generate_chart_node,
    save_report_node,
    handle_error_node,
    # ── Routing functions (used by conditional edges) ────────
    route_after_llm,
    route_after_code,
    route_after_error,
)


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_graph():
    """
    Constructs and compiles the Data Analyst LangGraph agent.

    This function:
      1. Creates a StateGraph typed with AnalystState
      2. Registers all node functions with their string names
      3. Wires nodes together using normal and conditional edges
      4. Compiles the graph into a runnable app

    Returns:
        A compiled LangGraph application ready to be invoked
        with an initial state dictionary.

    Usage:
        app = build_graph()
        result = app.invoke({
            "file_path": "data/bank_statement.csv",
            "user_question": "What did I spend most on?"
        })
        print(result["final_answer"])
    """

    # -------------------------------------------------------
    # STEP 1: Create the StateGraph
    # -------------------------------------------------------
    # StateGraph takes our AnalystState TypedDict as its schema.
    # This tells LangGraph what fields exist in the state and
    # enforces type safety — if a node returns a key that isn't
    # in AnalystState, LangGraph raises an error immediately.
    # -------------------------------------------------------
    graph = StateGraph(AnalystState)


    # -------------------------------------------------------
    # STEP 2: Register all nodes
    # -------------------------------------------------------
    # graph.add_node("string_name", function)
    #
    # The string name is what you use in add_edge() calls.
    # The function is the Python function from nodes.py.
    # They don't have to match — but matching them makes the
    # code much easier to read and debug.
    # -------------------------------------------------------

    graph.add_node("load_file",       load_file_node)
    graph.add_node("llm_reason",      llm_reason_node)
    graph.add_node("run_code",        run_code_node)
    graph.add_node("generate_chart",  generate_chart_node)
    graph.add_node("save_report",     save_report_node)
    graph.add_node("handle_error",    handle_error_node)


    # -------------------------------------------------------
    # STEP 3: Set the entry point
    # -------------------------------------------------------
    # START is a special built-in LangGraph node that marks
    # where graph execution begins.
    # This edge means: "when the graph starts, go to load_file"
    # -------------------------------------------------------
    graph.add_edge(START, "load_file")


    # -------------------------------------------------------
    # STEP 4: Wire the happy path (normal edges)
    # -------------------------------------------------------
    # These are the connections that always happen — no branching.
    # load_file ALWAYS leads to llm_reason (even on error,
    # because we store the error in state and let llm_reason
    # handle it gracefully).
    # -------------------------------------------------------
    graph.add_edge("load_file", "llm_reason")


    # -------------------------------------------------------
    # STEP 5: Wire the LLM decision point (conditional edge)
    # -------------------------------------------------------
    # After llm_reason_node runs, we don't know yet which node
    # to go to — the LLM decides by setting state["next_action"].
    # route_after_llm() reads that field and returns a string key.
    # The dictionary maps those keys to destination node names.
    #
    # Possible routes:
    #   "run_code"    → LLM wrote analysis code → execute it
    #   "gen_chart"   → LLM wants to make a chart directly
    #   "save_report" → LLM wants to write the final report
    #   "end"         → LLM says task is complete
    # -------------------------------------------------------
    graph.add_conditional_edges(
        "llm_reason",           # source node
        route_after_llm,        # routing function reads next_action
        {
            "run_code":    "run_code",
            "gen_chart":   "generate_chart",
            "save_report": "save_report",
            "end":         END,
        }
    )


    # -------------------------------------------------------
    # STEP 6: Wire the code execution decision point
    # -------------------------------------------------------
    # After run_code_node, two things can happen:
    #   SUCCESS → go to generate_chart (visualise the results)
    #   FAILURE → go to handle_error  (start retry loop)
    #
    # route_after_code() reads state["next_action"] which
    # run_code_node sets to either "gen_chart" or "error".
    # -------------------------------------------------------
    graph.add_conditional_edges(
        "run_code",             # source node
        route_after_code,       # routing function
        {
            "gen_chart": "generate_chart",
            "error":     "handle_error",
        }
    )


    # -------------------------------------------------------
    # STEP 7: Wire the error handler decision point
    # -------------------------------------------------------
    # After handle_error_node, two things can happen:
    #   RETRY → go back to llm_reason (LLM reads error, fixes code)
    #   END   → max retries reached, stop the graph gracefully
    #
    # route_after_error() reads state["next_action"] which
    # handle_error_node sets to either "retry" or "end".
    # -------------------------------------------------------
    graph.add_conditional_edges(
        "handle_error",         # source node
        route_after_error,      # routing function
        {
            "retry": "llm_reason",  # loop back to LLM with error context
            "end":   END,           # terminate gracefully
        }
    )


    # -------------------------------------------------------
    # STEP 8: Wire the final pipeline (normal edges)
    # -------------------------------------------------------
    # After chart generation → always go to save_report
    # After save_report      → always go to END
    # These are fixed — no branching needed at these points.
    # -------------------------------------------------------
    graph.add_edge("generate_chart", "save_report")
    graph.add_edge("save_report",    END)


    # -------------------------------------------------------
    # STEP 9: Compile the graph
    # -------------------------------------------------------
    # compile() validates the graph structure (checks for
    # unreachable nodes, missing edges, etc.) and returns
    # a Runnable object that can be invoked with .invoke()
    # or streamed with .stream().
    #
    # After compile(), the graph is immutable — you cannot
    # add more nodes or edges. To change the graph, rebuild
    # and recompile it.
    # -------------------------------------------------------
    app = graph.compile()

    print("Graph compiled successfully.")
    return app


# ============================================================
# GRAPH VISUALISATION HELPER
# ============================================================

def print_graph_structure(app):
    """
    Prints a text representation of the compiled graph
    structure to the terminal for debugging and verification.

    Shows all nodes and their connections so you can verify
    the graph is wired the way you intended before running it.

    Args:
        app: The compiled LangGraph application from build_graph()
    """
    print("\n" + "=" * 55)
    print("GRAPH STRUCTURE")
    print("=" * 55)

    try:
        # LangGraph provides get_graph() on compiled apps
        graph_data = app.get_graph()

        print("\nNODES:")
        for node_id in graph_data.nodes:
            print(f"  • {node_id}")

        print("\nEDGES:")
        for edge in graph_data.edges:
            source = edge.source
            target = edge.target
            # Conditional edges have a conditional=True flag
            edge_type = "──►" if not getattr(edge, 'conditional', False) else "─?►"
            print(f"  {source:20} {edge_type} {target}")

    except Exception as e:
        print(f"  (Could not render graph details: {e})")

    print("=" * 55)


# ============================================================
# INITIAL STATE BUILDER
# ============================================================

def build_initial_state(file_path: str, user_question: str) -> dict:
    """
    Creates a clean initial state dictionary to start the graph.

    Every field in AnalystState is set here — required fields
    get real values, optional fields are initialised to None.
    This prevents KeyError exceptions in nodes that read
    optional fields with state.get().

    Args:
        file_path:     Path to the CSV or Excel file to analyse
        user_question: The user's natural language question

    Returns:
        A complete state dictionary ready to pass to app.invoke()

    Example:
        state = build_initial_state(
            file_path="data/bank_statement.csv",
            user_question="What category did I spend most on?"
        )
        result = app.invoke(state)
    """
    return {
        # ── Required inputs (provided by user) ───────────────
        "file_path":          file_path,
        "user_question":      user_question,

        # ── Data loading fields (filled by load_file_node) ───
        "dataframe_summary":  None,
        "raw_data_info":      None,

        # ── LLM reasoning fields (filled by llm_reason_node) ─
        "code_to_run":        None,
        "next_action":        None,

        # ── Code execution fields (filled by run_code_node) ──
        "code_output":        None,
        "error_message":      None,

        # ── Output fields (filled by later nodes) ────────────
        "chart_path":         None,
        "final_answer":       None,

        # ── Control fields (managed internally) ──────────────
        "retry_count":        0,
        "messages":           [],
    }

# """
# Builds and compiles the LangGraph StateGraph for the data analyst agent.
# """

# from functools import partial

# from langgraph.graph import END, StateGraph

# from .nodes import generate_code, load_file, run_code, save_chart, write_report
# from .state import AnalystState


# def _should_retry(state: AnalystState) -> str:
#     """Route back to generate_code on error, otherwise continue."""
#     return "generate_code" if state.get("error") else "run_code"


# def build_graph(llm):
#     """
#     Construct and compile the analyst StateGraph.

#     Flow:
#         load_file → generate_code → run_code
#                                        ↓ (error?) → generate_code (retry)
#                                        ↓ (ok)     → save_chart → write_report → END
#     """
#     graph = StateGraph(AnalystState)

#     # Register nodes (bind the LLM where needed)
#     graph.add_node("load_file", load_file)
#     graph.add_node("generate_code", partial(generate_code, llm=llm))
#     graph.add_node("run_code", run_code)
#     graph.add_node("save_chart", save_chart)
#     graph.add_node("write_report", partial(write_report, llm=llm))

#     # Entry point
#     graph.set_entry_point("load_file")

#     # Edges
#     graph.add_edge("load_file", "generate_code")
#     graph.add_edge("generate_code", "run_code")

#     # Conditional: retry on code error, else proceed
#     graph.add_conditional_edges(
#         "run_code",
#         _should_retry,
#         {
#             "generate_code": "generate_code",
#             "run_code": "save_chart",
#         },
#     )

#     graph.add_edge("save_chart", "write_report")
#     graph.add_edge("write_report", END)

#     return graph.compile()