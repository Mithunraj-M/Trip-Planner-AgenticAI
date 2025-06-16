
from langgraph.graph import StateGraph, END
from agents.planner_agent import planner_node
from agents.tool_agent import tool_executor_node
from agents.refiner_agent import refiner_node
from typing import List, Optional, TypedDict


class TravelPlannerState(TypedDict, total=False):
    user_goal: str
    task_queue: List[str]
    history: List[tuple]
    refined: bool
    assumptions: List[str]
    removed_tasks: List[str]
    last_error: Optional[str]
    iteration_count: int

MAX_ITERATIONS = 3

def should_continue(state):
    return "continue" if state.get("task_queue") and state.get("iteration_count", 0) < MAX_ITERATIONS else "stop"


def build_graph():
    builder = StateGraph(TravelPlannerState)
    builder.add_node("planner", planner_node)
    builder.add_node("tool", tool_executor_node)
    builder.add_node("refiner", lambda state: {
        **refiner_node(state),
        "iteration_count": state.get("iteration_count", 0) + 1
    })
    builder.set_entry_point("planner")
    builder.add_edge("planner", "tool")
    builder.add_edge("tool", "refiner")
    builder.add_conditional_edges("refiner", should_continue, {"continue": "tool", "stop": END})
    return builder.compile()


def run_travel_planner(user_input: str):
    graph = build_graph()
    initial_state = {"user_goal": user_input, "iteration_count": 0}
    final_result = graph.invoke(initial_state)

    final_state = final_result.copy()
    task_queue = final_state.get("task_queue", [])
    history = final_state.get("history", [])

    while task_queue:
        final_state["task_queue"] = task_queue
        final_state["history"] = history
        final_state = tool_executor_node(final_state)
        task_queue = final_state.get("task_queue", [])
        history = final_state.get("history", [])

    # Return filtered history
    user_goal = user_input.lower()
    filtered = []
    for task, result, tool in final_state.get("history", []):
        if "manali" in task.lower() or any(word in task.lower() for word in user_goal.split()):
            filtered.append({"task": task, "result": result, "tool": tool})

    return {"plan": filtered}
