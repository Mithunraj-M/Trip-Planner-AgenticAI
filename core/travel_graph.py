from langgraph.graph import StateGraph, END
from langchain.agents import Tool
import inspect
from typing import TypedDict, List, Optional
import sys
import os

# Debug: print path of StateGraph
print(inspect.getfile(StateGraph))

# 📦 Add parent directory to import custom agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 🧠 Import actual planner, tool, refiner agents
from agents.planner_agent import planner_node
from agents.tool_agent import tool_executor_node
from agents.refiner_agent import refiner_node

# 🧾 Define state format
class TravelPlannerState(TypedDict, total=False):
    user_goal: str
    task_queue: List[str]
    history: List[tuple]
    refined: bool
    assumptions: List[str]
    removed_tasks: List[str]
    last_error: Optional[str]
    iteration_count: int

# 🔁 Define loop condition
MAX_ITERATIONS = 3

def should_continue(state):
    task_queue = state.get("task_queue", [])
    iteration_count = state.get("iteration_count", 0)
    if task_queue and iteration_count < MAX_ITERATIONS:
        return "continue"
    else:
        return "stop"

# 🏗️ Build the graph
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

builder.add_conditional_edges(
    "refiner",
    should_continue,
    {
        "continue": "tool",
        "stop": END
    }
)

# ✅ Compile the graph
graph = builder.compile()

# 🚀 Run the graph
if __name__ == "__main__":
    initial_state = {
        "user_goal": "Plan a trip to Manali under ₹10,000",
        "iteration_count": 0
    }

    final_result = graph.invoke(initial_state)

    # 🧹 Run remaining tasks without refinement
    final_state = final_result.copy()
    task_queue = final_state.get("task_queue", [])
    history = final_state.get("history", [])

    while task_queue:
        final_state["task_queue"] = task_queue
        final_state["history"] = history

        final_state = tool_executor_node(final_state)

        # Update for next loop
        task_queue = final_state.get("task_queue", [])
        history = final_state.get("history", [])

    # ✅ Output final plan
print("\n✅ Final Travel Plan:")

# Get original goal
user_goal = initial_state["user_goal"].lower()

# Clean up history
filtered_history = []
for task, result, tool in final_state.get("history", []):
    # Only keep tasks that seem relevant to the user goal (i.e., Manali-related)
    if "manali" in task.lower() or any(word in task.lower() for word in user_goal.split()):
        filtered_history.append((task, result, tool))

# Print sanitized tasks
for task, result, tool in filtered_history:
    print(f"🔹 Task: {task}\n   🛠 Tool: {tool}\n   📍 Result: {result}\n")

