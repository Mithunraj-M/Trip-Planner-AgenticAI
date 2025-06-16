from agents.planner_agent import PlannerAgent
from agents.tool_agent import tool_executor_node


planner = PlannerAgent()
user_goal = "Plan a budget trip to Manali under ₹10,000"

print(" User Goal:", user_goal)
all_tasks = planner.plan_tasks(user_goal)


tasks_to_run = all_tasks[:4]

print(f"\n Planned Tasks ({len(tasks_to_run)} shown):")
for i, task in enumerate(tasks_to_run, 1):
    print(f"{i}. {task}")


state = {"history": []}

for task in tasks_to_run:
    print(f"\n🔹 Executing Task: {task}")
    state["current_task"] = task
    state = tool_executor_node(state)

    print(f" Tool Used: {state['tool_used']}")
    print(f" Output:\n{state['last_output'][:500]}...")


print("\n📘 Final Summary:")
for i, (task, output, tool) in enumerate(state["history"], 1):
    print(f"{i}. [{tool}] {task} ➝ {output[:80]}...")
