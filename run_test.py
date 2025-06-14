from agents.planner_agent import PlannerAgent
from agents.tool_agent import tool_executor_node

# ----------------------------------------
# 🧠 Step 1: Planner generates task list
# ----------------------------------------
planner = PlannerAgent()
user_goal = "Plan a budget trip to Manali under ₹10,000"

print("🎯 User Goal:", user_goal)
all_tasks = planner.plan_tasks(user_goal)

# Limit to 4 tasks for testing
tasks_to_run = all_tasks[:4]

print(f"\n🧠 Planned Tasks ({len(tasks_to_run)} shown):")
for i, task in enumerate(tasks_to_run, 1):
    print(f"{i}. {task}")

# ----------------------------------------
# 🤖 Step 2: Tool executor runs each task
# ----------------------------------------
state = {"history": []}

for task in tasks_to_run:
    print(f"\n🔹 Executing Task: {task}")
    state["current_task"] = task
    state = tool_executor_node(state)

    print(f"✅ Tool Used: {state['tool_used']}")
    print(f"📄 Output:\n{state['last_output'][:500]}...")

# ----------------------------------------
# 📘 Step 3: Summary
# ----------------------------------------
print("\n📘 Final Summary:")
for i, (task, output, tool) in enumerate(state["history"], 1):
    print(f"{i}. [{tool}] {task} ➝ {output[:80]}...")
