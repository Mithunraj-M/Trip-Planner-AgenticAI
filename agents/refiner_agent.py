import ast
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI


load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

DEFAULT_LLM = ChatOpenAI(
    openai_api_key=OPENROUTER_KEY,
    base_url="https://openrouter.ai/api/v1",
    model="gpt-3.5-turbo",
    temperature=0.3,
    max_tokens=300
)


def get_removal_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a task deduplicator for a travel planning agent.
Match completed tasks to current tasks, even if paraphrased.
Use fuzzy logic — if a task like 'Find hotels' was completed and 'Find budget hotels in Manali' is in the list, they match.

Return only a Python list of task strings to remove:
["task1", "task2"]"""),
        ("human", "Completed Task: {task}\nCurrent Task Queue:\n{task_queue}")
    ])
    return prompt | llm | StrOutputParser()


def get_addition_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a follow-up suggester for a travel planner.
Based on the output of a completed task, suggest logical next tasks to add.
Only include new ideas relevant to the user's goal.

Return only a Python list:
["new task 1", "new task 2"]"""),
        ("human", "Completed Task: {task}\nOutput:\n{output}\nGoal: {goal}")
    ])
    return prompt | llm | StrOutputParser()


def get_assumption_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assumption extractor for a task refinement agent.
List any assumptions made when interpreting the task or generating follow-ups.

Return a Python list:
["assumption1", "assumption2"]"""),
        ("human", "Completed Task: {task}\nOutput:\n{output}")
    ])
    return prompt | llm | StrOutputParser()


def refiner_node(state: dict) -> dict:
    goal = state.get("user_goal", "")
    history = state.get("history", [])
    task_queue = state.get("task_queue", [])
    refinement_count = state.get("refinement_count", 0)
    llm = state.get("llm", DEFAULT_LLM)

    if refinement_count >= 3:
        print(" Max refinement limit reached. Skipping.")
        return {**state, "refined": False, "reason": "Max iterations reached"}

    
    state["refinement_count"] = refinement_count + 1

    removal_chain = get_removal_chain(llm)
    addition_chain = get_addition_chain(llm)
    assumption_chain = get_assumption_chain(llm)

    removed = []
    added = []
    assumptions = []

    for task, output, tool in history:
        task_queue_str = "\n".join(f"- {t}" for t in task_queue)

        print("\n Invoking LLM with:")
        print("Task:", task)
        print("Output:", output)
        print("Tool:", tool)
        print("Task Queue:\n", task_queue_str)

        
        try:
            to_remove_raw = removal_chain.invoke({
                "task": task,
                "task_queue": task_queue_str
            })
            print(" Removals Raw:", to_remove_raw)
            to_remove = ast.literal_eval(to_remove_raw)
            for t in to_remove:
                if t in task_queue:
                    task_queue.remove(t)
                    removed.append(t)
        except Exception as e:
            print(" Removal failed:", e)

       
        try:
            to_add_raw = addition_chain.invoke({
                "task": task,
                "output": output,
                "goal": goal
            })
            print("Additions Raw:", to_add_raw)
            to_add = ast.literal_eval(to_add_raw)
            for t in to_add:
                if t not in task_queue:
                    task_queue.append(t)
                    added.append(t)
        except Exception as e:
            print("Addition failed:", e)

       
        try:
            assumptions_raw = assumption_chain.invoke({
                "task": task,
                "output": output
            })
            print("Assumptions Raw:", assumptions_raw)
            assumptions.extend(ast.literal_eval(assumptions_raw))
        except Exception as e:
            print("Assumption extraction failed:", e)

    return {
        **state,
        "task_queue": task_queue,
        "refined": True,
        "removed_tasks": removed,
        "added_tasks": added,
        "assumptions": assumptions
    }


if __name__ == "__main__":
    dummy_state = {
        "user_goal": "Plan a trip to Manali under ₹10,000",
        "task_queue": [
            "Find budget hotels in Manali",
            "Search for cheapest transport options to Manali",
            "Check exchange rates for USD to INR",
            "List tourist attractions in Manali",
            "Plan a 3-day itinerary"
        ],
        "history": [
            ("Find hotels", "Hotel1 ₹800, Hotel2 ₹950", "GooglePlaces"),
            ("Check buses", "Bus from Delhi to Manali ₹550", "WebSearch"),
            ("Convert USD to INR", "1 USD = ₹83.2", "CurrencyConverter")
        ],
        "refinement_count": 0
    }

    result = refiner_node(dummy_state)

    print("\n Final Refined Tasks:", result.get("task_queue"))
    print("Removed Tasks:", result.get("removed_tasks"))
    print("Added Tasks:", result.get("added_tasks"))
    print("Assumptions:", result.get("assumptions"))
