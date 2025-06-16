import os
from dotenv import load_dotenv
from pathlib import Path
import openai
import sys
from langchain.agents import Tool, initialize_agent
from langchain_community.chat_models import ChatOpenAI
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


from tools.tools import (
    serpapi_search,
    google_places,
    smart_currency_conversion,
    map_distance,
)



env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)



OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in .env file")

openai.api_key = OPENROUTER_KEY
openai.api_base = "https://openrouter.ai/api/v1"



tools = [
    Tool.from_function(
        name="WebSearch",
        func=serpapi_search,
        description="Use SerpAPI to fetch search results for any topic.",
    ),
    Tool.from_function(
        name="GooglePlaces",
        func=google_places,
        description="Find metadata and location data for a place.",
    ),
    Tool.from_function(
        name="MapDistance",
        func=map_distance,
        description="Compute travel distance and ETA between locations.",
    ),
    Tool.from_function(
        name="CurrencyConverter",
        func=smart_currency_conversion,
        description="Convert destination's local currency to INR.",
    ),
]



llm = ChatOpenAI(
    openai_api_key=OPENROUTER_KEY,
    base_url="https://openrouter.ai/api/v1",
    model="gpt-3.5-turbo",
    temperature=0.2,
)



tool_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-zero-shot-react-description",
    verbose=True,
    return_intermediate_steps=True,
     
)



def tool_executor_node(state):
    task_queue = state.get("task_queue", [])
    history = state.get("history", [])

    if not task_queue:
        return {
            **state,
            "last_output": "No tasks in queue.",
            "tool_used": None,
            "current_task": None,
        }

    
    
    current_task = task_queue.pop(0)

    try:
        
        
        result = tool_agent.invoke(
            {"input": current_task},
            config={"callbacks": [], "tags": [], "metadata": {}, "run_name": "ToolExecutor"}
        )

        
        
        tool_used = "unknown"
        steps = result.get("intermediate_steps", [])
        print("Intermediate Steps:", steps)

        if steps and isinstance(steps, list):
            for step in steps:
                if isinstance(step, tuple) and hasattr(step[0], "tool"):
                    tool_used = step[0].tool
                    break

        return {
            **state,
            "current_task": current_task,
            "last_output": result.get("output", result),
            "tool_used": tool_used,
            "task_queue": task_queue,
            "history": history + [(current_task, result.get("output", result), tool_used)],
        }

    except Exception as e:
        return {
            **state,
            "current_task": current_task,
            "last_output": f"Error: {e}",
            "tool_used": "error",
            "task_queue": task_queue,
            "history": history + [(current_task, f"Error: {e}", "error")],
        }

#sample for testing unit functionality

if __name__ == "__main__":
    
    
    test_state = {
        "task_queue":[
    "Find budget hotels in Sydney for 5 nights",
    "Compare flight options from Delhi to Sydney, including layovers and prices",
    "List free or cheap tourist attractions in Sydney and Melbourne",
    "Search for local events or festivals happening in Australia in July",
    "Find the distance from Melbourne to Sydney by road and air",
    "Find the average cost of travel in sydney"
],
        "history": []
    }

    
    
    while test_state["task_queue"]:
        test_state = tool_executor_node(test_state)

    
    print("\n Final History:")
    
    print("\n Final History:")
    for i, (task, output, tool) in enumerate(test_state.get("history", []), 1):
        print(f"{i}. Task: {task}")
        print(f"    Tool Used: {tool}")
        print(f"    Output: {output}\n")
