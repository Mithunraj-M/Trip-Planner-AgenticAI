# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from travel_graph import run_travel_planner

app = FastAPI()

class TravelRequest(BaseModel):
    user_goal: str

@app.post("/plan-trip")
def plan_trip(request: TravelRequest):
    return run_travel_planner(request.user_goal)
