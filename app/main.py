from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from environment.env import ApiRateLimitEnv, TASKS
from grader.grader import grade


app = FastAPI(title="API Rate Limit Optimization Environment")

env = ApiRateLimitEnv(task_name="easy")


class ResetRequest(BaseModel):
    task: str = Field(default="easy", description="easy | medium | hard")


class StepRequest(BaseModel):
    action: int = Field(ge=0, le=2, description="0=accept, 1=reject, 2=queue")


@app.get("/")
def root() -> dict:
    return {
        "message": "Adaptive API Rate Limit Optimization Environment API is running.",
        "usage": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "grade": "GET /grade",
            "health": "GET /health",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reset")
def reset(payload: ResetRequest | None = None) -> dict:
    global env
    task = "easy" if payload is None else payload.task
    if task not in TASKS:
        return {
            "error": f"Unknown task '{task}'",
            "valid_tasks": list(TASKS.keys()),
        }
    env = ApiRateLimitEnv(task_name=task)
    observation = env.reset()
    return {
        "task": task,
        "state": observation.model_dump(),
    }


@app.post("/step")
def step(payload: StepRequest) -> dict:
    observation, reward, done, info = env.step(payload.action)
    return {
        "state": observation.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict:
    return {"state": env.state().model_dump()}


@app.get("/grade")
def get_grade() -> dict:
    return grade(env.metrics())


def main() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860)
