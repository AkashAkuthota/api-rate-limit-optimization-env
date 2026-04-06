from __future__ import annotations

import asyncio
import os
from typing import List, Optional

from openai import OpenAI

from environment.env import ApiRateLimitEnv, ApiObservation
from grader.grader import grade

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

TASK_NAME = os.getenv("TASK_NAME", "all")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "api_rate_limit_optimization")
MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: int, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def rule_policy(obs: ApiObservation) -> int:
    # Deterministic baseline policy optimized for priority + rate-limit safety.
    if obs.request_priority == "high":
        if obs.rate_limit_remaining > 0:
            return 0
        return 2

    if obs.request_priority == "medium":
        if obs.rate_limit_remaining > 1:
            return 0
        if obs.queue_size < 3:
            return 2
        return 1

    # low
    if obs.rate_limit_remaining > 2:
        return 0
    if obs.queue_size < 2:
        return 2
    return 1


def maybe_create_client() -> Optional[OpenAI]:
    if not API_KEY:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


async def main() -> None:
    _client = maybe_create_client()
    _ = _client  # Reserved for optional model-driven policies.

    if TASK_NAME == "all":
        tasks = ["easy", "medium", "hard"]
    else:
        tasks = [TASK_NAME]

    for task in tasks:
        env = ApiRateLimitEnv(task_name=task)
        obs = env.reset()

        rewards: List[float] = []
        steps = 0

        log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

        for step in range(1, MAX_STEPS + 1):
            action = rule_policy(obs)
            next_obs, reward, done, _info = env.step(action)

            rewards.append(reward)
            steps = step
            log_step(step=step, action=action, reward=reward, done=done, error=None)

            obs = next_obs
            if done:
                break

        result = grade(env.metrics(), max_steps=MAX_STEPS)
        score = float(result["score"])
        success = score >= 0.6
        log_end(success=bool(success), steps=steps, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
