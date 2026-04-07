from __future__ import annotations

import asyncio
import os
from typing import List, Optional

from openai import OpenAI

from environment.env import ApiRateLimitEnv, ApiObservation
from grader.grader import grade

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

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


def build_policy_prompt(obs: ApiObservation) -> str:
    return (
        "You are controlling an API gateway. Return exactly one digit: 0, 1, or 2.\n"
        "Actions: 0=accept, 1=reject, 2=queue.\n"
        "Prioritize high-priority requests, avoid rate-limit violations, and avoid overload.\n"
        f"current_requests={obs.current_requests}\n"
        f"rate_limit_remaining={obs.rate_limit_remaining}\n"
        f"time_window_remaining={obs.time_window_remaining}\n"
        f"request_priority={obs.request_priority}\n"
        f"queue_size={obs.queue_size}\n"
        f"avg_queue_wait={obs.avg_queue_wait}\n"
        f"recent_violations={obs.recent_violations}\n"
        f"system_load={obs.system_load}\n"
        "Respond with only the action digit."
    )


def parse_action(content: str) -> Optional[int]:
    for char in content:
        if char in {"0", "1", "2"}:
            return int(char)
    return None


def maybe_create_client() -> Optional[OpenAI]:
    if not API_BASE_URL or not API_KEY:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def llm_policy(client: OpenAI, obs: ApiObservation) -> tuple[Optional[int], Optional[str]]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=5,
            messages=[
                {
                    "role": "system",
                    "content": "Return only one action digit: 0, 1, or 2.",
                },
                {"role": "user", "content": build_policy_prompt(obs)},
            ],
        )
    except Exception as exc:
        return None, f"llm_error:{type(exc).__name__}"

    message = completion.choices[0].message.content or ""
    action = parse_action(message)
    if action is None:
        return None, "llm_error:invalid_action"
    return action, None


async def main() -> None:
    client = maybe_create_client()

    if TASK_NAME == "all":
        tasks = ["easy", "medium", "hard"]
    else:
        tasks = [TASK_NAME]

    for task in tasks:
        env = ApiRateLimitEnv(task_name=task)
        obs = env.reset()

        rewards: List[float] = []
        steps = 0
        llm_used = False

        log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

        for step in range(1, MAX_STEPS + 1):
            action: Optional[int] = None
            error: Optional[str] = None

            if client is not None and (
                not llm_used
                or obs.request_priority == "high"
                or obs.rate_limit_remaining <= 1
                or obs.system_load > 0.7
            ):
                action, error = llm_policy(client, obs)
                llm_used = True

            if action is None:
                action = rule_policy(obs)

            next_obs, reward, done, _info = env.step(action)

            rewards.append(reward)
            steps = step
            log_step(step=step, action=action, reward=reward, done=done, error=error)

            obs = next_obs
            if done:
                break

        result = grade(env.metrics(), max_steps=MAX_STEPS)
        score = float(result["score"])
        success = score >= 0.6
        log_end(success=bool(success), steps=steps, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
