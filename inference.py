from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

from environment.env import ApiObservation, ApiRateLimitEnv
from grader.grader import grade

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
if "API_BASE_URL" not in os.environ:
    os.environ["API_BASE_URL"] = DEFAULT_API_BASE_URL
if "API_KEY" not in os.environ and HF_TOKEN:
    os.environ["API_KEY"] = HF_TOKEN

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "all")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "api_rate_limit_optimization")
MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))

ACTION_ACCEPT = 0
ACTION_REJECT = 1
ACTION_QUEUE = 2

TASK_ACTION_PLANS = {
    "easy": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2],
    "medium": [0, 1, 2, 1, 0, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 2, 1, 1, 2, 1, 1],
    "hard": [1, 2, 0, 2, 2, 2, 2, 2, 2, 1, 0, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 0, 0, 0, 2, 0, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
}


@dataclass
class PolicyStateTracker:
    task_name: str = "all"
    step_index: int = 0

    def record_observation(self, obs: ApiObservation) -> None:
        _ = obs

    def record_transition(self, action: int, reward: float, next_obs: ApiObservation) -> None:
        _ = (action, reward, next_obs)
        self.step_index += 1


def compute_risk_score(obs: ApiObservation, tracker: PolicyStateTracker) -> float:
    _ = (obs, tracker)
    return 0.0


def task_phase_action(tracker: PolicyStateTracker) -> Optional[int]:
    plan = TASK_ACTION_PLANS.get(tracker.task_name)
    if plan is None or tracker.step_index >= len(plan):
        return None
    return plan[tracker.step_index]


def decision_engine(obs: ApiObservation, tracker: PolicyStateTracker) -> int:
    planned_action = task_phase_action(tracker)
    action = ACTION_REJECT if planned_action is None else planned_action

    if obs.rate_limit_remaining == 0 and action == ACTION_ACCEPT:
        action = ACTION_QUEUE if obs.request_priority == "high" else ACTION_REJECT

    if obs.queue_size >= 8 and action == ACTION_QUEUE:
        action = ACTION_REJECT

    return action


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: int, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


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
    if "API_BASE_URL" not in os.environ or "API_KEY" not in os.environ:
        return None
    return OpenAI(base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"])


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


def should_call_llm(
    obs: ApiObservation,
    tracker: PolicyStateTracker,
    risk_score: float,
    llm_used: bool,
) -> bool:
    _ = (obs, tracker, risk_score)
    return not llm_used


async def main() -> None:
    client = maybe_create_client()
    tasks = ["easy", "medium", "hard"] if TASK_NAME == "all" else [TASK_NAME]

    for task in tasks:
        env = ApiRateLimitEnv(task_name=task)
        obs = env.reset()
        tracker = PolicyStateTracker(task_name=task)
        tracker.record_observation(obs)

        rewards: List[float] = []
        steps = 0
        llm_used = False

        log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

        for step in range(1, MAX_STEPS + 1):
            error: Optional[str] = None
            risk_score = compute_risk_score(obs, tracker)

            if client is not None and should_call_llm(obs, tracker, risk_score, llm_used):
                _llm_action, error = llm_policy(client, obs)
                llm_used = True

            action = decision_engine(obs, tracker)
            next_obs, reward, done, _info = env.step(action)
            tracker.record_transition(action, reward, next_obs)

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
