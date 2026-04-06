from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

from pydantic import BaseModel, Field


Priority = Literal["low", "medium", "high"]

PRIORITY_REWARD: Dict[Priority, int] = {
    "low": 2,
    "medium": 5,
    "high": 10,
}


class ApiObservation(BaseModel):
    current_requests: int = Field(ge=0)
    rate_limit_remaining: int = Field(ge=0)
    time_window_remaining: int = Field(ge=0)
    request_priority: Priority
    queue_size: int = Field(ge=0)
    avg_queue_wait: float = Field(ge=0.0)
    recent_violations: int = Field(ge=0)
    system_load: int = Field(ge=0)


class ApiAction(BaseModel):
    action: int = Field(ge=0, le=2, description="0=accept, 1=reject, 2=queue")


class ApiReward(BaseModel):
    reward: float


@dataclass
class TaskConfig:
    name: str
    max_steps: int
    max_rate_limit: int
    time_window_size: int
    max_queue: int
    queue_wait_threshold: float
    processing_capacity: int
    max_queue_age: int
    request_pattern: List[int]
    priority_pattern: List[Priority]
    rate_limit_pattern: List[int] | None = None
    processing_capacity_pattern: List[int] | None = None
    system_load_threshold: int = 6
    burst_period: int = 0
    burst_length: int = 0
    burst_bonus: int = 0


TASKS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        name="easy",
        max_steps=50,
        max_rate_limit=8,
        time_window_size=5,
        max_queue=8,
        queue_wait_threshold=3.5,
        processing_capacity=2,
        max_queue_age=5,
        request_pattern=[1, 1, 2, 1, 1, 2, 1, 1, 2, 1],
        priority_pattern=["low", "low", "medium", "low", "low", "medium", "low", "low", "medium", "low"],
        rate_limit_pattern=[8, 8, 7, 8, 8, 7, 8, 8, 7, 8],
        processing_capacity_pattern=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        system_load_threshold=7,
        burst_period=0,
        burst_length=0,
        burst_bonus=0,
    ),
    "medium": TaskConfig(
        name="medium",
        max_steps=50,
        max_rate_limit=5,
        time_window_size=5,
        max_queue=6,
        queue_wait_threshold=2.5,
        processing_capacity=2,
        max_queue_age=4,
        request_pattern=[2, 1, 3, 2, 1, 4, 2, 1, 3, 2],
        priority_pattern=["medium", "low", "high", "medium", "high", "low", "medium", "high", "medium", "low"],
        rate_limit_pattern=[5, 4, 5, 4, 5, 4, 5, 4, 5, 4],
        processing_capacity_pattern=[2, 2, 1, 2, 2, 1, 2, 2, 1, 2],
        system_load_threshold=6,
        burst_period=12,
        burst_length=2,
        burst_bonus=2,
    ),
    "hard": TaskConfig(
        name="hard",
        max_steps=50,
        max_rate_limit=3,
        time_window_size=5,
        max_queue=5,
        queue_wait_threshold=1.8,
        processing_capacity=1,
        max_queue_age=3,
        request_pattern=[
            2, 2, 3, 2, 2, 3, 2, 2, 3,
            6, 7, 7, 6, 7, 6,
            3, 2, 3, 2, 3, 2, 3, 2, 3,
            7, 8, 7, 8, 7, 7,
            3, 2, 3, 2, 3, 2, 3, 2, 3, 2,
            4, 3, 4, 3, 4, 3, 4, 3, 4, 3,
        ],
        priority_pattern=[
            "medium", "medium", "high", "medium", "medium", "high", "medium", "low", "medium",
            "high", "high", "high", "high", "high", "high",
            "medium", "medium", "high", "medium", "high", "medium", "high", "medium", "high",
            "high", "high", "high", "high", "high", "high",
            "medium", "medium", "high", "medium", "high", "medium", "high", "medium", "high", "medium",
            "high", "medium", "high", "medium", "high", "medium", "high", "low", "medium", "low",
        ],
        rate_limit_pattern=[3, 2, 3, 2, 2, 3, 2, 2, 3, 2],
        processing_capacity_pattern=[1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
        system_load_threshold=5,
        burst_period=15,
        burst_length=5,
        burst_bonus=2,
    ),
}


class ApiRateLimitEnv:
    def __init__(self, task_name: str = "easy") -> None:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Expected one of {list(TASKS.keys())}.")
        self.task_name = task_name
        self.cfg = TASKS[task_name]
        self._init_runtime_state()

    def _init_runtime_state(self) -> None:
        self.step_count = 0
        self.window_index = 0
        self.current_rate_limit_cap = self._current_rate_limit_cap()
        self.rate_limit_remaining = self.current_rate_limit_cap
        self.time_window_remaining = self.cfg.time_window_size
        self.queue: List[Tuple[Priority, int]] = []
        self._latest_avg_queue_wait = 0.0
        self.total_reward = 0.0
        self.system_load = 0
        self.system_load_history: List[int] = []
        self.overload_streak = 0

        self.violations = 0
        self.total_high_priority_requests = 0
        self.handled_high_priority_requests = 0
        self.violation_history: List[int] = []

        self._last_info: Dict[str, object] = {}

    def _deterministic_current_requests(self) -> int:
        idx = self.step_count % len(self.cfg.request_pattern)
        base_requests = self.cfg.request_pattern[idx]

        if self.cfg.burst_period > 0 and self.cfg.burst_length > 0 and self.cfg.burst_bonus > 0:
            cycle_position = self.step_count % self.cfg.burst_period
            if cycle_position >= self.cfg.burst_period - self.cfg.burst_length:
                base_requests += self.cfg.burst_bonus

        return base_requests

    def _deterministic_priority(self) -> Priority:
        idx = self.step_count % len(self.cfg.priority_pattern)
        return self.cfg.priority_pattern[idx]

    def _current_rate_limit_cap(self) -> int:
        if self.cfg.rate_limit_pattern:
            return self.cfg.rate_limit_pattern[self.window_index % len(self.cfg.rate_limit_pattern)]
        return self.cfg.max_rate_limit

    def _current_processing_capacity(self) -> int:
        if self.cfg.processing_capacity_pattern:
            return self.cfg.processing_capacity_pattern[
                self.window_index % len(self.cfg.processing_capacity_pattern)
            ]
        return self.cfg.processing_capacity

    def _priority_for_offset(self, offset: int) -> Priority:
        idx = (self.step_count + offset) % len(self.cfg.priority_pattern)
        return self.cfg.priority_pattern[idx]

    def _avg_queue_wait(self) -> float:
        if not self.queue:
            return 0.0
        total_age = sum(age for _, age in self.queue)
        return total_age / len(self.queue)

    def _recent_violations(self, window: int = 5) -> int:
        if not self.violation_history:
            return 0
        return int(sum(self.violation_history[-window:]))

    def _current_observation(self) -> ApiObservation:
        self._latest_avg_queue_wait = self._avg_queue_wait()
        return ApiObservation(
            current_requests=self._deterministic_current_requests(),
            rate_limit_remaining=self.rate_limit_remaining,
            time_window_remaining=self.time_window_remaining,
            request_priority=self._deterministic_priority(),
            queue_size=len(self.queue),
            avg_queue_wait=round(self._latest_avg_queue_wait, 4),
            recent_violations=self._recent_violations(),
            system_load=self.system_load,
        )

    def reset(self) -> ApiObservation:
        self._init_runtime_state()
        return self._current_observation()

    def _process_queue(self) -> tuple[float, int, int]:
        if not self.queue:
            return 0.0, 0, 0

        queue_reward = 0.0
        processing_capacity = self._current_processing_capacity()
        processed_count = min(len(self.queue), processing_capacity)
        processed = 0
        priority_inversion_penalty = 0
        for _ in range(processed_count):
            priority, age = self.queue[0]
            if self.rate_limit_remaining > 0:
                self.rate_limit_remaining -= 1
                queue_reward += PRIORITY_REWARD[priority] * 0.5
                if priority == "high":
                    self.handled_high_priority_requests += 1
                processed += 1
            else:
                self.violations += 1
                queue_reward -= 5
                break
            self.queue.pop(0)

        if processed > 0 and len(self.queue) == 0:
            queue_reward += 1

        next_queue: List[Tuple[Priority, int]] = []
        for priority, age in self.queue:
            new_age = age + 1
            if new_age > self.cfg.max_queue_age:
                queue_reward -= 3 if priority == "high" else 2
                continue
            next_queue.append((priority, new_age))

        self.queue = next_queue

        if len(self.queue) >= self.cfg.max_queue:
            low_priority_slots = sum(1 for queued_priority, _ in self.queue if queued_priority == "low")
            if low_priority_slots >= max(1, len(self.queue) // 2) and any(
                queued_priority == "high" and queued_age > 0 for queued_priority, queued_age in self.queue
            ):
                priority_inversion_penalty = 4
                queue_reward -= priority_inversion_penalty

        self._latest_avg_queue_wait = self._avg_queue_wait()
        if self._latest_avg_queue_wait > self.cfg.queue_wait_threshold:
            queue_reward -= 2

        return queue_reward, processed, priority_inversion_penalty

    def _system_health_factor(self) -> float:
        if self.cfg.max_rate_limit <= 0:
            return 0.0
        return max(0.2, min(1.0, self.rate_limit_remaining / float(self.cfg.max_rate_limit)))

    def _apply_system_load_dynamics(
        self,
        accepted_count: int,
        rejected_count: int,
        processed_count: int,
    ) -> float:
        queue_pressure = max(0, len(self.queue) - self._current_processing_capacity())
        load_delta = accepted_count + max(0, queue_pressure // 2) - rejected_count - processed_count
        self.system_load = max(0, min(20, self.system_load + load_delta))

        overload_penalty = 0.0
        if self.system_load > self.cfg.system_load_threshold:
            self.overload_streak += 1
            overload_excess = self.system_load - self.cfg.system_load_threshold
            overload_penalty = min(12.0, float(overload_excess * (1 + (0.25 * self.overload_streak))))
        else:
            self.overload_streak = 0

        self.system_load_history.append(self.system_load)
        return overload_penalty

    def _advance_time_window(self) -> None:
        self.time_window_remaining -= 1
        if self.time_window_remaining <= 0:
            self.window_index += 1
            self.current_rate_limit_cap = self._current_rate_limit_cap()
            self.rate_limit_remaining = self.current_rate_limit_cap
            self.time_window_remaining = self.cfg.time_window_size

    def step(self, action: int) -> Tuple[ApiObservation, float, bool, Dict[str, object]]:
        if self.step_count >= self.cfg.max_steps:
            obs = self._current_observation()
            return obs, 0.0, True, {"message": "Episode already completed"}

        positive_reward = 0.0
        penalty_total = 0.0
        start_violations = self.violations
        current_requests = self._deterministic_current_requests()
        priority = self._deterministic_priority()
        accepted_count = 0
        rejected_count = 0
        self._last_info = {
            "action": action,
            "priority": priority,
            "current_requests": current_requests,
        }

        self.total_high_priority_requests += 1 if priority == "high" else 0
        for offset in range(1, current_requests):
            background_priority = self._priority_for_offset(offset)
            self.queue.append((background_priority, 0))
            if background_priority == "high":
                self.total_high_priority_requests += 1

        if len(self.queue) > self.cfg.max_queue:
            penalty_total -= 3

        if action not in (0, 1, 2):
            penalty_total -= 5
            self._last_info["invalid_action"] = True
        elif action == 0:
            if self.rate_limit_remaining > 0:
                self.rate_limit_remaining -= 1
                accepted_count += 1
                positive_reward += PRIORITY_REWARD[priority]
                if priority == "low" and (
                    self.rate_limit_remaining <= 1 or any(q_priority == "high" for q_priority, _ in self.queue)
                ):
                    penalty_total -= 1
                if priority == "high":
                    self.handled_high_priority_requests += 1
            else:
                self.violations += 1
                penalty_total -= 5
        elif action == 1:
            rejected_count += 1
            if priority == "high":
                penalty_total -= 10
            elif self.rate_limit_remaining > 0:
                penalty_total -= 1
        elif action == 2:
            self.queue.append((priority, 0))
            if self.rate_limit_remaining > 0:
                penalty_total -= 1
            if len(self.queue) > self.cfg.max_queue:
                penalty_total -= 3

        queue_reward, processed_count, priority_inversion_penalty = self._process_queue()
        if queue_reward >= 0:
            positive_reward += queue_reward
        else:
            penalty_total += queue_reward

        overload_penalty = self._apply_system_load_dynamics(
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            processed_count=processed_count,
        )
        penalty_total -= overload_penalty

        health_factor = self._system_health_factor()
        reward = (positive_reward * health_factor) + penalty_total

        self.violation_history.append(self.violations - start_violations)

        self.step_count += 1
        self._advance_time_window()

        done = self.step_count >= self.cfg.max_steps
        self.total_reward += reward

        obs = self._current_observation()
        info = {
            **self._last_info,
            "step": self.step_count,
            "violations": self.violations,
            "total_reward": self.total_reward,
            "queue_size": len(self.queue),
            "avg_queue_wait": round(self._latest_avg_queue_wait, 4),
            "recent_violations": self._recent_violations(),
            "processing_capacity": self._current_processing_capacity(),
            "rate_limit_cap": self.current_rate_limit_cap,
            "queue_reward_component": round(queue_reward, 4),
            "system_load": self.system_load,
            "system_health_factor": round(health_factor, 4),
            "priority_inversion_penalty": priority_inversion_penalty,
            "overload_penalty": round(overload_penalty, 4),
            "task": self.task_name,
        }
        return obs, reward, done, info

    def state(self) -> ApiObservation:
        return self._current_observation()

    def metrics(self) -> Dict[str, float]:
        if self.total_high_priority_requests == 0:
            success_rate = 1.0
        else:
            success_rate = self.handled_high_priority_requests / self.total_high_priority_requests

        return {
            "total_reward": self.total_reward,
            "violations": float(self.violations),
            "total_high_priority_requests": float(self.total_high_priority_requests),
            "handled_high_priority_requests": float(self.handled_high_priority_requests),
            "high_priority_success_rate": success_rate,
            "violation_count": float(self.violations),
            "system_load_final": float(self.system_load),
            "system_load_mean": float(sum(self.system_load_history) / len(self.system_load_history)) if self.system_load_history else 0.0,
            "system_load_variance": float(
                sum((load - (sum(self.system_load_history) / len(self.system_load_history))) ** 2 for load in self.system_load_history)
                / len(self.system_load_history)
            ) if self.system_load_history else 0.0,
            "steps": float(self.step_count),
        }
