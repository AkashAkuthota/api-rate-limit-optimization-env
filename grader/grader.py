from __future__ import annotations

from typing import Dict


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def grade(metrics: Dict[str, float], max_steps: int = 50) -> Dict[str, float | int]:
    total_reward = float(metrics.get("total_reward", 0.0))
    violations = int(metrics.get("violation_count", metrics.get("violations", 0.0)))
    total_high = float(metrics.get("total_high_priority_requests", 0.0))
    handled_high = float(metrics.get("handled_high_priority_requests", 0.0))
    system_load_variance = float(metrics.get("system_load_variance", 0.0))

    if total_high <= 0:
        success_rate = 1.0
    else:
        success_rate = handled_high / total_high

    min_reward = -12.0 * max_steps
    max_reward = 12.0 * max_steps
    normalized_total_reward = _clamp((total_reward - min_reward) / (max_reward - min_reward))

    max_possible_violations = max_steps
    violation_penalty = _clamp(violations / float(max_possible_violations))
    system_stability = _clamp(1.0 - (system_load_variance / float(max_steps * max_steps)))

    score = (
        0.35 * normalized_total_reward
        + 0.3 * _clamp(success_rate)
        + 0.2 * system_stability
        - 0.15 * violation_penalty
    )
    score = _clamp(score)

    return {
        "score": float(round(score, 6)),
        "success_rate": float(round(_clamp(success_rate), 6)),
        "violations": violations,
    }
