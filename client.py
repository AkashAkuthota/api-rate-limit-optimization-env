from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ApiRateLimitOptimizationAction, ApiRateLimitOptimizationObservation


class ApiRateLimitOptimizationEnv(
    EnvClient[ApiRateLimitOptimizationAction, ApiRateLimitOptimizationObservation, State]
):
    def _step_payload(self, action: ApiRateLimitOptimizationAction) -> Dict[str, Any]:
        return {"action": action.action}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ApiRateLimitOptimizationObservation]:
        observation_payload = payload.get("observation") or payload.get("state") or payload
        observation = ApiRateLimitOptimizationObservation(
            current_requests=observation_payload.get("current_requests", 0),
            rate_limit_remaining=observation_payload.get("rate_limit_remaining", 0),
            time_window_remaining=observation_payload.get("time_window_remaining", 0),
            request_priority=observation_payload.get("request_priority", "low"),
            queue_size=observation_payload.get("queue_size", 0),
            avg_queue_wait=observation_payload.get("avg_queue_wait", 0.0),
            recent_violations=observation_payload.get("recent_violations", 0),
            system_load=observation_payload.get("system_load", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("info", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        state_payload = payload.get("state") or payload
        return State(
            episode_id=state_payload.get("episode_id"),
            step_count=state_payload.get("step_count", 0),
        )
