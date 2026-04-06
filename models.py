from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ApiRateLimitOptimizationAction(Action):
    action: int = Field(..., ge=0, le=2, description="0=accept, 1=reject, 2=queue")


class ApiRateLimitOptimizationObservation(Observation):
    current_requests: int = Field(default=0, ge=0)
    rate_limit_remaining: int = Field(default=0, ge=0)
    time_window_remaining: int = Field(default=0, ge=0)
    request_priority: str = Field(default="low")
    queue_size: int = Field(default=0, ge=0)
    avg_queue_wait: float = Field(default=0.0, ge=0.0)
    recent_violations: int = Field(default=0, ge=0)
    system_load: int = Field(default=0, ge=0)
