from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Union


@dataclass
class PipelineState:
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineStep:
    name: str
    fn: Callable[[PipelineState], PipelineState]


PipelineStepFn = Callable[[PipelineState], PipelineState]
PipelineNode = Union[PipelineStep, PipelineStepFn]


def run_dag(initial_state: PipelineState, pipeline_steps: list[PipelineNode]) -> PipelineState:
    state = initial_state
    for step in pipeline_steps:
        if isinstance(step, PipelineStep):
            state = step.fn(state)
            trace = list(state.payload.get("dag_trace", []))
            trace.append(step.name)
            state.payload["dag_trace"] = trace
        else:
            state = step(state)
    return state
