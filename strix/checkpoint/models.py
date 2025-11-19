from datetime import UTC, datetime
from strix.agents.state import AgentState
from typing import Any
from pydantic import BaseModel, Field


class CheckpointVersionInfo(BaseModel):
    strix_version: str
    run_name: str
    checkpoint_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TracerCheckpointInfo(BaseModel):
    agents: dict[str, dict[str, Any]]
    tool_executions: dict[int, dict[str, Any]]
    chat_messages: list[dict[str, Any]]

class AgentCheckpointInfo(BaseModel):
    agent_state: AgentState
    prompt_modules: list[str] | None = None
    pending_agent_messages: dict[str, list[dict[str, Any]]] = {}
    is_root_agent: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentGraphCheckpointInfo(BaseModel):
    root_agent_id: str | None = None
    pending_agent_messages: dict[str, list[dict[str, Any]]] = {}
    agent_states: dict[str, AgentState] = {}


class StrixExecutionCheckpoint(BaseModel):
    checkpoint_id: str
    run_name: str | None = None
    version_info: CheckpointVersionInfo | None = None
    tracer_info: TracerCheckpointInfo
    graph_info: AgentGraphCheckpointInfo