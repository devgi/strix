from typing import TYPE_CHECKING, Optional
import uuid
from importlib.metadata import version
from pathlib import Path

from strix.agents.state import AgentState
from strix.checkpoint.models import StrixExecutionCheckpoint, CheckpointVersionInfo, AgentCheckpointInfo
from strix.telemetry import get_global_tracer
from strix.checkpoint.store import CheckpointStore
from strix.checkpoint.file_store import CheckpointFileStore

if TYPE_CHECKING:
    from strix.telemetry.tracer import Tracer


_resumed_execution : StrixExecutionCheckpoint | None = None
_active_checkpoint_store : CheckpointStore | None = None

def resume_checkpoint(checkpoint_file_path: str) -> StrixExecutionCheckpoint:
    """
    Resume execution from a checkpoint file.
    This should be called before any other resume function in this module.
    """
    global _resumed_execution
    _resumed_execution = CheckpointFileStore(Path(checkpoint_file_path)).load()
    if not _resumed_execution:
        raise RuntimeError(f"Failed to load checkpoint from file: {checkpoint_file_path}")

    return _resumed_execution


def resume_tracer(tracer: Optional["Tracer"] = None) -> None:
    if _resumed_execution is None:
        raise RecursionError("No loaded checkpoint to resume tracer from")

    if tracer is None:
        tracer = get_global_tracer()
    
    if tracer:
        tracer.restore_state_from_checkpoint(_resumed_execution.tracer_info)

def resume_root_agent_state_from_checkpoint() -> AgentState:
    if _resumed_execution is None:
        raise RuntimeError("No loaded checkpoint to resume main agent checkpoint from")
    
    root_agent_id = _resumed_execution.graph_info.root_agent_id

    if root_agent_id is None:
        raise RuntimeError("No root agent id found in checkpoint")

    root_agent_state =  _resumed_execution.graph_info.agent_states[root_agent_id]


    # Delete the agent_id reference from state allow the new instance generate its own unique id
    root_agent_state_modified = AgentState.model_construct(**root_agent_state.model_dump(exclude={"agent_id"}))

    return root_agent_state_modified


def initialize_execution_recording(results_dir: Path, run_name: str) -> None:
    global _active_checkpoint_store
    version_info = CheckpointVersionInfo(strix_version=version("strix-agent"), run_name=run_name, checkpoint_id=str(uuid.uuid4()))

    _active_checkpoint_store = CheckpointFileStore(results_dir / ".strix_checkpoint.db")
    _active_checkpoint_store.store_version_info(version_info)


async def record_execution_checkpoint(agent_checkpoint_info: AgentCheckpointInfo) -> None:


    tracer = get_global_tracer()
    if tracer is None:
        raise RuntimeError("No tracer found to record checkpoint")

    tracer_checkpoint_info = tracer.record_state_to_checkpoint()

    if _active_checkpoint_store:
        _active_checkpoint_store.store_checkpoint(agent_checkpoint_info, tracer_checkpoint_info)
