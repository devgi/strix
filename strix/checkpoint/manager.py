from typing import TYPE_CHECKING, Optional
import uuid
from importlib.metadata import version
from pathlib import Path
from datetime import datetime

from strix.agents.state import AgentState
from strix.checkpoint.models import StrixExecutionCheckpoint, CheckpointVersionInfo, AgentCheckpointInfo
from strix.telemetry import get_global_tracer
from strix.checkpoint.store import CheckpointStore
from strix.checkpoint.file_store import CheckpointFileStore
from strix.checkpoint.sqlite_store import CheckpointSQLiteStore

if TYPE_CHECKING:
    from strix.telemetry.tracer import Tracer


_resumed_execution : StrixExecutionCheckpoint | None = None
_active_checkpoint_store : CheckpointStore | None = None
_active_checkpoint_path: Path | None = None

def resume_checkpoint(checkpoint_file_path: str) -> StrixExecutionCheckpoint:
    """
    Resume execution from a checkpoint file.
    This should be called before any other resume function in this module.
    """
    global _resumed_execution
    _resumed_execution = CheckpointSQLiteStore(Path(checkpoint_file_path)).load()
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

    root_agent_state =  _resumed_execution.graph_info.agent_states[root_agent_id].model_copy()

    root_agent_state.add_message("user", _make_resume_agent_prompt(_resumed_execution))

    # Delete the agent_id reference from state allow the new instance generate its own unique id
    root_agent_state_modified = AgentState.model_construct(**root_agent_state.model_dump(exclude={"agent_id"}))

    return root_agent_state_modified


def _make_resume_agent_prompt(checkpoint: StrixExecutionCheckpoint) -> str:
        # Defensive: guard against missing data/None keys
        root_agent_id = getattr(checkpoint.graph_info, "root_agent_id", None)
        agent_states = getattr(checkpoint.graph_info, "agent_states", {})
        checkpoint_start_time = None

        if root_agent_id and root_agent_id in agent_states:
            root_agent_state = agent_states[root_agent_id]
            checkpoint_start_time = getattr(root_agent_state, "start_time", None)

        try:
            dt_start = datetime.fromisoformat(checkpoint_start_time) if checkpoint_start_time else None
            execution_age = ""
            if dt_start:
                delta = datetime.now(dt_start.tzinfo) - dt_start
                hrs = delta.total_seconds() // 3600
                mins = (delta.total_seconds() % 3600) // 60
                if hrs >= 1:
                    execution_age = f"{int(hrs)} hour(s) and {int(mins)} minute(s) ago"
                elif mins > 0:
                    execution_age = f"{int(mins)} minute(s) ago"
                else:
                    execution_age = "just now"
            else:
                execution_age = "an unknown time ago"
        except Exception:
            execution_age = "an unknown time ago"


        detailed_agent_info: list[str] = []
        if agent_states:
            for aid, ast in agent_states.items():
                name = getattr(ast, "agent_name", "[unknown]")
                iterations = getattr(ast, "iteration", 0)
                task = getattr(ast, "task", "[no task]")
                short_task = (task[:80] + "...") if task and len(task) > 80 else task
                detailed_agent_info.append(
                    f"    - id: {aid}\n"
                    f"      name: {name}\n"
                    f"      iterations: {iterations}\n"
                    f"      task: {short_task}"
                )
        agents_summary = (
            "\n- Agents detected in checkpoint:\n" +
            ("\n".join(detailed_agent_info) if detailed_agent_info else "    [none]")
        )

        resume_prompt = (
            f"This Strix agent has been resumed from a saved checkpoint."
            f"\n\n"
            f"Checkpoint details:\n"
            f"- Original start time: {checkpoint_start_time or '[unknown]'}"
            f" ({execution_age})"
            f"{agents_summary}"
            f"\n\n"
            f"WARNING:\n"
            f"- Only YOU (the main/root agent) are now running. Any subagents or parallel agents from the original execution (if any) are no longer running. "
            f"They will NOT resume unless explicitly re-created.\n"
            f"- DO NOT assume any previous network, environment, tools server, or system state are available. "
            f"Connectivity, underlying tools, or even the machine itself may have changed since the original execution."
            f"\n- You must proceed as if you have just started, but you can use prior messages and agent state restored here as context."
            f"\n\n"
            f"Continue only after thoroughly verifying or reacquiring any information you need — nothing from the prior state is guaranteed valid except the restored data."
        )

        return resume_prompt


def initialize_execution_recording(results_dir: Path, run_name: str) -> None:
    global _active_checkpoint_store
    global _active_checkpoint_path
    version_info = CheckpointVersionInfo(strix_version=version("strix-agent"), run_name=run_name, checkpoint_id=str(uuid.uuid4()))

    checkpoint_path = results_dir / "strix_checkpoint.db"
    _active_checkpoint_store = CheckpointSQLiteStore(checkpoint_path)
    _active_checkpoint_store.store_version_info(version_info)
    _active_checkpoint_path = checkpoint_path

def get_active_checkpoint_path() -> Optional[Path]:
    global _active_checkpoint_path
    return _active_checkpoint_path


async def record_execution_checkpoint(agent_checkpoint_info: AgentCheckpointInfo) -> None:


    tracer = get_global_tracer()
    if tracer is None:
        raise RuntimeError("No tracer found to record checkpoint")

    tracer_checkpoint_info = tracer.record_state_to_checkpoint()

    if _active_checkpoint_store:
        _active_checkpoint_store.store_checkpoint(agent_checkpoint_info, tracer_checkpoint_info)
