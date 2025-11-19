import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from logging import getLogger

from strix.agents.state import AgentState
from strix.checkpoint.models import (
    StrixExecutionCheckpoint,
    CheckpointVersionInfo,
    TracerCheckpointInfo,
    AgentGraphCheckpointInfo,
    AgentCheckpointInfo,
)
from strix.checkpoint.store import CheckpointStore

logger = getLogger(__name__)


ISO = "%Y-%m-%dT%H:%M:%S.%fZ"


class CheckpointFileStore(CheckpointStore):
    """Simple filesystem-backed checkpoint store.

    Layout under `file_path` (a directory):
      - version_info.json
      - agents/agent-{agent_id}.json   # one file per agent
      - tracer/tracer-{uuid}.json      # tracer checkpoints (we keep only last by file mtime)
      - execution_checkpoint.json      # optional consolidated StrixExecutionCheckpoint

    Guarantees:
      - Writes are atomic (temp file + os.replace).
      - Storing a checkpoint for an agent will overwrite that agent's file (keeps last).

    Notes / assumptions:
      - The caller provides `agent_state_opaque` and `tracer_state_opaque` as JSON-serializable strings
        (already serialized representations). This class will store them as-is under the per-agent
        / tracer files. When loading, it will attempt to parse them back into the Pydantic models
        expected by StrixExecutionCheckpoint. If parsing fails, the raw strings are left as-is where
        parsing is not possible.
      - load() will prefer a consolidated execution_checkpoint.json if present. Otherwise it will
        assemble an execution checkpoint from version_info.json + the newest tracer file + all
        agent files.
    """

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self.agents_dir = self.file_path / "agents"
        self.tracer_dir = self.file_path / "tracer"
        self.file_path.mkdir(parents=True, exist_ok=True)
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.tracer_dir.mkdir(parents=True, exist_ok=True)

    # -- small helpers
    def _atomic_write(self, path: Path, data: bytes) -> None:
        dirpath = path.parent
        dirpath.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dirpath)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp_path, str(path))
        finally:
            # if something failed and tmp still exists, try to clean
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _read_json(self, path: Path) -> Optional[dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    # -- public API
    def store_version_info(self, version_info: CheckpointVersionInfo) -> None:
        """Store version_info to version_info.json (atomic)."""
        payload = version_info.model_dump_json(indent=4)
        self._atomic_write( self.file_path / "version_info.json", payload.encode("utf-8"))

    def store_checkpoint(self, agent_checkpoint: AgentCheckpointInfo, tracer_checkpoint: TracerCheckpointInfo) -> None:
        """Store a checkpoint for a single agent.

        Behavior:
          - Writes agent file at agents/agent-{agent_id}.json (overwrites previous file atomically).
          - If tracer_state_opaque is provided, writes a new tracer file tracer/tracer-{uuid}.json
            (we keep them as separate timestamped files; load() will pick the newest one).
        """
        agent_path = self.agents_dir / f"state-{agent_checkpoint.agent_state.agent_id}.json"
        agent_checkpoint_payload = agent_checkpoint.model_dump_json(indent=4)
        self._atomic_write(agent_path, agent_checkpoint_payload.encode("utf-8"))

        tracer_path = self.tracer_dir / "tracer-latest.json"
        tracer_checkpoint_payload = tracer_checkpoint.model_dump_json(indent=4)
        self._atomic_write(tracer_path, tracer_checkpoint_payload.encode("utf-8"))

    
    def load(self) -> Optional[StrixExecutionCheckpoint]:
        """
        Load a StrixExecutionCheckpoint with the following logic:

        1. Try to load version_info.json
        2. Try to load tracer-latest.json
        3. Load all state-{agent_id}.json and deduce:
            - root_agent_id
            - latest pending_agent_messages
            - agent_states (AgentState objects)
        4. Build AgentGraphCheckpointInfo
        5. Assemble and return StrixExecutionCheckpoint
        """
        
        # --- 1. VERSION INFO ---
        version_info_raw = self._read_json(self.file_path / "version_info.json")
        version_info = None
        if version_info_raw:
            try:
                version_info = CheckpointVersionInfo.model_validate(version_info_raw)
            except Exception as e:
                logger.warning(f"Invalid version_info.json: {e}")

        # --- 2. TRACER INFO ---
        tracer_info_raw = self._read_json(self.tracer_dir / "tracer-latest.json")
        tracer_info = None
        if tracer_info_raw:
            try:
                tracer_info = TracerCheckpointInfo.model_validate(tracer_info_raw)
            except Exception as e:
                logger.warning(f"Invalid tracer-latest.json: {e}")

        # --- 3. LOAD ALL AGENTS ---
        agent_files = sorted(self.agents_dir.glob("state-*.json"))

        if not agent_files:
            logger.warning("No agents found in checkpoint")
            return None

        decoded_agent_states: dict[str, AgentState] = {}
        root_agent_id: str | None = None

        latest_pending: dict[str, list[dict[str, Any]]] = {}
        latest_created_at: datetime | None = None

        for p in agent_files:
            agent_id = p.stem.replace("state-", "")
            raw = self._read_json(p)
            if not raw:
                continue

            try:
                # raw is expected to follow AgentCheckpointInfo structure
                agent_cp = AgentCheckpointInfo.model_validate(raw)
            except Exception as e:
                logger.warning(f"Invalid agent checkpoint for {agent_id}: {e}")
                continue

            # 1. Capture agent_state
            decoded_agent_states[agent_id] = agent_cp.agent_state

            # 2. Root agent?
            if agent_cp.is_root_agent and root_agent_id is None:
                root_agent_id = agent_id

            # 3. Pending messages: choose by latest created_at
            if agent_cp.pending_agent_messages:
                created_at = agent_cp.created_at
                if latest_created_at is None or created_at > latest_created_at:
                    latest_created_at = created_at
                    latest_pending = agent_cp.pending_agent_messages

        # No valid agent states recovered? Bail out.
        if not decoded_agent_states:
            logger.warning("No valid agent states found in checkpoint")
            return None

        # If root agent was never detected, fallback to the first one
        if root_agent_id is None:
            root_agent_id = next(iter(decoded_agent_states.keys()))

        # --- 4. BUILD GRAPH INFO ---
        graph_info = AgentGraphCheckpointInfo(
            root_agent_id=root_agent_id,
            pending_agent_messages=latest_pending or {},
            agent_states=decoded_agent_states,
        )

        # --- 5. FINAL CHECKS ---
        if not tracer_info:
            logger.warning("Missing tracer_info; cannot build full checkpoint")
            return None

        if not version_info:
            logger.warning("Missing version_info; cannot build full checkpoint")
            return None

        # static checkpoint_id for file-based restoration
        checkpoint_id = version_info.checkpoint_id
        run_name = version_info.run_name

        return StrixExecutionCheckpoint(
            checkpoint_id=checkpoint_id,
            run_name=run_name,
            version_info=version_info,
            tracer_info=tracer_info,
            graph_info=graph_info,
        )