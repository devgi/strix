import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional, TypeVar
from datetime import UTC, datetime
from logging import getLogger
from pydantic import BaseModel

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

T = TypeVar("T", bound=BaseModel)


class CheckpointSQLiteStore(CheckpointStore):
    """SQLite-backed checkpoint store with configurable retention modes.
    
    Environment variable STRIX_CHECKPOINT_LEVEL controls behavior:
      - "none": No checkpointing performed
      - "latest": Only maintain the latest checkpoint (overwrites)
      - "all": Keep all checkpoints with auto-incrementing version counter
    
    Schema:
      - version_info: stores CheckpointVersionInfo
      - tracer_info: stores TracerCheckpointInfo
      - agent_checkpoints: stores AgentCheckpointInfo (one row per agent)
    
    All tables include a 'version' column for tracking checkpoint iterations.
    """

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_level = os.getenv("STRIX_CHECKPOINT_LEVEL", "latest").lower()
        if self.checkpoint_level not in ("none", "latest", "all"):
            logger.warning(
                f"Invalid STRIX_CHECKPOINT_LEVEL={self.checkpoint_level}, defaulting to 'latest'"
            )
            self.checkpoint_level = "latest"
        
        self.conn = sqlite3.connect(str(self.file_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()  # Thread-safe access to database
        self._optimize_for_writes()
        self._init_schema()

    def _optimize_for_writes(self) -> None:
        """Configure SQLite for optimal write performance while maintaining crash safety."""
        with self._lock:
            # DELETE mode: single file database (no WAL/SHM files)
            # Journal file is temporary and automatically deleted after transactions
            self.conn.execute("PRAGMA journal_mode=DELETE")
            # NORMAL synchronous: good balance of speed and safety
            # Ensures data is written to disk before returning, preventing corruption
            self.conn.execute("PRAGMA synchronous=NORMAL")
            # Increase cache size for better performance (64MB)
            self.conn.execute("PRAGMA cache_size=-65536")
            # Disable foreign keys for faster inserts (not used anyway)
            self.conn.execute("PRAGMA foreign_keys=OFF")
            # Optimize for write-heavy workloads
            self.conn.execute("PRAGMA temp_store=MEMORY")
            # Standard page size
            self.conn.execute("PRAGMA page_size=4096")
            self.conn.commit()

    def _init_schema(self) -> None:
        """Initialize database schema with three tables."""
        with self._lock:
            with self.conn:
                # Version info table
                # Add 'latest_id' column for "latest" mode to enable INSERT OR REPLACE
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS version_info (
                        version INTEGER PRIMARY KEY AUTOINCREMENT,
                        latest_id INTEGER DEFAULT 1,
                        checkpoint_id TEXT NOT NULL,
                        run_name TEXT NOT NULL,
                        strix_version TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        data TEXT NOT NULL
                    )
                """)
                
                # Add UNIQUE constraint for atomic replace in "latest" mode
                if self.checkpoint_level == "latest":
                    self.conn.execute("""
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_version_latest_id 
                        ON version_info(latest_id)
                    """)
                
                # Tracer info table
                # Add 'latest_id' column for "latest" mode to enable INSERT OR REPLACE
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS tracer_info (
                        version INTEGER PRIMARY KEY AUTOINCREMENT,
                        latest_id INTEGER DEFAULT 1,
                        created_at TEXT NOT NULL,
                        data TEXT NOT NULL
                    )
                """)
                
                # Add UNIQUE constraint for atomic replace in "latest" mode
                if self.checkpoint_level == "latest":
                    self.conn.execute("""
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_tracer_latest_id 
                        ON tracer_info(latest_id)
                    """)
                
            # Agent checkpoints table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_checkpoints (
                    version INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    is_root_agent INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            
            # Index for faster lookups
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_id 
                ON agent_checkpoints(agent_id, version DESC)
            """)
            
            # Add UNIQUE constraint for atomic replace operations in "latest" mode
            # This enables INSERT OR REPLACE to work atomically without DELETE + INSERT
            if self.checkpoint_level == "latest":
                self.conn.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_id_unique 
                    ON agent_checkpoints(agent_id)
                """)

    def store_version_info(self, version_info: CheckpointVersionInfo) -> None:
        """Store version info according to checkpoint level."""
        if self.checkpoint_level == "none":
            return
        
        data = version_info.model_dump_json()
        created_at = version_info.created_at.isoformat()
        
        with self._lock:
            with self.conn:
                if self.checkpoint_level == "latest":
                    # Use INSERT OR REPLACE for atomic operation
                    self.conn.execute(
                        """INSERT OR REPLACE INTO version_info 
                           (latest_id, checkpoint_id, run_name, strix_version, created_at, data) 
                           VALUES (1, ?, ?, ?, ?, ?)""",
                        (
                            version_info.checkpoint_id,
                            version_info.run_name,
                            version_info.strix_version,
                            created_at,
                            data,
                        ),
                    )
                else:
                    self.conn.execute(
                        """INSERT INTO version_info 
                           (checkpoint_id, run_name, strix_version, created_at, data) 
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            version_info.checkpoint_id,
                            version_info.run_name,
                            version_info.strix_version,
                            created_at,
                            data,
                        ),
                    )
                self.conn.commit()

    def store_checkpoint(
        self, 
        agent_checkpoint: AgentCheckpointInfo, 
        tracer_checkpoint: TracerCheckpointInfo
    ) -> None:
        """Store agent and tracer checkpoints according to checkpoint level.
        
        Thread-safe: This method can be called from multiple threads safely.
        """
        if self.checkpoint_level == "none":
            return
        
        # Serialize outside lock to reduce lock contention
        # No indent for faster serialization and smaller storage
        tracer_data = tracer_checkpoint.model_dump_json()
        tracer_created_at = datetime.now(UTC).isoformat()
        agent_data = agent_checkpoint.model_dump_json()
        agent_created_at = agent_checkpoint.created_at.isoformat()
        agent_id = agent_checkpoint.agent_state.agent_id
        is_root = 1 if agent_checkpoint.is_root_agent else 0
        
        with self._lock:
            with self.conn:
                # Store tracer checkpoint - use INSERT OR REPLACE in "latest" mode
                if self.checkpoint_level == "latest":
                    self.conn.execute(
                        """INSERT OR REPLACE INTO tracer_info (latest_id, created_at, data) 
                           VALUES (1, ?, ?)""",
                        (tracer_created_at, tracer_data),
                    )
                else:
                    self.conn.execute(
                        """INSERT INTO tracer_info (created_at, data) VALUES (?, ?)""",
                        (tracer_created_at, tracer_data),
                    )
                
                # Store agent checkpoint - use INSERT OR REPLACE in "latest" mode
                if self.checkpoint_level == "latest":
                    self.conn.execute(
                        """INSERT OR REPLACE INTO agent_checkpoints 
                           (agent_id, is_root_agent, created_at, data) 
                           VALUES (?, ?, ?, ?)""",
                        (agent_id, is_root, agent_created_at, agent_data),
                    )
                else:
                    self.conn.execute(
                        """INSERT INTO agent_checkpoints 
                           (agent_id, is_root_agent, created_at, data) 
                           VALUES (?, ?, ?, ?)""",
                        (agent_id, is_root, agent_created_at, agent_data),
                    )
                
                self.conn.commit()

    def _load_latest_json(self, table: str, model_class: type[T]) -> Optional[T]:
        """Load the latest JSON data from a table and parse it with the given model."""
        # Validate table name to prevent SQL injection
        valid_tables = {"version_info", "tracer_info", "agent_checkpoints"}
        if table not in valid_tables:
            raise ValueError(f"Invalid table name: {table}")
        
        with self._lock:
            cursor = self.conn.execute(
                f"SELECT data FROM {table} ORDER BY version DESC LIMIT 1"
            )
            row = cursor.fetchone()
        
        if not row:
            logger.warning(f"No {table} found in checkpoint")
            return None
        
        try:
            return model_class.model_validate_json(row["data"])
        except Exception as e:
            logger.warning(f"Invalid {table}: {e}")
            return None

    def load(self) -> Optional[StrixExecutionCheckpoint]:
        """Load the latest execution checkpoint from SQLite database.
        
        Returns:
            StrixExecutionCheckpoint if all required data exists, None otherwise.
        """
        if self.checkpoint_level == "none":
            logger.info("Checkpoint level is 'none', no checkpoint to load")
            return None
        
        # --- 1. LOAD VERSION INFO (latest) ---
        version_info = self._load_latest_json("version_info", CheckpointVersionInfo)
        if not version_info:
            return None
        
        # --- 2. LOAD TRACER INFO (latest) ---
        tracer_info = self._load_latest_json("tracer_info", TracerCheckpointInfo)
        if not tracer_info:
            return None
        
        # --- 3. LOAD ALL AGENTS (latest version per agent_id) ---
        with self._lock:
            if self.checkpoint_level == "latest":
                # Simple: get all rows (there's only one per agent)
                cursor = self.conn.execute(
                    "SELECT data, agent_id FROM agent_checkpoints"
                )
            else:  # "all" - get latest version for each agent_id
                cursor = self.conn.execute("""
                    SELECT data, agent_id 
                    FROM agent_checkpoints 
                    WHERE version IN (
                        SELECT MAX(version) 
                        FROM agent_checkpoints 
                        GROUP BY agent_id
                    )
                """)
            
            rows = cursor.fetchall()
        if not rows:
            logger.warning("No agent checkpoints found")
            return None
        
        decoded_agent_states: dict[str, AgentState] = {}
        root_agent_id: str | None = None
        latest_pending: dict[str, list[dict[str, Any]]] = {}
        latest_created_at: datetime | None = None
        
        for row in rows:
            try:
                agent_cp = AgentCheckpointInfo.model_validate_json(row["data"])
            except Exception as e:
                logger.warning(f"Invalid agent checkpoint for {row['agent_id']}: {e}")
                continue
            
            agent_id = agent_cp.agent_state.agent_id
            decoded_agent_states[agent_id] = agent_cp.agent_state
            
            if agent_cp.is_root_agent and root_agent_id is None:
                root_agent_id = agent_id
            
            if agent_cp.pending_agent_messages:
                created_at = agent_cp.created_at
                if latest_created_at is None or created_at > latest_created_at:
                    latest_created_at = created_at
                    latest_pending = agent_cp.pending_agent_messages
        
        if not decoded_agent_states:
            logger.warning("No valid agent states found")
            return None
        
        if root_agent_id is None:
            root_agent_id = next(iter(decoded_agent_states.keys()))
        
        # --- 4. BUILD GRAPH INFO ---
        graph_info = AgentGraphCheckpointInfo(
            root_agent_id=root_agent_id,
            pending_agent_messages=latest_pending or {},
            agent_states=decoded_agent_states,
        )
        
        # --- 5. ASSEMBLE CHECKPOINT ---
        return StrixExecutionCheckpoint(
            checkpoint_id=version_info.checkpoint_id,
            run_name=version_info.run_name,
            version_info=version_info,
            tracer_info=tracer_info,
            graph_info=graph_info,
        )

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self.conn:
                self.conn.close()

    def __del__(self):
        """Ensure connection is closed on deletion."""
        if hasattr(self, "conn"):
            self.close()
    