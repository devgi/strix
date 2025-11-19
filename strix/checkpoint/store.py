from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from strix.checkpoint.models import (
    StrixExecutionCheckpoint,
    AgentCheckpointInfo,
    TracerCheckpointInfo,
    CheckpointVersionInfo
)


class CheckpointStore(ABC):
    """Abstract base class defining the interface for checkpoint storage implementations.
    
    This interface is implemented by concrete checkpoint stores like CheckpointFile
    (filesystem-backed) and potentially other implementations (e.g., SQLite, database-backed).
    
    The interface provides three main operations:
    1. Store version information about the checkpoint
    2. Store checkpoint data for agents and tracer state
    3. Load a complete execution checkpoint
    """

    @abstractmethod
    def store_version_info(self, version_info: CheckpointVersionInfo) -> None:
        """Store version information to the checkpoint store.
        
        Args:
            version_info: The version information to store, including strix version,
                         run name, and creation timestamp.
        """
        pass

    @abstractmethod
    def store_checkpoint(self, agent_checkpoint: AgentCheckpointInfo, tracer_checkpoint: TracerCheckpointInfo) -> None:
        """Store a checkpoint for a single agent. This might be called multiple times
        for a single execution, with the latest state being the one that is stored.
        
        Args:
            agent_checkpoint: The agent checkpoint to store.
        """
        pass

    @abstractmethod
    def load(self) -> Optional[StrixExecutionCheckpoint]:
        """Load a complete execution checkpoint from the store.
        
        Returns:
            A StrixExecutionCheckpoint if sufficient data exists to construct one,
            None otherwise.
            
        The implementation should attempt to reconstruct a complete checkpoint from
        stored version info, tracer state, and agent states.
        """
        pass


