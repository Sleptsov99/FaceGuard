"""State management module."""

from typing import Dict, Any, Optional
from enum import Enum


class ProcessingState(Enum):
    """Processing states."""
    IDLE = "idle"
    PROCESSING = "processing"
    PAUSED = "paused"
    ERROR = "error"


class StateManager:
    """Manages application state."""
    
    def __init__(self):
        """Initialize state manager."""
        self._state = ProcessingState.IDLE
        self._data: Dict[str, Any] = {}
        self._frame_count = 0
    
    @property
    def state(self) -> ProcessingState:
        """Get current state."""
        return self._state
    
    @state.setter
    def state(self, new_state: ProcessingState):
        """Set new state."""
        self._state = new_state
    
    def set_data(self, key: str, value: Any):
        """Set state data."""
        self._data[key] = value
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get state data."""
        return self._data.get(key, default)
    
    def increment_frame(self):
        """Increment frame counter."""
        self._frame_count += 1
    
    @property
    def frame_count(self) -> int:
        """Get current frame count."""
        return self._frame_count
    
    def reset(self):
        """Reset state to initial."""
        self._state = ProcessingState.IDLE
        self._data = {}
        self._frame_count = 0
    
    def is_processing(self) -> bool:
        """Check if currently processing."""
        return self._state == ProcessingState.PROCESSING
    
    def is_error(self) -> bool:
        """Check if in error state."""
        return self._state == ProcessingState.ERROR
