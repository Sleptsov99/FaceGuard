"""Temporal analysis module for tracking metrics over time."""

import numpy as np
from collections import deque
from typing import Dict, List, Optional


class TemporalAnalyzer:
    """Analyzes temporal patterns in facial metrics."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize temporal analyzer.
        
        Args:
            window_size: Number of frames to keep in history
        """
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
    
    def add_frame_data(self, frame_index: int, metrics: Dict[str, float]):
        """
        Add metrics for a frame to history.
        
        Args:
            frame_index: Current frame index
            metrics: Dictionary of metric values
        """
        self.history.append({
            "frame": frame_index,
            "metrics": metrics
        })
    
    def get_trend(self, metric_name: str, window: int = 10) -> Optional[float]:
        """
        Calculate trend for a specific metric.
        
        Args:
            metric_name: Name of the metric
            window: Number of recent frames to consider
            
        Returns:
            Trend value (positive = increasing, negative = decreasing)
        """
        if len(self.history) < 2:
            return None
        
        values = [h["metrics"].get(metric_name) for h in list(self.history)[-window:]]
        values = [v for v in values if v is not None]
        
        if len(values) < 2:
            return None
        
        return np.polyfit(range(len(values)), values, 1)[0]
    
    def get_average(self, metric_name: str, window: int = 10) -> Optional[float]:
        """
        Calculate average for a specific metric.
        
        Args:
            metric_name: Name of the metric
            window: Number of recent frames to consider
            
        Returns:
            Average value
        """
        if len(self.history) < 1:
            return None
        
        values = [h["metrics"].get(metric_name) for h in list(self.history)[-window:]]
        values = [v for v in values if v is not None]
        
        if len(values) == 0:
            return None
        
        return np.mean(values)
    
    def clear(self):
        """Clear history."""
        self.history.clear()
