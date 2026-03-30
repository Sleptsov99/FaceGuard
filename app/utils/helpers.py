"""Utility helper functions."""

import cv2
import numpy as np
from typing import Tuple, Optional


def resize_frame(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
    """
    Resize frame if it exceeds max width.
    
    Args:
        frame: Input frame
        max_width: Maximum width
        
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    
    if width <= max_width:
        return frame
    
    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(frame, (new_width, new_height))


def draw_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = (0, 255, 0),
    font_scale: float = 0.7
) -> np.ndarray:
    """
    Draw text on frame.
    
    Args:
        frame: Input frame
        text: Text to draw
        position: (x, y) position
        color: BGR color
        font_scale: Font scale
        
    Returns:
        Frame with text
    """
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2
    )
    return frame


def calculate_fps(frame_time: float, prev_time: float) -> Tuple[float, float]:
    """
    Calculate FPS.
    
    Args:
        frame_time: Current frame time
        prev_time: Previous frame time
        
    Returns:
        Tuple of (fps, current_time)
    """
    current_time = frame_time
    fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0
    return fps, current_time
