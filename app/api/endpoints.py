"""API endpoints module (placeholder for future REST API)."""

from typing import Dict, Any


class APIEndpoints:
    """API endpoints for CV Engine."""
    
    def __init__(self):
        """Initialize API endpoints."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current engine status.
        
        Returns:
            Status dictionary
        """
        return {
            "status": "ok",
            "version": "0.1.0"
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.
        
        Returns:
            Metrics dictionary
        """
        return {
            "metrics": {}
        }
