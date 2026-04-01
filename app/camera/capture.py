"""Camera capture module."""

import cv2 
from typing import Optional ,Union 


class CameraCapture :
    """Handles camera and video file capture."""

    def __init__ (self ,source :Union [int ,str ]=0 ):
        """
        Initialize camera capture.
        
        Args:
            source: Camera index (int) or video file path (str)
        """
        self .source =source 
        self .capture =cv2 .VideoCapture (source )

        if not self .capture .isOpened ():
            raise RuntimeError (f"Failed to open capture source: {source }")

    def read (self )->Optional :
        """Read a frame from the capture source."""
        ret ,frame =self .capture .read ()
        return frame if ret else None 

    def is_opened (self )->bool :
        """Check if capture is opened."""
        return self .capture .isOpened ()

    def release (self ):
        """Release the capture resource."""
        self .capture .release ()

    @property 
    def fps (self )->float :
        """Get FPS of the capture source."""
        return self .capture .get (cv2 .CAP_PROP_FPS )

    @property 
    def width (self )->int :
        """Get frame width."""
        return int (self .capture .get (cv2 .CAP_PROP_FRAME_WIDTH ))

    @property 
    def height (self )->int :
        """Get frame height."""
        return int (self .capture .get (cv2 .CAP_PROP_FRAME_HEIGHT ))
