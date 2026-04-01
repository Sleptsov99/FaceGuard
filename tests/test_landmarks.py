"""Tests for Stage 1 landmark detection."""

import sys 
from pathlib import Path 
import unittest 
import numpy as np 

sys .path .insert (0 ,str (Path (__file__ ).parent .parent ))

from app .landmarks .result import DetectionResult ,DetectionStatus ,EyeLandmarks ,HeadPose 
from app .landmarks .detector import LandmarkDetector 




class TestDetectionStatus (unittest .TestCase ):
    def test_is_usable_ok (self ):
        self .assertTrue (DetectionStatus .OK .is_usable ())

    def test_is_usable_bad_angle (self ):
        self .assertTrue (DetectionStatus .BAD_ANGLE .is_usable ())

    def test_is_usable_multiple (self ):
        self .assertTrue (DetectionStatus .MULTIPLE_FACES .is_usable ())

    def test_not_usable_no_face (self ):
        self .assertFalse (DetectionStatus .NO_FACE .is_usable ())

    def test_not_usable_too_small (self ):
        self .assertFalse (DetectionStatus .FACE_TOO_SMALL .is_usable ())


class TestDetectionResult (unittest .TestCase ):
    def test_face_found_false_when_no_face (self ):
        r =DetectionResult (status =DetectionStatus .NO_FACE )
        self .assertFalse (r .face_found )

    def test_face_found_true_when_ok (self ):
        r =DetectionResult (status =DetectionStatus .OK ,confidence =0.9 )
        self .assertTrue (r .face_found )

    def test_is_valid_only_ok (self ):
        self .assertTrue (DetectionResult (status =DetectionStatus .OK ).is_valid )
        for s in (DetectionStatus .NO_FACE ,DetectionStatus .BAD_ANGLE ,
        DetectionStatus .FACE_TOO_SMALL ,DetectionStatus .MULTIPLE_FACES ):
            self .assertFalse (DetectionResult (status =s ).is_valid )


class TestHeadPose (unittest .TestCase ):
    def test_frontal (self ):
        self .assertTrue (HeadPose (yaw =0.0 ,pitch =0.0 ,roll =0.0 ).is_frontal ())

    def test_frontal_within_threshold (self ):
        self .assertTrue (HeadPose (yaw =29.0 ,pitch =-10.0 ,roll =5.0 ).is_frontal ())

    def test_not_frontal_yaw (self ):
        self .assertFalse (HeadPose (yaw =35.0 ,pitch =0.0 ,roll =0.0 ).is_frontal ())

    def test_not_frontal_pitch (self ):
        self .assertFalse (HeadPose (yaw =0.0 ,pitch =-31.0 ,roll =0.0 ).is_frontal ())

    def test_custom_thresholds (self ):
        pose =HeadPose (yaw =20.0 ,pitch =0.0 ,roll =0.0 )
        self .assertFalse (pose .is_frontal (max_yaw =15.0 ))
        self .assertTrue (pose .is_frontal (max_yaw =25.0 ))


class TestEyeLandmarks (unittest .TestCase ):
    def test_shapes (self ):
        contour =np .zeros ((16 ,2 ),dtype =np .float32 )
        ear_pts =np .zeros ((6 ,2 ),dtype =np .float32 )
        eye =EyeLandmarks (contour =contour ,ear_points =ear_pts )
        self .assertEqual (eye .contour .shape ,(16 ,2 ))
        self .assertEqual (eye .ear_points .shape ,(6 ,2 ))




class TestLandmarkDetectorNoFace (unittest .TestCase ):
    @classmethod 
    def setUpClass (cls ):
        cls .detector =LandmarkDetector ()

    @classmethod 
    def tearDownClass (cls ):
        cls .detector .release ()

    def _black_frame (self ,h =480 ,w =640 ):
        return np .zeros ((h ,w ,3 ),dtype =np .uint8 )

    def test_black_frame_returns_no_face (self ):
        result =self .detector .detect (self ._black_frame ())
        self .assertEqual (result .status ,DetectionStatus .NO_FACE )
        self .assertFalse (result .face_found )
        self .assertIsNone (result .face_bbox )
        self .assertIsNone (result .right_eye )
        self .assertIsNone (result .left_eye )

    def test_draw_no_face_does_not_crash (self ):
        frame =self ._black_frame ()
        result =self .detector .detect (frame )
        out =self .detector .draw (frame .copy (),result )
        self .assertEqual (out .shape ,frame .shape )

    def test_confidence_zero_when_no_face (self ):
        result =self .detector .detect (self ._black_frame ())
        self .assertEqual (result .confidence ,0.0 )


class TestLandmarkDetectorSmallFace (unittest .TestCase ):
    """Detector with very high min_face_size_ratio triggers FACE_TOO_SMALL."""

    @classmethod 
    def setUpClass (cls ):

        cls .detector =LandmarkDetector (min_face_size_ratio =0.99 )

    @classmethod 
    def tearDownClass (cls ):
        cls .detector .release ()

    def test_threshold_attribute (self ):
        self .assertEqual (self .detector .min_face_size_ratio ,0.99 )


if __name__ =="__main__":
    unittest .main ()
