"""
Landmark index constants for the 478-point refined face mesh topology
(MediaPipe Face Mesh / Face Landmarker Tasks — same indexing).
All indices are from the person's perspective (left/right = person's left/right).
"""

import numpy as np

# Right eye contour (person's right = camera's left)
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# Left eye contour (person's left = camera's right)
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# EAR keypoints: [outer, top1, top2, inner, bottom1, bottom2]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR  = [263, 387, 385, 362, 380, 373]

# Iris centers (478-point refined mesh) — для персональной калибровки взгляда
RIGHT_EYE_IRIS_CENTER = 468
LEFT_EYE_IRIS_CENTER = 473
# Углы глаза для оси «взгляд вдоль раскрытия века»
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362

# Head pose (6 keypoints for solvePnP): nose-tip, chin, L-eye-outer, R-eye-outer, L-mouth, R-mouth
POSE_KEYPOINTS = [4, 152, 263, 33, 287, 57]

# Face oval — used for bounding box computation
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172,  58, 132,  93, 234, 127, 162,  21,  54, 103,  67, 109,
]

# 3D canonical face model in mm, order matches POSE_KEYPOINTS
FACE_3D_MODEL = np.array([
    [   0.0,    0.0,    0.0],   # nose tip     (4)
    [   0.0, -330.0,  -65.0],   # chin         (152)
    [-225.0,  170.0, -135.0],   # L eye outer  (263)
    [ 225.0,  170.0, -135.0],   # R eye outer  (33)
    [-150.0, -150.0, -125.0],   # L mouth      (287)
    [ 150.0, -150.0, -125.0],   # R mouth      (57)
], dtype=np.float64)
