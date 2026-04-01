"""
Landmark index constants for the 478-point refined face mesh topology
(MediaPipe Face Mesh / Face Landmarker Tasks — same indexing).
All indices are from the person's perspective (left/right = person's left/right).
"""

import numpy as np 


RIGHT_EYE_CONTOUR =[33 ,7 ,163 ,144 ,145 ,153 ,154 ,155 ,133 ,173 ,157 ,158 ,159 ,160 ,161 ,246 ]

LEFT_EYE_CONTOUR =[362 ,382 ,381 ,380 ,374 ,373 ,390 ,249 ,263 ,466 ,388 ,387 ,386 ,385 ,384 ,398 ]


RIGHT_EYE_EAR =[33 ,160 ,158 ,133 ,153 ,144 ]
LEFT_EYE_EAR =[263 ,387 ,385 ,362 ,380 ,373 ]


RIGHT_EYE_IRIS_CENTER =468 
LEFT_EYE_IRIS_CENTER =473 

RIGHT_EYE_OUTER =33 
RIGHT_EYE_INNER =133 
LEFT_EYE_OUTER =263 
LEFT_EYE_INNER =362 


POSE_KEYPOINTS =[4 ,152 ,263 ,33 ,287 ,57 ]


FACE_OVAL =[
10 ,338 ,297 ,332 ,284 ,251 ,389 ,356 ,454 ,323 ,361 ,288 ,
397 ,365 ,379 ,378 ,400 ,377 ,152 ,148 ,176 ,149 ,150 ,136 ,
172 ,58 ,132 ,93 ,234 ,127 ,162 ,21 ,54 ,103 ,67 ,109 ,
]


FACE_3D_MODEL =np .array ([
[0.0 ,0.0 ,0.0 ],
[0.0 ,-330.0 ,-65.0 ],
[-225.0 ,170.0 ,-135.0 ],
[225.0 ,170.0 ,-135.0 ],
[-150.0 ,-150.0 ,-125.0 ],
[150.0 ,-150.0 ,-125.0 ],
],dtype =np .float64 )
