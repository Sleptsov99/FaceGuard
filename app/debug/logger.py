"""
MetricsLogger — Stage 9.

Writes two files:
  <stem>.csv   — one row per frame (all numeric metrics + quality flags)
  <stem>_blinks.jsonl — one JSON object per completed blink event

Usage
-----
    logger = MetricsLogger("session_001")
    logger.log(timestamp_ms, detection, metrics, distr, cv_state, quality_flags)
    logger.close()   # or use as context manager
"""

import csv 
import json 
import time 
from pathlib import Path 
from typing import FrozenSet ,Optional 

from app .landmarks .result import DetectionResult 
from app .metrics .result import EyeMetrics 
from app .state .distraction import DistractionResult 
from app .state .cv_state import CVState 
from app .debug .quality import QualityFlag ,flags_to_str 


_CSV_FIELDS =[
"timestamp_ms",
"ear_avg","ear_baseline","ear_deviation",
"perclos_30s","perclos_60s",
"blink_detected","blink_duration_ms","blink_rate_60s",
"long_blink_count",
"fatigue_score","fatigue_level",
"distraction_score","attention",
"eye_state",
"confidence","yaw","pitch","roll",
"quality_flags",
]


class MetricsLogger :
    """Thread-unsafe logger; call only from the CV main thread."""

    def __init__ (self ,stem :str ="cv_session",flush_every :int =30 ):
        """
        Parameters
        ----------
        stem        : output file name without extension
        flush_every : flush CSV to disk every N rows (0 = only on close)
        """
        self ._flush_every =flush_every 
        self ._row_count =0 

        csv_path =Path (stem +".csv")
        jsonl_path =Path (stem +"_blinks.jsonl")

        self ._csv_file =csv_path .open ("w",newline ="",encoding ="utf-8")
        self ._jsonl_file =jsonl_path .open ("w",encoding ="utf-8")

        self ._writer =csv .DictWriter (self ._csv_file ,fieldnames =_CSV_FIELDS )
        self ._writer .writeheader ()

        print (f"[logger] CSV    → {csv_path .resolve ()}")
        print (f"[logger] Blinks → {jsonl_path .resolve ()}")



    def log (
    self ,
    timestamp_ms :float ,
    detection :DetectionResult ,
    metrics :Optional [EyeMetrics ],
    distraction :DistractionResult ,
    cv_state :CVState ,
    quality_flags :FrozenSet [QualityFlag ],
    ):
        m =metrics 

        row ={
        "timestamp_ms":round (timestamp_ms ,1 ),
        "ear_avg":round (m .ear_avg_smooth ,4 )if m else "",
        "ear_baseline":round (m .ear_baseline ,4 )if m else "",
        "ear_deviation":round (m .ear_deviation ,4 )if m else "",
        "perclos_30s":round (m .perclos_30s ,4 )if m else "",
        "perclos_60s":round (m .perclos_60s ,4 )if m else "",
        "blink_detected":int (m .blink_detected )if m else 0 ,
        "blink_duration_ms":round (m .blink_duration_ms ,1 )if m else "",
        "blink_rate_60s":int (m .blink_rate_60s )if m else "",
        "long_blink_count":m .long_blink_count if m else "",
        "fatigue_score":round (m .fatigue_score ,3 )if m else "",
        "fatigue_level":m .fatigue_level .value if m else "",
        "distraction_score":round (distraction .distraction_score ,3 ),
        "attention":cv_state .attention .value ,
        "eye_state":cv_state .eye_state .value ,
        "confidence":round (detection .confidence ,3 ),
        "yaw":round (distraction .yaw ,2 ),
        "pitch":round (distraction .pitch ,2 ),
        "roll":round (distraction .roll ,2 ),
        "quality_flags":flags_to_str (quality_flags ),
        }
        self ._writer .writerow (row )
        self ._row_count +=1 


        if m and m .blink_detected and m .blink_duration_ms >0 :
            blink_record ={
            "timestamp_ms":round (timestamp_ms ,1 ),
            "duration_ms":round (m .blink_duration_ms ,1 ),
            "is_long":m .blink_duration_ms >=300.0 ,
            "ear_at_blink":round (m .ear_avg_smooth ,4 ),
            }
            self ._jsonl_file .write (json .dumps (blink_record )+"\n")

        if self ._flush_every and self ._row_count %self ._flush_every ==0 :
            self ._csv_file .flush ()
            self ._jsonl_file .flush ()

    def close (self ):
        self ._csv_file .flush ()
        self ._jsonl_file .flush ()
        self ._csv_file .close ()
        self ._jsonl_file .close ()
        print (f"[logger] Saved {self ._row_count } rows.")



    def __enter__ (self ):
        return self 

    def __exit__ (self ,*_ ):
        self .close ()
