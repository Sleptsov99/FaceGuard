"""
Alert rules on engine payload dicts: structured Alert, cooldowns, fatigue repetition.
"""

from __future__ import annotations 

import time 
from dataclasses import dataclass ,asdict 
from typing import Any ,Dict ,Optional 


EYE_CLOSURE_LONG ="eye_closure_long"
FATIGUE_HIGH ="fatigue_high"
FATIGUE_REPEAT ="fatigue_repeat"
PERCLOS_HIGH ="perclos_high"
FACE_LOST ="face_lost"
LONG_BLINK ="long_blink"


@dataclass 
class Alert :
    type :str 
    title :str 
    message :str 
    severity :str 
    timestamp :float 

    def as_dict (self )->Dict [str ,Any ]:
        d =asdict (self )
        return d 


def _alert (
type_id :str ,
title :str ,
message :str ,
severity :str ,
)->Alert :
    return Alert (
    type =type_id ,
    title =title ,
    message =message ,
    severity =severity ,
    timestamp =time .time (),
    )


class AlertPolicy :
    """At most one Alert per evaluate(); per-type cooldown; fatigue episode tracking."""

    def __init__ (
    self ,
    cooldown_s :float =45.0 ,
    fatigue_repeat_cooldown_s :float =120.0 ,
    ):
        self .cooldown_s =cooldown_s 
        self .fatigue_repeat_cooldown_s =fatigue_repeat_cooldown_s 
        self ._last_emit :Dict [str ,float ]={}
        self ._fatigue_warned =False 

    def reset (self )->None :
        self ._last_emit .clear ()
        self ._fatigue_warned =False 

    def _ready (self ,key :str ,custom_cooldown :Optional [float ]=None )->bool :
        now =time .monotonic ()
        cd =custom_cooldown if custom_cooldown is not None else self .cooldown_s 
        last =self ._last_emit .get (key ,0.0 )
        if now -last >=cd :
            self ._last_emit [key ]=now 
            return True 
        return False 

    def evaluate (self ,payload :Dict [str ,Any ])->Optional [Alert ]:
        """
        Priority: prolonged closure → fatigue (+ repeat) → PERCLOS → face lost → long blink.
        """
        closure_ms =float (payload .get ("closure_streak_ms")or 0.0 )
        if closure_ms >=1200.0 and self ._ready (EYE_CLOSURE_LONG ):
            return _alert (
            EYE_CLOSURE_LONG ,
            "Глаза закрыты",
            "Глаза закрыты слишком долго. Сделайте паузу.",
            "warning",
            )

        level =payload .get ("fatigue_level")
        fscore =payload .get ("fatigue_score")
        fatigue_now =(
        level in ("MODERATE","SEVERE")
        or (fscore is not None and float (fscore )>=0.72 )
        )
        if not fatigue_now :
            self ._fatigue_warned =False 

        if fatigue_now :
            if self ._fatigue_warned and self ._ready (
            FATIGUE_REPEAT ,self .fatigue_repeat_cooldown_s 
            ):
                return _alert (
                FATIGUE_REPEAT ,
                "Усталость",
                "Усталость повторяется. Рекомендуется сделать перерыв.",
                "warning",
                )
            if self ._ready (FATIGUE_HIGH ):
                self ._fatigue_warned =True 
                return _alert (
                FATIGUE_HIGH ,
                "Усталость",
                "Похоже, вы устали. Отдохните 5–10 минут.",
                "warning",
                )

        perclos =payload .get ("perclos_30s")
        if perclos is not None and float (perclos )>=0.22 and self ._ready (PERCLOS_HIGH ):
            return _alert (
            PERCLOS_HIGH ,
            "Сонливость",
            "Замечены признаки сонливости. Лучше ненадолго отвлечься от работы.",
            "warning",
            )

        absent_ms =float (payload .get ("face_absent_ms")or 0.0 )
        if (
        absent_ms >=3500.0 
        and self ._ready (FACE_LOST )
        and (
        payload .get ("detection_status")=="no_face"
        or not payload .get ("face_present",True )
        )
        ):
            return _alert (
            FACE_LOST ,
            "Нет лица",
            "Лицо не видно. Вернитесь в кадр.",
            "info",
            )

        if (
        payload .get ("blink_this_frame")
        and float (payload .get ("blink_duration_ms")or 0.0 )>=450.0 
        and self ._ready (LONG_BLINK )
        ):
            return _alert (
            LONG_BLINK ,
            "Моргание",
            "Похоже, вы устали. Сделайте небольшой перерыв.",
            "info",
            )

        return None 
