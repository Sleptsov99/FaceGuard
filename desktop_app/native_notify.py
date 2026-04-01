"""
Системные уведомления Windows (Toast в углу экрана / Центр уведомлений).

Порядок:
1) ``windows-toasts`` (WinRT) — обычно лучше виден, чем только трей.
2) ``winotify`` (PowerShell) — запасной вариант.

Плюс координатор всегда вызывает ``QSystemTrayIcon.showMessage`` — чтобы не остаться
только с текстом внутри окна, если Toast не показался.
"""

from __future__ import annotations 

import sys 


def notify_desktop (
title :str ,
body :str ,
*,
severity :str ="warning",
)->bool :
    """
    Показать системный Toast, если получилось.

    Returns True, если хотя бы один канал (WinRT или winotify) отработал без исключения.
    """
    if sys .platform !="win32":
        return False 
    ok =False 
    if _notify_winrt (title ,body ,severity =severity ):
        ok =True 
    if _notify_winotify (title ,body ,severity =severity ):
        ok =True 
    return ok 


def _notify_winrt (title :str ,body :str ,*,severity :str )->bool :
    try :
        from windows_toasts import Toast ,WindowsToaster 
        from windows_toasts .wrappers import ToastDuration ,ToastScenario 
    except ImportError :
        return False 

    try :

        scenario =(
        ToastScenario .Important if severity =="critical"else ToastScenario .Default 
        )
        toast =Toast (
        text_fields =[title [:128 ],(body or "")[:512 ]],
        duration =ToastDuration .Long ,
        scenario =scenario ,
        )
        WindowsToaster ("Emotion Checker").show_toast (toast )
        return True 
    except Exception :
        return False 


def _notify_winotify (title :str ,body :str ,*,severity :str )->bool :
    try :
        from winotify import Notification ,audio 
    except ImportError :
        return False 

    try :
        toast =Notification (
        app_id ="Emotion Checker",
        title =title [:128 ],
        msg =(body or "")[:512 ],
        )
        toast .duration ="long"
        if severity =="critical":
            toast .set_audio (audio .Default ,loop =False )
        else :
            toast .set_audio (audio .Silent ,loop =False )
        toast .show ()
        return True 
    except Exception :
        return False 
