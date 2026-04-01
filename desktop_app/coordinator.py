"""
Tray-first lifecycle: auto monitoring, dashboard optional, native notifications.
"""

from __future__ import annotations 

import time 
from typing import Optional 

from PySide6 .QtCore import QObject ,QTimer ,Qt 
from PySide6 .QtGui import QAction ,QIcon ,QPixmap 
from PySide6 .QtWidgets import QApplication ,QMenu ,QMessageBox ,QSystemTrayIcon 

from desktop_app .alert_overlay import AlertOverlay 
from desktop_app .main_window import MainWindow 
from desktop_app .native_notify import notify_desktop 
from desktop_app .worker import EngineThread 


class DesktopCoordinator (QObject ):
    """QObject subclass so QTimer has correct thread affinity and parent chain."""

    def __init__ (self ,app :QApplication ,camera_source :int =0 ):
        super ().__init__ (parent =app )
        self ._app =app 
        self ._camera_source =camera_source 
        self ._thread :Optional [EngineThread ]=None 
        self ._paused =False 
        self ._warning_until_mono =0.0 

        app .setQuitOnLastWindowClosed (False )
        app .aboutToQuit .connect (self ._shutdown_cleanup )

        self ._dashboard =MainWindow (
        camera_source =camera_source ,
        coordinator =self ,
        quiet_modal_errors =True ,
        )
        self ._dashboard .resize (580 ,680 )

        self ._tray =QSystemTrayIcon (self )
        self ._tray .setIcon (self ._default_icon ())

        self ._menu =QMenu ()
        act_dash =QAction ("Открыть панель",self )
        act_dash .triggered .connect (self .open_dashboard )
        self ._act_pause =QAction ("Приостановить мониторинг",self )
        self ._act_pause .triggered .connect (self ._toggle_pause )
        act_quit =QAction ("Выход",self )
        act_quit .triggered .connect (self .quit_app )
        self ._menu .addAction (act_dash )
        self ._menu .addAction (self ._act_pause )
        self ._menu .addSeparator ()
        self ._menu .addAction (act_quit )
        self ._tray .setContextMenu (self ._menu )
        self ._tray .activated .connect (self ._on_tray_activated )

        if not QSystemTrayIcon .isSystemTrayAvailable ():
            QMessageBox .warning (
            None ,
            "Emotion Checker",
            "Системный трей недоступен. Используйте панель управления.",
            )
            self ._dashboard .show ()

        self ._tray .show ()


        self ._alert_overlay =AlertOverlay ()
        self ._alert_overlay .show ()

        self ._tooltip_timer =QTimer (self )
        self ._tooltip_timer .timeout .connect (self .refresh_tray_tooltip )
        self ._tooltip_timer .start (3000 )

        self .start_monitoring ()

    def _shutdown_cleanup (self )->None :
        """Stop timers on the GUI thread before Qt tears down QObjects."""
        self ._tooltip_timer .stop ()

    def _default_icon (self )->QIcon :
        pm =QPixmap (32 ,32 )
        pm .fill (Qt .GlobalColor .darkGreen )
        return QIcon (pm )

    def _on_tray_activated (self ,reason :QSystemTrayIcon .ActivationReason )->None :
        if reason ==QSystemTrayIcon .ActivationReason .Trigger :
            self .open_dashboard ()

    def open_dashboard (self )->None :
        self ._dashboard .show_normal ()

    def _toggle_pause (self )->None :
        self .set_paused (not self ._paused )

    def set_paused (self ,paused :bool )->None :
        self ._paused =paused 
        if self ._thread is not None :
            self ._thread .set_paused (paused )
        self ._act_pause .setText (
        "Возобновить мониторинг"if paused else "Приостановить мониторинг"
        )
        self .refresh_tray_tooltip ()

    def quit_app (self )->None :
        self ._shutdown_cleanup ()
        self .stop_monitoring ()
        self ._tray .hide ()
        QApplication .quit ()

    def start_monitoring (self )->None :
        if self ._thread is not None and self ._thread .isRunning ():
            return 
        self ._thread =EngineThread (self ._camera_source ,parent =None )
        self ._thread .frame_summary .connect (self ._on_frame_summary )
        self ._thread .preview_frame .connect (self ._dashboard .on_preview_frame )
        self ._thread .alert_raised .connect (self ._on_alert )
        self ._thread .failed .connect (self ._on_failed )
        self ._thread .finished_clean .connect (self ._on_worker_finished )
        self ._thread .start ()
        self ._dashboard .on_engine_started ()
        self .refresh_tray_tooltip ()

    def stop_monitoring (self )->None :
        t =self ._thread 
        if t is not None and t .isRunning ():
            t .request_stop ()
            t .wait (5000 )
        self ._thread =None 
        self ._dashboard .on_engine_stopped ()

    def request_start_from_dashboard (self )->None :
        self .start_monitoring ()

    def request_stop_from_dashboard (self )->None :
        self .stop_monitoring ()

    def _on_frame_summary (self ,payload :dict )->None :
        lvl =payload .get ("fatigue_level")
        if lvl in ("MODERATE","SEVERE"):
            self ._warning_until_mono =time .monotonic ()+30.0 
        self ._dashboard .on_frame_summary (payload )

    def _on_alert (self ,alert :dict )->None :
        sev =alert .get ("severity","warning")
        icon =QSystemTrayIcon .MessageIcon .Warning 
        if sev =="critical":
            icon =QSystemTrayIcon .MessageIcon .Critical 
        elif sev =="info":
            icon =QSystemTrayIcon .MessageIcon .Information 
        self ._warning_until_mono =time .monotonic ()+45.0 
        title =alert .get ("title","Emotion Checker")
        body =alert .get ("message","")

        notify_desktop (title ,body ,severity =sev )
        self ._tray .showMessage (title ,body ,icon ,10_000 )
        self ._alert_overlay .show_alert (title ,body ,severity =sev )
        self ._dashboard .on_alert_dict (alert )

    def _on_failed (self ,err :str )->None :
        t ,b =(
        "Камера",
        "Не удалось получить доступ к камере. Проверьте разрешения и подключение.",
        )
        notify_desktop (t ,b ,severity ="critical")
        self ._tray .showMessage (
        t ,
        b ,
        QSystemTrayIcon .MessageIcon .Critical ,
        14_000 ,
        )
        self ._alert_overlay .show_alert (t ,b ,severity ="critical")
        self ._dashboard .on_engine_failed (err )

    def _on_worker_finished (self )->None :
        self ._thread =None 
        self ._dashboard .on_worker_thread_finished ()
        self .refresh_tray_tooltip ()

    def refresh_tray_tooltip (self )->None :
        if self ._thread is None or not self ._thread .isRunning ():
            self ._tray .setToolTip ("Emotion Checker\nМониторинг остановлен")
            return 
        if self ._paused :
            self ._tray .setToolTip ("Emotion Checker\nПауза")
            return 
        if time .monotonic ()<self ._warning_until_mono :
            self ._tray .setToolTip ("Emotion Checker\nПредупреждение (усталость)")
            return 
        self ._tray .setToolTip ("Emotion Checker\nМониторинг активен")

    def on_dashboard_close_request (self )->None :
        """Hide dashboard only; tray keeps app alive."""
        self ._dashboard .hide ()
