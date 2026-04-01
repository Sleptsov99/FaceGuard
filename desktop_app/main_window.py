"""
Optional dashboard: preview, metrics, alert history. Hidden by default when using tray.
"""

from __future__ import annotations 

from typing import TYPE_CHECKING ,Any ,Dict ,Optional 

from PySide6 .QtCore import Qt ,QSize 
from PySide6 .QtGui import QCloseEvent ,QImage ,QPixmap ,QResizeEvent 
from PySide6 .QtWidgets import (
QHBoxLayout ,
QLabel ,
QMainWindow ,
QMessageBox ,
QPlainTextEdit ,
QPushButton ,
QSizePolicy ,
QVBoxLayout ,
QWidget ,
)

if TYPE_CHECKING :
    from desktop_app .coordinator import DesktopCoordinator 


class MainWindow (QMainWindow ):
    def __init__ (
    self ,
    camera_source :int =0 ,
    coordinator :Optional ["DesktopCoordinator"]=None ,
    quiet_modal_errors :bool =False ,
    parent =None ,
    ):
        super ().__init__ (parent )
        self .setWindowTitle ("Emotion Checker — панель")
        self ._camera_source =camera_source 
        self ._coordinator =coordinator 
        self ._quiet_modal_errors =quiet_modal_errors 
        self ._last_preview :Optional [QImage ]=None 

        central =QWidget ()
        self .setCentralWidget (central )
        layout =QVBoxLayout (central )

        row =QHBoxLayout ()
        self ._btn_start =QPushButton ("Запустить мониторинг")
        self ._btn_stop =QPushButton ("Остановить")
        self ._btn_stop .setEnabled (False )
        self ._btn_start .clicked .connect (self ._on_start_clicked )
        self ._btn_stop .clicked .connect (self ._on_stop_clicked )
        row .addWidget (self ._btn_start )
        row .addWidget (self ._btn_stop )
        row .addStretch ()
        layout .addLayout (row )

        self ._preview_label =QLabel ("Камера не запущена")
        self ._preview_label .setAlignment (Qt .AlignCenter )
        self ._preview_label .setMinimumSize (QSize (480 ,270 ))
        self ._preview_label .setSizePolicy (
        QSizePolicy .Policy .Expanding ,
        QSizePolicy .Policy .Expanding ,
        )
        self ._preview_label .setStyleSheet (
        "QLabel { background-color: #1e1e1e; color: #aaa; border: 1px solid #444; }"
        )
        self ._preview_label .setScaledContents (False )
        layout .addWidget (self ._preview_label ,stretch =1 )

        self ._lbl_status =QLabel ("Engine: —")
        self ._lbl_blink =QLabel ("Blinks (total): 0")
        self ._lbl_detection =QLabel ("Detection: —")
        self ._lbl_calibration =QLabel ("Калибровка: —")
        self ._lbl_fatigue_distraction =QLabel ("Fatigue / distraction: —")
        self ._calibration_done =False 
        for w in (
        self ._lbl_status ,
        self ._lbl_blink ,
        self ._lbl_detection ,
        self ._lbl_calibration ,
        self ._lbl_fatigue_distraction ,
        ):
            w .setTextInteractionFlags (Qt .TextSelectableByMouse )
        layout .addWidget (self ._lbl_status )
        layout .addWidget (self ._lbl_blink )
        layout .addWidget (self ._lbl_detection )
        layout .addWidget (self ._lbl_calibration )
        layout .addWidget (self ._lbl_fatigue_distraction )

        self ._alert_banner =QLabel ("")
        self ._alert_banner .setWordWrap (True )
        self ._alert_banner .setVisible (False )
        self ._alert_banner .setStyleSheet (
        "QLabel { background-color: #fff3cd; color: #664d03; "
        "padding: 8px; border: 1px solid #ffc107; border-radius: 4px; }"
        )
        layout .addWidget (self ._alert_banner )

        layout .addWidget (QLabel ("Журнал / уведомления:"))
        self ._log =QPlainTextEdit ()
        self ._log .setReadOnly (True )
        self ._log .setMaximumBlockCount (300 )
        layout .addWidget (self ._log )

        self ._append_log (
        "Фоновый режим: мониторинг запускается из трея автоматически."
        if coordinator 
        else "Нажмите «Запустить мониторинг»."
        )

    def show_normal (self )->None :
        self .show ()
        self .raise_ ()
        self .activateWindow ()

    def resizeEvent (self ,event :QResizeEvent )->None :
        super ().resizeEvent (event )
        if self ._last_preview is not None and not self ._last_preview .isNull ():
            self ._update_preview_display ()

    def _update_preview_display (self )->None :
        if self ._last_preview is None or self ._last_preview .isNull ():
            return 
        pix =QPixmap .fromImage (self ._last_preview )
        target =self ._preview_label .contentsRect ().size ()
        if target .width ()<2 or target .height ()<2 :
            return 
        scaled =pix .scaled (
        target ,
        Qt .AspectRatioMode .KeepAspectRatio ,
        Qt .TransformationMode .SmoothTransformation ,
        )
        self ._preview_label .setPixmap (scaled )

    def _append_log (self ,text :str )->None :
        self ._log .appendPlainText (text )

    def _clear_preview (self )->None :
        self ._last_preview =None 
        self ._preview_label .clear ()
        self ._preview_label .setPixmap (QPixmap ())
        self ._preview_label .setText ("Камера не запущена")

    def _hide_alert_banner (self )->None :
        self ._alert_banner .setVisible (False )
        self ._alert_banner .setText ("")

    def _on_start_clicked (self )->None :
        if self ._coordinator :
            self ._coordinator .request_start_from_dashboard ()
            return 
        self ._append_log ("Нет координатора: запуск только через трей-режим.")

    def _on_stop_clicked (self )->None :
        if self ._coordinator :
            self ._coordinator .request_stop_from_dashboard ()
            return 

    def on_engine_started (self )->None :
        self ._calibration_done =False 
        self ._btn_start .setEnabled (False )
        self ._btn_stop .setEnabled (True )
        self ._lbl_status .setText ("Engine: running")
        self ._hide_alert_banner ()
        self ._preview_label .setText ("")
        self ._preview_label .setPixmap (QPixmap ())
        self ._append_log ("Мониторинг запущен.")

    def on_engine_stopped (self )->None :
        self ._calibration_done =False 
        self ._btn_start .setEnabled (True )
        self ._btn_stop .setEnabled (False )
        self ._lbl_status .setText ("Engine: idle")
        self ._lbl_calibration .setText ("Калибровка: —")
        self ._clear_preview ()
        self ._append_log ("Мониторинг остановлен.")

    def on_worker_thread_finished (self )->None :
        """Worker ended on its own (camera lost, error path, etc.)."""
        self ._calibration_done =False 
        self ._btn_start .setEnabled (True )
        self ._btn_stop .setEnabled (False )
        self ._lbl_status .setText ("Engine: остановлено")
        self ._lbl_calibration .setText ("Калибровка: —")
        self ._clear_preview ()
        self ._append_log ("Поток движка завершён.")

    def on_preview_frame (self ,img :QImage )->None :
        if img .isNull ():
            return 
        self ._last_preview =img .copy ()
        self ._preview_label .setText ("")
        self ._update_preview_display ()

    def on_frame_summary (self ,payload :Dict [str ,Any ])->None :
        self ._lbl_blink .setText (f"Blinks (total): {payload ['blink_total']}")
        self ._lbl_detection .setText (
        f"Detection: {payload ['detection_status']} "
        f"(conf {payload ['confidence']:.2f}, face_present={payload ['face_present']})"
        )
        perclos =payload ["perclos_30s"]
        perclos_s =f"{perclos :.3f}"if perclos is not None else "—"
        fatigue =payload ["fatigue_level"]or "—"
        fscore =payload ["fatigue_score"]
        fscore_s =f"{fscore :.2f}"if fscore is not None else "—"
        eyes =f"{payload .get ('state_left')or '—'} / {payload .get ('state_right')or '—'}"
        dscore =float (payload .get ("distraction_score")or 0.0 )
        self ._lbl_fatigue_distraction .setText (
        f"Fatigue: {fatigue } ({fscore_s })  PERCLOS30: {perclos_s }  |  "
        f"Distracted: {payload ['is_distracted']}  score: {dscore :.2f}  "
        f"reason: {payload ['distraction_reason']}\n"
        f"Eyes: {eyes }  closure_streak: {payload .get ('closure_streak_ms',0 ):.0f} ms"
        )
        cs =payload .get ("calibration_state")
        cp =float (payload .get ("calibration_progress")or 0.0 )
        if cs =="running":
            self ._lbl_calibration .setText (
            f"Калибровка (~30 с): {int (cp *100 )}% — смотрите в экран, не отворачивайтесь"
            )
        elif cs =="done":
            self ._calibration_done =True 
            self ._lbl_calibration .setText (
            "Калибровка: завершена. Пороги усталости и взгляда персональные."
            )
        elif cs =="failed":
            self ._lbl_calibration .setText (
            "Калибровка: мало валидных кадров. Стандартные пороги."
            )
        elif self ._calibration_done :
            self ._lbl_calibration .setText ("Калибровка: применена")
        else :
            self ._lbl_calibration .setText ("Калибровка: —")
        if payload .get ("blink_this_frame"):
            self ._append_log (
            f"Frame {payload ['frame_index']}: blink (total {payload ['blink_total']})"
            )

    def on_alert_dict (self ,alert :Dict [str ,Any ])->None :
        msg =alert .get ("message","")
        title =alert .get ("title","")
        self ._alert_banner .setText (f"{title }: {msg }"if title else msg )
        self ._alert_banner .setVisible (True )
        ts =alert .get ("timestamp",0 )
        self ._append_log (
        f"[{alert .get ('type','?')}] {title } — {msg } (severity={alert .get ('severity')}, t={ts :.0f})"
        )

    def on_engine_failed (self ,message :str )->None :
        self ._append_log (f"Error: {message }")
        if not self ._quiet_modal_errors :
            QMessageBox .warning (self ,"Engine error",message )
        self ._lbl_status .setText ("Engine: error")

    def closeEvent (self ,event :QCloseEvent )->None :
        if self ._coordinator is not None :
            event .ignore ()
            self ._coordinator .on_dashboard_close_request ()
            return 
        event .accept ()
