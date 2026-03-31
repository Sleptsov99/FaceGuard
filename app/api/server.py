"""
WebSocket server — Stage 8.

Runs in a background thread (own asyncio event loop).
The CV main thread pushes serialized JSON strings via push().

JSON contract
─────────────
Server → client (events):
  {"type":"frame",   "timestamp_ms":…, "eye_state":…, …}   — every frame
  {"type":"session_started"}
  {"type":"session_stopped"}
  {"type":"calibration_started"}
  {"type":"calibration_done"}
  {"type":"calibration_failed"}

Client → server (commands):
  {"cmd":"start_calibration"}
  {"cmd":"stop_session"}
  {"cmd":"get_current_metrics"}   → server replies with last frame event

Connect: ws://localhost:<port>  (default 8765)
"""

import asyncio
import json
import logging
import queue
import threading
from typing import Callable, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


class WebSocketServer:
    """
    Thread-safe WebSocket broadcast server.

    Usage
    -----
    server = WebSocketServer(port=8765, on_command=handle_cmd)
    server.start()                    # spawns background thread
    server.push(json_string)          # called from CV thread each frame
    server.push_event("session_started")
    server.stop()
    """

    def __init__(
        self,
        port: int = 8765,
        on_command: Optional[Callable[[str], None]] = None,
    ):
        self._port        = port
        self._on_command  = on_command   # called from asyncio thread with raw cmd string
        self._loop:          Optional[asyncio.AbstractEventLoop] = None
        self._thread:        Optional[threading.Thread]          = None
        self._clients:       Set[WebSocketServerProtocol]        = set()
        self._last_frame:    Optional[str]                       = None
        self._stop_event     = threading.Event()
        self._asyncio_stop:  Optional[asyncio.Event]             = None

    # ─── public API (CV thread) ───────────────────────────────────────────────

    def start(self):
        """Start the WebSocket server in a background daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="ws-server")
        self._thread.start()

    def stop(self):
        """Signal the server to shut down."""
        self._stop_event.set()
        if self._loop and self._asyncio_stop:
            self._loop.call_soon_threadsafe(self._asyncio_stop.set)

    def push(self, json_str: str):
        """Broadcast a pre-serialized JSON string to all connected clients."""
        self._last_frame = json_str
        if self._loop and self._clients:
            asyncio.run_coroutine_threadsafe(
                self._broadcast(json_str), self._loop
            )

    def push_event(self, event_type: str, **kwargs):
        """Broadcast a simple event by type."""
        payload = json.dumps({"type": event_type, **kwargs}, separators=(",", ":"))
        self.push(payload)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ─── asyncio internals ────────────────────────────────────────────────────

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    async def _serve(self):
        self._asyncio_stop = asyncio.Event()
        async with websockets.serve(self._handler, "localhost", self._port):
            logger.info("WebSocket server listening on ws://localhost:%d", self._port)
            print(f"[ws] listening on ws://localhost:{self._port}")
            await self._asyncio_stop.wait()

    async def _handler(self, ws: WebSocketServerProtocol):
        self._clients.add(ws)
        logger.debug("Client connected (%d total)", len(self._clients))
        try:
            async for message in ws:
                await self._handle_command(ws, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            logger.debug("Client disconnected (%d total)", len(self._clients))

    async def _handle_command(self, ws: WebSocketServerProtocol, message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            await ws.send(json.dumps({"type": "error", "msg": "invalid JSON"}))
            return

        cmd = data.get("cmd", "")

        if cmd == "get_current_metrics":
            payload = self._last_frame or json.dumps({"type": "no_data"})
            await ws.send(payload)
        elif cmd in ("start_calibration", "stop_session"):
            if self._on_command:
                self._on_command(cmd)
            await ws.send(json.dumps({"type": "ack", "cmd": cmd}))
        else:
            await ws.send(json.dumps({"type": "error", "msg": f"unknown cmd: {cmd}"}))

    async def _broadcast(self, json_str: str):
        if not self._clients:
            return
        dead = set()
        for ws in list(self._clients):
            try:
                await ws.send(json_str)
            except websockets.exceptions.ConnectionClosed:
                dead.add(ws)
        self._clients -= dead
