"""UDP camera ingest for simulator JPEG frames."""

from __future__ import annotations

from collections.abc import Callable
import logging
import queue
import socket
import struct
import threading

import numpy as np

from perception.types import VisionFrame

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only when OpenCV is unavailable.
    cv2 = None

_LOGGER = logging.getLogger(__name__)
_FRAME_HEADER = struct.Struct("<IHHIIQ")


class UDPVisionReceiver:
    """Background UDP receiver that emits decoded RGB camera frames."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 5600,
        queue_size: int = 8,
        frame_callback: Callable[[VisionFrame], None] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._frame_callback = frame_callback
        self._frame_queue: queue.Queue[VisionFrame] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    def start(self) -> None:
        if cv2 is None:
            msg = "opencv-python is required for UDPVisionReceiver but is not installed"
            raise RuntimeError(msg)
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._receive_loop, name="udp-vision")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._socket is not None:
            self._socket.close()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def get_frame(self, timeout: float | None = None) -> VisionFrame:
        return self._frame_queue.get(timeout=timeout)

    def _receive_loop(self) -> None:
        partial_frames: dict[int, dict[str, object]] = {}
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.25)
        sock.bind((self._host, self._port))
        self._socket = sock

        try:
            while not self._stop_event.is_set():
                try:
                    packet, _address = sock.recvfrom(65536)
                except socket.timeout:
                    continue
                except OSError:
                    return

                header = packet[: _FRAME_HEADER.size]
                payload = packet[_FRAME_HEADER.size :]
                (
                    frame_id,
                    chunk_id,
                    total_chunks,
                    jpeg_size,
                    payload_size,
                    sim_time_ns,
                ) = _FRAME_HEADER.unpack(header)

                frame_state = partial_frames.setdefault(
                    frame_id,
                    {
                        "chunks": {},
                        "expected_chunks": total_chunks,
                        "jpeg_size": jpeg_size,
                        "timestamp_ns": sim_time_ns,
                    },
                )
                chunks = frame_state["chunks"]
                assert isinstance(chunks, dict)
                chunks[chunk_id] = payload[:payload_size]

                if len(chunks) != total_chunks:
                    continue

                frame = self._decode_frame(
                    frame_id=frame_id,
                    timestamp_ns=sim_time_ns,
                    total_chunks=total_chunks,
                    expected_size=jpeg_size,
                    chunks=chunks,
                )
                del partial_frames[frame_id]

                if frame is None:
                    continue
                self._emit_frame(frame)
        finally:
            sock.close()
            self._socket = None

    def _decode_frame(
        self,
        *,
        frame_id: int,
        timestamp_ns: int,
        total_chunks: int,
        expected_size: int,
        chunks: dict[int, bytes],
    ) -> VisionFrame | None:
        missing_chunk_ids = [index for index in range(total_chunks) if index not in chunks]
        if missing_chunk_ids:
            _LOGGER.warning("dropping frame %s with missing UDP chunks %s", frame_id, missing_chunk_ids)
            return None

        jpeg_bytes = b"".join(chunks[index] for index in range(total_chunks))
        if len(jpeg_bytes) != expected_size:
            _LOGGER.debug(
                "frame %s assembled JPEG size mismatch: expected=%s actual=%s",
                frame_id,
                expected_size,
                len(jpeg_bytes),
            )

        rgb = self._decode_rgb(jpeg_bytes)
        if rgb is None:
            _LOGGER.warning("failed to decode simulator frame %s", frame_id)
            return None

        return VisionFrame(frame_id=frame_id, timestamp_ns=timestamp_ns, rgb=rgb)

    def _emit_frame(self, frame: VisionFrame) -> None:
        if self._frame_callback is not None:
            self._frame_callback(frame)

        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        self._frame_queue.put_nowait(frame)

    @staticmethod
    def _decode_rgb(jpeg_bytes: bytes) -> np.ndarray | None:
        if cv2 is None:
            return None
        image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
