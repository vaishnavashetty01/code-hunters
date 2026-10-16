import importlib.util
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
from contextlib import closing
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from threading import Condition, Event, Lock, Thread
from urllib.parse import unquote, urlsplit
import webbrowser


def _module_is_available(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _use_project_environment():
    required_modules = ("cv2", "numpy", "ultralytics", "torch", "google.genai", "dotenv")
    missing_modules = [name for name in required_modules if not _module_is_available(name)]
    if not missing_modules:
        return

    project_dir = Path(__file__).resolve().parent
    project_python = project_dir / ".venv312" / "Scripts" / "python.exe"
    if (
        __name__ == "__main__"
        and project_python.is_file()
        and Path(sys.executable).resolve() != project_python.resolve()
    ):
        print(
            f"Missing Python packages: {', '.join(missing_modules)}. "
            "Restarting with the project's Python 3.12 environment...",
            flush=True,
        )
        result = subprocess.run(
            [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=project_dir,
            check=False,
        )
        raise SystemExit(result.returncode)

    raise SystemExit(
        f"Missing Python packages: {', '.join(missing_modules)}.\n"
        "Use Python 3.12 and install the project dependencies with:\n"
        "  python -m pip install -r requirements.txt\n"
        "If .venv312 already exists, run main.py from the project folder so it can select that environment."
    )


_use_project_environment()

import cv2
import time
import json
import numpy as np
from twin_backend.twin_state import CampusDigitalTwin
from vision_edge.campus_vision import CampusVisionEngine
from genai_brain.report_engine import CampusGenAIBrain

CAMERA_HOST = "127.0.0.1"
CAMERA_PORT = 8000
CAMERA_API_VERSION = 2
CAMERA_ALLOWED_ROLES = {"admin", "lecturer", "council"}
RECORDING_ALLOWED_ROLES = {"lecturer"}
RECORDING_VIEW_ROLES = {"admin", "lecturer", "student"}
PROJECT_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = PROJECT_DIR / "recordings"
RECORDINGS_DB = PROJECT_DIR / "recordings.sqlite3"
RECORDING_FPS = 15.0
_camera_frames = Condition()
_latest_camera_jpeg = None
_camera_frame_number = 0
_latest_camera_frame = None
_inference_frame_number = 0
_camera_state = "stopped"
_camera_error = ""
_camera_frame_ready = Event()
_camera_frame_size = None
_camera_observation = {
    "possible_litter_indicator": False,
    "message": "Start the camera to run an anonymous scene check.",
}
_camera_stop = Event()
_camera_lock = Lock()
_camera_thread = None
_inference_stop = Event()
_recording_lock = Lock()
_recording_writer = None
_recording_filename = ""
_recording_started_at = None
_recording_last_frame_at = None
_recording_state = "idle"
_recording_error = ""
_recording_frame_count = 0


def publish_camera_frame(frame):
    global _latest_camera_jpeg, _camera_frame_number
    success, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not success:
        raise RuntimeError("Could not encode a webcam frame for the browser stream.")
    with _camera_frames:
        _latest_camera_jpeg = jpeg.tobytes()
        _camera_frame_number += 1
        _camera_frames.notify_all()


def publish_inference_frame(frame):
    global _latest_camera_frame, _inference_frame_number
    with _camera_frames:
        _latest_camera_frame = frame.copy()
        _inference_frame_number += 1
        _camera_frames.notify_all()


class CampusRequestHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_DIR), **kwargs)

    def end_headers(self):
        route = urlsplit(self.path).path
        if route.endswith("/") or Path(route).suffix.lower() in {".html", ".js", ".css"}:
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _recording_status(self):
        with _recording_lock:
            elapsed = (
                max(0, int(time.time() - _recording_started_at))
                if _recording_state == "recording" and _recording_started_at is not None
                else 0
            )
            return {
                "state": _recording_state,
                "filename": _recording_filename,
                "error": _recording_error,
                "duration_seconds": elapsed,
            }

    def do_GET(self):
        route = urlsplit(self.path).path
        if route.lower().startswith("/recordings.sqlite3"):
            self.send_error(404, "Recording database is not a web resource.")
            return
        if route == "/camera/status":
            self._send_json(200, {
                "api_version": CAMERA_API_VERSION,
                "state": _camera_state,
                "error": _camera_error,
                "observation": _camera_observation,
                "recording": self._recording_status(),
            })
            return

        if route == "/recordings":
            if self.headers.get("X-CanopyQ-Role", "") not in RECORDING_VIEW_ROLES:
                self._send_json(403, {"error": "Sign in to view class recordings."})
                return
            self._send_json(200, {"recordings": list_recordings()})
            return

        if route.startswith("/recordings/"):
            self._serve_recording(unquote(route[len("/recordings/"):]))
            return

        if route != "/camera/stream":
            super().do_GET()
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

        last_frame_number = 0
        try:
            while True:
                with _camera_frames:
                    _camera_frames.wait_for(
                        lambda: _camera_frame_number > last_frame_number,
                        timeout=10,
                    )
                    if _camera_frame_number <= last_frame_number:
                        continue
                    frame_number = _camera_frame_number
                    jpeg = _latest_camera_jpeg

                self.wfile.write(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )
                self.wfile.flush()
                last_frame_number = frame_number
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

    def _serve_recording(self, filename):
        if (
            Path(filename).name != filename
            or not re.fullmatch(r"class-session-\d{8}-\d{6}-\d{6}\.mp4", filename)
        ):
            self.send_error(404, "Recording not found.")
            return
        path = RECORDINGS_DIR / filename
        if not path.is_file():
            self.send_error(404, "Recording not found.")
            return

        file_size = path.stat().st_size
        range_header = self.headers.get("Range")
        start, end, partial = 0, file_size - 1, False
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match or file_size == 0:
                self.send_error(416, "Requested range is not satisfiable.")
                return
            range_start, range_end = match.groups()
            if not range_start:
                suffix_length = int(range_end or "0")
                if suffix_length < 1:
                    self.send_error(416, "Requested range is not satisfiable.")
                    return
                start = max(file_size - suffix_length, 0)
            else:
                start = int(range_start)
                end = min(int(range_end), file_size - 1) if range_end else file_size - 1
            if start >= file_size or start > end:
                self.send_error(416, "Requested range is not satisfiable.")
                return
            partial = True

        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "private, no-store")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        try:
            with path.open("rb") as recording:
                recording.seek(start)
                remaining = length
                while remaining:
                    chunk = recording.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

    def do_POST(self):
        route = urlsplit(self.path).path
        role = self.headers.get("X-CanopyQ-Role", "")
        if route in {"/camera/start", "/camera/stop"}:
            if role not in CAMERA_ALLOWED_ROLES:
                self._send_json(403, {"error": "This role does not have camera access."})
                return
        elif route in {"/recording/start", "/recording/stop"}:
            if role not in RECORDING_ALLOWED_ROLES:
                self._send_json(403, {"error": "Only lecturers can control class recording."})
                return
        else:
            self._send_json(404, {"error": "Endpoint not found."})
            return

        if route == "/camera/start":
            started = start_camera()
            self._send_json(202 if started else 200, {"state": _camera_state})
        elif route == "/camera/stop":
            stop_camera()
            self._send_json(202, {"state": _camera_state})
        elif route == "/recording/start":
            try:
                start_recording()
            except RuntimeError as error:
                self._send_json(409, {"error": str(error), "recording": self._recording_status()})
                return
            self._send_json(202, {"recording": self._recording_status()})
        elif route == "/recording/stop":
            try:
                stop_recording()
            except RuntimeError as error:
                self._send_json(500, {"error": str(error), "recording": self._recording_status()})
                return
            self._send_json(200, {"recording": self._recording_status()})

    def log_message(self, format, *args):
        return


if __name__ == "__main__" and "--check-dependencies" in sys.argv[1:]:
    import torch

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"OpenCV: {cv2.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print("Project dependencies are ready.")
    raise SystemExit(0)

# Initialize global functional architectures
twin = CampusDigitalTwin()
vision = CampusVisionEngine()
brain = CampusGenAIBrain()

def export_twin_json_packet(active_zone=None, active_report=None):
    """
    Saves the absolute current tracking parameters to sync with index.html instantly.
    """
    packet = {
        "timestamp": time.time(),
        "sectors": {}
    }
    
    for zone in ["Academic_Block_A", "Hostel_Block_B", "Central_Cafeteria"]:
        packet["sectors"][zone] = twin.get_zone_report(zone)
        if zone == active_zone:
            packet["sectors"][zone]["latest_ai_report"] = active_report
        else:
            packet["sectors"][zone]["latest_ai_report"] = None
            
    data_path = PROJECT_DIR / "twin_data.json"
    temporary_path = data_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(packet, f, indent=4)
    temporary_path.replace(data_path)


def open_camera():
    backends = []
    if sys.platform == "win32":
        backends.extend((cv2.CAP_DSHOW, cv2.CAP_MSMF))
    else:
        backends.append(cv2.CAP_ANY)

    for camera_index in range(3):
        for backend in backends:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"Camera connected (device {camera_index}, backend {backend}).", flush=True)
                return cap
            cap.release()
    return None


def start_camera():
    global _camera_error, _camera_frame_size, _camera_observation, _camera_state, _camera_thread
    with _camera_lock:
        if _camera_thread is not None and _camera_thread.is_alive():
            return False
        _camera_stop.clear()
        _camera_frame_ready.clear()
        _camera_frame_size = None
        _camera_error = ""
        _camera_observation = {
            "possible_litter_indicator": False,
            "message": "Camera starting; no scene analysis is available yet.",
        }
        _camera_state = "starting"
        _camera_thread = Thread(
            target=capture_camera,
            name="campus-camera-capture",
            daemon=True,
        )
        _camera_thread.start()
        return True


def stop_camera():
    global _camera_observation, _camera_state
    try:
        stop_recording()
    except RuntimeError as error:
        print(f"[Recording] {error}", flush=True)
    with _camera_lock:
        if _camera_thread is not None and _camera_thread.is_alive():
            _camera_state = "stopping"
            _camera_stop.set()
            _camera_frame_ready.clear()
            with _camera_frames:
                _camera_frames.notify_all()
        else:
            _camera_state = "stopped"
        _camera_observation = {
            "possible_litter_indicator": False,
            "message": "Camera is off; no scene is being analyzed.",
        }


def capture_camera():
    global _camera_error, _camera_frame_size, _camera_observation, _camera_state
    cap = None
    try:
        cap = open_camera()
        if cap is None:
            _camera_error = (
                "No webcam found or camera permission was denied. "
                "Check Windows camera privacy settings and try again."
            )
            _camera_state = "unavailable"
            _camera_observation = {
                "possible_litter_indicator": False,
                "message": "Camera unavailable; no scene is being analyzed.",
            }
            return

        print("[Camera] Webcam started from the dashboard.", flush=True)
        while not _camera_stop.is_set():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("The webcam stopped returning frames.")

            height, width = frame.shape[:2]
            scale = min(640 / width, 480 / height, 1.0)
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            _camera_frame_size = (frame.shape[1], frame.shape[0])
            publish_camera_frame(frame)
            publish_inference_frame(frame)
            if _camera_state == "starting":
                _camera_state = "ready"
                _camera_frame_ready.set()
            record_camera_frame(frame)
    except Exception as error:
        _camera_error = str(error)
        _camera_state = "unavailable"
        _camera_observation = {
            "possible_litter_indicator": False,
            "message": "Camera analysis unavailable; no scene result is available.",
        }
        print(f"[Camera] {error}", flush=True)
    finally:
        try:
            stop_recording()
        except RuntimeError as error:
            print(f"[Recording] {error}", flush=True)
        if cap is not None:
            cap.release()
        if _camera_state != "unavailable":
            _camera_state = "stopped"
        with _camera_frames:
            _camera_frames.notify_all()


def start_recording():
    global _recording_error, _recording_filename, _recording_frame_count
    global _recording_last_frame_at, _recording_started_at, _recording_state, _recording_writer
    if _camera_state != "ready" or not _camera_frame_ready.wait(timeout=3):
        raise RuntimeError("Start the camera and wait for its live preview before recording.")
    if _camera_frame_size is None:
        raise RuntimeError("The camera has not produced a frame yet. Try again in a moment.")
    initialize_recordings_database()

    with _recording_lock:
        if _camera_state != "ready" or not _camera_frame_ready.is_set():
            raise RuntimeError("The camera stopped before recording could start.")
        if _recording_state == "recording":
            raise RuntimeError("A class recording is already in progress.")
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        _recording_filename = f"class-session-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.mp4"
        recording_path = RECORDINGS_DIR / _recording_filename
        writer = cv2.VideoWriter(
            str(recording_path),
            cv2.VideoWriter_fourcc(*"avc1"),
            15.0,
            _camera_frame_size,
        )
        if not writer.isOpened():
            writer.release()
            if recording_path.exists():
                recording_path.unlink()
            _recording_state = "error"
            _recording_error = "Could not open the H.264 video encoder on this computer."
            raise RuntimeError(_recording_error)
        _recording_writer = writer
        _recording_started_at = time.time()
        _recording_last_frame_at = 0.0
        _recording_frame_count = 0
        _recording_error = ""
        _recording_state = "recording"


def record_camera_frame(frame):
    global _recording_error, _recording_last_frame_at, _recording_state
    global _recording_writer, _recording_frame_count
    with _recording_lock:
        if _recording_state != "recording" or _recording_writer is None:
            return
        frame_time = time.monotonic()
        if frame_time - _recording_last_frame_at < 1 / 15:
            return
        try:
            _recording_writer.write(frame)
            _recording_last_frame_at = frame_time
            _recording_frame_count += 1
        except cv2.error as error:
            _recording_writer.release()
            _recording_writer = None
            _recording_state = "error"
            _recording_error = f"Video recording stopped because writing failed: {error}"
            print(f"[Recording] {_recording_error}", flush=True)


def stop_recording():
    global _recording_last_frame_at, _recording_started_at, _recording_state, _recording_writer
    with _recording_lock:
        if _recording_writer is None:
            return _recording_filename
        _recording_writer.release()
        _recording_writer = None
        _recording_started_at = None
        _recording_last_frame_at = None
        filename = _recording_filename
        frame_count = _recording_frame_count
    try:
        save_recording_metadata(filename, frame_count)
    except (OSError, sqlite3.Error) as error:
        with _recording_lock:
            _recording_state = "error"
            _recording_error = f"Video saved, but its class-library entry could not be stored: {error}"
        print(f"[Recording] {_recording_error}", flush=True)
        raise RuntimeError(_recording_error) from error
    with _recording_lock:
        _recording_state = "stopped"
        _recording_error = ""
    print(f"[Recording] Saved {filename} ({frame_count} frames).", flush=True)
    return filename


def initialize_recordings_database():
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(RECORDINGS_DB, timeout=10)) as database:
        with database:
            database.execute(
                """
                CREATE TABLE IF NOT EXISTS class_recordings (
                    filename TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    published INTEGER NOT NULL DEFAULT 1 CHECK (published IN (0, 1))
                )
                """
            )
            for path in RECORDINGS_DIR.glob("class-session-*.mp4"):
                if not re.fullmatch(r"class-session-\d{8}-\d{6}-\d{6}\.mp4", path.name):
                    continue
                try:
                    details = path.stat()
                except OSError as error:
                    print(f"[Recording] Could not read recording metadata for {path.name}: {error}", flush=True)
                    continue
                if details.st_size:
                    created_at = datetime.fromtimestamp(details.st_mtime).astimezone().isoformat(timespec="minutes")
                    database.execute(
                        """
                        INSERT OR IGNORE INTO class_recordings
                        (filename, created_at, duration_seconds, size_bytes, published)
                        VALUES (?, ?, 0, ?, 1)
                        """,
                        (path.name, created_at, details.st_size),
                    )


def save_recording_metadata(filename, frame_count):
    path = RECORDINGS_DIR / filename
    details = path.stat()
    if details.st_size == 0 or frame_count == 0:
        raise OSError("The recording contains no video frames.")
    created_at = datetime.fromtimestamp(details.st_mtime).astimezone().isoformat(timespec="minutes")
    with closing(sqlite3.connect(RECORDINGS_DB, timeout=10)) as database:
        with database:
            database.execute(
                """
                INSERT INTO class_recordings
                (filename, created_at, duration_seconds, size_bytes, published)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(filename) DO UPDATE SET
                    created_at = excluded.created_at,
                    duration_seconds = excluded.duration_seconds,
                    size_bytes = excluded.size_bytes,
                    published = 1
                """,
                (filename, created_at, frame_count / RECORDING_FPS, details.st_size),
            )


def list_recordings():
    initialize_recordings_database()
    with closing(sqlite3.connect(RECORDINGS_DB, timeout=10)) as database:
        with database:
            database.row_factory = sqlite3.Row
            rows = database.execute(
                """
                SELECT filename, created_at, duration_seconds, size_bytes
                FROM class_recordings
                WHERE published = 1
                ORDER BY created_at DESC, filename DESC
                """
            ).fetchall()

    recordings = []
    for row in rows:
        if not (RECORDINGS_DIR / row["filename"]).is_file():
            print(f"[Recording] Missing video file for catalog entry {row['filename']}.", flush=True)
            continue
        recordings.append({
            "filename": row["filename"],
            "size_bytes": row["size_bytes"],
            "created_at": row["created_at"],
            "duration_seconds": row["duration_seconds"],
        })
    return recordings


def process_camera_frames():
    global _camera_observation
    last_frame_number = 0
    last_export_time = 0.0
    last_report_time = 0.0
    live_demo_zone = "Hostel_Block_B"

    while not _inference_stop.is_set():
        with _camera_frames:
            _camera_frames.wait_for(
                lambda: _inference_frame_number > last_frame_number or _inference_stop.is_set(),
                timeout=1,
            )
            if _inference_stop.is_set():
                return
            frame_number = _inference_frame_number
            frame = _latest_camera_frame

        if frame is None:
            continue
        last_frame_number = frame_number

        # Capture keeps running independently, so slow inference cannot stall the browser feed.
        waste_flag, energy_flag = vision.process_camera_frame(frame)
        _camera_observation = vision.latest_observation
        simulated_water_flow = 24.8 if (waste_flag or energy_flag) else 2.1
        twin.update_sensor_metrics(
            live_demo_zone,
            simulated_water_flow,
            42.6 if energy_flag else 12.4,
        )
        twin.update_vision_metrics(live_demo_zone, waste_flag, energy_flag)

        current_time = time.time()
        should_report = (
            (waste_flag or energy_flag)
            and current_time - last_report_time > 10
        )
        if should_report:
            print("[Live Threat Registered] Fetching automated brief from Gemini AI Brain...", flush=True)
            metrics = twin.get_zone_report(live_demo_zone)
            report = brain.compile_sustainability_brief(live_demo_zone, metrics)
            export_twin_json_packet(live_demo_zone, report)
            last_report_time = current_time
            last_export_time = current_time
        elif current_time - last_export_time >= 1:
            export_twin_json_packet()
            last_export_time = current_time


def main():
    print("=== STARTING LIVE AI-POWERED SUSTAINABLE CAMPUS CORE ENGINE ===")
    export_twin_json_packet()
    initialize_recordings_database()

    try:
        camera_server = ThreadingHTTPServer((CAMERA_HOST, CAMERA_PORT), CampusRequestHandler)
        camera_server.daemon_threads = True
    except OSError as error:
        raise RuntimeError(
            f"Could not start the CanopyQ dashboard at http://{CAMERA_HOST}:{CAMERA_PORT}. "
            "Close the other server using port 8000 and try again."
        ) from error

    inference_thread = Thread(
        target=process_camera_frames,
        name="campus-camera-inference",
        daemon=True,
    )
    inference_thread.start()

    try:
        server_thread = Thread(
            target=camera_server.serve_forever,
            name="eco-twin-web-server",
            daemon=True,
        )
        server_thread.start()
        dashboard_url = f"http://{CAMERA_HOST}:{CAMERA_PORT}"
        print(f"Dashboard ready at {dashboard_url}. Use the dashboard button to start the camera.", flush=True)
        webbrowser.open(dashboard_url)
        server_thread.join()
    finally:
        stop_camera()
        _inference_stop.set()
        with _camera_frames:
            _camera_frames.notify_all()
        if _camera_thread is not None:
            _camera_thread.join(timeout=3)
        inference_thread.join(timeout=3)
        camera_server.shutdown()
        camera_server.server_close()

if __name__ == "__main__":
    main()
