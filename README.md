# CanopyQ Smart Campus

CanopyQ combines the live Python digital-twin dashboard with a browser-only virtual-campus simulation.

## Start the dashboard

1. Use Python 3.12 for the vision dependencies. If setting up from scratch, create a Python 3.12 virtual environment and install `requirements.txt` inside it.
2. Start the app by double-clicking `start_dashboard.bat`, or run `python main.py` from the project folder. If the current Python environment is missing dependencies and `.venv312` exists, the launcher automatically restarts with `.venv312`. To check the environment without starting the camera, run `python main.py --check-dependencies`.
3. The app opens the CanopyQ dashboard at `http://127.0.0.1:8000`. Choose a demo role and enter `admin` or `demo` as the local demo code. Camera access is available to Administrator, Lecturer, and Student Council; codes are case-insensitive and surrounding spaces are ignored.

The webcam stays off when the app starts. An authorized user starts or stops it using the camera button on the dashboard; students do not see that control. Camera requests and the MJPEG stream are served locally by the same Python app at `127.0.0.1:8000`, so no second static server or camera-start command is needed. If the camera status reports HTTP 404, close the outdated server process, start `start_dashboard.bat`, and reload the page. Check Windows camera privacy settings if the current dashboard reports that no camera is available. Camera capture runs independently of AI inference so slow processing does not stall the feed, and the old OpenCV popup is removed.

The general-purpose vision model can only report a **possible** bottle or cup in the lower part of a frame for staff review. This is not proof of littering and can produce false positives. It does not detect property damage, identify people, recognize faces, track attendance, or save camera images. The scene observation is held in memory only while the app runs; do not use it for discipline or other decisions about an individual.

Lecturers can use the camera panel to manually start and stop video-only H.264 class recordings; recording is never automatic and has a visible on-screen indicator. Announce recording and follow campus consent requirements before starting. Video files are saved in the local `recordings` folder, while a SQLite catalog in `recordings.sqlite3` stores the class date, length, and file size. The catalog file is not served to browsers. Published videos appear in the Missed class recordings library for students, lecturers, and administrators on this computer. Video-only recordings have no audio. Files remain on disk until manually deleted according to campus retention policy. The demo role selector is not real authentication or access control. The server binds to `127.0.0.1`, so recordings are not remotely available to students on other devices; use an approved campus learning platform to share a downloaded file.

The dashboard refreshes the twin packet every five seconds. Each role gets a tailored dashboard and navigation: administrators get all tools; lecturers get facilities, AI briefs, camera access, and class recording controls; the campus council gets building wellbeing and environmental indicators plus camera access; students get a simplified campus snapshot and the class recording library. The browser-only campus simulator uses the same CanopyQ brand, shared navigation, color system, and active role. These demo codes and role checks are public client-side values, not production authentication.

## Explore the campus simulator

Use the dashboard navigation to open `site/index.html`. Its overview, smart-building, microgrid, carbon-scheduler, and architecture pages run a separate software simulation in the browser and need no Python services.

The live twin and the simulation are complementary views: live sensor/vision readings and AI briefs appear in the CanopyQ overview, while energy dispatch, cooling, and carbon scheduling are modeled in the campus simulator.
