# Abnormal Behavior Detection - Web Deployment

This project provides a local Flask webpage for your trained GRU abnormal
behavior detection model.

## Pipeline

Video upload
-> MediaPipe Pose Landmarker Heavy
-> 33 landmarks
-> x/y/z/visibility = 132 features per frame
-> pelvis/body-scale normalization
-> uniform sampling to 300 frames
-> trained GRU
-> sigmoid probability
-> threshold 0.50
-> NORMAL / ABNORMAL

## Files

- `app.py` - Flask backend and inference pipeline
- `best_gru_model.keras` - trained model
- `setup_pose_model.py` - downloads the MediaPipe `.task` model
- `templates/index.html` - webpage
- `static/style.css` - webpage styling
- `static/script.js` - upload and result logic
- `requirements.txt` - Python dependencies

## Run on Windows

Open Command Prompt in this folder.

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate it

```bash
venv\\Scripts\\activate
```

### 3. Install packages

```bash
pip install -r requirements.txt
```

### 4. Download the MediaPipe pose model

```bash
python setup_pose_model.py
```

This creates:

`pose_landmarker_heavy.task`

### 5. Start the web application

```bash
python app.py
```

For a one-click launch on Windows, run this once in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\create_desktop_shortcut.ps1
```

This creates **Abnormal Behavior Detection** on the desktop. Double-clicking
it runs `launch_app.bat`, starts Flask, and opens the webpage in a separate
browser app window without normal tabs and address-bar controls. Microsoft Edge
or Google Chrome is used when installed.

### 6. Open the webpage

Go to:

`http://127.0.0.1:5000`

## How to use

1. Select a video.
2. Preview the video.
3. Play the video to see the 33 pose landmarks drawn over the person. The
	current frame, total frame count, FPS, and duration update in real time.
4. Click **Analyze Video**.
5. Wait while pose landmarks are extracted for the GRU model.
6. The page displays NORMAL or ABNORMAL and the confidence.

## Important

The deployment preprocessing intentionally follows the supplied Colab code:
- 33 pose landmarks
- 132 features
- hip midpoint as pelvis center
- larger shoulder/hip width as scale
- uniform sampling across the full video
- final shape `(300, 132)`
- classification threshold `0.50`

Do not replace these steps with a different preprocessing method unless the
model is retrained or the change has been experimentally validated.

## Notes

The application runs locally by default. It does not expose the model to the
internet. For public deployment, add authentication, HTTPS, upload limits,
rate limiting, and appropriate privacy controls.
