import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)

destination = Path(__file__).resolve().parent / "pose_landmarker_heavy.task"

print("Downloading MediaPipe Pose Landmarker Heavy...")
urllib.request.urlretrieve(URL, destination)
print(f"Saved to: {destination}")
