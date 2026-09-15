import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "best_gru_model.keras"
POSE_MODEL_PATH = BASE_DIR / "pose_landmarker_heavy.task"

UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
TARGET_FRAMES = 300
FEATURES = 132
THRESHOLD = 0.50

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB


# ------------------------------------------------------------
# Load trained GRU model once when the server starts
# ------------------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ------------------------------------------------------------
# EXACT pose normalization used in the Colab training code
# ------------------------------------------------------------
def normalize_pose(sequence):
    """
    Input:
        sequence shape = (frames, 132)

    132 = 33 landmarks x 4:
        x, y, z, visibility

    Same normalization logic as the training notebook:
    - center around left/right hip midpoint
    - scale by the larger of shoulder width and hip width
    - keep visibility unchanged
    """
    seq = sequence.copy().astype(np.float32)

    seq = seq.reshape(-1, 33, 4)

    left_hip = seq[:, 23, :3]
    right_hip = seq[:, 24, :3]

    pelvis = (left_hip + right_hip) / 2.0

    seq[:, :, :3] -= pelvis[:, None, :]

    left_shoulder = seq[:, 11, :3]
    right_shoulder = seq[:, 12, :3]

    shoulder_width = np.linalg.norm(
        left_shoulder - right_shoulder, axis=1
    )

    hip_width = np.linalg.norm(
        seq[:, 23, :3] - seq[:, 24, :3], axis=1
    )

    scale = np.maximum(shoulder_width, hip_width)
    scale[scale < 1e-6] = 1.0

    seq[:, :, :3] /= scale[:, None, None]

    return seq.reshape(-1, 132)


# ------------------------------------------------------------
# EXACT temporal sampling used in the Colab training code
# ------------------------------------------------------------
def sample_frames(sequence, target_frames=300):
    num_frames = sequence.shape[0]

    if num_frames == target_frames:
        return sequence

    if num_frames <= 0:
        raise ValueError("No video frames were found.")

    indices = np.linspace(
        0, num_frames - 1, target_frames
    ).astype(np.int64)

    return sequence[indices]


# ------------------------------------------------------------
# Video -> 132 pose features/frame
# ------------------------------------------------------------
def extract_pose_from_video(video_path):
    base_options = python.BaseOptions(
        model_asset_path=str(POSE_MODEL_PATH)
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        landmarker.close()
        raise ValueError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose_sequence = []
    detected_frames = 0
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int((frame_index / fps) * 1000)

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]

                frame_landmarks = []

                for landmark in landmarks:
                    frame_landmarks.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility
                    ])

                pose_sequence.append(frame_landmarks)
                detected_frames += 1
            else:
                # Same behavior as training: zeros for a missing pose.
                pose_sequence.append([0.0] * (33 * 4))

            frame_index += 1

    finally:
        cap.release()
        landmarker.close()

    if not pose_sequence:
        raise ValueError("The video contains no readable frames.")

    pose_sequence = np.asarray(
        pose_sequence, dtype=np.float32
    )

    if pose_sequence.ndim != 2 or pose_sequence.shape[1] != FEATURES:
        raise ValueError(
            f"Unexpected pose shape: {pose_sequence.shape}. "
            f"Expected (frames, 132)."
        )

    return pose_sequence, fps, total_frames, detected_frames


# ------------------------------------------------------------
# Full inference
# ------------------------------------------------------------
def predict_video(video_path):
    sequence, fps, total_frames, detected_frames = (
        extract_pose_from_video(video_path)
    )

    sequence = normalize_pose(sequence)
    sequence = sample_frames(sequence, TARGET_FRAMES)

    if sequence.shape != (TARGET_FRAMES, FEATURES):
        raise ValueError(
            f"Final model input is {sequence.shape}; "
            f"expected (300, 132)."
        )

    if not np.isfinite(sequence).all():
        raise ValueError("Invalid NaN/Inf values were produced.")

    X = np.expand_dims(sequence, axis=0).astype(np.float32)

    probability = float(model.predict(X, verbose=0).ravel()[0])

    label = "ABNORMAL" if probability >= THRESHOLD else "NORMAL"

    confidence = (
        probability if label == "ABNORMAL"
        else 1.0 - probability
    )

    detection_rate = (
        detected_frames / total_frames * 100
        if total_frames > 0 else 0
    )

    return {
        "label": label,
        "probability": probability,
        "confidence": confidence,
        "fps": fps,
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "detection_rate": detection_rate,
        "input_shape": list(X.shape[1:]),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/pose-model")
def pose_model():
    if not POSE_MODEL_PATH.exists():
        return jsonify({"error": "Pose model not found. Run setup_pose_model.py first."}), 404
    return send_file(POSE_MODEL_PATH, mimetype="application/octet-stream", max_age=3600)


@app.route("/video-info", methods=["POST"])
def video_info():
    if "video" not in request.files:
        return jsonify({"error": "Please select a video file."}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "No video file was selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported format. Use MP4, AVI, MOV or MKV."}), 400

    extension = Path(file.filename).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{extension}"
    video_path = UPLOAD_DIR / secure_filename(unique_name)

    try:
        file.save(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Could not open the uploaded video.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()
        return jsonify({
            "fps": float(fps),
            "total_frames": total_frames,
            "duration": float(duration)
        })
    except Exception as exc:
        return jsonify({"error": f"Could not read video information: {str(exc)}"}), 500
    finally:
        try:
            if video_path.exists():
                video_path.unlink()
        except Exception:
            pass


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "Please select a video file."}), 400

    file = request.files["video"]

    if not file.filename:
        return jsonify({"error": "No video file was selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Unsupported format. Use MP4, AVI, MOV or MKV."
        }), 400

    extension = Path(file.filename).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{extension}"
    filename = secure_filename(unique_name)
    video_path = UPLOAD_DIR / filename

    try:
        file.save(video_path)

        result = predict_video(video_path)
        return jsonify(result)

    except Exception as exc:
        return jsonify({
            "error": f"Analysis failed: {str(exc)}"
        }), 500

    finally:
        try:
            if video_path.exists():
                video_path.unlink()
        except Exception:
            pass


@app.errorhandler(413)
def too_large(_):
    return jsonify({
        "error": "Video is too large. Maximum size is 500 MB."
    }), 413


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "best_gru_model.keras was not found."
        )

    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "pose_landmarker_heavy.task was not found. "
            "Run setup_pose_model.py first."
        )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
