import { FilesetResolver, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/+esm";

const form = document.getElementById("uploadForm");
const input = document.getElementById("videoInput");
const fileName = document.getElementById("fileName");
const preview = document.getElementById("preview");
const canvas = document.getElementById("landmarkCanvas");
const ctx = canvas.getContext("2d");
const workspace = document.getElementById("videoWorkspace");
const poseLoading = document.getElementById("poseLoading");
const livePoseStatus = document.getElementById("livePoseStatus");
const button = document.getElementById("analyzeBtn");
const statusBox = document.getElementById("status");
const result = document.getElementById("result");

let objectUrl = null;
let videoInfo = { fps: 0, total_frames: 0, duration: 0 };
let poseLandmarker = null;
let lastVideoTime = -1;
let animationId = null;
let poseBusy = false;

const connections = [
    [11,12],[11,13],[13,15],[12,14],[14,16],
    [11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],
    [27,29],[29,31],[28,30],[30,32], [15,17],[15,19],[15,21],
    [16,18],[16,20],[16,22],[27,31],[28,32]
];

input.addEventListener("change", async () => {
    const file = input.files[0];
    if (!file) return resetVideo();

    fileName.textContent = file.name;
    result.hidden = true;
    statusBox.hidden = true;
    workspace.hidden = false;
    livePoseStatus.textContent = "Loading";

    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    preview.src = objectUrl;
    preview.load();

    try {
        await loadVideoInfo(file);
    } catch (error) {
        showStatus(error.message);
    }
});

preview.addEventListener("loadedmetadata", resizeCanvas);
preview.addEventListener("play", () => {
    startPoseLoop();
});
preview.addEventListener("pause", stopPoseLoop);
preview.addEventListener("ended", stopPoseLoop);
preview.addEventListener("seeked", () => {
    lastVideoTime = -1;
    updateFrameCounter();
    drawPose();
});
window.addEventListener("resize", resizeCanvas);

async function loadVideoInfo(file) {
    const data = new FormData();
    data.append("video", file);
    const response = await fetch("/video-info", { method: "POST", body: data });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not read video information.");

    videoInfo = payload;
    document.getElementById("videoFrames").textContent = payload.total_frames.toLocaleString();
    document.getElementById("videoFps").textContent = payload.fps.toFixed(2);
    document.getElementById("videoDuration").textContent = formatTime(payload.duration);
    updateFrameCounter();
    await initPoseLandmarker();
}

async function initPoseLandmarker() {
    if (poseLandmarker) {
        poseLoading.hidden = true;
        return;
    }

    poseLoading.hidden = false;
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm"
        );
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "/pose-model",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1,
            minPoseDetectionConfidence: 0.5,
            minPosePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        poseLoading.hidden = true;
        livePoseStatus.textContent = "Ready";
    } catch (error) {
        poseLoading.textContent = "Live pose tracker unavailable — analysis still works.";
        livePoseStatus.textContent = "Unavailable";
        console.error(error);
    }
}

function startPoseLoop() {
    stopPoseLoop();
    const render = () => {
        if (preview.paused || preview.ended) return;
        updateFrameCounter();
        drawPose();
        animationId = requestAnimationFrame(render);
    };
    animationId = requestAnimationFrame(render);
}

function stopPoseLoop() {
    if (animationId) cancelAnimationFrame(animationId);
    animationId = null;
}

function updateFrameCounter() {
    const fps = Number(videoInfo.fps) || 30;
    const total = Number(videoInfo.total_frames) || 0;
    const current = total ? Math.min(total, Math.max(1, Math.floor(preview.currentTime * fps) + 1)) : 0;
    document.getElementById("currentFrame").textContent = total ? `${current.toLocaleString()} / ${total.toLocaleString()}` : "0 / —";
}

function resizeCanvas() {
    if (!preview.videoWidth || !preview.videoHeight) return;
    canvas.width = preview.videoWidth;
    canvas.height = preview.videoHeight;
    canvas.style.aspectRatio = `${preview.videoWidth} / ${preview.videoHeight}`;
}

function drawPose() {
    if (!poseLandmarker || poseBusy || !preview.videoWidth || preview.readyState < 2) return;

    const timestamp = Math.round(preview.currentTime * 1000);
    if (timestamp === lastVideoTime) return;
    lastVideoTime = timestamp;
    poseBusy = true;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    try {
        const detection = poseLandmarker.detectForVideo(preview, timestamp);
        const landmarks = detection.landmarks?.[0];
        if (!landmarks) {
            livePoseStatus.textContent = "No pose";
            return;
        }

        livePoseStatus.textContent = "33 landmarks";
        ctx.lineWidth = Math.max(2, canvas.width / 450);
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        for (const [a, b] of connections) {
            const p1 = landmarks[a], p2 = landmarks[b];
            if (!p1 || !p2) continue;
            ctx.beginPath();
            ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
            ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
            ctx.stroke();
        }

        for (const point of landmarks) {
            ctx.beginPath();
            ctx.arc(point.x * canvas.width, point.y * canvas.height, Math.max(2.5, canvas.width / 280), 0, Math.PI * 2);
            ctx.fill();
        }
    } catch (error) {
        console.error(error);
    } finally {
        poseBusy = false;
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = input.files[0];
    if (!file) return showStatus("Please select a video first.");

    const data = new FormData();
    data.append("video", file);
    button.disabled = true;
    button.textContent = "Analyzing...";
    showStatus("Extracting pose landmarks and running the GRU model...");

    try {
        const response = await fetch("/predict", { method: "POST", body: data });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Analysis failed.");
        showResult(payload);
        statusBox.hidden = true;
    } catch (error) {
        showStatus(error.message);
        result.hidden = true;
    } finally {
        button.disabled = false;
        button.textContent = "Analyze Video";
    }
});

function showStatus(message) {
    statusBox.textContent = message;
    statusBox.hidden = false;
}

function showResult(data) {
    result.hidden = false;
    const confidence = data.confidence * 100;
    document.getElementById("prediction").textContent = data.label;
    document.getElementById("confidenceValue").textContent = confidence.toFixed(1) + "%";
    document.getElementById("confidenceBar").style.width = confidence.toFixed(1) + "%";
    document.getElementById("frames").textContent = Number(data.total_frames).toLocaleString();
    document.getElementById("poseRate").textContent = data.detection_rate.toFixed(1) + "%";
    document.getElementById("shape").textContent = data.input_shape.join(" × ");
    document.getElementById("resultBadge").textContent = data.label === "ABNORMAL" ? "ABNORMAL ACTIVITY DETECTED" : "NORMAL ACTIVITY";
}

function resetVideo() {
    stopPoseLoop();
    workspace.hidden = true;
    preview.removeAttribute("src");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    fileName.textContent = "Choose a video file";
    videoInfo = { fps: 0, total_frames: 0, duration: 0 };
}

function formatTime(seconds) {
    seconds = Math.max(0, Number(seconds) || 0);
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
}
