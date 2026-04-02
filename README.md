# Gordon

A voice-controlled robotic arm assistant that combines speech recognition, computer vision, and real-time hand tracking to control a 4-DOF robotic arm.

## Features

- **Wake Word Activation** — Responds to "Gordon" using semantic similarity matching
- **Voice Commands** — Speak natural language commands to control the arm (mirror, record, replay, idle)
- **Real-Time Hand Mirroring** — Tracks your pose and hand landmarks via MediaPipe and maps them to robotic arm joint angles in real-time
- **Action Recording & Replay** — Record sequences of motor commands and replay them on demand, with natural language scheduling ("every 5 minutes for 1 hour", "10 times", etc.)
- **Live UI Overlay** — OpenCV display with FPS, current mode, audio level meter, transcripts, and pose/hand overlays

## Tech Stack

- **Speech**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (ASR), [Silero VAD](https://github.com/snakers4/silero-vad) (voice activity detection)
- **NLP**: [sentence-transformers](https://www.sbert.net/) (intent matching via semantic embeddings)
- **Vision**: [MediaPipe](https://developers.google.com/mediapipe) (pose & hand landmark detection), [OpenCV](https://opencv.org/)
- **Audio**: sounddevice, torchaudio
- **Deep Learning**: PyTorch

## Project Structure

```
gordon/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── gordonpy/               # Core Python package
│   ├── config.py           # Configuration constants (audio, kinematics, camera)
│   ├── asr.py              # ASR engine (Whisper + embeddings)
│   ├── vad.py              # Voice activity detection (Silero)
│   ├── audio_io.py         # Audio capture and ring buffer
│   ├── intents.py          # Intent definitions and matching
│   ├── vision.py           # Pose/hand tracking and motor angle calculation
│   ├── ui.py               # OpenCV UI rendering
│   ├── actions.py          # Action recording, replay, and scheduling
│   └── utils.py            # Text normalization, model downloading
└── packages/
    └── control/            # C++ control layer (Bazel build)
        ├── src/            # C++ source files
        ├── include/        # C++ headers
        └── BUILD.bazel     # Bazel build config
```

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for faster inference)
- Webcam
- Microphone
- Robotic arm with serial connection (for hardware control)

## Setup

```bash
# Clone the repository
git clone https://github.com/Max-Grabov/gordon.git
cd gordon

# Install dependencies
pip install -r requirements.txt
```

MediaPipe model files are downloaded automatically on first run to the `models/` directory.

## Usage

```bash
python main.py
```

### Voice Commands

1. Say **"Gordon"** to activate command mode
2. Speak a command within the 5-second window:
   - **"mirror"** — arm mirrors your hand movements in real-time
   - **"record"** — start recording motor commands
   - **"stop"** — stop recording, then name the action
   - **"idle"** — return to idle mode
3. Recorded actions can be replayed with scheduling (e.g., "every 30 seconds", "5 times")

### Controls

- Press **ESC** / **Q** to quit

## How It Works

1. **Audio** — Continuous 16kHz capture with ring buffer and VAD for speech segmentation
2. **Wake Word** — Semantic similarity (sentence-transformers) + exact text matching against "gordon"
3. **ASR** — faster-whisper transcribes speech; embeddings classify intent
4. **Vision** — MediaPipe detects pose (shoulder/elbow/wrist) and hand landmarks; inverse kinematics maps them to 4 motor angles (base yaw, base pitch, elbow, wrist pitch) plus claw open/close
5. **Smoothing** — Exponential moving average (alpha=0.30) stabilizes motor commands
6. **Replay** — Timestamped recordings stored as JSON; natural language scheduling parsed via regex
