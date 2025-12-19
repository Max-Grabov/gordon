SR = 16000

WAKE_VAD = {"min_silence_duration_ms": 250, "speech_pad_ms": 80}

COMMAND_TIMEOUT_S = 5.0
AUDIO_BUFFER_S = 3.0

# VAD
SILERO_THRESHOLD = 0.6
SILERO_MIN_SPEECH_DURATION_MS = 250
SILERO_MIN_SILENCE_DURATION_MS = 100
SILERO_SPEECH_PAD_MS = 30

# Audio UI
SILENCE_RMS = 0.002
SPEAKING_RMS = 0.008
MAX_RMS_DISPLAY = 0.15

# Kinematics
ALPHA_0 = 0.023
ALPHA_3 = 0.140
ALPHA = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0}
GEAR_RATIO = {"base_yaw": 1.0, "base_pitch": 1.0, "elbow": 1.0, "wrist_pitch": 1.0}
CLAW_OPEN_THRESH = 0.12
MIRROR_EWMA = 0.30

# Cam
CAM_INDEX = 0
FRAME_W, FRAME_H = 1920, 1080
MIRROR_FLIP = True

# Models
POSE_MODEL = "heavy"
POSE_TASK_PATH = "models/pose_landmarker_heavy.task"
HAND_TASK_PATH = "models/hand_landmarker.task"