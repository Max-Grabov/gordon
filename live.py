import time
import os
import urllib.request
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import RunningMode as RunningMode
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    PoseLandmarker, PoseLandmarkerOptions,
)

MODEL_URLS = {
    "pose_lite":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_full":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "pose_heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    "hand":       "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

def ensure_model(local_path: Path, url: str):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        urllib.request.urlretrieve(url, local_path)

mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
POSE_CONN = mp.solutions.pose.POSE_CONNECTIONS
HAND_CONN = mp.solutions.hands.HAND_CONNECTIONS

def _to_landmark_list(norm_landmarks):
    ll = landmark_pb2.NormalizedLandmarkList()
    ll.landmark.extend(
        [landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in norm_landmarks]
    )
    return ll

def draw_overlays(rgb_image, pose_landmarks_list, hand_landmarks_list):
    img = rgb_image.copy()
    for hand_landmarks in hand_landmarks_list or []:
        hand_proto = _to_landmark_list(hand_landmarks)
        mp_draw.draw_landmarks(
            img, hand_proto, HAND_CONN,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )
    for pose_landmarks in pose_landmarks_list or []:
        pose_proto = _to_landmark_list(pose_landmarks)
        mp_draw.draw_landmarks(
            img, pose_proto, POSE_CONN,
            mp_styles.get_default_pose_landmarks_style()
        )
    return img


def main():
    cam_ind = 0
    w = 1280
    h = 720 
    pose_model = "heavy" 
    num_hands = 1

    pose_model_path = Path("models") / f"pose_landmarker_{pose_model}.task"
    hand_model_path = Path("models") / "hand_landmarker.task"

    ensure_model(pose_model_path, MODEL_URLS[f"pose_{pose_model}"])
    ensure_model(hand_model_path, MODEL_URLS["hand"])

    pose_opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=1
        # min_pose_detection_confidence=0.5,
        # min_pose_presence_confidence=0.5,
        # min_tracking_confidence=0.5,
    )
    hand_opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model_path)),
        running_mode=RunningMode.VIDEO,
        num_hands=max(1, int(num_hands))
        # min_hand_detection_confidence=0.5,
        # min_hand_presence_confidence=0.5,
        # min_tracking_confidence=0.5,
    )
    pose_lm = PoseLandmarker.create_from_options(pose_opts)
    hand_lm = HandLandmarker.create_from_options(hand_opts)

    cap = cv2.VideoCapture(cam_ind, cv2.CAP_DSHOW if os.name == "nt" else 0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    t0 = time.perf_counter()
    last = time.perf_counter()
    fps = 0.0

    window_name = "ESC to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_bgr = cv2.flip(frame_bgr, 1)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int((time.perf_counter() - t0) * 1000)

            pose_res = pose_lm.detect_for_video(mp_image, timestamp_ms)
            hand_res = hand_lm.detect_for_video(mp_image, timestamp_ms)

            pose_list = pose_res.pose_landmarks if hasattr(pose_res, "pose_landmarks") else []
            hand_list = hand_res.hand_landmarks if hasattr(hand_res, "hand_landmarks") else []

            annotated_rgb = draw_overlays(frame_rgb, pose_list, hand_list)

            now = time.perf_counter()
            dt = now - last
            last = now
            fps = (0.9 * fps + 0.1 * (1.0 / dt)) if dt > 0 else fps

            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                annotated_bgr,
                f"FPS: {fps:5.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, annotated_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            pose_lm.close()
            hand_lm.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
