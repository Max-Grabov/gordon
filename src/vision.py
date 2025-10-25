import numpy as np, cv2, mediapipe as mp
from .utils import _xy, _to_landmark_list, ensure_model
from . import config

mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
HAND_CONN = mp.solutions.hands.HAND_CONNECTIONS
RunningMode = mp.tasks.vision.RunningMode
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, PoseLandmarker, PoseLandmarkerOptions

def draw_overlays(rgb_image, pose_landmarks_list, hand_landmarks_list, handedness_list=None):
    img = rgb_image.copy(); h, w = img.shape[:2]
    # Hands (draw left if present else first)
    hands_to_draw = []
    if hand_landmarks_list:
        if handedness_list and len(hand_landmarks_list)==len(handedness_list):
            for i,hnd in enumerate(handedness_list):
                if "left" in str(hnd).lower(): hands_to_draw=[hand_landmarks_list[i]]; break
            if not hands_to_draw: hands_to_draw=[hand_landmarks_list[0]]
        else: hands_to_draw = hand_landmarks_list
    for hand_pts in hands_to_draw:
        mp_draw.draw_landmarks(img, _to_landmark_list(hand_pts), HAND_CONN,
                               mp_styles.get_default_hand_landmarks_style(),
                               mp_styles.get_default_hand_connections_style())
    # Right arm points
    if pose_landmarks_list:
        pose = pose_landmarks_list[0]; arm_idxs=[11,13,15]
        for idx in arm_idxs:
            if 0 <= idx < len(pose):
                cx,cy=_xy(w,h,pose[idx]); cv2.circle(img,(cx,cy),5,(0,255,0),-1)
        p0,p1,p2=arm_idxs
        if 0<=p0<len(pose) and 0<=p1<len(pose):
            x0,y0=_xy(w,h,pose[p0]); x1,y1=_xy(w,h,pose[p1]); cv2.line(img,(x0,y0),(x1,y1),(0,255,0),2,cv2.LINE_AA)
        if 0<=p1<len(pose) and 0<=p2<len(pose):
            x1,y1=_xy(w,h,pose[p1]); x2,y2=_xy(w,h,pose[p2]); cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2,cv2.LINE_AA)
    return img

def _lm_xyz(lms, idx):
    if lms is None or idx < 0: return None
    try:
        l=lms[idx]; return np.array([float(l.x), float(l.y), float(l.z)], dtype=np.float32)
    except Exception: return None

def _select_hand(hand_list, handedness_list, preferred="right"):
    if not hand_list: return None
    if handedness_list and len(handedness_list)==len(hand_list):
        for i,hnd in enumerate(handedness_list):
            if preferred.lower() in str(hnd).lower(): return hand_list[i]
    return hand_list[0]

def _euclid3(a,b): return float(np.linalg.norm(a-b))

def compute_mirror_angles(s,e,w,p):
    A = config.ALPHA
    t0 = np.arctan2((w[0]-s[0])*A[1], (w[2]-s[2])*A[2])
    t1 = np.arctan2((s[1]-e[1])*A[3], (e[2]-s[2])*A[4])
    t2 = np.arctan2((w[1]-e[1])*A[5], (w[2]-e[2])*A[6])
    t3 = np.arctan2((p[1]-w[1])*A[7], (p[2]-w[2])*A[8])
    return [float(t0),float(t1),float(t2),float(t3)]

def transform_to_motor_angle_range(lower_angle_bound: float, higher_angle_bound: float, input_angle: float, counterclockwise: bool):

    
def thetas_to_motor_turns(thetas):
    names=["base_yaw","base_pitch","elbow","wrist_pitch"]
    return [(th/(2*np.pi))*config.GEAR_RATIO[nm] for th,nm in zip(thetas,names)]

class MirrorProcessor:
    def __init__(self):
        ensure_model(config.POSE_TASK_PATH,
            f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{config.POSE_MODEL}/float16/1/pose_landmarker_{config.POSE_MODEL}.task")
        ensure_model(config.HAND_TASK_PATH,
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

        self.pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.POSE_TASK_PATH),
            running_mode=RunningMode.VIDEO, num_poses=1,
        ))
        self.hand = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.HAND_TASK_PATH),
            running_mode=RunningMode.VIDEO, num_hands=1,
        ))
        self.prev_theta = None

    def process(self, frame_rgb, timestamp_ms):
        pose_res = self.pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb), timestamp_ms)
        hand_res = self.hand.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb), timestamp_ms)

        pose_list = pose_res.pose_landmarks if hasattr(pose_res, "pose_landmarks") else []
        hand_list = hand_res.hand_landmarks if hasattr(hand_res, "hand_landmarks") else []
        handedness = getattr(hand_res, "handedness", None)

        right_hand = _select_hand(hand_list, handedness, "right")
        s = _lm_xyz(pose_list[0] if pose_list else None, 12)
        e = _lm_xyz(pose_list[0] if pose_list else None, 14)
        w = _lm_xyz(pose_list[0] if pose_list else None, 16)
        p = _lm_xyz(right_hand, 20) if right_hand is not None else None

        thumb_tip = _lm_xyz(right_hand, 4) if right_hand is not None else None
        pinky_tip = _lm_xyz(right_hand, 20) if right_hand is not None else None

        annotated_rgb = draw_overlays(frame_rgb, pose_list, hand_list, handedness)
        out_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        info = {"angles": None, "motors": None, "claw_open": False}
        ready = (s is not None) and (e is not None) and (w is not None) and (p is not None)

        if thumb_tip is not None and pinky_tip is not None:
            d = _euclid3(thumb_tip, pinky_tip)
            info["claw_open"] = (d >= config.CLAW_OPEN_THRESH)
            H,W = out_bgr.shape[:2]
            def to_px(pt): return (int(pt[0]*W), int(pt[1]*H))
            cv2.line(out_bgr, to_px(thumb_tip), to_px(pinky_tip), (0,255,255), 2)
            cv2.putText(out_bgr, f"CLAW: {'OPEN' if info['claw_open'] else 'CLOSE'}  d={d:.3f}",
                        (10, out_bgr.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,255) if info["claw_open"] else (0,165,255), 2, cv2.LINE_AA)

        if ready:
            thetas = compute_mirror_angles(s,e,w,p)
            if self.prev_theta is None:
                filt = np.array(thetas, dtype=np.float32)
            else:
                prev = np.array(self.prev_theta, dtype=np.float32)
                filt = (1.0 - config.MIRROR_EWMA) * np.array(thetas, dtype=np.float32) + config.MIRROR_EWMA * prev
            self.prev_theta = filt.tolist()
            motors = thetas_to_motor_turns(self.prev_theta)
            info["angles"] = self.prev_theta
            info["motors"] = motors

            degs = np.degrees(filt).tolist()
            hud = [
                f"Theta_0 base yaw   : {degs[0]:6.1f}",
                f"Theta_1 base pitch : {degs[1]:6.1f}",
                f"Theta_2 elbow      : {degs[2]:6.1f}",
                f"Theta_3 wrist pitch: {degs[3]:6.1f}",
            ]
            for i, line in enumerate(hud):
                cv2.putText(out_bgr, line, (10, 200 + i*24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(out_bgr, f"mot(turns): {', '.join(f'{m:.3f}' for m in motors)}",
                        (10, 200 + len(hud)*24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)

        return out_bgr, info

    def close(self):
        try: self.pose.close()
        except: pass
        try: self.hand.close()
        except: pass
