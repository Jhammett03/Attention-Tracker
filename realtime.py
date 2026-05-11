import argparse
import collections
import os
import pickle
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from cnn_time_model import TemporalCNN
from frame_extraction import (
    LEFT_EYE, LEFT_IRIS, RIGHT_EYE, RIGHT_IRIS,
    build_face_frame, eye_aspect_ratio, iris_to_face_frame, pts_xy,
)
from transformer_model import TransformerClassifier

SAMPLE_FPS = 15
WINDOW_SIZE = 150
SMOOTH_WINDOW = 3  # majority-vote over this many frames to buffer predictions

def extract_frame_features(face, w, h):
    left_eye_pts = pts_xy(face, LEFT_EYE, w, h)
    right_eye_pts = pts_xy(face, RIGHT_EYE, w, h)

    left_ear = eye_aspect_ratio(left_eye_pts)
    right_ear = eye_aspect_ratio(right_eye_pts)

    if len(face) < 478:
        return None

    R, eye_center_3d, inter_eye_dist = build_face_frame(face)
    left_gaze = iris_to_face_frame(face[LEFT_IRIS[0]], R, eye_center_3d, inter_eye_dist)
    right_gaze = iris_to_face_frame(face[RIGHT_IRIS[0]], R, eye_center_3d, inter_eye_dist)
    z_axis = R[:, 2]

    all_x = [lm.x for lm in face]
    all_y = [lm.y for lm in face]
    bb_min_x, bb_max_x = min(all_x), max(all_x)
    bb_min_y, bb_max_y = min(all_y), max(all_y)

    return [
        left_ear, right_ear,
        float(left_gaze[0]), float(left_gaze[1]),
        float(right_gaze[0]), float(right_gaze[1]),
        float(z_axis[0]), float(z_axis[1]),
        bb_min_x, bb_min_y,
        bb_max_x, bb_min_y,
        bb_min_x, bb_max_y,
        bb_max_x, bb_max_y,
    ]


def aggregate_window(window):
    arr = np.array(window, dtype=np.float32)
    return [
        float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])),
        float(np.std(arr[:, 0])),  float(np.std(arr[:, 1])),
        float(np.mean(arr[:, 2])), float(np.mean(arr[:, 3])),
        float(np.mean(arr[:, 4])), float(np.mean(arr[:, 5])),
        float(np.var(arr[:, 2])),  float(np.var(arr[:, 3])),
        float(np.var(arr[:, 4])),  float(np.var(arr[:, 5])),
        float(np.mean(arr[:, 6])), float(np.mean(arr[:, 7])),
        float(np.var(arr[:, 6])),  float(np.var(arr[:, 7])),
        float(np.mean(arr[:, 8])),  float(np.mean(arr[:, 9])),
        float(np.mean(arr[:, 10])), float(np.mean(arr[:, 11])),
        float(np.mean(arr[:, 12])), float(np.mean(arr[:, 13])),
        float(np.mean(arr[:, 14])), float(np.mean(arr[:, 15])),
        float(np.var(arr[:, 8])),   float(np.var(arr[:, 9])),
        float(np.var(arr[:, 10])),  float(np.var(arr[:, 11])),
        float(np.var(arr[:, 12])),  float(np.var(arr[:, 13])),
        float(np.var(arr[:, 14])),  float(np.var(arr[:, 15])),
    ]

#opencv overlay (can be toggled by pressing "m")
def draw_overlay(frame, face, w, h):
    for eye_idxs in (LEFT_EYE, RIGHT_EYE):
        pts = pts_xy(face, eye_idxs, w, h).astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=1)
        for pt in pts:
            cv2.circle(frame, tuple(pt), 2, (255, 255, 0), -1)

    for iris_idxs in (LEFT_IRIS, RIGHT_IRIS):
        iris_pts = pts_xy(face, iris_idxs, w, h)
        cx, cy = int(iris_pts[0, 0]), int(iris_pts[0, 1])
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    all_x = [lm.x for lm in face]
    all_y = [lm.y for lm in face]
    cv2.rectangle(frame,
                  (int(min(all_x) * w), int(min(all_y) * h)),
                  (int(max(all_x) * w), int(max(all_y) * h)),
                  (0, 255, 255), 1)

    R, eye_center_3d, _ = build_face_frame(face)
    z_axis = R[:, 2]
    origin = (int(eye_center_3d[0] * w), int(eye_center_3d[1] * h))
    tip = (int(origin[0] + z_axis[0] * 80), int(origin[1] + z_axis[1] * 80))
    cv2.arrowedLine(frame, origin, tip, (255, 0, 255), 2, tipLength=0.3)

#predictors
#loads baseline linear regression model from pkl
class LRPredictor:
    name = "logistic regression"
    MODEL_FILE = "baseline_model.pkl"

    def __init__(self):
        if not os.path.exists(self.MODEL_FILE):
            raise FileNotFoundError(
                f"{self.MODEL_FILE} not found — run: python baseline_model.py --save"
            )
        with open(self.MODEL_FILE, "rb") as f:
            d = pickle.load(f)
        self.clf, self.scaler, self.le = d["clf"], d["scaler"], d["le"]
        print(f"loaded logistic regression from {self.MODEL_FILE}")

    def predict(self, window):
        X = self.scaler.transform([aggregate_window(window)])
        pred = self.clf.predict(X)[0]
        prob = self.clf.predict_proba(X)[0]
        return self.le.inverse_transform([pred])[0], float(np.max(prob))

#load transformer model from pt and predict based on it
class TransformerPredictor:
    name = "transformer"
    MODEL_FILE = "transformer_model.pt"

    def __init__(self):
        if not os.path.exists(self.MODEL_FILE):
            raise FileNotFoundError(
                f"{self.MODEL_FILE} not found — run: python transformer_model.py --save"
            )
        self.device = torch.device("cpu")
        ck = torch.load(self.MODEL_FILE, map_location=self.device, weights_only=False)
        self.model = TransformerClassifier(
            input_dim=ck["input_dim"], seq_len=ck["seq_len"]
        ).to(self.device)
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()
        self.scaler = ck["scaler"]
        self.le = ck["le"]
        print(f"loaded transformer from {self.MODEL_FILE}")

    def predict(self, window):
        arr = np.array(window, dtype=np.float32)
        arr_scaled = self.scaler.transform(arr)
        tensor = torch.tensor(arr_scaled[np.newaxis], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        return self.le.classes_[pred_idx], float(np.max(probs))

#loads temporal CNN model from pt 
class CNNPredictor:
    name = "temporal CNN"
    MODEL_FILE = "cnn_model.pt"

    def __init__(self):
        if not os.path.exists(self.MODEL_FILE):
            raise FileNotFoundError(
                f"{self.MODEL_FILE} not found — run: python cnn_time_model.py --save"
            )
        self.device = torch.device("cpu")
        ck = torch.load(self.MODEL_FILE, map_location=self.device, weights_only=False)
        self.model = TemporalCNN(in_channels=ck["in_channels"]).to(self.device)
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()
        self.scaler = ck["scaler"]
        self.le = ck["le"]
        print(f"loaded temporal CNN from {self.MODEL_FILE}")

    def predict(self, window):
        arr = np.array(window, dtype=np.float32)          # (150, 16)
        arr_scaled = self.scaler.transform(arr)
        tensor = (
            torch.tensor(arr_scaled[np.newaxis], dtype=torch.float32)
            .permute(0, 2, 1)                             # (1, 16, 150)
            .to(self.device)
        )
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        return self.le.classes_[pred_idx], float(np.max(probs))



def main():
    parser = argparse.ArgumentParser()
    #helpful arguments for checking camera devices (especially useful on mac if camera defaults to iphone)
    parser.add_argument("--camera", type=int, default=0) #specify a camera device index
    parser.add_argument("--list-cameras", action="store_true")
    args = parser.parse_args()

    #list available camera devices
    if args.list_cameras:
        print("checking camera indices 0-5:")
        for i in range(6):
            cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                ok, frame = cap.read()
                status = f"ok ({frame.shape[1]}x{frame.shape[0]})" if ok else "opened but no frame"
                print(f"  [{i}] {status}")
            else:
                print(f"  [{i}] not available")
            cap.release()
        return

    predictors = []
    for cls in (TransformerPredictor, CNNPredictor, LRPredictor):
        try:
            predictors.append(cls())
        except FileNotFoundError as e:
            print(f"skipping {cls.name}: {e}")

    #model files must be run with --save flag to store model to be loaded here
    if not predictors:
        raise RuntimeError(
            "no saved models found — run each model file with --save first:\n"
            "  python transformer_model.py --save\n"
            "  python cnn_time_model.py --save\n"
            "  python baseline_model.py --save"
        )

    active_idx = 0
    print(f"\nactive model: {predictors[active_idx].name}  (Tab to switch)")
    print("Q: quit  |  M: toggle overlay  |  Tab: cycle model")

    print("loading mediapipe...")
    base = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {args.camera}")

    window = collections.deque(maxlen=WINDOW_SIZE)
    pred_history = collections.deque(maxlen=SMOOTH_WINDOW)
    label = "warming up..."
    conf = 0.0
    is_ready = False
    show_overlay = False
    last_face = None

    sample_interval = 1.0 / SAMPLE_FPS
    last_sample_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        if now - last_sample_t >= sample_interval:
            last_sample_t = now

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

            #extract frame features
            if result.face_landmarks and len(result.face_landmarks[0]) >= 478:
                last_face = result.face_landmarks[0]
                feats = extract_frame_features(last_face, w, h)
                if feats is not None:
                    window.append(feats)

            #make prediction using specified model
            if len(window) == WINDOW_SIZE:
                raw_label, raw_conf = predictors[active_idx].predict(window)
                pred_history.append((raw_label, raw_conf))
                counts = collections.Counter(l for l, _ in pred_history)
                label = counts.most_common(1)[0][0]
                conf = float(np.mean([c for l, c in pred_history if l == label]))
                is_ready = True

        color = (0, 200, 0) if label == "focused" else (0, 0, 220)

        if is_ready:
            display = f"{label}  {conf:.0%}"
        else:
            display = f"collecting... {len(window)}/{WINDOW_SIZE}"
            color = (180, 180, 180)

        h, w = frame.shape[:2]

        cv2.putText(frame, display, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

        bar_filled = int((len(window) / WINDOW_SIZE) * 200)
        cv2.rectangle(frame, (20, 68), (220, 80), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 68), (20 + bar_filled, 80), (150, 150, 150), -1)

        cv2.putText(frame, f"model: {predictors[active_idx].name}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if show_overlay and last_face is not None:
            draw_overlay(frame, last_face, w, h)

        cv2.imshow("Focus Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): #quit
            break
        elif key == ord("m"): #toggle face mesh feature overlay
            show_overlay = not show_overlay
        elif key == 9:  # Tab: switch model
            active_idx = (active_idx + 1) % len(predictors)
            pred_history.clear()
            is_ready = False
            label = "switching..."
            print(f"switched to: {predictors[active_idx].name}")

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
