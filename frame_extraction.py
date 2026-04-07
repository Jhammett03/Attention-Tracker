import os
import csv
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

MAX_FRAMES = 30
FEATURE_DIM = 6


def extract_frames_fps(video_path, fps=3, duration=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    frames = []
    dt = 1.0 / fps
    t = 0.0

    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, int(t * 1000))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
        t += dt

    cap.release()
    return frames


def pts_xy(landmarks, idxs, w, h):
    return np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in idxs],
        dtype=np.float32
    )


def eye_aspect_ratio(eye_pts):
    p1, p2, p3, p4, p5, p6 = eye_pts
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h  = np.linalg.norm(p1 - p4)
    return float((v1 + v2) / (2.0 * h + 1e-6))


def eye_center(eye_pts):
    return (
        float(np.mean(eye_pts[:, 0])),
        float(np.mean(eye_pts[:, 1]))
    )


def iris_center(iris_pts):
    return (
        float(np.mean(iris_pts[:, 0])),
        float(np.mean(iris_pts[:, 1]))
    )


def normalized_iris_position(eye_pts, iris_ctr):
    x_vals = eye_pts[:, 0]
    y_vals = eye_pts[:, 1]

    x_min = float(np.min(x_vals))
    x_max = float(np.max(x_vals))
    y_min = float(np.min(y_vals))
    y_max = float(np.max(y_vals))

    eye_w = max(x_max - x_min, 1e-6)
    eye_h = max(y_max - y_min, 1e-6)

    iris_x_norm = (iris_ctr[0] - x_min) / eye_w
    iris_y_norm = (iris_ctr[1] - y_min) / eye_h

    return iris_x_norm, iris_y_norm


def pad_sequence(seq, feature_dim):
    if len(seq) >= MAX_FRAMES:
        return seq[:MAX_FRAMES]
    padding = [[0.0] * feature_dim for _ in range(MAX_FRAMES - len(seq))]
    return seq + padding


def aggregate_old_features(arr):
    return {
        "mean_left_ear": float(np.mean(arr[:, 0])),
        "mean_right_ear": float(np.mean(arr[:, 1])),
        "std_left_ear": float(np.std(arr[:, 0])),
        "std_right_ear": float(np.std(arr[:, 1])),
        "left_x_variance": float(np.var(arr[:, 2])),
        "left_y_variance": float(np.var(arr[:, 3])),
        "right_x_variance": float(np.var(arr[:, 4])),
        "right_y_variance": float(np.var(arr[:, 5])),
    }


def aggregate_iris_features(arr):
    return {
        "mean_left_ear": float(np.mean(arr[:, 0])),
        "mean_right_ear": float(np.mean(arr[:, 1])),
        "std_left_ear": float(np.std(arr[:, 0])),
        "std_right_ear": float(np.std(arr[:, 1])),

        "mean_left_iris_x": float(np.mean(arr[:, 2])),
        "mean_left_iris_y": float(np.mean(arr[:, 3])),
        "mean_right_iris_x": float(np.mean(arr[:, 4])),
        "mean_right_iris_y": float(np.mean(arr[:, 5])),

        "var_left_iris_x": float(np.var(arr[:, 2])),
        "var_left_iris_y": float(np.var(arr[:, 3])),
        "var_right_iris_x": float(np.var(arr[:, 4])),
        "var_right_iris_y": float(np.var(arr[:, 5])),
    }


def process_video(video_path, label, landmarker, fps=3, duration=10):
    frames = extract_frames_fps(video_path, fps, duration)

    old_sequence = []
    iris_sequence = []

    for frame_bgr in frames:
        h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_img)
        if not result.face_landmarks:
            continue

        face = result.face_landmarks[0]

        left_eye_pts = pts_xy(face, LEFT_EYE, w, h)
        right_eye_pts = pts_xy(face, RIGHT_EYE, w, h)

        left_ear = eye_aspect_ratio(left_eye_pts)
        right_ear = eye_aspect_ratio(right_eye_pts)

        left_ctr = eye_center(left_eye_pts)
        right_ctr = eye_center(right_eye_pts)

        old_sequence.append([
            left_ear,
            right_ear,
            left_ctr[0],
            left_ctr[1],
            right_ctr[0],
            right_ctr[1]
        ])

        if len(face) >= 478:
            left_iris_pts = pts_xy(face, LEFT_IRIS, w, h)
            right_iris_pts = pts_xy(face, RIGHT_IRIS, w, h)

            left_iris_ctr = iris_center(left_iris_pts)
            right_iris_ctr = iris_center(right_iris_pts)

            left_iris_x_norm, left_iris_y_norm = normalized_iris_position(
                left_eye_pts, left_iris_ctr
            )
            right_iris_x_norm, right_iris_y_norm = normalized_iris_position(
                right_eye_pts, right_iris_ctr
            )

            iris_sequence.append([
                left_ear,
                right_ear,
                left_iris_x_norm,
                left_iris_y_norm,
                right_iris_x_norm,
                right_iris_y_norm
            ])

    if len(old_sequence) == 0:
        return None

    old_sequence = pad_sequence(old_sequence, FEATURE_DIM)
    old_arr = np.array(old_sequence, dtype=np.float32)
    old_agg = aggregate_old_features(old_arr)

    iris_result = None
    if len(iris_sequence) > 0:
        iris_sequence = pad_sequence(iris_sequence, FEATURE_DIM)
        iris_arr = np.array(iris_sequence, dtype=np.float32)
        iris_agg = aggregate_iris_features(iris_arr)
        iris_result = (iris_sequence, iris_agg)

    return (old_sequence, old_agg), iris_result


def load_existing_npz(npz_path):
    if os.path.isfile(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        X_existing = data["X"]
        y_existing = data["y"]
        names_existing = data["names"].tolist() if "names" in data.files else []
    else:
        X_existing = np.empty((0, MAX_FRAMES, FEATURE_DIM), dtype=np.float32)
        y_existing = np.empty((0,), dtype=object)
        names_existing = []

    return X_existing, y_existing, names_existing


def append_to_npz(npz_path, new_sequences, new_labels, new_names):
    X_existing, y_existing, names_existing = load_existing_npz(npz_path)

    if len(new_sequences) == 0:
        return

    new_X = np.array(new_sequences, dtype=np.float32)
    new_y = np.array(new_labels, dtype=object)

    X_all = np.concatenate((X_existing, new_X), axis=0)
    y_all = np.concatenate((y_existing, new_y), axis=0)
    names_all = names_existing + new_names

    np.savez(npz_path, X=X_all, y=y_all, names=np.array(names_all, dtype=object))


def append_to_csv(csv_path, rows):
    if len(rows) == 0:
        return

    write_header = not os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    videos_root = os.path.join(script_dir, "videos")

    old_npz_path = "./sequence_dataset.npz"
    old_csv_path = "./dataset.csv"

    iris_npz_path = "./sequence_dataset_iris.npz"
    iris_csv_path = "./dataset_iris.csv"

    _, _, old_names_existing = load_existing_npz(old_npz_path)
    _, _, iris_names_existing = load_existing_npz(iris_npz_path)

    old_processed = set(old_names_existing)
    iris_processed = set(iris_names_existing)

    base = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base,
        num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    new_old_sequences = []
    new_old_labels = []
    new_old_names = []
    new_old_csv_rows = []

    new_iris_sequences = []
    new_iris_labels = []
    new_iris_names = []
    new_iris_csv_rows = []

    for label in os.listdir(videos_root):
        label_dir = os.path.join(videos_root, label)
        if not os.path.isdir(label_dir):
            continue

        for fname in os.listdir(label_dir):
            if not fname.lower().endswith((".mp4", ".mov", ".avi")):
                continue

            old_needed = fname not in old_processed
            iris_needed = fname not in iris_processed

            if not old_needed and not iris_needed:
                continue

            video_path = os.path.join(label_dir, fname)
            print("processing:", video_path)

            result = process_video(video_path, label, landmarker)

            if result is None:
                print("no face detected, skipping")
                continue

            old_result, iris_result = result

            if old_needed:
                old_seq, old_agg = old_result
                new_old_sequences.append(old_seq)
                new_old_labels.append(label)
                new_old_names.append(fname)

                row = {"video": fname, "label": label}
                row.update(old_agg)
                new_old_csv_rows.append(row)

            if iris_needed and iris_result is not None:
                iris_seq, iris_agg = iris_result
                new_iris_sequences.append(iris_seq)
                new_iris_labels.append(label)
                new_iris_names.append(fname)

                row = {"video": fname, "label": label}
                row.update(iris_agg)
                new_iris_csv_rows.append(row)

    landmarker.close()

    append_to_npz(old_npz_path, new_old_sequences, new_old_labels, new_old_names)
    append_to_csv(old_csv_path, new_old_csv_rows)

    append_to_npz(iris_npz_path, new_iris_sequences, new_iris_labels, new_iris_names)
    append_to_csv(iris_csv_path, new_iris_csv_rows)

    print("done")
    print("new old dataset rows:", len(new_old_csv_rows))
    print("new iris dataset rows:", len(new_iris_csv_rows))


if __name__ == "__main__":
    main()