import os
import csv
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MAX_FRAMES = 30
FEATURE_DIM = 6


def extract_frames_fps(video_path, fps=3, duration=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("could not open video")

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
    xs = []
    ys = []
    for x, y in eye_pts:
        xs.append(x)
        ys.append(y)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def pad_sequence(seq):
    if len(seq) >= MAX_FRAMES:
        return seq[:MAX_FRAMES]
    padding = [[0.0] * FEATURE_DIM] * (MAX_FRAMES - len(seq))
    return seq + padding


def process_video(video_path, label, landmarker, fps=3, duration=10):
    frames = extract_frames_fps(video_path, fps, duration)

    sequence = []

    for frame_bgr in frames:
        h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_img)
        if not result.face_landmarks:
            continue

        face = result.face_landmarks[0]

        left_pts = pts_xy(face, LEFT_EYE, w, h)
        right_pts = pts_xy(face, RIGHT_EYE, w, h)

        left_center = eye_center(left_pts)
        right_center = eye_center(right_pts)

        left_ear = eye_aspect_ratio(left_pts)
        right_ear = eye_aspect_ratio(right_pts)

        sequence.append([
            left_ear,
            right_ear,
            left_center[0],
            left_center[1],
            right_center[0],
            right_center[1]
        ])

    if len(sequence) == 0:
        return None

    sequence = pad_sequence(sequence)

    arr = np.array(sequence)

    agg = {
        "mean_left_ear": float(np.mean(arr[:, 0])),
        "mean_right_ear": float(np.mean(arr[:, 1])),
        "std_left_ear": float(np.std(arr[:, 0])),
        "std_right_ear": float(np.std(arr[:, 1])),
        "left_x_variance": float(np.var(arr[:, 2])),
        "left_y_variance": float(np.var(arr[:, 3])),
        "right_x_variance": float(np.var(arr[:, 4])),
        "right_y_variance": float(np.var(arr[:, 5])),
    }

    return sequence, agg


def main():
    videos_root = "videos"
    npz_path = "sequence_dataset.npz"
    csv_path = "dataset.csv"

    if os.path.isfile(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        X_existing = data["X"]
        y_existing = data["y"]

        if "names" in data.files:
            names_existing = data["names"].tolist()
        else:
            names_existing = []
    else:
        X_existing = []
        y_existing = []
        names_existing = []


    processed_set = set(names_existing)

    # load existing csv processed names
    csv_processed = set()
    if os.path.isfile(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                csv_processed.add(row["video"])

    base = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base,
        num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    new_sequences = []
    new_labels = []
    new_names = []
    new_csv_rows = []

    for label in os.listdir(videos_root):
        label_dir = os.path.join(videos_root, label)
        if not os.path.isdir(label_dir):
            continue

        for fname in os.listdir(label_dir):
            if not fname.lower().endswith((".mp4", ".mov", ".avi")):
                continue

            if fname in processed_set:
                continue

            video_path = os.path.join(label_dir, fname)
            print("processing:", video_path)

            result = process_video(video_path, label, landmarker)

            if result is None:
                print("no face detected, skipping")
                continue

            seq, agg = result

            new_sequences.append(seq)
            new_labels.append(label)
            new_names.append(fname)

            row = {"video": fname, "label": label}
            row.update(agg)
            new_csv_rows.append(row)

    landmarker.close()

    if len(new_sequences) == 0:
        print("no new videos found")
        return

    X_existing = np.atleast_3d(X_existing) if len(X_existing) > 0 else np.empty((0, MAX_FRAMES, FEATURE_DIM))
    y_existing = np.atleast_1d(y_existing) if len(y_existing) > 0 else np.empty((0,), dtype=object)

    new_X = np.array(new_sequences, dtype=np.float32)
    new_y = np.array(new_labels)

    X_all = np.concatenate((X_existing, new_X), axis=0)
    y_all = np.concatenate((y_existing, new_y), axis=0)

    names_all = names_existing + new_names

    np.savez(npz_path, X=X_all, y=y_all, names=np.array(names_all))

    write_header = not os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = None
        for row in new_csv_rows:
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                    write_header = False
            writer.writerow(row)

    print("dataset updated")
    print("sequence shape:", X_all.shape)


if __name__ == "__main__":
    main()
