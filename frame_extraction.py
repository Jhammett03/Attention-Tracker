import os
import csv
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

FACE_LEFT_CORNER  = 33
FACE_RIGHT_CORNER = 263
FACE_TOP = 10

MAX_FRAMES = 150
FEATURE_DIM = 6
GAZE_FEATURE_DIM = 16

def extract_frames_fps(video_path, fps=15, duration=10):
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
    return (float(np.mean(eye_pts[:, 0])), float(np.mean(eye_pts[:, 1])))


def iris_center(iris_pts):
    # landmark 0 is mediapipe's dedicated iris center, not a rim point
    return (float(iris_pts[0, 0]), float(iris_pts[0, 1]))


def normalized_iris_position(eye_pts, iris_ctr):
    x_min = float(np.min(eye_pts[:, 0]))
    x_max = float(np.max(eye_pts[:, 0]))
    y_min = float(np.min(eye_pts[:, 1]))
    y_max = float(np.max(eye_pts[:, 1]))

    eye_w = max(x_max - x_min, 1e-6)
    eye_h = max(y_max - y_min, 1e-6)

    return (iris_ctr[0] - x_min) / eye_w, (iris_ctr[1] - y_min) / eye_h


def build_face_frame(face):
    pts = np.array([(lm.x, lm.y, lm.z) for lm in face], dtype=np.float32)

    left_corner  = pts[FACE_LEFT_CORNER]
    right_corner = pts[FACE_RIGHT_CORNER]
    top_of_head  = pts[FACE_TOP]

    eye_center = (left_corner + right_corner) / 2.0
    inter_eye_dist = float(np.linalg.norm(right_corner - left_corner))

    x_axis = right_corner - left_corner
    x_axis /= np.linalg.norm(x_axis) + 1e-9

    y_approx = top_of_head - eye_center
    y_approx /= np.linalg.norm(y_approx) + 1e-9
    y_axis = y_approx - np.dot(y_approx, x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis) + 1e-9

    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis) + 1e-9

    R = np.column_stack((x_axis, y_axis, z_axis))
    return R, eye_center, inter_eye_dist


def iris_to_face_frame(iris_lm, R, eye_center, inter_eye_dist):
    p = np.array([iris_lm.x, iris_lm.y, iris_lm.z], dtype=np.float32)
    coords = R.T @ (p - eye_center)
    if inter_eye_dist > 1e-7:
        coords /= inter_eye_dist
    return coords


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


def aggregate_gaze_features(arr):
    return {
        "mean_left_ear": float(np.mean(arr[:, 0])),
        "mean_right_ear": float(np.mean(arr[:, 1])),
        "std_left_ear": float(np.std(arr[:, 0])),
        "std_right_ear": float(np.std(arr[:, 1])),
        "mean_left_gaze_x": float(np.mean(arr[:, 2])),
        "mean_left_gaze_y": float(np.mean(arr[:, 3])),
        "mean_right_gaze_x": float(np.mean(arr[:, 4])),
        "mean_right_gaze_y": float(np.mean(arr[:, 5])),
        "var_left_gaze_x": float(np.var(arr[:, 2])),
        "var_left_gaze_y": float(np.var(arr[:, 3])),
        "var_right_gaze_x": float(np.var(arr[:, 4])),
        "var_right_gaze_y": float(np.var(arr[:, 5])),
        "mean_head_x": float(np.mean(arr[:, 6])),
        "mean_head_y": float(np.mean(arr[:, 7])),
        "var_head_x": float(np.var(arr[:, 6])),
        "var_head_y": float(np.var(arr[:, 7])),
        "mean_bb_tl_x": float(np.mean(arr[:, 8])),
        "mean_bb_tl_y": float(np.mean(arr[:, 9])),
        "mean_bb_tr_x": float(np.mean(arr[:, 10])),
        "mean_bb_tr_y": float(np.mean(arr[:, 11])),
        "mean_bb_bl_x": float(np.mean(arr[:, 12])),
        "mean_bb_bl_y": float(np.mean(arr[:, 13])),
        "mean_bb_br_x": float(np.mean(arr[:, 14])),
        "mean_bb_br_y": float(np.mean(arr[:, 15])),
        "var_bb_tl_x": float(np.var(arr[:, 8])),
        "var_bb_tl_y": float(np.var(arr[:, 9])),
        "var_bb_tr_x": float(np.var(arr[:, 10])),
        "var_bb_tr_y": float(np.var(arr[:, 11])),
        "var_bb_bl_x": float(np.var(arr[:, 12])),
        "var_bb_bl_y": float(np.var(arr[:, 13])),
        "var_bb_br_x": float(np.var(arr[:, 14])),
        "var_bb_br_y": float(np.var(arr[:, 15])),
    }


def process_video(video_path, label, landmarker, fps=15, duration=10):
    frames = extract_frames_fps(video_path, fps, duration)

    old_sequence = []
    iris_sequence = []
    gaze_sequence = []

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

        old_sequence.append([left_ear, right_ear, left_ctr[0], left_ctr[1], right_ctr[0], right_ctr[1]])

        if len(face) >= 478:
            left_iris_pts = pts_xy(face, LEFT_IRIS, w, h)
            right_iris_pts = pts_xy(face, RIGHT_IRIS, w, h)

            left_iris_ctr = iris_center(left_iris_pts)
            right_iris_ctr = iris_center(right_iris_pts)

            lx, ly = normalized_iris_position(left_eye_pts, left_iris_ctr)
            rx, ry = normalized_iris_position(right_eye_pts, right_iris_ctr)

            iris_sequence.append([left_ear, right_ear, lx, ly, rx, ry])

            R, eye_center_3d, inter_eye_dist = build_face_frame(face)
            left_gaze = iris_to_face_frame(face[LEFT_IRIS[0]], R, eye_center_3d, inter_eye_dist)
            right_gaze = iris_to_face_frame(face[RIGHT_IRIS[0]], R, eye_center_3d, inter_eye_dist)

            z_axis = R[:, 2]

            all_x = [lm.x for lm in face]
            all_y = [lm.y for lm in face]
            bb_min_x, bb_max_x = float(min(all_x)), float(max(all_x))
            bb_min_y, bb_max_y = float(min(all_y)), float(max(all_y))

            gaze_sequence.append([
                left_ear, right_ear,
                float(left_gaze[0]), float(left_gaze[1]),
                float(right_gaze[0]), float(right_gaze[1]),
                float(z_axis[0]), float(z_axis[1]),
                bb_min_x, bb_min_y,
                bb_max_x, bb_min_y,
                bb_min_x, bb_max_y,
                bb_max_x, bb_max_y,
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
        iris_result = (iris_sequence, aggregate_iris_features(iris_arr))

    gaze_result = None
    if len(gaze_sequence) > 0:
        gaze_sequence = pad_sequence(gaze_sequence, GAZE_FEATURE_DIM)
        gaze_arr = np.array(gaze_sequence, dtype=np.float32)
        gaze_result = (gaze_sequence, aggregate_gaze_features(gaze_arr))

    return (old_sequence, old_agg), iris_result, gaze_result


def load_existing_npz(npz_path, feature_dim=FEATURE_DIM):
    if os.path.isfile(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        X_existing = data["X"]
        y_existing = data["y"]
        names_existing = data["names"].tolist() if "names" in data.files else []
    else:
        X_existing = np.empty((0, MAX_FRAMES, feature_dim), dtype=np.float32)
        y_existing = np.empty((0,), dtype=object)
        names_existing = []
    return X_existing, y_existing, names_existing


def append_to_npz(npz_path, new_sequences, new_labels, new_names):
    if len(new_sequences) == 0:
        return
    new_X = np.array(new_sequences, dtype=np.float32)
    feature_dim = new_X.shape[2]
    X_existing, y_existing, names_existing = load_existing_npz(npz_path, feature_dim)
    new_y = np.array(new_labels, dtype=object)
    X_all = np.concatenate((X_existing, new_X), axis=0)
    y_all = np.concatenate((y_existing, new_y), axis=0)
    np.savez(npz_path, X=X_all, y=y_all, names=np.array(names_existing + new_names, dtype=object))


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
    gaze_npz_path = "./sequence_dataset_gaze.npz"
    gaze_csv_path = "./dataset_gaze.csv"

    _, _, old_names = load_existing_npz(old_npz_path)
    _, _, iris_names = load_existing_npz(iris_npz_path)
    _, _, gaze_names = load_existing_npz(gaze_npz_path, GAZE_FEATURE_DIM)

    old_processed = set(old_names)
    iris_processed = set(iris_names)
    gaze_processed = set(gaze_names)

    base = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(options)

    new_old_seqs, new_old_labels, new_old_names, new_old_rows = [], [], [], []
    new_iris_seqs, new_iris_labels, new_iris_names, new_iris_rows = [], [], [], []
    new_gaze_seqs, new_gaze_labels, new_gaze_names, new_gaze_rows = [], [], [], []

    for label in os.listdir(videos_root):
        label_dir = os.path.join(videos_root, label)
        if not os.path.isdir(label_dir):
            continue

        for fname in os.listdir(label_dir):
            if not fname.lower().endswith((".mp4", ".mov", ".avi")):
                continue

            old_needed = fname not in old_processed
            iris_needed = fname not in iris_processed
            gaze_needed = fname not in gaze_processed

            if not old_needed and not iris_needed and not gaze_needed:
                continue

            video_path = os.path.join(label_dir, fname)
            print("processing:", video_path)

            result = process_video(video_path, label, landmarker)
            if result is None:
                print("no face detected, skipping")
                continue

            old_result, iris_result, gaze_result = result

            if old_needed:
                old_seq, old_agg = old_result
                new_old_seqs.append(old_seq)
                new_old_labels.append(label)
                new_old_names.append(fname)
                new_old_rows.append({"video": fname, "label": label, **old_agg})

            if iris_needed and iris_result is not None:
                iris_seq, iris_agg = iris_result
                new_iris_seqs.append(iris_seq)
                new_iris_labels.append(label)
                new_iris_names.append(fname)
                new_iris_rows.append({"video": fname, "label": label, **iris_agg})

            if gaze_needed and gaze_result is not None:
                gaze_seq, gaze_agg = gaze_result
                new_gaze_seqs.append(gaze_seq)
                new_gaze_labels.append(label)
                new_gaze_names.append(fname)
                new_gaze_rows.append({"video": fname, "label": label, **gaze_agg})

    landmarker.close()

    append_to_npz(old_npz_path, new_old_seqs, new_old_labels, new_old_names)
    append_to_csv(old_csv_path, new_old_rows)

    append_to_npz(iris_npz_path, new_iris_seqs, new_iris_labels, new_iris_names)
    append_to_csv(iris_csv_path, new_iris_rows)

    append_to_npz(gaze_npz_path, new_gaze_seqs, new_gaze_labels, new_gaze_names)
    append_to_csv(gaze_csv_path, new_gaze_rows)

    print("done")
    print("new old rows:", len(new_old_rows))
    print("new iris rows:", len(new_iris_rows))
    print("new gaze rows:", len(new_gaze_rows))


if __name__ == "__main__":
    main()
