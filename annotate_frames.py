import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from frame_extraction import (
    LEFT_IRIS, RIGHT_IRIS,
    extract_frames_fps, pts_xy, iris_center,
    build_face_frame, iris_to_face_frame,
)

GAZE_ARROW_SCALE = 2.5


def draw_annotation(frame, w, h, R, left_iris_face, right_iris_face, left_iris_pts, right_iris_pts):
    for pt in np.vstack([left_iris_pts, right_iris_pts]):
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 200, 0), -1)

    # symmetric ±0.5 offsets cancel when averaged, leaving gaze deviation
    avg_face = (left_iris_face + right_iris_face) / 2.0
    gaze_face_3d = np.array([avg_face[0], avg_face[1], 0.0], dtype=np.float32)

    gaze_img_3d = R @ gaze_face_3d
    gaze_img_px = np.array([gaze_img_3d[0] * w, gaze_img_3d[1] * h], dtype=np.float32)

    for ctr in [iris_center(left_iris_pts), iris_center(right_iris_pts)]:
        start = (int(ctr[0]), int(ctr[1]))
        end = (
            int(ctr[0] + gaze_img_px[0] * GAZE_ARROW_SCALE),
            int(ctr[1] + gaze_img_px[1] * GAZE_ARROW_SCALE),
        )
        if np.linalg.norm(np.array(end) - np.array(start)) > 3:
            cv2.arrowedLine(frame, start, end, (0, 0, 220), 2, tipLength=0.15)
        else:
            cv2.circle(frame, start, 4, (0, 0, 220), -1)


def annotate_video(video_path, out_dir, landmarker):
    frames = extract_frames_fps(video_path)
    os.makedirs(out_dir, exist_ok=True)

    per_frame_data = []
    for frame_bgr in frames:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        if not result.face_landmarks or len(result.face_landmarks[0]) < 478:
            per_frame_data.append(None)
            continue

        face = result.face_landmarks[0]
        R, eye_center, inter_eye_dist = build_face_frame(face)
        left_iris_face = iris_to_face_frame(face[LEFT_IRIS[0]], R, eye_center, inter_eye_dist)
        right_iris_face = iris_to_face_frame(face[RIGHT_IRIS[0]], R, eye_center, inter_eye_dist)
        left_iris_pts = pts_xy(face, LEFT_IRIS, w, h)
        right_iris_pts = pts_xy(face, RIGHT_IRIS, w, h)
        per_frame_data.append((frame_bgr, h, w, R, left_iris_face, right_iris_face, left_iris_pts, right_iris_pts))

    # subtract per-video baseline so arrows show deviation from neutral
    valid = [d for d in per_frame_data if d is not None]
    baseline = np.mean([(d[4] + d[5]) / 2.0 for d in valid], axis=0) if valid else np.zeros(3, dtype=np.float32)

    for i, data in enumerate(per_frame_data):
        if data is None:
            cv2.imwrite(os.path.join(out_dir, f"frame_{i:02d}.jpg"), frames[i])
            continue

        frame_bgr, h, w, R, left_iris_face, right_iris_face, left_iris_pts, right_iris_pts = data
        draw_annotation(frame_bgr, w, h, R,
                        left_iris_face - baseline, right_iris_face - baseline,
                        left_iris_pts, right_iris_pts)
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:02d}.jpg"), frame_bgr)


def main():
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_root = os.path.join(script_dir, "frames_annotated")

    base = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(options)

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        stem = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(out_root, "debug", stem)
        print(f"annotating {video_path} -> {out_dir}")
        annotate_video(video_path, out_dir, landmarker)
    else:
        videos_root = os.path.join(script_dir, "videos")
        for label in os.listdir(videos_root):
            label_dir = os.path.join(videos_root, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if not fname.lower().endswith((".mp4", ".mov", ".avi")):
                    continue
                video_path = os.path.join(label_dir, fname)
                stem = os.path.splitext(fname)[0]
                out_dir = os.path.join(out_root, label, stem)
                print(f"annotating {video_path} -> {out_dir}")
                annotate_video(video_path, out_dir, landmarker)

    landmarker.close()
    print("done")


if __name__ == "__main__":
    main()
