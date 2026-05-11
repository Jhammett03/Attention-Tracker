import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from frame_extraction import extract_frames_fps
from realtime import draw_overlay


def annotate_video(video_path, out_dir, landmarker):
    frames = extract_frames_fps(video_path)
    os.makedirs(out_dir, exist_ok=True)

    for i, frame_bgr in enumerate(frames):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        if result.face_landmarks and len(result.face_landmarks[0]) >= 478:
            draw_overlay(frame_bgr, result.face_landmarks[0], w, h)

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
