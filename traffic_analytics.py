"""
Traffic Analytics with YOLOv8 + Object Tracking
-----------------------------------------------
Detects and tracks objects in a video, assigns each one a persistent ID,
and turns the raw detections into traffic metrics and a dashboard.

Modes
  people   : YOLOv8 detection, class "person"
  vehicles : YOLOv8 detection, car / motorcycle / bus / truck
  aerial   : YOLOv8-OBB (oriented boxes, trained on aerial imagery),
             small vehicle / large vehicle

Usage
  python traffic_analytics.py --video input.mp4 --mode aerial
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

MODES = {
    "people":   {"weights": "yolov8n.pt",     "classes": [0],          "obb": False},
    "vehicles": {"weights": "yolov8s.pt",     "classes": [2, 3, 5, 7], "obb": False},
    "aerial":   {"weights": "yolov8s-obb.pt", "classes": [9, 10],      "obb": True},
}


def track_video(video_path, mode, out_dir, max_side=1280, conf=0.25):
    """Run detection + tracking frame by frame.
    Writes an annotated video and returns a DataFrame with one row per detection."""
    cfg = MODES[mode]
    model = YOLO(cfg["weights"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Downscale so the longest side is max_side (faster, enough detail)
    scale = max_side / max(w, h)
    W, H = int(w * scale), int(h * scale)

    out_path = os.path.join(out_dir, f"tracked_{mode}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    rows, seen, frame_no = [], set(), 0
    track_kwargs = dict(persist=True, classes=cfg["classes"], conf=conf, verbose=False)
    if cfg["obb"]:
        track_kwargs["imgsz"] = 1024

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (W, H))
        res = model.track(frame, **track_kwargs)[0]

        boxes = res.obb if cfg["obb"] else res.boxes
        now = 0
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().tolist()
            for i, c, p in zip(ids, boxes.cls.int().tolist(), boxes.conf.tolist()):
                rows.append({
                    "frame": frame_no,
                    "time_sec": round(frame_no / fps, 2),
                    "id": i,
                    "type": res.names[c],
                    "conf": round(p, 2),
                })
            seen.update(ids)
            now = len(ids)

        img = res.plot()
        cv2.putText(img, f"Now: {now}  Total: {len(seen)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        writer.write(img)

        frame_no += 1
        if frame_no % 100 == 0:
            print(f"Progress: {frame_no}/{total}")

    cap.release()
    writer.release()
    print(f"Annotated video saved: {out_path}")
    return pd.DataFrame(rows), fps, frame_no


def summarize(df, fps, n_frames, min_seconds=0.5):
    """Aggregate detections per object, drop short-lived IDs (tracking noise),
    and compute headline metrics."""
    obj = df.groupby("id").agg(
        type=("type", lambda s: s.mode()[0]),
        first_sec=("time_sec", "min"),
        last_sec=("time_sec", "max"),
        frames=("frame", "count"),
        avg_conf=("conf", "mean"),
    ).reset_index()
    obj["seconds_on_screen"] = (obj["last_sec"] - obj["first_sec"]).round(2)

    # An ID seen for less than min_seconds is usually a tracking glitch
    clean = obj[obj["frames"] >= int(fps * min_seconds)].copy()

    per_frame = df.groupby("frame")["id"].nunique().reindex(range(n_frames), fill_value=0)

    summary = pd.DataFrame({
        "Metric": ["Video length (sec)", "Raw IDs", "Objects after cleaning",
                   "Removed (noise)", "Avg objects on screen", "Max objects on screen",
                   "Peak time (sec)", "Avg time on screen (sec)", "Avg confidence"],
        "Value": [round(n_frames / fps, 1), len(obj), len(clean), len(obj) - len(clean),
                  round(per_frame.mean(), 1), int(per_frame.max()),
                  round(per_frame.idxmax() / fps, 1),
                  round(clean["seconds_on_screen"].mean(), 2),
                  round(clean["avg_conf"].mean(), 2)],
    })
    by_type = clean.groupby("type").agg(
        count=("id", "count"),
        avg_seconds=("seconds_on_screen", "mean"),
        avg_conf=("avg_conf", "mean"),
    ).round(2)
    return summary, by_type, clean, per_frame


def dashboard(clean, per_frame, fps, path):
    """2x2 dashboard: load over time, type mix, dwell time, cumulative count."""
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Traffic Analytics Dashboard", fontsize=16, fontweight="bold")

    t = per_frame.index / fps
    ax[0, 0].plot(t, per_frame.values, color="tab:blue")
    ax[0, 0].fill_between(t, per_frame.values, alpha=0.2)
    ax[0, 0].set(title="Objects on screen over time", xlabel="Time (sec)", ylabel="Count")

    counts = clean["type"].value_counts()
    bars = ax[0, 1].bar(counts.index, counts.values)
    ax[0, 1].bar_label(bars)
    ax[0, 1].set_title("Object type")

    ax[1, 0].hist(clean["seconds_on_screen"], bins=15, color="tab:purple", edgecolor="white")
    ax[1, 0].set(title="Time on screen distribution", xlabel="Seconds", ylabel="Count")

    first = clean["first_sec"].sort_values().values
    ax[1, 1].step(first, range(1, len(first) + 1), where="post", color="tab:red")
    ax[1, 1].set(title="Cumulative unique objects", xlabel="Time (sec)", ylabel="Total")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Dashboard saved: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--mode", choices=MODES, default="aerial")
    ap.add_argument("--out", default="outputs", help="Output folder")
    ap.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df, fps, n = track_video(args.video, args.mode, args.out, conf=args.conf)
    if df.empty:
        print("No objects detected.")
        return
    df.to_csv(os.path.join(args.out, "detections.csv"), index=False)

    summary, by_type, clean, per_frame = summarize(df, fps, n)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print("\nSummary\n", summary.to_string(index=False))
    print("\nBy type\n", by_type.to_string())

    dashboard(clean, per_frame, fps, os.path.join(args.out, "dashboard.png"))


if __name__ == "__main__":
    main()
