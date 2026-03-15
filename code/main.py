# main.py
import argparse
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from moviepy import VideoFileClip, AudioFileClip

from .config import (
    DEFAULT_VIDEO_SOURCE, TSSCI_OUTPUT_PATH, VIDEO_OUTPUT_PATH,
    SPATIAL_ORDER_UNIQUE, SKELETON_LINKS
)
from .model import TSSCIProcessor

def parse_args():
    p = argparse.ArgumentParser(description="AI Expert: Multi-Person Pose TSSCI Pipeline")
    p.add_argument("--source", default=DEFAULT_VIDEO_SOURCE, help="Video file or webcam index")
    p.add_argument("--max-frames", type=int, default=None, help="Max frames to process (default: None for all)")
    p.add_argument("--fps", type=float, default=24.0, help="Playback FPS")
    p.add_argument("--no-preview", action="store_true", help="Disable real-time preview")
    return p.parse_args()

def merge_audio(video_path, source_path):
    """Merges audio from source_path into video_path."""
    if not os.path.exists(source_path) or isinstance(source_path, int):
        return
    
    print(f"[INFO] Merging audio from {source_path}...")
    try:
        temp_path = video_path.replace(".mp4", "_temp.mp4")
        os.rename(video_path, temp_path)
        
        with VideoFileClip(temp_path) as video_clip:
            with VideoFileClip(source_path) as source_clip:
                if source_clip.audio is not None:
                    final_clip = video_clip.with_audio(source_clip.audio)
                    final_clip.write_videofile(video_path, codec="libx264", audio_codec="aac")
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"[WARNING] Could not merge audio: {e}")

def display_tssci(tssci, person_idx):
    """Visualizes the abstract TSSCI image for a specific person."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(tssci, aspect="auto", interpolation="nearest")
    ax.set_title(f"TSSCI (Person {person_idx+1}) - Temporal Spatial Skeleton Color Image", fontsize=13)
    ax.set_xlabel("Keypoints (Spatial Order)", fontsize=11)
    ax.set_ylabel("Frames (Time)", fontsize=11)
    plt.tight_layout()
    plt.show()

def play_tssci(tssci, person_idx, fps=24.0):
    """TSSCI Player: Animates the movement from the TSSCI image data for one person."""
    n_frames = tssci.shape[0]
    data = tssci.astype(np.float32) / 255.0
    so_idx = {lm_id: col for col, lm_id in enumerate(SPATIAL_ORDER_UNIQUE)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("#1a1a2e")
    ax_sk, ax_ts = axes[0], axes[1]

    # TSSCI strip
    ax_ts.imshow(tssci, aspect="auto", interpolation="nearest")
    ax_ts.set_title(f"Person {person_idx+1} TSSCI", color="white")
    cursor_line = ax_ts.axhline(y=0, color="white", linewidth=1.5, linestyle="--")

    # Skeleton canvas
    ax_sk.set_xlim(0, 1.0)
    ax_sk.set_ylim(1.0, 0)
    ax_sk.set_facecolor("#0f0f23")
    ax_sk.set_title(f"Skeleton Playback (Person {person_idx+1})", color="white")

    link_lines = []
    for (id_a, id_b, color) in SKELETON_LINKS:
        line, = ax_sk.plot([], [], "-", color=color, linewidth=2, alpha=0.85)
        link_lines.append((id_a, id_b, line))

    joint_scatter = ax_sk.scatter([], [], s=20, c="white", zorder=5, alpha=0.9)

    def _update(frame_idx):
        row = data[frame_idx]
        for (id_a, id_b, line) in link_lines:
            col_a, col_b = so_idx.get(id_a), so_idx.get(id_b)
            if col_a is not None and col_b is not None:
                if row[col_a, 2] > 0.3 and row[col_b, 2] > 0.3:
                    line.set_data([row[col_a, 0], row[col_b, 0]], [row[col_a, 1], row[col_b, 1]])
                else:
                    line.set_data([], [])
        
        xs = [row[so_idx[i], 0] for i in range(33) if row[so_idx[i], 2] > 0.3]
        ys = [row[so_idx[i], 1] for i in range(33) if row[so_idx[i], 2] > 0.3]
        joint_scatter.set_offsets(list(zip(xs, ys)) if xs else np.empty((0, 2)))
        cursor_line.set_ydata([frame_idx, frame_idx])
        return link_lines[0][2], joint_scatter, cursor_line

    ani = animation.FuncAnimation(fig, _update, frames=n_frames, interval=int(1000/fps), blit=False)
    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    
    source = args.source
    if source.isdigit():
        source = int(source)

    print(f"[INFO] Starting Multi-Person Pipeline: Source={source}")
    processor = TSSCIProcessor(num_poses=3)
    
    # Task 1 & 2: Multi-person extraction
    tssci_list = processor.build_tssci_multi_person(
        source, 
        max_frames=args.max_frames, 
        preview=not args.no_preview,
        output_video=VIDEO_OUTPUT_PATH
    )

    if tssci_list:
        # Step 3: Audio merge for result video
        merge_audio(VIDEO_OUTPUT_PATH, source)

        for i, tssci in enumerate(tssci_list):
            if np.any(tssci):
                person_path = f"assets/tssci_person_{i+1}.png"
                cv2.imwrite(person_path, cv2.cvtColor(tssci, cv2.COLOR_RGB2BGR))
                print(f"[INFO] Saved Person {i+1} TSSCI: {person_path}")

                if i == 0 or args.no_preview is False:
                   display_tssci(tssci, i)
                   play_tssci(tssci, i, fps=args.fps)
    else:
        print("[ERROR] No people detected.")

if __name__ == "__main__":
    main()
