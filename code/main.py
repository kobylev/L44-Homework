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
    SKELETON_ONLY_VIDEO_PATH, COMBINED_VIDEO_PATH,
    SPATIAL_ORDER_UNIQUE, SKELETON_LINKS
)
from .model import TSSCIProcessor

def parse_args():
    p = argparse.ArgumentParser(description="AI Expert: Multi-Person Pose TSSCI Pipeline")
    p.add_argument("--source", default=DEFAULT_VIDEO_SOURCE, help="Video file or webcam index")
    p.add_argument("--max-frames", type=int, default=None, help="Max frames to process (default: None for all)")
    p.add_argument("--no-preview", action="store_true", help="Disable real-time preview")
    return p.parse_args()


def merge_audio(video_path, source_path):
    """Merges audio from source_path into video_path."""
    if not os.path.exists(source_path) or isinstance(source_path, int):
        return
    
    print(f"[INFO] Merging audio from {source_path} into {video_path}...")
    try:
        temp_path = video_path.replace(".mp4", "_temp.mp4")
        if os.path.exists(video_path):
            os.rename(video_path, temp_path)
        else:
            print(f"[WARNING] video_path {video_path} not found for merge.")
            return
        
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

def play_tssci(tssci, person_idx, fps=24.0, save_path=None):
    """TSSCI Player: Animates the movement from the TSSCI image data for one person."""
    n_frames = tssci.shape[0]
    data = tssci.astype(np.float32) / 255.0
    so_idx = {lm_id: col for col, lm_id in enumerate(SPATIAL_ORDER_UNIQUE)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.patch.set_facecolor("white")
    ax_sk, ax_ts = axes[0], axes[1]

    # TSSCI strip
    ax_ts.imshow(tssci, aspect="auto", interpolation="nearest")
    ax_ts.set_title("TSSCI Image", color="black", fontsize=15, pad=15)
    ax_ts.set_xlabel("Key Points (DFS order)", color="black", fontsize=12)
    ax_ts.set_ylabel("Frame", color="black", fontsize=12)
    cursor_line = ax_ts.axhline(y=0, color="yellow", linewidth=2.5)

    # Skeleton canvas
    ax_sk.set_xlim(0, 1.0)
    ax_sk.set_ylim(1.0, 0)
    ax_sk.set_facecolor("black")
    ax_sk.set_title(f"Skeleton Playback (Person {person_idx+1})", color="black", fontsize=14)

    link_lines = []
    for (id_a, id_b, color) in SKELETON_LINKS:
        line, = ax_sk.plot([], [], "-", color=color, linewidth=2.5, alpha=0.9)
        link_lines.append((id_a, id_b, line))

    joint_scatter = ax_sk.scatter([], [], s=30, c="white", zorder=5, alpha=1.0)

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
        return [l[2] for l in link_lines] + [joint_scatter, cursor_line]

    ani = animation.FuncAnimation(fig, _update, frames=n_frames, interval=int(1000/fps), blit=True)
    plt.tight_layout()

    if save_path:
        print(f"[INFO] Saving combined video to {save_path}...")
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='AI Expert'), bitrate=4000)
        ani.save(save_path, writer=writer)
        plt.close(fig)
    else:
        plt.show()


def main():
    args = parse_args()
    
    source = args.source
    if source.isdigit():
        source = int(source)

    print(f"[INFO] Starting Multi-Person Pipeline: Source={source}")
    processor = TSSCIProcessor(num_poses=3)
    
    # Task 1 & 2: Multi-person extraction
    tssci_list, actual_fps = processor.build_tssci_multi_person(
        source, 
        max_frames=args.max_frames, 
        preview=not args.no_preview,
        output_video=VIDEO_OUTPUT_PATH
    )

    if tssci_list:
        # Step 3: Audio merge for standard videos
        merge_audio(VIDEO_OUTPUT_PATH, source)
        merge_audio(SKELETON_ONLY_VIDEO_PATH, source)

        for i, tssci in enumerate(tssci_list):
            if np.any(tssci):
                person_path = f"assets/tssci_person_{i+1}.png"
                cv2.imwrite(person_path, cv2.cvtColor(tssci, cv2.COLOR_RGB2BGR))
                print(f"[INFO] Saved Person {i+1} TSSCI: {person_path}")

                if i == 0:
                   # For the main person, save the combined video with high-quality dashboard
                   temp_combined = COMBINED_VIDEO_PATH.replace(".mp4", "_no_audio.mp4")
                   play_tssci(tssci, i, fps=actual_fps, save_path=temp_combined)
                   
                   # Merge audio from source into the temp video, saving to final path
                   print(f"[INFO] Merging audio into {COMBINED_VIDEO_PATH}...")
                   try:
                       with VideoFileClip(temp_combined) as video_clip:
                           with VideoFileClip(source) as source_clip:
                               if source_clip.audio is not None:
                                   final_clip = video_clip.with_audio(source_clip.audio)
                                   final_clip.write_videofile(COMBINED_VIDEO_PATH, codec="libx264", audio_codec="aac")
                               else:
                                   video_clip.write_videofile(COMBINED_VIDEO_PATH, codec="libx264")
                       if os.path.exists(temp_combined):
                           os.remove(temp_combined)
                   except Exception as e:
                       print(f"[WARNING] Could not merge audio for combined video: {e}")


                elif args.no_preview is False:
                   display_tssci(tssci, i)
                   play_tssci(tssci, i, fps=actual_fps)


    else:
        print("[ERROR] No people detected.")

if __name__ == "__main__":
    main()
