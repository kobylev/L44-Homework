# model.py
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from .config import SPATIAL_ORDER_UNIQUE, N_KPS, SKELETON_ONLY_VIDEO_PATH, COMBINED_VIDEO_PATH

class TSSCIProcessor:
    """
    Handles Multi-Person Skeleton Extraction (up to 3 people), 
    TSSCI Generation, and Video Annotation using MediaPipe Tasks.
    """
    def __init__(self, model_path='pose_landmarker.task', num_poses=3, min_conf=0.5):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose # Used for drawing connection constants

    def draw_skeletons(self, frame: np.ndarray, result, black_bg=False) -> np.ndarray:
        """Draw all detected skeletons on the frame or a black background."""
        if black_bg:
            annotated = np.zeros_like(frame)
        else:
            annotated = frame.copy()
            
        if not result.pose_landmarks:
            return annotated
            
        from .config import SKELETON_LINKS
        h, w = frame.shape[:2]
        
        # Color mapping BGR
        color_map = {
            "cyan": (255, 255, 0),
            "yellow": (0, 255, 255),
            "lime": (0, 255, 0),
            "orange": (0, 165, 255),
            "deepskyblue": (255, 191, 0),
            "tomato": (71, 99, 255)
        }

        for landmarks in result.pose_landmarks:
            # 1. Draw connections
            for (id_a, id_b, color_name) in SKELETON_LINKS:
                if id_a >= len(landmarks) or id_b >= len(landmarks):
                    continue
                lm_a = landmarks[id_a]
                lm_b = landmarks[id_b]
                
                # Use a slightly lower threshold to prevent flickering
                if lm_a.visibility > 0.4 and lm_b.visibility > 0.4:
                    pt_a = (int(lm_a.x * w), int(lm_a.y * h))
                    pt_b = (int(lm_b.x * w), int(lm_b.y * h))
                    color = color_map.get(color_name, (255, 255, 255))
                    cv2.line(annotated, pt_a, pt_b, color, 2)

            # 2. Draw joints
            for lm in landmarks:
                if lm.visibility > 0.4:
                    pt = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(annotated, pt, 3, (255, 255, 255), -1)
                    
        return annotated

    def extract_all_keypoints(self, result) -> list:
        """Extract (x, y, confidence) for all detected people."""
        frame_all_people = []
        if result.pose_landmarks:
            for landmarks in result.pose_landmarks:
                row = np.zeros((N_KPS, 3), dtype=np.float32)
                for col_idx, lm_id in enumerate(SPATIAL_ORDER_UNIQUE):
                    lm = landmarks[lm_id]
                    row[col_idx] = [lm.x, lm.y, lm.visibility]
                frame_all_people.append(row)
        return frame_all_people

    def build_tssci_multi_person(self, source, max_frames=200, preview=True, output_video=None):
        """Process video and build multi-person TSSCI data and skeleton-only video."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Source not found: {source}")

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_overlay = cv2.VideoWriter(output_video, fourcc, fps, (frame_w, frame_h)) if output_video else None
        writer_skel_only = cv2.VideoWriter(SKELETON_ONLY_VIDEO_PATH, fourcc, fps, (frame_w, frame_h))
        
        all_people_buffers = [[], [], []]
        frame_count = 0

        while True:
            if max_frames is not None and frame_count >= max_frames:
                break
            
            ok, frame = cap.read()
            if not ok:
                break

            rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(frame_count * (1000 / fps))
            
            result = self.landmarker.detect_for_video(rgb, timestamp_ms)

            # Standard Overlay
            annotated_overlay = self.draw_skeletons(frame, result, black_bg=False)
            if writer_overlay:
                writer_overlay.write(annotated_overlay)

            # Skeleton Only (Black Background)
            annotated_skel = self.draw_skeletons(frame, result, black_bg=True)
            writer_skel_only.write(annotated_skel)

            # Data extraction
            people_data = self.extract_all_keypoints(result)
            for i in range(3):
                if i < len(people_data):
                    all_people_buffers[i].append(people_data[i])
                else:
                    all_people_buffers[i].append(np.zeros((N_KPS, 3), dtype=np.float32))

            if preview:
                cv2.imshow("Multi-Person Tracing (Press q to quit)", annotated_overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

        cap.release()
        if writer_overlay: writer_overlay.release()
        writer_skel_only.release()
        cv2.destroyAllWindows()

        tssci_list = []
        for buffer in all_people_buffers:
            if buffer:
                data = np.stack(buffer, axis=0)
                tssci = (data * 255).clip(0, 255).astype(np.uint8)
                tssci_list.append(tssci)
        
        return tssci_list, fps
