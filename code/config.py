# config.py
# ---------------------------------------------------------------------------
# TSSCI keypoint ordering (right-hand-rule / continuous boundary walk)
# ---------------------------------------------------------------------------
# MediaPipe supplies 33 landmarks. We walk the body outline starting from
# the head, descending down the right side, across the feet, up the left
# side and back to the head – creating a spatially continuous 1-D sequence
# that preserves kinematic neighbours.

SPATIAL_ORDER = [
    # Head region (left to right then centre)
    3, 2, 1, 0, 4, 5, 6,    # L-eye-outer … R-eye-outer
    7, 9, 10, 8,             # ears & mouth
    # Left arm (shoulder → wrist → fingers)
    11, 13, 15, 17, 19, 21,
    # Right arm
    12, 14, 16, 18, 20, 22,
    # Torso centre
    11, 23, 24, 12,          # shoulder belt → hip belt
    # Left leg
    23, 25, 27, 29, 31,
    # Right leg
    24, 26, 28, 30, 32,
]

# Process spatial order to ensure uniqueness while preserving order
def get_unique_spatial_order():
    seen = set()
    unique_order = []
    for idx in SPATIAL_ORDER:
        if idx not in seen:
            unique_order.append(idx)
            seen.add(idx)
    return unique_order

SPATIAL_ORDER_UNIQUE = get_unique_spatial_order()
N_KPS = len(SPATIAL_ORDER_UNIQUE)

# Skeleton links for the player/visualization
SKELETON_LINKS = [
    # Head
    (0, 1, "cyan"), (0, 4, "cyan"),
    (1, 2, "cyan"), (2, 3, "cyan"),
    (4, 5, "cyan"), (5, 6, "cyan"),
    (9, 10, "cyan"),
    (7, 9, "cyan"), (8, 10, "cyan"),
    (3, 7, "cyan"), (6, 8, "cyan"),
    # Torso
    (11, 12, "yellow"), (11, 23, "yellow"), (12, 24, "yellow"),
    (23, 24, "yellow"),
    # Left arm
    (11, 13, "lime"), (13, 15, "lime"),
    (15, 17, "lime"), (15, 19, "lime"), (15, 21, "lime"),
    (17, 19, "lime"),
    # Right arm
    (12, 14, "orange"), (14, 16, "orange"),
    (16, 18, "orange"), (16, 20, "orange"), (16, 22, "orange"),
    (18, 20, "orange"),
    # Left leg
    (23, 25, "deepskyblue"), (25, 27, "deepskyblue"),
    (27, 29, "deepskyblue"), (27, 31, "deepskyblue"),
    (29, 31, "deepskyblue"),
    # Right leg
    (24, 26, "tomato"), (26, 28, "tomato"),
    (28, 30, "tomato"), (28, 32, "tomato"),
    (30, 32, "tomato"),
]

# Paths
DEFAULT_VIDEO_SOURCE = "assets/test_video.mp4"
TSSCI_OUTPUT_PATH = "assets/tssci.png"
VIDEO_OUTPUT_PATH = "assets/skeleton_overlay.mp4"
SKELETON_ONLY_VIDEO_PATH = "assets/skeleton_only.mp4"
FINAL_SKELETON_ONLY_WITH_AUDIO_PATH = "assets/skeleton_only_with_audio.mp4"
