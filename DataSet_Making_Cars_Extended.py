import os
import numpy as np
from PIL import Image
from scipy.ndimage import label
import cv2

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
DATASET_ROOT = "/data/OriginalDataset"              # Original 4K RGB images + labels
BASE         = "/data/UAVid_cars_dataset"            # Pre-curated 512×512 crops (Images + Labels/masks)
SAVE_ROOT    = "/data/New_Dataset_Cars_Extended"     # Output directory
DEBUG_DIR    = "/data/New_Dataset_Cars_Extended/debug" # Debug overlays

os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

TARGET_SIZE = 512

# ──────────────────────────────────────────────────────────────────────
# Car colours in the original UAVid labels
# ──────────────────────────────────────────────────────────────────────
CAR_COLORS = [
    np.array([64, 0, 128]),      # vehicle type 1
    np.array([192, 0, 192]),     # vehicle type 2
]


def build_car_mask(label_rgb):
    """Build a binary mask of ALL car pixels from an RGB label image."""
    car_mask = np.zeros(label_rgb.shape[:2], dtype=bool)
    for color in CAR_COLORS:
        car_mask |= np.all(label_rgb == color, axis=-1)
    return car_mask


def get_bbox(instance_mask):
    """Return (y_min, y_max, x_min, x_max) of a binary mask."""
    ys, xs = np.where(instance_mask)
    return ys.min(), ys.max(), xs.min(), xs.max()


def save_debug_overlay(rgb_512, mask_512, save_path, alpha=0.45):
    """
    Save a side-by-side debug image:
      LEFT  = plain RGB (512×512)
      RIGHT = RGB with blue mask overlay
    """
    overlay = rgb_512.copy()
    blue = np.array([60, 60, 255], dtype=np.uint8)  # blue tint
    car_pixels = mask_512 > 0
    overlay[car_pixels] = (
        overlay[car_pixels].astype(np.float32) * (1 - alpha)
        + blue.astype(np.float32) * alpha
    ).astype(np.uint8)

    side_by_side = np.concatenate([rgb_512, overlay], axis=1)  # 512 × 1024
    Image.fromarray(side_by_side).save(save_path)


# ──────────────────────────────────────────────────────────────────────
# Core per-image processing
# ──────────────────────────────────────────────────────────────────────
def process_single_image_extended(
    rgb_path, label_path, save_dir, save_labels_dir,
    image_id, curated_labels_dir
):
    """
    For every car instance found in the original label:
      – skip if no curated mask exists (deleted for quality)
      – expand the bounding box by ±h / ±w  (zoom-out on RGB)
      – reverse the proportional scaling of the curated 512×512 mask,
        place it in a canvas matching the RGB crop, then resize both
        identically to 512×512  →  guaranteed alignment
    """
    rgb       = np.array(Image.open(rgb_path).convert("RGB"))
    label_rgb = np.array(Image.open(label_path).convert("RGB"))
    H, W      = rgb.shape[:2]

    car_mask = build_car_mask(label_rgb)
    labeled_mask, num_objects = label(car_mask)

    if num_objects == 0:
        return

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_labels_dir, exist_ok=True)
    car_idx = 1

    for obj_id in range(1, num_objects + 1):
        car_name = f"car_{car_idx:03d}.png"

        # ── 1. Check if curated mask exists ──────────────────────
        curated_mask_path = os.path.join(curated_labels_dir, car_name)
        if not os.path.exists(curated_mask_path):
            car_idx += 1          # keep numbering in sync
            continue

        # ── 2. Original bounding box from the label ──────────────
        instance_mask = (labeled_mask == obj_id)
        y_min, y_max, x_min, x_max = get_bbox(instance_mask)

        # Tight crop dimensions (must match the original notebook's
        # crop [y_min:y_max+1, x_min:x_max+1])
        h_tight = y_max - y_min + 1
        w_tight = x_max - x_min + 1

        # ── 3. Expanded bbox (±h_tight / ±w_tight) ──────────────
        exp_y_min = y_min - h_tight
        exp_y_max = y_max + 1 + h_tight   # +1 because y_max is inclusive
        exp_x_min = x_min - w_tight
        exp_x_max = x_max + 1 + w_tight

        # Clamp to image boundaries
        clamped_y_min = max(0, exp_y_min)
        clamped_y_max = min(H, exp_y_max)
        clamped_x_min = max(0, exp_x_min)
        clamped_x_max = min(W, exp_x_max)

        # ── 4. Crop the expanded RGB ─────────────────────────────
        rgb_crop = rgb[clamped_y_min:clamped_y_max,
                       clamped_x_min:clamped_x_max]
        exp_h = clamped_y_max - clamped_y_min
        exp_w = clamped_x_max - clamped_x_min

        # ── 5. Reverse curated mask back to original pixel space ─
        #
        # The curated mask (512×512) was created by:
        #   tight crop (h_tight × w_tight) → proportional resize
        #   → center-pad to 512×512
        #
        # We reverse that: extract the content, resize back to
        # h_tight × w_tight, then place it in a canvas matching
        # the RGB crop size.  Both get the same final resize.
        # ─────────────────────────────────────────────────────────
        curated_mask = np.array(Image.open(curated_mask_path).convert("L"))
        curated_mask = (curated_mask > 127).astype(np.uint8)

        # Reproduce the original proportional scale + padding
        scale = TARGET_SIZE / max(h_tight, w_tight)
        content_h = int(round(h_tight * scale))
        content_w = int(round(w_tight * scale))

        pad_top_orig  = (TARGET_SIZE - content_h) // 2
        pad_left_orig = (TARGET_SIZE - content_w) // 2

        # Extract just the proportionally-scaled car region
        mask_content = curated_mask[
            pad_top_orig : pad_top_orig + content_h,
            pad_left_orig : pad_left_orig + content_w,
        ]

        # Resize back to original tight-crop pixel dimensions
        mask_tight = cv2.resize(
            mask_content,
            (w_tight, h_tight),          # (width, height) for cv2
            interpolation=cv2.INTER_NEAREST,
        )

        # ── 6. Place mask in expanded-crop-sized canvas ──────────
        # The tight bbox sits at a known offset in the expanded crop
        tight_y_in_crop = y_min - clamped_y_min
        tight_x_in_crop = x_min - clamped_x_min

        mask_canvas = np.zeros((exp_h, exp_w), dtype=np.uint8)
        mask_canvas[
            tight_y_in_crop : tight_y_in_crop + h_tight,
            tight_x_in_crop : tight_x_in_crop + w_tight,
        ] = mask_tight

        # ── 7. Resize both to 512×512 ────────────────────────────
        # Both have the same spatial dimensions (exp_h × exp_w)
        # so the same resize produces guaranteed alignment
        rgb_final = cv2.resize(
            rgb_crop,
            (TARGET_SIZE, TARGET_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

        mask_final = cv2.resize(
            mask_canvas,
            (TARGET_SIZE, TARGET_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

        # ── 8. Save ──────────────────────────────────────────────
        Image.fromarray(rgb_final).save(
            os.path.join(save_dir, car_name)
        )
        Image.fromarray((mask_final * 255).astype(np.uint8)).save(
            os.path.join(save_labels_dir, car_name)
        )

        car_idx += 1


# ──────────────────────────────────────────────────────────────────────
# Sequence-level driver
# ──────────────────────────────────────────────────────────────────────
def process_sequence_extended(seq_name, split):
    """Walk every original image in a sequence and process all cars."""
    rgb_dir   = os.path.join(DATASET_ROOT, split, seq_name, "Images")
    label_dir = os.path.join(DATASET_ROOT, split, seq_name, "Labels")

    if not os.path.exists(rgb_dir) or not os.path.exists(label_dir):
        print(f"⚠️  Skipping {split}/{seq_name} — dirs not found")
        return

    image_files = sorted(os.listdir(rgb_dir))

    for img_name in image_files:
        rgb_path   = os.path.join(rgb_dir, img_name)
        label_path = os.path.join(label_dir, img_name)
        image_id   = os.path.splitext(img_name)[0]

        car_folder = f"CarsFromimage{image_id}"

        # Curated masks live here
        curated_labels_dir = os.path.join(
            BASE, split, seq_name, "Labels", car_folder
        )

        # If no curated folder at all → no cars to process
        if not os.path.isdir(curated_labels_dir):
            continue

        # Output directories
        save_dir = os.path.join(
            SAVE_ROOT, split, seq_name, "Images", car_folder
        )
        save_labels_dir = os.path.join(
            SAVE_ROOT, split, seq_name, "Labels", car_folder
        )

        process_single_image_extended(
            rgb_path, label_path, save_dir, save_labels_dir,
            image_id, curated_labels_dir,
        )


# ──────────────────────────────────────────────────────────────────────
# Run for seq4 (train split) only — extend later
# ──────────────────────────────────────────────────────────────────────
# process_sequence_extended("seq4", "train")
# print("✅ Extended dataset built for train/seq4")

# ──────────────────────────────────────────────────────────────────────
# TO RUN ON ALL SEQUENCES (uncomment when ready):
# ──────────────────────────────────────────────────────────────────────
TRAIN_SEQS = ["seq1", "seq4", "seq6", "seq9", "seq10", "seq11",
              "seq12", "seq14", "seq32", "seq33", "seq35"]
VAL_SEQS   = ["seq18", "seq20"]
TEST_SEQS  = ["seq21", "seq28", "seq37"]

for seq in TRAIN_SEQS:
    process_sequence_extended(seq, "train")
for seq in VAL_SEQS:
    process_sequence_extended(seq, "valid")
for seq in TEST_SEQS:
    process_sequence_extended(seq, "test")

print("✅ New Extended Dataset Fully Built")
