import numpy as np
import torch
from PIL import Image
import io
import base64


def run_promptcnn(model, image_np: np.ndarray, meta: dict) -> np.ndarray:
    from model_utils import preprocess_image
    image_size = meta["image_size"]
    t  = preprocess_image(image_np, image_size)
    with torch.no_grad():
        mu = model(t)
    pts = mu[0].cpu().float().clone()
    pts[:, 0] *= image_size
    pts[:, 1] *= image_size
    return pts.numpy().astype(np.float32)


def run_sam(sam_model, processor, image_pil: Image.Image,
            point_coords: np.ndarray, point_labels: np.ndarray) -> np.ndarray:
    image_np = np.array(image_pil)
    inference_state = processor.set_image(image_pil)

    point_coords = np.asarray(point_coords, dtype=np.float32)[None, ...]
    point_labels = np.asarray(point_labels, dtype=np.int32)[None, ...]

    sam_masks, scores, _ = sam_model.predict_inst(
        inference_state,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    best_mask = sam_masks[np.argmax(scores)]
    h, w = image_np.shape[:2]
    mask_resized = np.array(
        Image.fromarray(best_mask.astype(np.uint8)).resize((w, h), resample=Image.NEAREST)
    )
    return mask_resized


def render_overlay(image_np: np.ndarray, sam_mask: np.ndarray,
                   point_coords: np.ndarray, point_labels: np.ndarray,
                   mask_color=(0, 120, 255), alpha=0.45) -> np.ndarray:
    img = image_np.astype(np.float32).copy()
    mask = sam_mask.astype(bool)

    if mask.shape != img.shape[:2]:
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (img.shape[1], img.shape[0]), resample=Image.NEAREST
        )
        mask = np.array(mask_pil) > 127

    overlay = img.copy()
    overlay[mask] = (
        (1 - alpha) * img[mask] +
        alpha * np.array(mask_color, dtype=np.float32)
    )

    result = overlay.astype(np.uint8)
    result = _draw_points(result, point_coords, point_labels)
    return result


def _draw_points(img: np.ndarray, coords: np.ndarray,
                 labels: np.ndarray, radius: int = 10) -> np.ndarray:
    try:
        from PIL import ImageDraw
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        for (x, y), lbl in zip(coords, labels):
            color = (0, 220, 0) if lbl == 1 else (220, 0, 0)
            border = (0, 80, 0) if lbl == 1 else (80, 0, 0)
            x, y = int(round(x)), int(round(y))
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                         fill=color, outline=border, width=2)
        return np.array(pil)
    except Exception:
        return img


def numpy_to_base64(img_np: np.ndarray) -> str:
    pil = Image.fromarray(img_np.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def scale_points_to_sam(coords: list[dict], display_w: int, display_h: int,
                         sam_w: int, sam_h: int) -> np.ndarray:
    """Transforma coordonatele din spatiul imaginii afisate in spatiul SAM."""
    result = []
    for pt in coords:
        x = pt["x"] / display_w * sam_w
        y = pt["y"] / display_h * sam_h
        result.append([x, y])
    return np.array(result, dtype=np.float32)

VEHICLE_CLASSES = {4, 5, 6, 7, 8, 9}

MASK_COLORS = [
    (  0, 200, 255, 160),
    (255, 130,   0, 160),
    (100, 255, 100, 160),
    (255,  50, 180, 160),
    (200, 255,  50, 160),
    (150,  80, 255, 160),
    (255, 210,   0, 160),
    ( 50, 235, 200, 160),
]


def crop_and_pad(image_np: np.ndarray,
                 car_x: int, car_y: int, car_w: int, car_h: int,
                 target_size: int = 512, context_factor: float = 0.5) -> tuple:
    """
    Extinde bbox-ul vehiculului cu un context proportional pe fiecare latura,
    apoi redimensioneaza zona la target_size x target_size. Returneaza crop-ul si
    coordonatele zonei extinse, necesare pentru reproiectarea mastii.
    """
    H_img, W_img = image_np.shape[:2]

    x_min, y_min = int(car_x), int(car_y)
    x_max = x_min + int(car_w) - 1
    y_max = y_min + int(car_h) - 1
    h_tight = y_max - y_min + 1
    w_tight = x_max - x_min + 1

    margin_y = int(round(h_tight * context_factor))
    margin_x = int(round(w_tight * context_factor))

    y0 = max(0, y_min - margin_y)
    y1 = min(H_img, y_max + 1 + margin_y)
    x0 = max(0, x_min - margin_x)
    x1 = min(W_img, x_max + 1 + margin_x)

    region = image_np[y0:y1, x0:x1]
    canvas = np.array(
        Image.fromarray(region).resize((target_size, target_size), Image.BILINEAR)
    )
    transform = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
    return canvas, transform


def uncrop_mask(mask_512: np.ndarray, transform: dict) -> np.ndarray:
    """Redimensioneaza masca inapoi la dimensiunea zonei extinse (w, h)."""
    w, h = transform["w"], transform["h"]
    m = Image.fromarray((mask_512 * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
    return (np.array(m) > 127).astype(np.uint8)


def mask_to_rgba_b64(mask: np.ndarray, color_rgba: tuple) -> str:
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[mask > 0] = color_rgba
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def numpy_to_base64_jpeg(img_np: np.ndarray, quality: int = 85) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_np.astype(np.uint8)).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def mask_to_gray_b64(mask: np.ndarray) -> str:
    """Masca binara -> PNG grayscale base64 (pentru panoul de debug)."""
    buf = io.BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
