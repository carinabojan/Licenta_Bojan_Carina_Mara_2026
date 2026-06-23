import os
import io
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

load_dotenv()

from model_utils import load_prompt_cnn, load_sam, DEVICE
from inference import (run_promptcnn, run_sam, render_overlay, numpy_to_base64,
                       scale_points_to_sam, crop_and_pad, uncrop_mask,
                       mask_to_rgba_b64, mask_to_gray_b64,
                       numpy_to_base64_jpeg, MASK_COLORS, VEHICLE_CLASSES)

app = FastAPI(title="SAM_Clean Interface")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Radacina proiectului (folderul Licenta_Bojan_Carina), relativ la backend/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(PROJECT_ROOT, "best_model.pth"))

try:
    prompt_cnn, meta = load_prompt_cnn(MODEL_PATH)
    PROMPTCNN_AVAILABLE = True
    print(f"[OK] PromptCNN incarcat din {MODEL_PATH}")
except Exception as e:
    prompt_cnn, meta = None, None
    PROMPTCNN_AVAILABLE = False
    print(f"[WARN] PromptCNN indisponibil: {e}")
    print("[WARN] Interfata va functiona cu puncte manuale + SAM.")

sam_model, processor = load_sam()
print("[OK] SAM incarcat.")

POZA_DIR = os.getenv("POZA_DIR", os.path.join(PROJECT_ROOT, "PozeInterfata"))

_session: dict = {}


def _parse_annotations(txt_path: str) -> list[dict]:
    """
    Parseaza format BRUT VisDrone: x,y,w,h,score,category,truncation,occlusion
    Retine doar vehicule (VEHICLE_CLASSES, 1-indexed): car=4, van=5, truck=6,
    tricycle=7, awning-tricycle=8, bus=9, motor=10. (0=ignored, 11=others excluse)
    """
    import re
    cars = []
    try:
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = re.split(r"[,\s]+", line)
                if len(parts) < 6:
                    continue
                x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                category = int(parts[5])
                if category not in VEHICLE_CLASSES:
                    continue
                if w > 0 and h > 0:
                    cars.append({
                        "idx": len(cars),
                        "x": x, "y": y, "w": w, "h": h,
                        "category": category,
                    })
    except Exception as e:
        print(f"[WARN] Eroare parsare adnotare {txt_path}: {e}")
    return cars

class Point(BaseModel):
    x: float
    y: float
    label: int          # 1 = pozitiv, 0 = negativ


class ShowMaskRequest(BaseModel):
    points: list[Point]


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Primeste imaginea, o redimensioneaza la 512x512 (ca in training) si ruleaza PromptCNN+SAM."""
    contents = await file.read()
    pil_orig = Image.open(io.BytesIO(contents)).convert("RGB")

    image_size = meta["image_size"] if PROMPTCNN_AVAILABLE else 512
    pil_512 = pil_orig.resize((image_size, image_size), resample=Image.BILINEAR)
    image_np = np.array(pil_512, dtype=np.uint8)

    _session["image_np"]  = image_np
    _session["image_pil"] = pil_512
    _session["orig_w"]    = image_size
    _session["orig_h"]    = image_size

    if PROMPTCNN_AVAILABLE:
        point_labels = meta["point_labels"]
        coords_512 = run_promptcnn(prompt_cnn, image_np, meta)

        sam_mask = run_sam(sam_model, processor, pil_512, coords_512, point_labels)
        _session["sam_mask"]    = sam_mask
        _session["auto_coords"] = coords_512
        _session["auto_labels"] = point_labels

        overlay = render_overlay(image_np, sam_mask,
                                 np.zeros((0, 2), dtype=np.float32),
                                 np.array([], dtype=np.int32))
        auto_points = [
            {"x": float(coords_512[i, 0]), "y": float(coords_512[i, 1]), "label": int(point_labels[i])}
            for i in range(len(point_labels))
        ]
    else:
        overlay     = image_np
        auto_points = []

    return {
        "image_b64":   numpy_to_base64(overlay),
        "orig_w":      image_size,
        "orig_h":      image_size,
        "auto_points": auto_points,
    }


@app.post("/show_mask")
def show_mask(req: ShowMaskRequest):
    """Ruleaza SAM cu punctele trimise de utilizator (coordonate in spatiul 512x512)."""
    if "image_pil" not in _session:
        raise HTTPException(status_code=400, detail="Nicio imagine incarcata.")

    if not req.points:
        raise HTTPException(status_code=400, detail="Cel putin un punct necesar.")

    coords = np.array([[p.x, p.y] for p in req.points], dtype=np.float32)
    labels = np.array([p.label for p in req.points], dtype=np.int32)

    image_np  = _session["image_np"]
    image_pil = _session["image_pil"]

    sam_mask = run_sam(sam_model, processor, image_pil, coords, labels)
    _session["sam_mask"] = sam_mask

    overlay = render_overlay(image_np, sam_mask,
                              np.zeros((0, 2), dtype=np.float32),
                              np.array([], dtype=np.int32))

    return {"image_b64": numpy_to_base64(overlay)}


@app.get("/reset")
def reset():
    """Sterge sesiunea curenta."""
    _session.clear()
    return {"status": "ok"}

@app.get("/list_images")
def list_images():
    """Listeaza imaginile disponibile in POZA_DIR."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    if not os.path.isdir(POZA_DIR):
        return {"images": [], "error": f"Folderul '{POZA_DIR}' nu exista."}
    stems = sorted({
        os.path.splitext(f)[0]
        for f in os.listdir(POZA_DIR)
        if os.path.splitext(f)[1].lower() in exts
    })
    return {"images": stems}


@app.get("/load_image/{stem}")
def load_image_endpoint(stem: str):
    """Incarca o imagine si adnotarile ei din POZA_DIR."""
    import re
    if not re.match(r"^[\w\-\.]+$", stem):
        raise HTTPException(400, "Nume de fisier invalid.")

    img_path = None
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        candidate = os.path.join(POZA_DIR, stem + ext)
        if os.path.isfile(candidate):
            img_path = candidate
            break
    if img_path is None:
        raise HTTPException(404, f"Imaginea '{stem}' nu a fost gasita in {POZA_DIR}.")

    pil_orig = Image.open(img_path).convert("RGB")
    W, H = pil_orig.size
    image_np = np.array(pil_orig)

    _session["scene_np"]    = image_np
    _session["scene_pil"]   = pil_orig
    _session["scene_w"]     = W
    _session["scene_h"]     = H
    _session["current_stem"] = stem

    ann_path = os.path.join(POZA_DIR, stem + ".txt")
    cars = _parse_annotations(ann_path) if os.path.isfile(ann_path) else []
    _session["scene_cars"] = cars

    return {
        "orig_w":          W,
        "orig_h":          H,
        "image_b64":       numpy_to_base64_jpeg(image_np, quality=85),
        "image_mime":      "image/jpeg",
        "cars":            cars,
        "has_annotations": len(cars) > 0,
    }


class ProcessCropRequest(BaseModel):
    car_idx: int
    debug: bool = False
    context_factor: float = 0.5


@app.post("/process_crop")
def process_crop_endpoint(req: ProcessCropRequest):
    """Proceseaza un crop de masina prin PromptCNN + SAM3, returneaza masca la dimensiunea bbox-ului."""
    if "scene_np" not in _session:
        raise HTTPException(400, "Nicio imagine incarcata.")

    car = next((c for c in _session.get("scene_cars", []) if c["idx"] == req.car_idx), None)
    if car is None:
        raise HTTPException(404, f"Masina idx={req.car_idx} nu a fost gasita.")

    crop_512, transform = crop_and_pad(
        _session["scene_np"], car["x"], car["y"], car["w"], car["h"],
        context_factor=req.context_factor,
    )
    crop_pil = Image.fromarray(crop_512)

    if PROMPTCNN_AVAILABLE:
        coords_512   = run_promptcnn(prompt_cnn, crop_512, meta)
        point_labels = meta["point_labels"]
    else:
        coords_512   = np.array([[256.0, 256.0]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

    sam_mask_512 = run_sam(sam_model, processor, crop_pil, coords_512, point_labels)

    car_mask = uncrop_mask(sam_mask_512, transform)

    stem = _session.get("current_stem", "unknown")
    debug_dir = os.path.join(POZA_DIR, "debug_crops")
    try:
        os.makedirs(debug_dir, exist_ok=True)
        prefix = os.path.join(debug_dir, f"{stem}_car{req.car_idx}")
        Image.fromarray(crop_512).save(prefix + "_crop.jpg")
        Image.fromarray((sam_mask_512 * 255).astype(np.uint8)).save(prefix + "_mask512.png")
        Image.fromarray((car_mask * 255).astype(np.uint8)).save(prefix + "_mask_bbox.png")
        print(f"[DEBUG] car_idx={req.car_idx}  bbox=({car['x']},{car['y']},{car['w']}x{car['h']})  "
              f"exp=({transform['x']},{transform['y']},{transform['w']}x{transform['h']})  "
              f"mask512_sum={int(sam_mask_512.sum())}  car_mask_sum={int(car_mask.sum())}")
    except Exception as _e:
        print(f"[WARN] debug crop save failed: {_e}")

    color    = MASK_COLORS[req.car_idx % len(MASK_COLORS)]
    mask_b64 = mask_to_rgba_b64(car_mask, color)

    x, y, w, h = transform["x"], transform["y"], transform["w"], transform["h"]
    pt_orig = {
        "x": round(x + float(coords_512[0, 0]) * (w / 512.0), 1),
        "y": round(y + float(coords_512[0, 1]) * (h / 512.0), 1),
    }

    response = {
        "car_idx":    req.car_idx,
        "car_x":      transform["x"],
        "car_y":      transform["y"],
        "car_w":      transform["w"],
        "car_h":      transform["h"],
        "mask_b64":   mask_b64,
        "point_orig": pt_orig,
    }

    if req.debug:
        response["debug"] = {
            "crop_b64":      numpy_to_base64_jpeg(crop_512, quality=80),
            "mask512_b64":   mask_to_gray_b64(sam_mask_512),
            "bbox_mask_b64": mask_to_gray_b64(car_mask),
        }

    return response


class CorrectPoint(BaseModel):
    x: float
    y: float
    label: int  


class CorrectRequest(BaseModel):
    car_idx: int
    points: list[CorrectPoint]
    debug: bool = False
    context_factor: float = 0.5


@app.post("/correct")
def correct_endpoint(req: CorrectRequest):
    """
    Corectie manuala: ocoleste PromptCNN si apeleaza SAM direct cu punctele
    utilizatorului (coordonate ale imaginii complexe). Inlocuieste doar masca
    vehiculului respectiv.
    """
    if "scene_np" not in _session:
        raise HTTPException(400, "Nicio imagine incarcata.")
    if not req.points:
        raise HTTPException(400, "Niciun punct furnizat.")

    car = next((c for c in _session.get("scene_cars", []) if c["idx"] == req.car_idx), None)
    if car is None:
        raise HTTPException(404, f"Masina idx={req.car_idx} nu a fost gasita.")

    crop_512, transform = crop_and_pad(
        _session["scene_np"], car["x"], car["y"], car["w"], car["h"],
        context_factor=req.context_factor,
    )
    crop_pil = Image.fromarray(crop_512)

    x, y = transform["x"], transform["y"]
    w, h = transform["w"], transform["h"]

    coords, labels = [], []
    for p in req.points:
        lx = (p.x - x) * (512.0 / w)
        ly = (p.y - y) * (512.0 / h)
        coords.append([lx, ly])
        labels.append(int(p.label))

    coords = np.array(coords, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    sam_mask_512 = run_sam(sam_model, processor, crop_pil, coords, labels)
    car_mask     = uncrop_mask(sam_mask_512, transform)

    color    = MASK_COLORS[req.car_idx % len(MASK_COLORS)]
    mask_b64 = mask_to_rgba_b64(car_mask, color)

    print(f"[CORRECT] car_idx={req.car_idx}  n_points={len(req.points)}  "
          f"car_mask_sum={int(car_mask.sum())}")

    response = {
        "car_idx":  req.car_idx,
        "car_x":    x, "car_y": y, "car_w": transform["w"], "car_h": transform["h"],
        "mask_b64": mask_b64,
        "manual":   True,
    }
    if req.debug:
        response["debug"] = {
            "crop_b64":      numpy_to_base64_jpeg(crop_512, quality=80),
            "mask512_b64":   mask_to_gray_b64(sam_mask_512),
            "bbox_mask_b64": mask_to_gray_b64(car_mask),
        }
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
