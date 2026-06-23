# -*- coding: utf-8 -*-
"""
Pasul 1 — Vizualizare adnotari VisDrone.
Deseneaza bounding box-urile (cutii rosii) ale vehiculelor peste fiecare imagine
si salveaza rezultatul in VisDrone2019_annotated/.

Foloseste acelasi filtru de clase ca interfata (VEHICLE_CLASSES, format brut VisDrone).
"""
import os
import re
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont

# ── Config ──────────────────────────────────────────────────────────────
VISDRONE_DIR = r"d:\ALicenta\Notebooks\VisDrone2019-DET-val"
IMAGES_DIR   = os.path.join(VISDRONE_DIR, "images")
ANNOT_DIR    = os.path.join(VISDRONE_DIR, "annotations")
OUT_DIR      = r"d:\ALicenta\Notebooks\VisDrone2019_annotated"

# Clase VisDrone (format brut, 1-indexat): 4=car 5=van 6=truck
# 7=tricycle 8=awning-tricycle 9=bus
VEHICLE_CLASSES = {4, 5, 6, 7, 8, 9}
CLASS_NAMES = {4: "car", 5: "van", 6: "truck", 7: "tricycle",
               8: "awning-tri", 9: "bus"}


def parse_annotations(txt_path):
    """Returneaza lista de vehicule [{idx, x, y, w, h, category}]."""
    cars = []
    if not os.path.isfile(txt_path):
        return cars
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[,\s]+", line)
            if len(parts) < 6:
                continue
            try:
                x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                category   = int(parts[5])
            except ValueError:
                continue
            if category not in VEHICLE_CLASSES:
                continue
            if w > 0 and h > 0:
                cars.append({"idx": len(cars), "x": x, "y": y, "w": w, "h": h,
                             "category": category})
    return cars


def annotate_image(img_path, cars, out_path):
    img  = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for car in cars:
        x, y, w, h = car["x"], car["y"], car["w"], car["h"]
        # cutie rosie groasa (ca in interfata)
        draw.rectangle([x, y, x + w, y + h], outline=(255, 23, 68), width=3)
        label = f"M{car['idx'] + 1} {CLASS_NAMES.get(car['category'], '?')}"
        # fundal pentru text
        tb = draw.textbbox((x, y - 20), label, font=font)
        draw.rectangle(tb, fill=(0, 0, 0))
        draw.text((x, y - 20), label, fill=(255, 230, 0), font=font)

    img.save(out_path, quality=88)
    return len(cars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="proceseaza doar primele N imagini (0 = toate)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    images = sorted(f for f in os.listdir(IMAGES_DIR)
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if args.limit > 0:
        images = images[:args.limit]

    total_cars = 0
    for i, fname in enumerate(images, 1):
        stem      = os.path.splitext(fname)[0]
        img_path  = os.path.join(IMAGES_DIR, fname)
        txt_path  = os.path.join(ANNOT_DIR, stem + ".txt")
        out_path  = os.path.join(OUT_DIR, fname)

        cars = parse_annotations(txt_path)
        n    = annotate_image(img_path, cars, out_path)
        total_cars += n
        if i % 25 == 0 or i == len(images):
            print(f"[{i}/{len(images)}] {fname} — {n} vehicule")

    print(f"\nGata. {len(images)} imagini adnotate in: {OUT_DIR}")
    print(f"Total vehicule desenate: {total_cars}")


if __name__ == "__main__":
    main()
