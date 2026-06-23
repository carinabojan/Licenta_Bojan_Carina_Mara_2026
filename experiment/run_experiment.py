# -*- coding: utf-8 -*-
"""
Experiment comparativ box-to-mask, parametrizat pe intervalul de offset.
Genereaza box-uri descentrate, ruleaza comparatia A vs B, calculeaza matricea
de confuzie si salveaza rezultatele intr-un folder dedicat.

Structura iesirii (per experiment):
  experiment/exp_<lo>_<hi>/
    boxes_offset/                 adnotari .txt cu box-urile descentrate
    boxes_offset/_preview/        previzualizari verde(original) vs rosu(descentrat)
    comparatii/                   vizualizari side-by-side REF | A | B per vehicul
    raport_iou.csv                IoU + TP/FP/FN per vehicul
    tabel_per_imagine.txt         cate vehicule B>A / egal / A>B per imagine
    matrice_confuzie.txt          matricile 2x2 (text)
    matrice_confuzie.png          heatmap colorat A vs B

Utilizare:
  python run_experiment.py --lo 0.30 --hi 0.45
  python run_experiment.py --lo 0.15 --hi 0.30
  python run_experiment.py --lo 0.45 --hi 0.60
"""
import os
import re
import sys
import random
import argparse
import numpy as np
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont

BACKEND = r"d:\ALicenta\Notebooks\interfata\backend"
sys.path.insert(0, BACKEND)
from model_utils import load_prompt_cnn, load_sam            # noqa: E402
from inference import (run_promptcnn, run_sam,                # noqa: E402
                       crop_and_pad, uncrop_mask)

# ── Config fix ──────────────────────────────────────────────────────────
ORIG_DIR   = r"d:\ALicenta\Notebooks\PozeInterfata"
EXP_ROOT   = r"d:\ALicenta\Notebooks\experiment"
MODEL_PATH = os.getenv("MODEL_PATH", r"d:\ALicenta\Notebooks\best_model.pth")
CONTEXT_FACTOR = 1.0           # context-ul folosit la antrenament
SEED = 42
# overlap impus intre box-ul descentrat si bbox-ul real, ca box-ul sa atinga
# vehiculul fara a cadea complet pe fundal
MIN_OV, MAX_OV = 0.20, 0.60

VEHICLE_CLASSES = {4, 5, 6, 7, 8, 9}
CLASS_NAMES = {4: "car", 5: "van", 6: "truck", 7: "tricycle", 8: "awning-tri", 9: "bus"}

COL_REF = np.array([60, 230, 90],  dtype=np.float32)
COL_A   = np.array([0, 170, 255],  dtype=np.float32)
COL_B   = np.array([255, 90, 0],   dtype=np.float32)


def _font(sz):
    try: return ImageFont.truetype("arial.ttf", sz)
    except Exception: return ImageFont.load_default()


# ── Parsare adnotari ────────────────────────────────────────────────────
def parse_full(txt_path):
    """Toate liniile, cu flag de vehicul (pentru rescriere offset)."""
    rows = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = re.split(r"[,\s]+", line)
            if len(p) < 8:
                rows.append({"raw": line, "veh": False}); continue
            try:
                v = [int(float(t)) for t in p[:8]]
            except ValueError:
                rows.append({"raw": line, "veh": False}); continue
            x, y, w, h, score, cat, trunc, occ = v
            rows.append({"x": x, "y": y, "w": w, "h": h, "score": score,
                         "cat": cat, "trunc": trunc, "occ": occ,
                         "veh": cat in VEHICLE_CLASSES})
    return rows


def parse_vehicles(txt_path):
    """Doar vehicule, ca lista [{idx,x,y,w,h,cat}]."""
    cars = []
    if not os.path.isfile(txt_path):
        return cars
    for r in parse_full(txt_path):
        if r.get("veh") and r.get("w", 0) > 0 and r.get("h", 0) > 0:
            cars.append({"idx": len(cars), "x": r["x"], "y": r["y"],
                         "w": r["w"], "h": r["h"], "cat": r["cat"]})
    return cars


# ── Descentrare ─────────────────────────────────────────────────────────
def _overlap(x, y, w, h, nx, ny):
    ix = max(0, min(x + w, nx + w) - max(x, nx))
    iy = max(0, min(y + h, ny + h) - max(y, ny))
    return (ix * iy) / float(w * h)


def offset_box(x, y, w, h, W, H, rng, lo, hi):
    best = (x, y)
    for _ in range(60):
        fx = rng.uniform(lo, hi) * rng.choice([-1, 1])
        fy = rng.uniform(lo, hi) * rng.choice([-1, 1])
        nx = max(0, min(int(round(x + fx * w)), W - w))
        ny = max(0, min(int(round(y + fy * h)), H - h))
        if MIN_OV <= _overlap(x, y, w, h, nx, ny) <= MAX_OV:
            return nx, ny
        best = (nx, ny)
    return best


def make_offset_set(lo, hi, off_dir, prev_dir):
    os.makedirs(off_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)
    rng = random.Random(SEED)
    font = _font(16)
    stems = sorted({os.path.splitext(f)[0] for f in os.listdir(ORIG_DIR)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))})
    for stem in stems:
        img_path = next((os.path.join(ORIG_DIR, stem + e)
                         for e in (".jpg", ".jpeg", ".png")
                         if os.path.isfile(os.path.join(ORIG_DIR, stem + e))), None)
        txt_path = os.path.join(ORIG_DIR, stem + ".txt")
        if img_path is None or not os.path.isfile(txt_path):
            continue
        img = Image.open(img_path).convert("RGB"); W, H = img.size
        rows = parse_full(txt_path)
        new_lines, pairs = [], []
        for r in rows:
            if not r["veh"]:
                new_lines.append(r.get("raw") or
                    f"{r['x']},{r['y']},{r['w']},{r['h']},{r['score']},{r['cat']},{r['trunc']},{r['occ']}")
                continue
            nx, ny = offset_box(r["x"], r["y"], r["w"], r["h"], W, H, rng, lo, hi)
            new_lines.append(f"{nx},{ny},{r['w']},{r['h']},{r['score']},{r['cat']},{r['trunc']},{r['occ']}")
            pairs.append((r["x"], r["y"], r["w"], r["h"], nx, ny))
        with open(os.path.join(off_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(new_lines) + "\n")
        # preview
        prev = img.copy(); d = ImageDraw.Draw(prev)
        for i, (x, y, w, h, nx, ny) in enumerate(pairs):
            d.rectangle([x, y, x+w, y+h], outline=(40, 230, 80), width=2)
            d.rectangle([nx, ny, nx+w, ny+h], outline=(255, 30, 60), width=3)
            d.text((nx, ny-18), f"M{i+1}", fill=(255, 230, 0), font=font)
        d.rectangle([6, 6, 240, 48], fill=(0, 0, 0))
        d.text((12, 8),  "verde = box original",  fill=(40, 230, 80), font=font)
        d.text((12, 27), "rosu  = box descentrat", fill=(255, 60, 90), font=font)
        prev.save(os.path.join(prev_dir, stem + "_preview.jpg"), quality=90)
        print(f"  offset {stem}: {len(pairs)} vehicule")


# ── Segmentari ──────────────────────────────────────────────────────────
def iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u > 0 else 0.0


def confusion(pred, ref):
    p = pred.astype(bool); r = ref.astype(bool)
    return (int(np.logical_and(p, r).sum()), int(np.logical_and(p, ~r).sum()),
            int(np.logical_and(~p, r).sum()), int(np.logical_and(~p, ~r).sum()))


def _place_full(car, tr, scene_np):
    """Aseaza masca vehiculului in masca la dimensiunea imaginii complete."""
    H, W = scene_np.shape[:2]
    full = np.zeros((H, W), dtype=np.uint8)
    x0, y0 = tr["x"], tr["y"]; hh, ww = car.shape
    full[y0:y0+hh, x0:x0+ww] = car
    return full


def sam_from_box(sam_model, processor, scene_np, box):
    """
    Varianta A / REF: pe acelasi crop ca pipeline-ul, SAM3 primeste bbox-ul ca
    prompt (in coordonatele crop-ului). Intrarea imagine e identica cu varianta B;
    difera doar tipul de prompt (box vs punct).
    """
    x, y, w, h = box
    crop, tr = crop_and_pad(scene_np, x, y, w, h, context_factor=CONTEXT_FACTOR)
    sx = 512.0 / tr["w"]; sy = 512.0 / tr["h"]
    bx = np.array([(x - tr["x"]) * sx, (y - tr["y"]) * sy,
                   (x + w - tr["x"]) * sx, (y + h - tr["y"]) * sy], dtype=np.float32)
    st = processor.set_image(Image.fromarray(crop))
    masks, scores, _ = sam_model.predict_inst(st, box=bx, multimask_output=True)
    best = masks[np.argmax(scores)]
    m512 = np.array(Image.fromarray(best.astype(np.uint8)).resize((512, 512), Image.NEAREST))
    car = uncrop_mask((m512 > 0).astype(np.uint8), tr)
    return _place_full(car, tr, scene_np)


def pipeline_from_box(model, meta, sam_model, processor, scene_np, box):
    """Varianta B: acelasi crop, dar prompt = punctul prezis de PromptCNN."""
    x, y, w, h = box
    crop, tr = crop_and_pad(scene_np, x, y, w, h, context_factor=CONTEXT_FACTOR)
    coords = run_promptcnn(model, crop, meta)
    m512 = run_sam(sam_model, processor, Image.fromarray(crop), coords, meta["point_labels"])
    car = uncrop_mask(m512, tr)
    full = _place_full(car, tr, scene_np)
    px = tr["x"] + float(coords[0, 0]) * (tr["w"] / 512.0)
    py = tr["y"] + float(coords[0, 1]) * (tr["h"] / 512.0)
    return full, (px, py)


def ov(img, mask, color, a=0.5):
    o = img.astype(np.float32).copy(); m = mask.astype(bool)
    o[m] = (1 - a) * o[m] + a * color
    return o.astype(np.uint8)


def make_panel(scene_np, bo, bf, ref, mA, mB, ptB, iA, iB, out_path):
    x, y, w, h = bo; ox, oy, owd, ohd = bf
    xs = [x, x+w, ox, ox+owd]; ys = [y, y+h, oy, oy+ohd]
    pad = int(max(w, h) * 0.8); H, W = scene_np.shape[:2]
    cx0 = max(0, min(xs)-pad); cy0 = max(0, min(ys)-pad)
    cx1 = min(W, max(xs)+pad); cy1 = min(H, max(ys)+pad)
    base = scene_np[cy0:cy1, cx0:cx1]
    cr = lambda m: m[cy0:cy1, cx0:cx1]
    pR = Image.fromarray(ov(base, cr(ref), COL_REF))
    pA = Image.fromarray(ov(base, cr(mA),  COL_A))
    pB = Image.fromarray(ov(base, cr(mB),  COL_B))
    ImageDraw.Draw(pR).rectangle([x-cx0, y-cy0, x+w-cx0, y+h-cy0], outline=(60,230,90), width=2)
    for p in (pA, pB):
        ImageDraw.Draw(p).rectangle([ox-cx0, oy-cy0, ox+owd-cx0, oy+ohd-cy0], outline=(255,30,60), width=2)
    pxx, pyy = ptB
    ImageDraw.Draw(pB).ellipse([pxx-cx0-4, pyy-cy0-4, pxx-cx0+4, pyy-cy0+4], fill=(255,255,255), outline=(0,0,0))
    font = _font(14); gap, tb = 8, 42; cw = pR.width
    canvas = Image.new("RGB", (cw*3 + gap*2, pR.height + tb), (20, 22, 32))
    d = ImageDraw.Draw(canvas)
    titles = [("REF", "box original"), ("A: box direct", f"IoU = {iA:.3f}"),
              ("B: pipeline", f"IoU = {iB:.3f}")]
    cols = [(60,230,90), (0,170,255), (255,120,40)]
    for i, ((l1, l2), col) in enumerate(zip(titles, cols)):
        xo = i*(cw+gap); canvas.paste((pR, pA, pB)[i], (xo, tb))
        d.text((xo+4, 4), l1, fill=col, font=font)
        d.text((xo+4, 22), l2, fill=col, font=font)
    canvas.save(out_path)


def plot_confusion(CA, CB, mA, mB, lo, hi, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["vehicul", "fundal"]
    mat = lambda C: np.array([[C[0], C[2]], [C[1], C[3]]], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, C, m, title in zip(axes, (CA, CB), (mA, mB),
            ("Varianta A — box direct in SAM3", "Varianta B — pipeline (PromptCNN + SAM3)")):
        M = mat(C); Mn = M / M.sum(axis=1, keepdims=True)
        im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Prezis (Predicted)"); ax.set_ylabel("Referinta (Actual)")
        iou_g, prec, rec, dice = m
        ax.set_title(f"{title}\nIoU={iou_g:.3f}  Dice={dice:.3f}  P={prec:.3f}  R={rec:.3f}", fontsize=9)
        for r in range(2):
            for c in range(2):
                col = "white" if Mn[r,c] > 0.5 else "black"
                ax.text(c, r, f"{Mn[r,c]*100:.1f}%\n{int(M[r,c]):,}", ha="center", va="center", color=col, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Matrice de confuzie pe pixeli — A vs B  (offset {int(lo*100)}-{int(hi*100)}%)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out_path, dpi=130); plt.close(fig)


def metrics(C):
    tp, fp, fn, tn = C
    iou_g = tp/(tp+fp+fn) if (tp+fp+fn) else 0.0
    prec  = tp/(tp+fp) if (tp+fp) else 0.0
    rec   = tp/(tp+fn) if (tp+fn) else 0.0
    dice  = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0
    return iou_g, prec, rec, dice


# ── Main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, required=True)
    ap.add_argument("--hi", type=float, required=True)
    ap.add_argument("--out", type=str, default=None,
                    help="nume folder iesire (implicit exp_<lo>_<hi>)")
    args = ap.parse_args()
    lo, hi = args.lo, args.hi
    tag = args.out if args.out else f"exp_{int(lo*100)}_{int(hi*100)}"
    base = os.path.join(EXP_ROOT, tag)
    off_dir   = os.path.join(base, "boxes_offset")
    prev_dir  = os.path.join(off_dir, "_preview")
    comp_dir  = os.path.join(base, "comparatii")
    for d in (base, comp_dir):
        os.makedirs(d, exist_ok=True)

    print(f"\n########## EXPERIMENT {tag}  (offset {int(lo*100)}-{int(hi*100)}%) ##########")
    print("[1] Generez box-urile descentrate...")
    make_offset_set(lo, hi, off_dir, prev_dir)

    print("[2] Incarc modelele...")
    model, meta = load_prompt_cnn(MODEL_PATH)
    sam_model, processor = load_sam()

    stems = sorted({os.path.splitext(f)[0] for f in os.listdir(ORIG_DIR)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))})

    rep = ["stem,car_idx,class,IoU_A,IoU_B,delta,TPa,FPa,FNa,TPb,FPb,FNb"]
    A_all, B_all = [], []
    CA = [0, 0, 0, 0]; CB = [0, 0, 0, 0]
    per_img = OrderedDict()

    print("[3] Rulez comparatia A vs B...")
    for stem in stems:
        img_path = next((os.path.join(ORIG_DIR, stem + e)
                         for e in (".jpg", ".jpeg", ".png")
                         if os.path.isfile(os.path.join(ORIG_DIR, stem + e))), None)
        if img_path is None:
            continue
        scene_pil = Image.open(img_path).convert("RGB")
        scene_np  = np.array(scene_pil)
        cars_o = parse_vehicles(os.path.join(ORIG_DIR, stem + ".txt"))
        cars_f = parse_vehicles(os.path.join(off_dir,  stem + ".txt"))
        n = min(len(cars_o), len(cars_f))
        per_img[stem] = {"n": 0, "B": 0, "eg": 0, "A": 0, "sA": 0.0, "sB": 0.0}
        print(f"  === {stem} — {n} vehicule ===")
        for i in range(n):
            bo = (cars_o[i]["x"], cars_o[i]["y"], cars_o[i]["w"], cars_o[i]["h"])
            bf = (cars_f[i]["x"], cars_f[i]["y"], cars_f[i]["w"], cars_f[i]["h"])
            try:
                ref       = sam_from_box(sam_model, processor, scene_np, bo)
                mA        = sam_from_box(sam_model, processor, scene_np, bf)
                mB, ptB   = pipeline_from_box(model, meta, sam_model, processor, scene_np, bf)
                iA, iB    = iou(mA, ref), iou(mB, ref)
            except Exception as e:
                print(f"    M{i+1}: EROARE {e}"); continue
            tpa, fpa, fna, _ = confusion(mA, ref)
            tpb, fpb, fnb, _ = confusion(mB, ref)
            ca = confusion(mA, ref); cb = confusion(mB, ref)
            for j in range(4): CA[j] += ca[j]; CB[j] += cb[j]
            make_panel(scene_np, bo, bf, ref, mA, mB, ptB, iA, iB,
                       os.path.join(comp_dir, f"{stem}_M{i+1}.png"))
            cls = CLASS_NAMES.get(cars_o[i]["cat"], "?")
            rep.append(f"{stem},{i+1},{cls},{iA:.4f},{iB:.4f},{iB-iA:+.4f},"
                       f"{tpa},{fpa},{fna},{tpb},{fpb},{fnb}")
            A_all.append(iA); B_all.append(iB)
            d = per_img[stem]; d["n"]+=1; d["sA"]+=iA; d["sB"]+=iB
            if iB > iA+0.02: d["B"]+=1
            elif iA > iB+0.02: d["A"]+=1
            else: d["eg"]+=1
            print(f"    M{i+1} ({cls}): A={iA:.3f} B={iB:.3f} d={iB-iA:+.3f}")

    # CSV
    with open(os.path.join(base, "raport_iou.csv"), "w") as f:
        f.write("\n".join(rep) + "\n")

    # tabel per imagine
    T = []
    T.append("="*72)
    T.append(f"  EXPERIMENT: offset {int(lo*100)}-{int(hi*100)}%   |   CATE VEHICULE MAI BUNE PER VARIANTA")
    T.append("="*72)
    T.append(f"  {'Imagine':<26}{'N':>4}{'B>A':>6}{'egal':>6}{'A>B':>6}{'IoU_A':>8}{'IoU_B':>8}")
    T.append("  " + "-"*70)
    TB=TE=TA=TN=0
    for stem, d in per_img.items():
        if d["n"]==0: continue
        T.append(f"  {stem:<26}{d['n']:>4}{d['B']:>6}{d['eg']:>6}{d['A']:>6}{d['sA']/d['n']:>8.3f}{d['sB']/d['n']:>8.3f}")
        TB+=d["B"]; TE+=d["eg"]; TA+=d["A"]; TN+=d["n"]
    T.append("  " + "-"*70)
    mA_s = np.mean(A_all) if A_all else 0; mB_s = np.mean(B_all) if B_all else 0
    T.append(f"  {'TOTAL':<26}{TN:>4}{TB:>6}{TE:>6}{TA:>6}{mA_s:>8.3f}{mB_s:>8.3f}")
    T.append("="*72)
    ttxt = "\n".join(T); print("\n"+ttxt)
    with open(os.path.join(base, "tabel_per_imagine.txt"), "w", encoding="utf-8") as f:
        f.write(ttxt + "\n")

    # matrice de confuzie
    mA = metrics(CA); mB = metrics(CB)
    def cm(name, C, m):
        tp, fp, fn, tn = C; iou_g, prec, rec, dice = m
        L = [f"  {name}", "  "+"-"*58,
             "                         |   REF: vehicul   |   REF: fundal", "  "+"-"*58,
             f"   Prezis: vehicul       | TP = {tp:>11} | FP = {fp:>11}",
             f"   Prezis: fundal        | FN = {fn:>11} | TN = {tn:>11}", "  "+"-"*58,
             f"   IoU = {iou_g:.4f}   Precision = {prec:.4f}   Recall = {rec:.4f}   Dice = {dice:.4f}"]
        return "\n".join(L)
    M = ["="*64, f"  MATRICE DE CONFUZIE GLOBALA (pixeli) — offset {int(lo*100)}-{int(hi*100)}%", "="*64, "",
         cm("VARIANTA A — box descentrat direct in SAM3", CA, mA), "",
         cm("VARIANTA B — pipeline (PromptCNN + SAM3)", CB, mB), "", "="*64,
         f"  Vehicule: {len(A_all)}   |   IoU global  A={mA[0]:.4f}  ->  B={mB[0]:.4f}", "="*64]
    mtxt = "\n".join(M); print("\n"+mtxt)
    with open(os.path.join(base, "matrice_confuzie.txt"), "w", encoding="utf-8") as f:
        f.write(mtxt + "\n")
    plot_confusion(CA, CB, mA, mB, lo, hi, os.path.join(base, "matrice_confuzie.png"))

    print(f"\n########## GATA — toate rezultatele in: {base} ##########")


if __name__ == "__main__":
    main()
