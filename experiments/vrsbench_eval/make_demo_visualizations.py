"""Generate bbox visualization demos: hard (red) vs soft (green) vs GT (blue).

Picks samples where soft label beats hard label clearly. Saves PNGs + one
composite figure for the poster / laptop demo.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE = Path("/lus/lfs1aip2/projects/b6ar/dw22963")
HARD_FILE = BASE / "GeoChat/experiments/vrsbench_eval/predictions/vrs_grounding_predictions.jsonl"
SOFT_FILE = BASE / "GeoChat/experiments/vrsbench_eval/predictions/softlabel_newcode_vg_predictions.jsonl"
IMG_DIR = BASE / "data/VRSBench_full/Images_val"
OUT_DIR = BASE / "GeoChat/experiments/vrsbench_eval/demo_visualizations"
OUT_DIR.mkdir(exist_ok=True)


def parse_bbox(s):
    ints = re.findall(r"\d+", s or "")
    if len(ints) < 4:
        return None
    return [int(ints[0]), int(ints[1]), int(ints[2]), int(ints[3])]


def iou(a, b):
    if a is None or b is None:
        return 0.0
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


print("Loading predictions...")
hard = load_jsonl(HARD_FILE)
soft = load_jsonl(SOFT_FILE)
assert len(hard) == len(soft), f"{len(hard)} vs {len(soft)}"

print("Scoring samples...")
scored = []
for h, s in zip(hard, soft):
    assert h["question_id"] == s["question_id"]
    gt = parse_bbox(h["ground_truth"])
    hb = parse_bbox(h["answer"])
    sb = parse_bbox(s["answer"])
    if gt is None:
        continue
    h_iou = iou(gt, hb)
    s_iou = iou(gt, sb)
    scored.append({
        "qid": h["question_id"],
        "image": h["image_id"],
        "question": h["question"],
        "unique": h.get("unique", False),
        "obj_cls": h.get("obj_cls", ""),
        "gt": gt, "hb": hb, "sb": sb,
        "h_iou": h_iou, "s_iou": s_iou,
        "delta": s_iou - h_iou,
    })

# Pick samples where soft clearly beats hard: soft >= 0.6, hard < 0.4, large delta
wins = [x for x in scored
        if x["s_iou"] >= 0.60 and x["h_iou"] < 0.40 and x["delta"] >= 0.30]
wins.sort(key=lambda x: -x["delta"])

# Diversity: one per image + prefer unique/non-unique mix + prefer varied obj_cls
picked, seen_imgs, seen_cls = [], set(), {}
for w in wins:
    if w["image"] in seen_imgs:
        continue
    # soft cap: at most 2 per obj_cls to diversify
    if seen_cls.get(w["obj_cls"], 0) >= 2:
        continue
    picked.append(w)
    seen_imgs.add(w["image"])
    seen_cls[w["obj_cls"]] = seen_cls.get(w["obj_cls"], 0) + 1
    if len(picked) == 5:
        break

print(f"Picked {len(picked)} demos:")
for i, w in enumerate(picked):
    print(f"  [{i}] qid={w['qid']} img={w['image']} cls={w['obj_cls']} "
          f"unique={w['unique']} hard_iou={w['h_iou']:.2f} soft_iou={w['s_iou']:.2f} "
          f"(+{w['delta']:.2f})")
    print(f"      Q: {w['question']}")


def draw_with_pil(image_path, gt, hb, sb, title, save_path):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    def scale(b):
        return [b[0] * W / 100, b[1] * H / 100, b[2] * W / 100, b[3] * H / 100]

    # Draw order: gt (blue, thickest), hard (red), soft (green)
    for bbox, color, width in [(scale(gt), "blue", 6),
                                (scale(hb), "red", 4),
                                (scale(sb), "lime", 4)]:
        draw.rectangle(bbox, outline=color, width=width)

    img.save(save_path)


# Per-sample image output
for i, w in enumerate(picked):
    img_path = IMG_DIR / w["image"]
    if not img_path.exists():
        print(f"  WARNING: {img_path} not found, skipping")
        continue
    out = OUT_DIR / f"demo_{i+1}_{w['image'].replace('.png', '')}.png"
    draw_with_pil(img_path, w["gt"], w["hb"], w["sb"],
                  f"{w['question']}", out)

# Composite figure (matplotlib) with captions
n = len(picked)
fig, axes = plt.subplots(1, n, figsize=(5 * n, 6.5))
if n == 1:
    axes = [axes]

for ax, w in zip(axes, picked):
    img_path = IMG_DIR / w["image"]
    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])

    def scale(b):
        return (b[0] * W / 100, b[1] * H / 100,
                (b[2] - b[0]) * W / 100, (b[3] - b[1]) * H / 100)

    for bbox, color, lw, label in [
        (w["gt"], "blue", 3.0, "Ground Truth"),
        (w["hb"], "red", 2.0, f"Hard (IoU={w['h_iou']:.2f})"),
        (w["sb"], "lime", 2.0, f"Soft (IoU={w['s_iou']:.2f})"),
    ]:
        x, y, bw, bh = scale(bbox)
        rect = patches.Rectangle((x, y), bw, bh, linewidth=lw,
                                  edgecolor=color, facecolor="none", label=label)
        ax.add_patch(rect)

    q = w["question"]
    if len(q) > 70:
        q = q[:67] + "..."
    unique_tag = "Unique" if w["unique"] else "Non-unique"
    ax.set_title(f'"{q}"\n[{unique_tag}, {w["obj_cls"]}]  '
                 f'Δ IoU = +{w["delta"]:.2f}',
                 fontsize=10, wrap=True)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

plt.suptitle("Soft Labeling vs Hard Labeling on VRSBench Visual Grounding\n"
             "Blue = Ground Truth, Red = Hard Label baseline, Green = Soft Label (ours)",
             fontsize=13, y=0.98)
plt.tight_layout()
composite_path = OUT_DIR / "composite_demo.png"
plt.savefig(composite_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {len(picked)} individual demos + composite to {OUT_DIR}")
print(f"Composite: {composite_path}")
