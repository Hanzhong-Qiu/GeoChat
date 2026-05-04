"""Compute VRSBench visual grounding metrics.

Usage:
    python compute_metrics.py <predictions.jsonl>
    python compute_metrics.py <predictions.jsonl> --tag <label>

For each input file prints a TSV table with one row per subset:
    tag  split         N      Acc@0.3 Acc@0.5 Acc@0.7 Acc@0.9 meanIoU cumIoU
Makes batch aggregation across many model runs trivial: just run this on
each predictions.jsonl and concatenate the output.
"""
import argparse
import json
import re
import sys

import numpy as np
from eval_utils import computeIoU

THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
SPLITS = {
    "all":        (1, 0, True, False),
    "unique":     (1, True),
    "non_unique": (0, False),
}


def parse_bbox(text):
    ints = re.findall(r"\d+", text or "")
    if len(ints) < 4:
        return None
    return [int(ints[0]), int(ints[1]), int(ints[2]), int(ints[3])]


def evaluate_split(items, allowed_unique):
    """Return a dict of metrics on the subset whose `unique` is in allowed_unique."""
    counts = np.zeros(len(THRESHOLDS))
    cumI = 0.0
    cumU = 0.0
    mean_iou = 0.0
    total = 0

    for item in items:
        is_unique = item.get("unique", item.get("is_unique", True))
        if is_unique not in allowed_unique:
            continue

        gt = parse_bbox(item.get("ground_truth", ""))
        if gt is None:
            continue
        total += 1

        pred = parse_bbox(item.get("answer", ""))
        if pred is None:
            pred = [0, 0, 0, 0]

        try:
            iou_score, I, U = computeIoU(gt, pred, return_iou=True)
        except Exception:
            continue

        mean_iou += iou_score
        cumI += I
        cumU += U
        for k, thres in enumerate(THRESHOLDS):
            if iou_score >= thres:
                counts[k] += 1

    if total == 0:
        return None

    out = {"N": total}
    for k, thres in enumerate(THRESHOLDS):
        out[f"Acc@{thres}"] = 100.0 * counts[k] / total
    out["meanIoU"] = 100.0 * mean_iou / total
    out["cumIoU"] = 100.0 * cumI / cumU if cumU > 0 else float("nan")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Path to predictions.jsonl")
    parser.add_argument("--tag", default=None, help="Label for this run in the output table (default: basename)")
    parser.add_argument("--header", action="store_true", help="Print column header row")
    args = parser.parse_args()

    tag = args.tag or args.data.split("/")[-1].replace(".jsonl", "")

    with open(args.data) as f:
        items = [json.loads(line) for line in f]

    cols = ["tag", "split", "N"] + [f"Acc@{t}" for t in THRESHOLDS] + ["meanIoU", "cumIoU"]
    if args.header:
        print("\t".join(cols))

    for split_name, allowed in SPLITS.items():
        metrics = evaluate_split(items, allowed)
        if metrics is None:
            print(f"{tag}\t{split_name}\t0\t-\t-\t-\t-\t-\t-")
            continue
        row = [tag, split_name, str(metrics["N"])]
        for t in THRESHOLDS:
            row.append(f"{metrics[f'Acc@{t}']:.2f}")
        row.append(f"{metrics['meanIoU']:.2f}")
        row.append(f"{metrics['cumIoU']:.2f}")
        print("\t".join(row))


if __name__ == "__main__":
    main()
