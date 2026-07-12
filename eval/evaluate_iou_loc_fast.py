#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ROOT_DIR))

from utils.openclip_encoder import OpenCLIPNetwork  # noqa: E402


def polygon_to_mask(img_shape: tuple[int, int], points_list) -> np.ndarray:
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def stack_mask(mask_base: np.ndarray, mask_add: np.ndarray) -> np.ndarray:
    mask = mask_base.copy()
    mask[mask_add != 0] = 1
    return mask


def smooth(mask: np.ndarray) -> np.ndarray:
    # Original eval uses a 7x7 majority filter implemented with Python loops.
    # For binary masks, a box-filter majority vote is equivalent and much faster.
    votes = cv2.blur(mask.astype(np.float32), (7, 7), borderType=cv2.BORDER_REPLICATE)
    return (votes >= 0.5).astype(np.uint8)


def seed_everything(seed_value: int = 42) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


def eval_gt_lerfdata(json_folder: Path) -> tuple[dict[str, dict], tuple[int, int]]:
    gt_json_paths = sorted(json_folder.glob("frame_*.json"))
    gt_ann = {}
    image_shape = None
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        h, w = gt_data["info"]["height"], gt_data["info"]["width"]
        image_shape = (h, w)
        idx = int(gt_data["info"]["name"].split("_")[-1].split(".jpg")[0]) - 1
        for prompt_data in gt_data["objects"]:
            label = prompt_data["category"]
            box = np.asarray(prompt_data["bbox"]).reshape(-1)
            mask = polygon_to_mask((h, w), prompt_data["segmentation"])
            if img_ann[label].get("mask", None) is not None:
                mask = stack_mask(img_ann[label]["mask"], mask)
                img_ann[label]["bboxes"] = np.concatenate(
                    [img_ann[label]["bboxes"].reshape(-1, 4), box.reshape(-1, 4)],
                    axis=0,
                )
            else:
                img_ann[label]["bboxes"] = box
            img_ann[label]["mask"] = mask
        gt_ann[f"{idx}"] = img_ann

    if image_shape is None:
        raise FileNotFoundError(f"No frame_*.json found in {json_folder}")
    return gt_ann, image_shape


def load_render_index(source_path: Path, render_dir: Path) -> dict[str, Path]:
    image_names = sorted(
        p.name for p in (source_path / "images").iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    render_paths = sorted(render_dir.glob("*.npy"), key=lambda p: int(p.stem))
    if len(render_paths) < len(image_names):
        raise RuntimeError(
            f"Render count ({len(render_paths)}) is smaller than image count "
            f"({len(image_names)}) in {render_dir}"
        )
    return {Path(name).stem: render_paths[i] for i, name in enumerate(image_names)}


@torch.no_grad()
def relevancy_from_compressed(
    compressed_path: Path,
    feature_512_path: Path,
    feature_3d_path: Path,
    clip_model: OpenCLIPNetwork,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    pred = np.load(compressed_path).astype(np.float32)
    h, w, _ = pred.shape
    pred_t = torch.from_numpy(pred.reshape(-1, 3)).to(device)
    feat3 = torch.from_numpy(np.load(feature_3d_path).astype(np.float32)).to(device)
    feat512 = torch.from_numpy(np.load(feature_512_path).astype(np.float32)).to(device)

    phrases = torch.cat([clip_model.pos_embeds, clip_model.neg_embeds], dim=0).to(device)
    n_pos = len(clip_model.positives)
    n_neg = len(clip_model.negatives)
    out = torch.empty((pred_t.shape[0], n_pos), dtype=torch.float32, device=device)

    feat3_t = feat3.T.contiguous()
    for start in range(0, pred_t.shape[0], chunk_size):
        end = min(start + chunk_size, pred_t.shape[0])
        nearest = torch.argmax(pred_t[start:end] @ feat3_t, dim=1)
        restored = feat512[nearest].to(phrases.dtype)
        logits = restored @ phrases.T
        negatives = logits[:, n_pos:]
        for pos_id in range(n_pos):
            positive = logits[:, pos_id : pos_id + 1].repeat(1, n_neg)
            sims = torch.stack((positive, negatives), dim=-1)
            probs = torch.softmax(10 * sims, dim=-1)
            best_id = probs[..., 0].argmin(dim=1)
            out[start:end, pos_id] = probs[
                torch.arange(end - start, device=device), best_id, 0
            ].float()

    return out.T.reshape(n_pos, h, w)


def evaluate_relevancy(
    valid_map: torch.Tensor,
    img_ann: dict,
    mask_thresh: float,
) -> tuple[list[float], list[int], list[float], list[float], int, int]:
    # valid_map: levels x prompts x H x W
    n_levels, n_prompt, _, _ = valid_map.shape
    chosen_iou_list = []
    chosen_lvl_list = []
    chosen_pixel_list = []
    chosen_gt_list = []

    for k, label in enumerate(img_ann.keys()):
        iou_lvl = np.zeros(n_levels, dtype=np.float32)
        pixel_lvl = np.zeros(n_levels, dtype=np.float32)
        gt_lvl = np.zeros(n_levels, dtype=np.float32)

        for i in range(n_levels):
            np_relev = valid_map[i, k].detach().cpu().numpy()
            kernel = np.ones((30, 30), dtype=np.float32) / (30 * 30)
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            relev = 0.5 * (avg_filtered + np_relev)

            output = relev - relev.min()
            output = output / (output.max() + 1e-9)
            output = np.clip(output * 2.0 - 1.0, 0.0, 1.0)
            mask_pred = smooth((output > mask_thresh).astype(np.uint8))
            mask_gt = img_ann[label]["mask"].astype(np.uint8)

            intersection = np.logical_and(mask_gt, mask_pred).sum()
            union = np.logical_or(mask_gt, mask_pred).sum()
            iou_lvl[i] = float(intersection / union) if union > 0 else 0.0
            pixel_lvl[i] = float(intersection)
            gt_lvl[i] = float(mask_gt.sum())
            valid_map[i, k] = torch.from_numpy(relev).to(valid_map.device)

        score_lvl = valid_map[:, k].amax(dim=(1, 2))
        chosen_lvl = int(torch.argmax(score_lvl).item())
        chosen_iou_list.append(float(iou_lvl[chosen_lvl]))
        chosen_pixel_list.append(float(pixel_lvl[chosen_lvl]))
        chosen_gt_list.append(float(gt_lvl[chosen_lvl]))
        chosen_lvl_list.append(chosen_lvl)

    acc_num = 0
    total_prompts = 0
    for k, label in enumerate(img_ann.keys()):
        select_output = valid_map[:, k].detach().cpu().numpy()
        avg_filtered = cv2.filter2D(select_output.transpose(1, 2, 0), -1, np.ones((30, 30)) / (30 * 30))
        if avg_filtered.ndim == 2:
            avg_filtered = avg_filtered[..., None]
        score_lvl = avg_filtered.reshape(-1, n_levels).max(axis=0)
        select_level = int(np.argmax(score_lvl))
        level_map = avg_filtered[..., select_level]
        max_score = level_map.max()
        coords = np.asarray(np.nonzero(level_map == max_score)).T[:, ::-1]
        total_prompts += 1

        for box in img_ann[label]["bboxes"].reshape(-1, 4):
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            if np.any(
                (coords[:, 0] >= x_min)
                & (coords[:, 0] <= x_max)
                & (coords[:, 1] >= y_min)
                & (coords[:, 1] <= y_max)
            ):
                acc_num += 1
                break

    return chosen_iou_list, chosen_lvl_list, chosen_pixel_list, chosen_gt_list, acc_num, total_prompts


def run(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset_name = args.dataset_name
    source_path = Path(args.source_path) if args.source_path else ROOT_DIR / "data" / dataset_name / f"dust3r_{args.n_views}_views"
    json_folder = Path(args.json_folder) / args.label_dataset
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_ann, _ = eval_gt_lerfdata(json_folder)
    feature_levels = [int(x) for x in args.feature_levels.split(",") if x.strip()]
    clip_model = OpenCLIPNetwork(device)

    per_image = []
    all_ious = []
    all_pixels = []
    all_gt_pixels = []
    all_levels = []
    acc_num = 0
    total_prompts = 0

    render_maps = {}
    for level in feature_levels:
        render_dir = (
            Path(args.feat_dir)
            / f"{args.n_views}_views_{level}"
            / "train"
            / f"ours_{args.total_iters}"
            / "renders_npy"
        )
        render_maps[level] = load_render_index(source_path, render_dir)

    image_stems = sorted(render_maps[feature_levels[0]].keys())
    labeled_stems = [
        stem for stem in image_stems
        if str(int(stem.split("_")[-1]) - 1) in gt_ann
    ]
    if args.eval_stems:
        requested_stems = {
            stem.strip() for stem in args.eval_stems.split(",") if stem.strip()
        }
        labeled_stems = [stem for stem in labeled_stems if stem in requested_stems]
    if not labeled_stems:
        raise RuntimeError(
            f"No rendered train frames overlap labels in {json_folder}. "
            f"Train frames: {image_stems}"
        )

    for stem in tqdm(labeled_stems, desc=f"Semantic eval {dataset_name}/{args.n_views} views"):
        ann_idx = str(int(stem.split("_")[-1]) - 1)
        img_ann = gt_ann[ann_idx]
        clip_model.set_positives(list(img_ann.keys()))

        relevancy_levels = []
        for level in feature_levels:
            feature_512_path = source_path / "language_features" / f"{stem}_f.npy"
            feature_3d_path = source_path / "language_features_dim3" / f"{stem}_f.npy"
            relevancy_levels.append(
                relevancy_from_compressed(
                    render_maps[level][stem],
                    feature_512_path,
                    feature_3d_path,
                    clip_model,
                    device,
                    args.chunk_size,
                )
            )
        valid_map = torch.stack(relevancy_levels, dim=0)

        ious, levels, pixels, gt_pixels, img_acc, img_total = evaluate_relevancy(
            valid_map, img_ann, args.mask_thresh
        )
        all_ious.extend(ious)
        all_pixels.extend(pixels)
        all_gt_pixels.extend(gt_pixels)
        all_levels.extend(levels)
        acc_num += img_acc
        total_prompts += img_total
        per_image.append(
            {
                "image": stem,
                "num_prompts": img_total,
                "mean_iou": float(np.mean(ious)),
                "loc_acc": float(img_acc / img_total) if img_total else 0.0,
            }
        )

    result = {
        "dataset": dataset_name,
        "label_dataset": args.label_dataset,
        "n_views": args.n_views,
        "feature_levels": feature_levels,
        "eval_stems": labeled_stems,
        "mask_thresh": args.mask_thresh,
        "num_images": len(labeled_stems),
        "num_prompts": total_prompts,
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "iou2": float(np.sum(all_pixels) / (np.sum(all_gt_pixels) + 1e-9)),
        "loc_acc": float(acc_num / total_prompts) if total_prompts else 0.0,
        "chosen_levels": all_levels,
        "per_image": per_image,
        "timestamp": time.strftime("%Y%m%d_%H%M%S", time.localtime()),
    }

    summary_path = output_dir / f"{dataset_name}_{args.n_views}views_semantic.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    csv_path = output_dir / f"{dataset_name}_semantic_summary.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "dataset",
                "label_dataset",
                "n_views",
                "feature_levels",
                "num_images",
                "num_prompts",
                "mean_iou",
                "iou2",
                "loc_acc",
            ],
        )
        if write_header:
            writer.writeheader()
        row = result.copy()
        row["feature_levels"] = " ".join(map(str, feature_levels))
        writer.writerow({k: row[k] for k in writer.fieldnames})

    print(json.dumps({k: result[k] for k in [
        "dataset", "n_views", "feature_levels", "num_images",
        "num_prompts", "mean_iou", "iou2", "loc_acc"
    ]}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast LERF-style IoU/localization eval without visualization dumps.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--label_dataset", default=None)
    parser.add_argument("--feat_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--json_folder", required=True)
    parser.add_argument("--source_path", default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.6)
    parser.add_argument("--total_iters", type=int, default=1000)
    parser.add_argument("--n_views", type=int, required=True)
    parser.add_argument("--feature_levels", default="2")
    parser.add_argument(
        "--eval_stems",
        default="",
        help="Optional comma-separated frame stems, e.g. frame_00002,frame_00140.",
    )
    parser.add_argument("--chunk_size", type=int, default=131072)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.label_dataset is None:
        args.label_dataset = args.dataset_name
    seed_everything()
    run(args)


if __name__ == "__main__":
    main()
