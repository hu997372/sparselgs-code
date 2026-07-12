#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append("./RoMa-main")
from romatch import roma_indoor, roma_outdoor  # noqa: E402
from romatch.utils.utils import tensor_to_pil  # noqa: E402


LEVEL_NAMES = {0: "default", 1: "s", 2: "m", 3: "l"}
LERF_OVS_4VIEW_SCENES = {"teatime", "ramen", "waldo_kitchen", "figurines"}


@dataclass
class FusionConfig:
    dataname: str
    n_views: int
    feature_levels: tuple[int, ...]
    device: str
    matcher: str
    coarse_res: int
    match_threshold: float
    reproject_threshold: float
    semantic_weight: float
    fusion_area_threshold: float
    certainty_threshold: float
    min_roma_pixels: int
    min_reproject_pixels: int
    geom_start: int
    geom_times: float
    geom_dist_base: float
    geom_rel_diff_base: float
    mask_fusion: str
    roma_similarity_space: str
    reproject_similarity_space: str
    scoring: str
    refresh_origin: bool
    visualize: bool
    save_path: str
    stats_path: str | None
    snapshot_dir: str | None
    dry_run: bool


@dataclass
class StageStats:
    proposals: int = 0
    accepted: int = 0
    overwritten: int = 0
    mask_fusions: int = 0
    skipped_small: int = 0
    skipped_empty: int = 0
    skipped_oob: int = 0


@dataclass
class ViewFeatures:
    name: str
    image_path: Path
    seg_maps: torch.Tensor
    feat3: torch.Tensor
    feat512: torch.Tensor


@dataclass
class Candidate:
    score: float
    src_view: int
    src_seg: int
    dst_view: int
    dst_seg: int
    area_score: float
    semantic_score: float
    matched_pixels: int
    stage: str


def parse_levels(args: argparse.Namespace) -> tuple[int, ...]:
    if args.feature_levels:
        tokens = args.feature_levels.replace(",", " ").split()
        levels = tuple(int(token) for token in tokens)
    else:
        levels = (args.feature_level,)
    bad = [level for level in levels if level not in LEVEL_NAMES]
    if bad:
        raise ValueError(f"Unsupported feature levels: {bad}. Use 0, 1, 2, or 3.")
    return levels


def infer_n_views(dataname: str, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    return 4 if dataname in LERF_OVS_4VIEW_SCENES else 3


def copy_origin(origin_dir: Path, feature_dir: Path, refresh: bool) -> None:
    if refresh and origin_dir.exists():
        shutil.rmtree(origin_dir)
    if not origin_dir.exists():
        shutil.copytree(feature_dir, origin_dir)


def sorted_images(image_dir: Path) -> list[Path]:
    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return images


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) <= 1e-12:
        return 0.0
    return float(torch.dot(a, b) / denom)


def valid_counter(values: torch.Tensor) -> Counter:
    counter = Counter(values.detach().cpu().numpy().reshape(-1).tolist())
    counter.pop(-1, None)
    return counter


def majority_key(counter: Counter) -> int | None:
    if not counter:
        return None
    return int(max(counter, key=lambda key: counter[key]))


def safe_index_feature(features: torch.Tensor, index: int) -> torch.Tensor | None:
    if index < 0 or index >= features.shape[0]:
        return None
    return features[index]


def save_feature_visualization(
    feature_dir: Path,
    image_name: str,
    level: int,
    height: int,
    width: int,
    out_dir: Path,
) -> None:
    seg_map = torch.from_numpy(np.load(feature_dir / f"{image_name}_s.npy"))
    feature_map = torch.from_numpy(np.load(feature_dir / f"{image_name}_f.npy"))
    yy, xx = torch.meshgrid(
        torch.arange(height),
        torch.arange(width),
        indexing="ij",
    )
    seg = seg_map[level, yy, xx].long()
    point_feature = feature_map[seg].reshape(height, width, -1)
    if point_feature.shape[2] != 3:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.cvtColor(point_feature.detach().cpu().numpy(), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / f"{image_name}.jpg"), img * 255)


def save_match_visual(
    image_a: Path,
    image_b: Path,
    map_a_xy: torch.Tensor,
    map_b_xy: torch.Tensor,
    height: int,
    width: int,
    save_dir: Path,
    name: str,
    device: torch.device,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    im_a = Image.open(image_a).resize((width, height))
    im_b = Image.open(image_b).resize((width, height))
    x_a = (torch.tensor(np.array(im_a)) / 255).to(device).permute(2, 0, 1)
    x_b = (torch.tensor(np.array(im_b)) / 255).to(device).permute(2, 0, 1)
    blank_a = torch.zeros((3, height, width), device=device)
    blank_b = torch.zeros((3, height, width), device=device)
    map_a_xy = map_a_xy.long()
    map_b_xy = map_b_xy.long()
    blank_a[:, map_a_xy[:, 1], map_a_xy[:, 0]] = x_a[:, map_a_xy[:, 1], map_a_xy[:, 0]]
    blank_b[:, map_b_xy[:, 1], map_b_xy[:, 0]] = x_b[:, map_b_xy[:, 1], map_b_xy[:, 0]]
    tensor_to_pil(blank_a, unnormalize=False).save(save_dir / f"{name}_a.jpg")
    tensor_to_pil(blank_b, unnormalize=False).save(save_dir / f"{name}_b.jpg")


class SemanticFusion:
    def __init__(self, config: FusionConfig) -> None:
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )
        self.data_root = Path("data") / config.dataname / f"dust3r_{config.n_views}_views"
        self.image_dir = self.data_root / "images"
        self.camera_dir = self.data_root / "sparse" / "0"
        self.feat512_dir = self.data_root / "language_features"
        self.feat3_dir = self.data_root / "language_features_dim3"
        self.origin512_dir = self.data_root / "language_features_origin"
        self.origin3_dir = self.data_root / "language_features_origin_dim3"
        self.look_dir = self.data_root / "language_feature_renew_look"
        self.views: list[ViewFeatures] = []
        self.height = 0
        self.width = 0
        self.matcher_height = 0
        self.matcher_width = 0
        self.matcher_model = None
        self.stats: dict[str, StageStats] = defaultdict(StageStats)
        self.best_updates: dict[tuple[str, int, int], Candidate] = {}
        self.mask_collision: dict[int, dict[int, list[Candidate]]] = defaultdict(lambda: defaultdict(list))
        self.accepted_updates: list[dict] = []

    def run(self) -> dict:
        self.prepare_origins()
        self.load_views()
        self.init_matcher()

        for level in self.config.feature_levels:
            print(f"Semantic alignment level={level} ({LEVEL_NAMES[level]})")
            self.best_updates.clear()
            self.mask_collision.clear()
            self.roma_stage(level)
            self.save_snapshot(level, "01_roma")
            self.mask_fusion_stage(level)
            self.save_snapshot(level, "02_mask_fusion")
            self.reprojection_stage(level)
            self.save_snapshot(level, "03_reprojection")
            if self.config.visualize:
                self.visualize_level(level)

        if not self.config.dry_run:
            self.save_features()

        summary = self.summary()
        if self.config.stats_path:
            stats_path = Path(self.config.stats_path)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        return summary

    def prepare_origins(self) -> None:
        for path in [self.image_dir, self.feat512_dir, self.feat3_dir, self.camera_dir]:
            if not path.exists():
                raise FileNotFoundError(path)
        copy_origin(self.origin512_dir, self.feat512_dir, self.config.refresh_origin)
        copy_origin(self.origin3_dir, self.feat3_dir, self.config.refresh_origin)

    def load_views(self) -> None:
        image_paths = sorted_images(self.image_dir)
        first = cv2.imread(str(image_paths[0]))
        if first is None:
            raise RuntimeError(f"Failed to read image: {image_paths[0]}")
        self.height, self.width = first.shape[:2]

        self.views = []
        for image_path in image_paths:
            stem = image_path.stem
            seg_path = self.origin3_dir / f"{stem}_s.npy"
            feat3_path = self.origin3_dir / f"{stem}_f.npy"
            feat512_path = self.origin512_dir / f"{stem}_f.npy"
            for path in [seg_path, feat3_path, feat512_path]:
                if not path.exists():
                    raise FileNotFoundError(path)
            self.views.append(
                ViewFeatures(
                    name=stem,
                    image_path=image_path,
                    seg_maps=torch.from_numpy(np.load(seg_path)).long().to(self.device),
                    feat3=torch.from_numpy(np.load(feat3_path).astype(np.float32)).to(self.device),
                    feat512=torch.from_numpy(np.load(feat512_path).astype(np.float32)).to(self.device),
                )
            )
        if len(self.views) != self.config.n_views:
            print(
                f"[WARN] n_views={self.config.n_views}, but found {len(self.views)} images in {self.image_dir}"
            )

    def init_matcher(self) -> None:
        if self.config.matcher == "roma_outdoor":
            self.matcher_model = roma_outdoor(
                device=str(self.device),
                coarse_res=self.config.coarse_res,
                upsample_res=(self.height, self.width),
            )
        else:
            self.matcher_model = roma_indoor(
                device=str(self.device),
                coarse_res=self.config.coarse_res,
                upsample_res=(self.height, self.width),
            )
        self.matcher_height, self.matcher_width = self.matcher_model.get_output_resolution()

    def segment_area(self, view_id: int, level: int) -> Counter:
        return valid_counter(self.views[view_id].seg_maps[level])

    def segment_similarity(self, src_view: int, src_seg: int, dst_view: int, dst_seg: int, space: str) -> float:
        src = self.views[src_view]
        dst = self.views[dst_view]
        if space == "low":
            src_feat = safe_index_feature(src.feat3, src_seg)
            dst_feat = safe_index_feature(dst.feat3, dst_seg)
        else:
            src_feat = safe_index_feature(src.feat512, src_seg)
            dst_feat = safe_index_feature(dst.feat512, dst_seg)
        if src_feat is None or dst_feat is None:
            return 0.0
        return cosine(src_feat, dst_feat)

    def score(self, area_score: float, semantic_score: float) -> float:
        sem_w = self.config.semantic_weight
        return (1.0 - sem_w) * area_score + sem_w * semantic_score

    def apply_candidate(
        self,
        candidate: Candidate,
        threshold: float,
        best_scope: str,
    ) -> bool:
        if candidate.score <= threshold:
            return False
        dst = self.views[candidate.dst_view]
        src = self.views[candidate.src_view]
        src_feat3 = safe_index_feature(src.feat3, candidate.src_seg)
        src_feat512 = safe_index_feature(src.feat512, candidate.src_seg)
        if src_feat3 is None or src_feat512 is None:
            return False
        dst_feat3 = safe_index_feature(dst.feat3, candidate.dst_seg)
        dst_feat512 = safe_index_feature(dst.feat512, candidate.dst_seg)
        if dst_feat3 is None or dst_feat512 is None:
            return False

        key = (best_scope, candidate.dst_view, candidate.dst_seg)
        previous = self.best_updates.get(key)
        if previous is not None and previous.score >= candidate.score:
            return False

        dst.feat3[candidate.dst_seg] = src_feat3
        dst.feat512[candidate.dst_seg] = src_feat512
        self.best_updates[key] = candidate
        stat = self.stats[candidate.stage]
        stat.accepted += 1
        if previous is not None:
            stat.overwritten += 1
        self.accepted_updates.append(
            {
                "stage": candidate.stage,
                "score": candidate.score,
                "src_view": candidate.src_view,
                "src_name": src.name,
                "src_seg": candidate.src_seg,
                "dst_view": candidate.dst_view,
                "dst_name": dst.name,
                "dst_seg": candidate.dst_seg,
                "area_score": candidate.area_score,
                "semantic_score": candidate.semantic_score,
                "matched_pixels": candidate.matched_pixels,
                "threshold": threshold,
                "best_scope": best_scope,
                "overwrote": previous is not None,
            }
        )
        return True

    def roma_stage(self, level: int) -> None:
        print("RoMa pixel matching process")
        for src_view, src in tqdm(list(enumerate(self.views)), desc="roma views"):
            src_seg = src.seg_maps[level]
            src_areas = self.segment_area(src_view, level)
            if not src_areas:
                self.stats["roma"].skipped_empty += 1
                continue
            for dst_view, dst in enumerate(self.views):
                if dst_view == src_view:
                    continue
                dst_seg = dst.seg_maps[level]
                dst_areas = self.segment_area(dst_view, level)
                if not dst_areas:
                    self.stats["roma"].skipped_empty += 1
                    continue
                warp, certainty = self.matcher_model.match(
                    str(src.image_path),
                    str(dst.image_path),
                    device=str(self.device),
                )
                certain = certainty > self.config.certainty_threshold
                if int(certain.sum().item()) == 0:
                    self.stats["roma"].skipped_empty += 1
                    continue
                legacy_src_labels = None
                legacy_dst_counter = None
                if self.config.scoring == "legacy":
                    # The original script grouped source masks by seg_map[certain]
                    # and scored target coverage by seg_map2[certain], rather than
                    # by the RoMa-projected target coordinates. Keep that path for
                    # reproducibility; --scoring paper uses the corrected geometry.
                    legacy_src_labels = src_seg[certain]
                    legacy_dst_counter = valid_counter(dst_seg[certain])
                kpts_src, kpts_dst = self.matcher_model.to_pixel_coordinates(
                    warp,
                    self.matcher_height,
                    self.matcher_width,
                    self.matcher_height,
                    self.matcher_width,
                )
                kpts_src = kpts_src[certain].long()
                kpts_dst = kpts_dst[certain].long()
                valid = (
                    (kpts_src[:, 0] >= 0)
                    & (kpts_src[:, 0] < self.width)
                    & (kpts_src[:, 1] >= 0)
                    & (kpts_src[:, 1] < self.height)
                    & (kpts_dst[:, 0] >= 0)
                    & (kpts_dst[:, 0] < self.width)
                    & (kpts_dst[:, 1] >= 0)
                    & (kpts_dst[:, 1] < self.height)
                )
                if int(valid.sum().item()) == 0:
                    self.stats["roma"].skipped_oob += 1
                    continue
                kpts_src = kpts_src[valid]
                kpts_dst = kpts_dst[valid]
                if self.config.scoring == "legacy" and legacy_src_labels is not None:
                    src_labels = legacy_src_labels[valid]
                else:
                    src_labels = src_seg[kpts_src[:, 1], kpts_src[:, 0]]
                dst_labels = dst_seg[kpts_dst[:, 1], kpts_dst[:, 0]]
                src_counter = valid_counter(src_labels)

                for src_id in src_counter:
                    in_src = src_labels == src_id
                    dst_counter = valid_counter(dst_labels[in_src])
                    dst_id = majority_key(dst_counter)
                    if dst_id is None:
                        self.stats["roma"].skipped_empty += 1
                        continue
                    matched_pixels = int(dst_counter[dst_id])
                    if matched_pixels < self.config.min_roma_pixels:
                        self.stats["roma"].skipped_small += 1
                        continue
                    dst_area = max(1, int(dst_areas.get(dst_id, 1)))
                    if self.config.scoring == "legacy":
                        area_score = min(1.0, int((legacy_dst_counter or {}).get(dst_id, 0)) / dst_area)
                    else:
                        area_score = min(1.0, matched_pixels / dst_area)
                    semantic_score = self.segment_similarity(
                        src_view,
                        src_id,
                        dst_view,
                        dst_id,
                        self.config.roma_similarity_space,
                    )
                    total = self.score(area_score, semantic_score)
                    candidate = Candidate(
                        score=total,
                        src_view=src_view,
                        src_seg=src_id,
                        dst_view=dst_view,
                        dst_seg=dst_id,
                        area_score=area_score,
                        semantic_score=semantic_score,
                        matched_pixels=matched_pixels,
                        stage="roma",
                    )
                    self.stats["roma"].proposals += 1
                    self.apply_candidate(candidate, self.config.match_threshold, "roma")
                    if (
                        total > 2.0 * self.config.match_threshold
                        and area_score > self.config.fusion_area_threshold
                    ):
                        self.mask_collision[dst_view][dst_id].append(candidate)

                    if self.config.visualize:
                        dst_match = dst_labels == dst_id
                        keep = in_src & dst_match
                        if int(keep.sum().item()) > 0:
                            save_match_visual(
                                src.image_path,
                                dst.image_path,
                                kpts_src[keep],
                                kpts_dst[keep],
                                self.height,
                                self.width,
                                Path(self.config.save_path),
                                f"roma_l{level}_v{src_view}_{src_id}_to_v{dst_view}_{dst_id}",
                                self.device,
                            )

    def should_run_mask_fusion(self, level: int) -> bool:
        if self.config.mask_fusion == "on":
            return True
        if self.config.mask_fusion == "off":
            return False
        return level >= 2

    def mask_fusion_stage(self, level: int) -> None:
        print("inconsistent mask fusion process")
        if not self.should_run_mask_fusion(level):
            return
        for dst_view, per_seg in self.mask_collision.items():
            dst_features = self.views[dst_view]
            for dst_seg, candidates in per_seg.items():
                dst_feat3 = safe_index_feature(dst_features.feat3, dst_seg)
                dst_feat512 = safe_index_feature(dst_features.feat512, dst_seg)
                if dst_feat3 is None or dst_feat512 is None:
                    continue
                for candidate in candidates:
                    src_features = self.views[candidate.src_view]
                    if candidate.src_seg >= src_features.feat3.shape[0]:
                        continue
                    src_features.feat3[candidate.src_seg] = dst_feat3
                    src_features.feat512[candidate.src_seg] = dst_feat512
                    self.stats["mask_fusion"].mask_fusions += 1
                    self.accepted_updates.append(
                        {
                            "stage": "mask_fusion",
                            "score": candidate.score,
                            "src_view": dst_view,
                            "src_name": dst_features.name,
                            "src_seg": dst_seg,
                            "dst_view": candidate.src_view,
                            "dst_name": src_features.name,
                            "dst_seg": candidate.src_seg,
                            "area_score": candidate.area_score,
                            "semantic_score": candidate.semantic_score,
                            "matched_pixels": candidate.matched_pixels,
                            "threshold": None,
                            "best_scope": "mask_fusion",
                            "overwrote": True,
                        }
                    )

    def reprojection_stage(self, level: int) -> None:
        print("depth reprojection matching process")
        intrinsics = torch.load(self.camera_dir / "intrinsics.pt", map_location="cpu").detach().cpu().numpy()
        poses = torch.load(self.camera_dir / "poses.pt", map_location="cpu").detach().cpu().numpy()
        depths_t = torch.load(self.camera_dir / "depths.pt", map_location="cpu")
        if isinstance(depths_t, torch.Tensor):
            depths = depths_t.detach().cpu().numpy()
        else:
            depths = [
                depth.detach().cpu().numpy() if isinstance(depth, torch.Tensor) else np.asarray(depth)
                for depth in depths_t
            ]
        n_loaded = min(len(self.views), len(depths), len(intrinsics), len(poses))
        if n_loaded != len(self.views):
            print(f"[WARN] Camera/depth count {n_loaded} differs from image count {len(self.views)}")

        for ref_view in tqdm(range(n_loaded), desc="reprojection views"):
            ref_seg = self.views[ref_view].seg_maps[level]
            ref_intr = intrinsics[ref_view]
            ref_extr = np.linalg.inv(poses[ref_view])
            ref_depth = depths[ref_view]
            for dst_view in range(n_loaded):
                if dst_view == ref_view:
                    continue
                dst_intr = intrinsics[dst_view]
                dst_extr = np.linalg.inv(poses[dst_view])
                dst_depth = depths[dst_view]
                geom = self.geometric_correspondences(
                    ref_depth,
                    ref_intr,
                    ref_extr,
                    dst_depth,
                    dst_intr,
                    dst_extr,
                )
                if geom is None:
                    self.stats["reproject"].skipped_empty += 1
                    continue
                ref_y, ref_x, dst_y, dst_x = geom
                dst_seg = self.views[dst_view].seg_maps[level]
                ref_labels = ref_seg[ref_y, ref_x]
                dst_labels = dst_seg[dst_y, dst_x]
                ref_counter = valid_counter(ref_labels)
                dst_areas = self.segment_area(dst_view, level)
                for ref_id in ref_counter:
                    in_ref = ref_labels == ref_id
                    dst_counter = valid_counter(dst_labels[in_ref])
                    dst_id = majority_key(dst_counter)
                    if dst_id is None:
                        self.stats["reproject"].skipped_empty += 1
                        continue
                    matched_pixels = int(dst_counter[dst_id])
                    if matched_pixels < self.config.min_reproject_pixels:
                        self.stats["reproject"].skipped_small += 1
                        continue
                    dst_area = max(1, int(dst_areas.get(dst_id, 1)))
                    area_score = min(1.0, matched_pixels / dst_area)
                    semantic_score = self.segment_similarity(
                        ref_view,
                        ref_id,
                        dst_view,
                        dst_id,
                        self.config.reproject_similarity_space,
                    )
                    total = self.score(area_score, semantic_score)
                    candidate = Candidate(
                        score=total,
                        src_view=ref_view,
                        src_seg=ref_id,
                        dst_view=dst_view,
                        dst_seg=dst_id,
                        area_score=area_score,
                        semantic_score=semantic_score,
                        matched_pixels=matched_pixels,
                        stage="reproject",
                    )
                    self.stats["reproject"].proposals += 1
                    self.apply_candidate(candidate, self.config.reproject_threshold, "reproject")

    def geometric_correspondences(
        self,
        depth_ref: np.ndarray,
        intr_ref: np.ndarray,
        extr_ref: np.ndarray,
        depth_dst: np.ndarray,
        intr_dst: np.ndarray,
        extr_dst: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        h0, w0 = depth_ref.shape[:2]
        cols, rows = np.meshgrid(np.arange(w0), np.arange(h0))
        flat_cols = cols.reshape(-1)
        flat_rows = rows.reshape(-1)
        flat_depth = depth_ref.reshape(-1)
        valid_ref = flat_depth > 0
        if not np.any(valid_ref):
            return None

        pix = np.vstack((flat_cols, flat_rows, np.ones_like(flat_cols)))
        xyz_ref = np.linalg.inv(intr_ref) @ (pix * flat_depth.reshape(1, -1))
        xyz_dst = (extr_dst @ np.linalg.inv(extr_ref) @ np.vstack((xyz_ref, np.ones_like(flat_cols))))[:3]
        z_dst = xyz_dst[2]
        valid_z = z_dst > 1e-8
        projected = intr_dst @ xyz_dst
        projected[:2] /= np.maximum(projected[2:3], 1e-8)
        dst_x = projected[0].reshape(h0, w0).astype(np.float32)
        dst_y = projected[1].reshape(h0, w0).astype(np.float32)

        sampled_dst_depth = cv2.remap(
            depth_dst.astype(np.float32),
            dst_x,
            dst_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        sampled_flat = sampled_dst_depth.reshape(-1)
        xyz_dst_sampled = np.linalg.inv(intr_dst) @ (projected * sampled_flat.reshape(1, -1))
        xyz_ref_back = (extr_ref @ np.linalg.inv(extr_dst) @ np.vstack((xyz_dst_sampled, np.ones_like(flat_cols))))[:3]
        projected_back = intr_ref @ xyz_ref_back
        projected_back[:2] /= np.maximum(projected_back[2:3], 1e-8)
        back_x = projected_back[0].reshape(h0, w0)
        back_y = projected_back[1].reshape(h0, w0)
        back_depth = xyz_ref_back[2].reshape(h0, w0)

        dist = np.sqrt((back_x - cols) ** 2 + (back_y - rows) ** 2)
        rel_depth = np.abs(back_depth - depth_ref) / np.maximum(depth_ref, 1e-8)
        valid = valid_ref.reshape(h0, w0) & valid_z.reshape(h0, w0) & (sampled_dst_depth > 0)

        mask = np.zeros((h0, w0), dtype=bool)
        max_votes = max(self.config.geom_start + 1, len(self.views))
        for i in range(self.config.geom_start, max_votes + 1):
            mask |= (
                valid
                & (dist < i * self.config.geom_dist_base)
                & (rel_depth < i * self.config.geom_rel_diff_base)
            )
        if not np.any(mask):
            return None

        ref_rows, ref_cols = np.nonzero(mask)
        scale_y = self.height / h0
        scale_x = self.width / w0
        src_x = np.floor(dst_x[mask] * scale_x).astype(np.int64)
        src_y = np.floor(dst_y[mask] * scale_y).astype(np.int64)
        ref_x = np.floor(ref_cols * scale_x).astype(np.int64)
        ref_y = np.floor(ref_rows * scale_y).astype(np.int64)
        in_bounds = (
            (src_x >= 0)
            & (src_x < self.width)
            & (src_y >= 0)
            & (src_y < self.height)
            & (ref_x >= 0)
            & (ref_x < self.width)
            & (ref_y >= 0)
            & (ref_y < self.height)
        )
        if not np.any(in_bounds):
            return None
        return (
            torch.from_numpy(ref_y[in_bounds]).long().to(self.device),
            torch.from_numpy(ref_x[in_bounds]).long().to(self.device),
            torch.from_numpy(src_y[in_bounds]).long().to(self.device),
            torch.from_numpy(src_x[in_bounds]).long().to(self.device),
        )

    def visualize_level(self, level: int) -> None:
        for view in self.views:
            save_feature_visualization(
                self.feat3_dir,
                view.name,
                level,
                self.height,
                self.width,
                self.look_dir / f"level_{level}",
            )

    def save_features(self) -> None:
        for view in self.views:
            np.save(self.feat3_dir / f"{view.name}_f.npy", view.feat3.detach().cpu().numpy())
            np.save(self.feat512_dir / f"{view.name}_f.npy", view.feat512.detach().cpu().numpy())

    def save_snapshot(self, level: int, stage: str) -> None:
        if not self.config.snapshot_dir:
            return
        snapshot_root = Path(self.config.snapshot_dir) / f"level_{level}" / stage
        dim3_dir = snapshot_root / "language_features_dim3"
        dim512_dir = snapshot_root / "language_features"
        dim3_dir.mkdir(parents=True, exist_ok=True)
        dim512_dir.mkdir(parents=True, exist_ok=True)
        for view in self.views:
            np.save(dim3_dir / f"{view.name}_f.npy", view.feat3.detach().cpu().numpy())
            np.save(dim512_dir / f"{view.name}_f.npy", view.feat512.detach().cpu().numpy())

    def summary(self) -> dict:
        return {
            "timestamp": time.strftime("%Y%m%d_%H%M%S", time.localtime()),
            "data_root": str(self.data_root),
            "config": asdict(self.config),
            "num_views": len(self.views),
            "image_size": [self.height, self.width],
            "stats": {stage: asdict(stats) for stage, stats in sorted(self.stats.items())},
            "accepted_updates": self.accepted_updates,
        }


def build_config(args: argparse.Namespace) -> FusionConfig:
    n_views = infer_n_views(args.dataname, args.n_views)
    levels = parse_levels(args)
    dist_base = args.geom_dist_base
    rel_diff_base = args.geom_rel_diff_base
    if dist_base is None:
        dist_base = 0.125 * args.geom_times
    if rel_diff_base is None:
        rel_diff_base = 0.1 * args.geom_times
    return FusionConfig(
        dataname=args.dataname,
        n_views=n_views,
        feature_levels=levels,
        device=args.device,
        matcher=args.matcher,
        coarse_res=args.coarse_res,
        match_threshold=args.match_threshold,
        reproject_threshold=args.reproject_threshold,
        semantic_weight=args.semantic_weight,
        fusion_area_threshold=args.fusion_area_threshold,
        certainty_threshold=args.certainty_threshold,
        min_roma_pixels=args.min_roma_pixels,
        min_reproject_pixels=args.min_reproject_pixels,
        geom_start=args.geom_start,
        geom_times=args.geom_times,
        geom_dist_base=dist_base,
        geom_rel_diff_base=rel_diff_base,
        mask_fusion=args.mask_fusion,
        roma_similarity_space=args.roma_similarity_space,
        reproject_similarity_space=args.reproject_similarity_space,
        scoring=args.scoring,
        refresh_origin=args.refresh_origin,
        visualize=not args.no_visualization,
        save_path=args.save_path,
        stats_path=args.stats_path,
        snapshot_dir=args.snapshot_dir,
        dry_run=args.dry_run,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-view semantic feature alignment using RoMa and depth reprojection."
    )
    parser.add_argument("--dataname", required=True)
    parser.add_argument("--n_views", type=int, default=None)
    parser.add_argument("--feature_level", type=int, default=2)
    parser.add_argument(
        "--feature_levels",
        default="",
        help="Optional comma/space separated levels. Overrides --feature_level, e.g. 1,2,3.",
    )
    parser.add_argument("--refresh_origin", action="store_true")
    parser.add_argument("--no_visualization", action="store_true")
    parser.add_argument("--save_path", default="./data/match_results")
    parser.add_argument("--stats_path", default="")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matcher", choices=["roma_indoor", "roma_outdoor"], default="roma_indoor")
    parser.add_argument("--coarse_res", type=int, default=560)
    parser.add_argument(
        "--similarity_space",
        choices=["high", "low"],
        default=None,
        help="Compatibility alias. Sets both RoMa and reprojection similarity spaces.",
    )
    parser.add_argument("--roma_similarity_space", choices=["high", "low"], default="low")
    parser.add_argument("--reproject_similarity_space", choices=["high", "low"], default="high")
    parser.add_argument("--scoring", choices=["paper", "legacy"], default="legacy")
    parser.add_argument("--mask_fusion", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--snapshot_dir", default="")

    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--reproject_threshold", type=float, default=0.5)
    parser.add_argument("--semantic_weight", type=float, default=0.3)
    parser.add_argument("--fusion_area_threshold", type=float, default=0.3)
    parser.add_argument("--certainty_threshold", type=float, default=0.35)
    parser.add_argument("--min_roma_pixels", type=int, default=150)
    parser.add_argument("--min_reproject_pixels", type=int, default=150)
    parser.add_argument("--geom_start", type=int, default=1)
    parser.add_argument("--geom_times", type=float, default=1.5)
    parser.add_argument("--geom_dist_base", type=float, default=None)
    parser.add_argument("--geom_rel_diff_base", type=float, default=None)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.similarity_space is not None:
        args.roma_similarity_space = args.similarity_space
        args.reproject_similarity_space = args.similarity_space
    if not args.stats_path:
        args.stats_path = None
    if not args.snapshot_dir:
        args.snapshot_dir = None
    config = build_config(args)
    SemanticFusion(config).run()


if __name__ == "__main__":
    main()
