import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
os.sys.path.insert(0, str(BASE_DIR / "mast3r"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment  # noqa: E402
from mast3r.image_pairs import make_pairs  # noqa: E402
from mast3r.model import AsymmetricMASt3R  # noqa: E402

import mast3r.utils.path_to_dust3r  # noqa: E402,F401
from dust3r.utils.device import to_numpy  # noqa: E402
from dust3r.utils.image import load_images  # noqa: E402
from utils.dust3r_utils import save_colmap_cameras, save_colmap_images, storePly  # noqa: E402


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224])
    parser.add_argument(
        "--model_path",
        type=str,
        default="./ckpt/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument("--img_base_path", type=str, required=True)
    parser.add_argument("--llffhold", type=int, default=0)
    parser.add_argument("--scene_graph", type=str, default="complete")
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--lr1", type=float, default=0.07)
    parser.add_argument("--niter1", type=int, default=300)
    parser.add_argument("--lr2", type=float, default=0.01)
    parser.add_argument("--niter2", type=int, default=300)
    parser.add_argument("--matching_conf_thr", type=float, default=5.0)
    parser.add_argument("--min_conf_thr", type=float, default=2.0)
    parser.add_argument("--shared_intrinsics", action="store_true")
    parser.add_argument("--no_clean_depth", action="store_true")
    return parser


def image_names(path):
    suffixes = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sorted(p.name for p in Path(path).iterdir() if p.suffix in suffixes)


def ensure_sparse_view_images(img_base_path, n_views, img_folder_path, llffhold):
    img_folder_path.mkdir(parents=True, exist_ok=True)
    existing = image_names(img_folder_path)
    if len(existing) == n_views:
        return existing

    full_img_dir = Path(img_base_path) / "images"
    if not full_img_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {img_folder_path} to contain {n_views} images, or {full_img_dir} to exist."
        )

    train_img_list = image_names(full_img_dir)
    if llffhold > 0:
        train_img_list = [c for idx, c in enumerate(train_img_list) if (idx + 1) % llffhold != 0]

    indices = np.linspace(0, len(train_img_list) - 1, n_views, dtype=int)
    train_img_list = [train_img_list[i] for i in indices]
    for img_name in train_img_list:
        shutil.copy(full_img_dir / img_name, img_folder_path / img_name)
    return train_img_list


def flatten_valid_points(points, colors, confidences, min_conf_thr):
    xyz_all = []
    rgb_all = []
    for pts, img, conf in zip(points, colors, confidences):
        pts_np = to_numpy(pts).reshape(-1, 3)
        rgb_np = to_numpy(img).reshape(-1, 3)
        conf_np = to_numpy(conf).reshape(-1)
        valid = (conf_np > min_conf_thr) & np.isfinite(pts_np).all(axis=1)
        xyz_all.append(pts_np[valid])
        rgb_all.append(rgb_np[valid])
    xyz = np.concatenate(xyz_all, axis=0)
    rgb = (np.clip(np.concatenate(rgb_all, axis=0), 0.0, 1.0) * 255.0).astype(np.uint8)
    return xyz, rgb


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    img_base_path = Path(args.img_base_path)
    img_folder_path = img_base_path / f"dust3r_{args.n_views}_views" / "images"
    output_colmap_path = img_folder_path.parent / "sparse" / "0"
    cache_path = output_colmap_path.parent / "mast3r_cache"
    output_colmap_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)

    train_img_list = ensure_sparse_view_images(img_base_path, args.n_views, img_folder_path, args.llffhold)
    if len(train_img_list) != args.n_views:
        raise ValueError(f"Expected {args.n_views} images, found {len(train_img_list)} in {img_folder_path}")

    filelist = [str(img_folder_path / name) for name in train_img_list]
    ori_size = Image.open(filelist[0]).size
    print("Selected images:", train_img_list)
    print("Original image size:", ori_size)

    model = AsymmetricMASt3R.from_pretrained(args.model_path).to(args.device).eval()
    imgs = load_images(filelist, size=args.image_size, verbose=True)
    pairs = make_pairs(imgs, scene_graph=args.scene_graph, prefilter=None, symmetrize=True)

    start_time = time.time()
    scene = sparse_global_alignment(
        filelist,
        pairs,
        str(cache_path),
        model,
        lr1=args.lr1,
        niter1=args.niter1,
        lr2=args.lr2,
        niter2=args.niter2,
        device=args.device,
        shared_intrinsics=args.shared_intrinsics,
        matching_conf_thr=args.matching_conf_thr,
        subsample=args.subsample,
    )
    points, depths, confidences = scene.get_dense_pts3d(
        clean_depth=not args.no_clean_depth,
        subsample=args.subsample,
    )
    print(f"Time taken for {args.n_views} MASt3R views: {time.time() - start_time:.2f} seconds")

    intrinsics = scene.intrinsics
    poses = scene.get_im_poses()
    focals = scene.get_focals()

    save_colmap_cameras(ori_size, to_numpy(intrinsics), str(output_colmap_path / "cameras.txt"))
    save_colmap_images(to_numpy(poses), str(output_colmap_path / "images.txt"), train_img_list)

    xyz, rgb = flatten_valid_points(points, scene.imgs, confidences, args.min_conf_thr)
    storePly(str(output_colmap_path / "points3D.ply"), xyz, rgb)
    np.save(output_colmap_path / "pts_4_3dgs_all.npy", np.concatenate([to_numpy(p).reshape(-1, 3) for p in points]))
    np.save(output_colmap_path / "focal.npy", to_numpy(focals))
    torch.save(intrinsics.detach().cpu(), output_colmap_path / "intrinsics.pt")
    torch.save(poses.detach().cpu(), output_colmap_path / "poses.pt")
    torch.save([d.detach().cpu() for d in depths], output_colmap_path / "depths.pt")
    print(f"Wrote MASt3R initialization to {output_colmap_path} ({xyz.shape[0]} points)")
