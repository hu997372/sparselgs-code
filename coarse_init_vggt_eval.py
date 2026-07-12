import argparse
import os
import random
import shutil
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
os.sys.path.insert(0, str(BASE_DIR / "vggt"))

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap  # noqa: E402
from vggt.dependency.track_predict import predict_tracks  # noqa: E402
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images_square  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
from plyfile import PlyData, PlyElement  # noqa: E402
import pycolmap  # noqa: E402


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./ckpt/model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument("--img_base_path", type=str, required=True)
    parser.add_argument("--llffhold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--img_load_resolution", type=int, default=1024)
    parser.add_argument("--conf_thres_value", type=float, default=5.0)
    parser.add_argument("--max_points", type=int, default=0, help="0 keeps all confidence-filtered points.")
    parser.add_argument("--use_ba", action="store_true")
    parser.add_argument(
        "--ba_point_source",
        choices=("dense_vggt", "sparse_reconstruction"),
        default="dense_vggt",
        help="Point cloud to export after BA. dense_vggt keeps VGGT depth points for full-pointcloud ablations.",
    )
    parser.add_argument("--max_reproj_error", type=float, default=8.0)
    parser.add_argument("--shared_camera", action="store_true", default=False)
    parser.add_argument("--camera_type", type=str, default="PINHOLE")
    parser.add_argument("--vis_thresh", type=float, default=0.2)
    parser.add_argument("--query_frame_num", type=int, default=8)
    parser.add_argument("--max_query_pts", type=int, default=4096)
    parser.add_argument("--max_points_num", type=int, default=163840)
    parser.add_argument("--keypoint_extractor", type=str, default="aliked+sp")
    parser.add_argument("--no_fine_tracking", action="store_true")
    parser.add_argument("--no_complete_non_vis", action="store_true")
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


def save_colmap_cameras_direct(intrinsics, sizes, camera_file):
    with open(camera_file, "w", encoding="utf-8") as f:
        for i, (K, (width, height)) in enumerate(zip(intrinsics, sizes), 1):
            f.write(
                f"{i} PINHOLE {int(width)} {int(height)} "
                f"{K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n"
            )


def camera_model_name(camera):
    if hasattr(camera, "model_name"):
        return camera.model_name
    model = camera.model
    return model.name if hasattr(model, "name") else str(model)


def save_reconstruction_cameras(reconstruction, camera_file):
    with open(camera_file, "w", encoding="utf-8") as f:
        for camera_id in sorted(reconstruction.cameras):
            camera = reconstruction.cameras[camera_id]
            params = " ".join(str(float(v)) for v in camera.params)
            f.write(f"{camera.camera_id} {camera_model_name(camera)} {int(camera.width)} {int(camera.height)} {params}\n")


def save_reconstruction_images(reconstruction, images_file):
    with open(images_file, "w", encoding="utf-8") as f:
        for image_id in sorted(reconstruction.images):
            image = reconstruction.images[image_id]
            extrinsic = image.cam_from_world.matrix()
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            q = rotation_to_quaternion(R)
            f.write(
                f"{image.image_id} {q[0]} {q[1]} {q[2]} {q[3]} "
                f"{t[0]} {t[1]} {t[2]} {image.camera_id} {image.name}\n"
            )
            f.write("\n")


def reconstruction_points(reconstruction):
    xyz = []
    rgb = []
    for point3D_id in sorted(reconstruction.points3D):
        point = reconstruction.points3D[point3D_id]
        xyz.append(point.xyz)
        color = point.color if hasattr(point, "color") else point.rgb
        rgb.append(color)
    if not xyz:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    return np.asarray(xyz, dtype=np.float32), np.asarray(rgb, dtype=np.uint8)


def reconstruction_matrices(reconstruction):
    intrinsics = []
    extrinsics = []
    for image_id in sorted(reconstruction.images):
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]
        intrinsics.append(camera.calibration_matrix())
        extrinsics.append(image.cam_from_world.matrix())
    return np.stack(extrinsics).astype(np.float32), np.stack(intrinsics).astype(np.float32)


def rename_and_rescale_reconstruction(reconstruction, image_names_, original_coords, img_size, shared_camera=False):
    rescale_camera = True
    for image_id in sorted(reconstruction.images):
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]
        zero_idx = image_id - 1
        image.name = image_names_[zero_idx]

        if rescale_camera:
            pred_params = np.asarray(camera.params).copy()
            real_image_size = original_coords[zero_idx, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params *= resize_ratio
            pred_params[-2:] = real_image_size / 2.0
            camera.params = pred_params
            camera.width = int(real_image_size[0])
            camera.height = int(real_image_size[1])

        top_left = original_coords[zero_idx, :2]
        for point2D in image.points2D:
            point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False
    return reconstruction


def store_ply(path, xyz, rgb):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, np.concatenate((xyz, normals, rgb), axis=1)))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def rotation_to_quaternion(R):
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s])
    if m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        return np.array([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s])
    if m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        return np.array([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s])
    s = np.sqrt(1.0 + m22 - m00 - m11) * 2
    return np.array([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s])


def save_colmap_images_from_w2c(extrinsics, images_file, image_names_):
    with open(images_file, "w", encoding="utf-8") as f:
        for i, extrinsic in enumerate(extrinsics, 1):
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            q = rotation_to_quaternion(R)
            f.write(f"{i} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {i} {image_names_[i - 1]}\n")
            f.write("\n")


def run_vggt(model, images, dtype, resolution):
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype, enabled=images.is_cuda):
            batch = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(batch)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batch.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, batch, ps_idx)

    return (
        extrinsic.squeeze(0).cpu().numpy(),
        intrinsic.squeeze(0).cpu().numpy(),
        depth_map.squeeze(0).cpu().numpy(),
        depth_conf.squeeze(0).cpu().numpy(),
        images.detach().cpu(),
    )


def rescale_intrinsics_to_original(intrinsics, original_coords, resolution):
    intrinsics_orig = intrinsics.copy()
    sizes = []
    for i in range(len(intrinsics_orig)):
        real_w, real_h = original_coords[i, -2:]
        resize_ratio = max(real_w, real_h) / resolution
        intrinsics_orig[i, :2, :] *= resize_ratio
        intrinsics_orig[i, 0, 2] = real_w / 2.0
        intrinsics_orig[i, 1, 2] = real_h / 2.0
        sizes.append((int(real_w), int(real_h)))
    return intrinsics_orig, sizes


def confidence_mask(depth_conf, original_coords, resolution, threshold):
    conf = depth_conf[..., 0] if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1 else depth_conf
    masks = conf >= threshold
    yy, xx = np.mgrid[:resolution, :resolution]
    for i in range(len(masks)):
        x1, y1, x2, y2 = original_coords[i, :4]
        in_image = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
        masks[i] &= in_image
    return masks


def ba_reconstruction(args, images, original_coords_np, train_img_list, extrinsic, intrinsic, depth_map, depth_conf):
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    image_size = np.array(images.shape[-2:])
    scale = args.img_load_resolution / args.resolution

    dtype = torch.bfloat16 if images.is_cuda and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype, enabled=images.is_cuda):
        pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
            images,
            conf=depth_conf,
            points_3d=points_3d,
            masks=None,
            max_query_pts=args.max_query_pts,
            query_frame_num=args.query_frame_num,
            max_points_num=args.max_points_num,
            keypoint_extractor=args.keypoint_extractor,
            fine_tracking=not args.no_fine_tracking,
            complete_non_vis=not args.no_complete_non_vis,
        )
    torch.cuda.empty_cache()

    intrinsic_ba = intrinsic.copy()
    intrinsic_ba[:, :2, :] *= scale
    track_mask = pred_vis_scores > args.vis_thresh
    reconstruction, _ = batch_np_matrix_to_pycolmap(
        points_3d,
        extrinsic,
        intrinsic_ba,
        pred_tracks,
        image_size,
        masks=track_mask,
        max_reproj_error=args.max_reproj_error,
        shared_camera=args.shared_camera,
        camera_type=args.camera_type,
        points_rgb=points_rgb,
    )
    if reconstruction is None:
        raise RuntimeError("VGGT+BA could not build a reconstruction.")

    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, ba_options)
    return rename_and_rescale_reconstruction(
        reconstruction,
        train_img_list,
        original_coords_np,
        img_size=args.img_load_resolution,
        shared_camera=args.shared_camera,
    )


def maybe_limit_points(points, colors, max_points, seed):
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(points), size=max_points, replace=False)
    return points[keep], colors[keep]


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    img_base_path = Path(args.img_base_path)
    img_folder_path = img_base_path / f"dust3r_{args.n_views}_views" / "images"
    output_colmap_path = img_folder_path.parent / "sparse" / "0"
    output_colmap_path.mkdir(parents=True, exist_ok=True)

    train_img_list = ensure_sparse_view_images(img_base_path, args.n_views, img_folder_path, args.llffhold)
    if len(train_img_list) != args.n_views:
        raise ValueError(f"Expected {args.n_views} images, found {len(train_img_list)} in {img_folder_path}")
    filelist = [str(img_folder_path / name) for name in train_img_list]
    print("Selected images:", train_img_list)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    print(f"Using device={args.device}, dtype={dtype}")

    start_time = time.time()
    model = VGGT()
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().to(args.device)

    load_resolution = args.img_load_resolution if args.use_ba else args.resolution
    images, original_coords = load_and_preprocess_images_square(filelist, load_resolution)
    images = images.to(args.device)
    original_coords_np = original_coords.cpu().numpy()

    extrinsic, intrinsic, depth_map, depth_conf, square_images = run_vggt(model, images, dtype, args.resolution)
    if args.use_ba:
        reconstruction = ba_reconstruction(
            args, images, original_coords_np, train_img_list, extrinsic, intrinsic, depth_map, depth_conf
        )
        extrinsic_ba, intrinsics_orig = reconstruction_matrices(reconstruction)
        sparse_xyz, sparse_rgb = reconstruction_points(reconstruction)
        if args.ba_point_source == "sparse_reconstruction":
            xyz, rgb = sparse_xyz, sparse_rgb
        else:
            points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
            masks = confidence_mask(depth_conf, original_coords_np, args.resolution, args.conf_thres_value)
            valid = masks & np.isfinite(points_3d).all(axis=-1)
            colors = (square_images.permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            xyz = points_3d[valid]
            rgb = colors[valid]
            xyz, rgb = maybe_limit_points(xyz, rgb, args.max_points, args.seed)
            print(
                f"Using dense VGGT point cloud after BA: {xyz.shape[0]} points "
                f"(sparse reconstruction had {sparse_xyz.shape[0]} points)"
            )
        c2w = closed_form_inverse_se3(extrinsic_ba).astype(np.float32)
        save_reconstruction_cameras(reconstruction, str(output_colmap_path / "cameras.txt"))
        save_reconstruction_images(reconstruction, str(output_colmap_path / "images.txt"))
    else:
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
        intrinsics_orig, sizes = rescale_intrinsics_to_original(intrinsic, original_coords_np, args.resolution)
        masks = confidence_mask(depth_conf, original_coords_np, args.resolution, args.conf_thres_value)
        valid = masks & np.isfinite(points_3d).all(axis=-1)
        colors = (square_images.permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        xyz = points_3d[valid]
        rgb = colors[valid]
        xyz, rgb = maybe_limit_points(xyz, rgb, args.max_points, args.seed)
        c2w = closed_form_inverse_se3(extrinsic).astype(np.float32)
        save_colmap_cameras_direct(intrinsics_orig, sizes, str(output_colmap_path / "cameras.txt"))
        save_colmap_images_from_w2c(extrinsic, str(output_colmap_path / "images.txt"), train_img_list)

    store_ply(str(output_colmap_path / "points3D.ply"), xyz.astype(np.float32), rgb)
    np.save(output_colmap_path / "pts_4_3dgs_all.npy", xyz.astype(np.float32))
    np.save(output_colmap_path / "focal.npy", intrinsics_orig[:, 0, 0])
    torch.save(torch.from_numpy(intrinsics_orig.astype(np.float32)), output_colmap_path / "intrinsics.pt")
    torch.save(torch.from_numpy(c2w), output_colmap_path / "poses.pt")
    torch.save(torch.from_numpy(depth_map.astype(np.float32)), output_colmap_path / "depths.pt")

    for image_name in train_img_list:
        Image.open(img_folder_path / image_name).verify()
    print(f"Time taken for {args.n_views} VGGT views: {time.time() - start_time:.2f} seconds")
    print(f"Wrote VGGT initialization to {output_colmap_path} ({xyz.shape[0]} points)")
