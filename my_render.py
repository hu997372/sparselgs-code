from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import torch
import numpy as np
from gaussian_renderer import render
from gaussian_renderer import GaussianModel
import os
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.pose_utils import get_tensor_from_camera
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, read_extrinsics_binary, read_intrinsics_binary
from scene.dataset_readers import readColmapCameras
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy
import utils.color_utils as colormaps
from utils.openclip_encoder import OpenCLIPNetwork, OpenCLIPNetworkConfig

def visualizer(camera_poses, colors, save_path="/mnt/data/1.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for pose, color in zip(camera_poses, colors):
        rotation = pose[:3, :3]
        translation = pose[:3, 3]  # Corrected to use 3D translation component
        camera_positions = np.einsum(
            "...ij,...j->...i", np.linalg.inv(rotation), -translation
        )

        ax.scatter(
            camera_positions[0],
            camera_positions[1],
            camera_positions[2],
            c=color,
            marker="o",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Poses")

    plt.savefig(save_path)
    plt.close()

    return save_path

def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def viewmatrix(lookdir, up, position):
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def generate_interpolated_path(poses, n_interp, spline_degree=5,
                               smoothness=.03, rot_weight=.1):
    """Creates a smooth spline path between input keyframe camera poses.

  Spline is calculated with poses in format (position, lookat-point, up-point).

  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
  """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree,
                        s=smoothness)
    return points_to_poses(new_points) 

class DummyCamera:
    def __init__(self, R, T, FoVx, FoVy, W, H):
        self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
        self.R = R
        self.T = T
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0,0,0]), 1.0)).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.image_width = W
        self.image_height = H
        self.FoVx = FoVx
        self.FoVy = FoVy

parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser)
pipeline = PipelineParams(parser)

parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--include_feature", action="store_true")

parser.add_argument("--get_video", action="store_true")
parser.add_argument("--n_views", default=None, type=int)
parser.add_argument("--dataname", default='bed', type=str)
args = get_combined_args(parser)


# dataname = 'teatime'
dataname = args.dataname
args.n_view = 4 if dataname in ['teatime', 'figurines', 'ramen', 'waldo_kitchen'] else 3
feature_level = 2
model_path = './output/{}/{}_views_{}'.format(dataname, args.n_view, feature_level)
img0_path = os.path.join(model_path, 'train/ours_1000/renders/00000.png')
import cv2
a = cv2.imread(img0_path)
h, w = a.shape[:2]
path = './data/{}/dust3r_{}_views'.format(dataname, args.n_view)
pose_path = os.path.join('./output/{}'.format(dataname), '{}_views_{}pose'.format(args.n_view, feature_level))
org_pose = np.load(os.path.join(pose_path, 'pose_1000.npy'))
visualizer(org_pose, ["green" for _ in org_pose], model_path + "pose/poses_optimized.png")
n_interp = int(4 * 30 / args.n_view)  # 10second, fps=30
s = []
for i in range(args.n_view-1):
    s0 = generate_interpolated_path(org_pose[i:i+2], n_interp=n_interp)
    s.extend(s0)
s.append(org_pose[-1][:3, :])
# R = extr[:3, :3].transpose()
# T = extr[:3, 3]
# path = '/home/hu997372/code/langsplat/data/{}'.format(dataname)
# bg_color = [0, 0, 0]
bg_color = [1, 1, 1]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
with torch.no_grad():
    gaussians = GaussianModel(3)
    checkpoint = os.path.join(model_path, 'chkpnt1000.pth')
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, args, mode='test')

try:
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
except:
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

cam_infos_unsorted, poses = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, 'images'), eval=False)
sorting_indices = sorted(range(len(cam_infos_unsorted)), key=lambda x: cam_infos_unsorted[x].image_name)
cam_infos = [cam_infos_unsorted[i] for i in sorting_indices]
render_path = os.path.join('my_result', '{}_{}'.format(dataname, feature_level))
if not os.path.exists(render_path):
    os.makedirs(render_path)
fovx, fovy = cam_infos[0].FovX, cam_infos[0].FovY

device = "cuda" if torch.cuda.is_available() else "cpu"
model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
tokenizer = model.tokenizer
# texts = ['the book of The Unbearable Lightness of Being', 'a can of red bull drink', 'a white keyboard', 'a pack of pocket tissues', 'desktop', 'blue partition']
# texts = ["orange cat"]
# texts = ["mini offroad car"]
texts = ["weaving basket"]
num_query = len(texts)

# 对文本进行编码
text_inputs = tokenizer(texts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
import torch.nn.functional as F
text_features = F.normalize(text_features, dim=1)

gt_512dir = os.path.join(path, 'language_features')
gt_3dir = os.path.join(path, 'language_features_dim3')
name_list = os.listdir(gt_3dir)
name_list = sorted(set([name.split('_')[0] for name in name_list]))
feature_map, feature_map_dim3 = [], []
for frame_idx in name_list:
    feature_map.append(torch.from_numpy(np.load(os.path.join(gt_512dir, frame_idx + '_f.npy'))).to(device))
    feature_map_dim3.append(torch.from_numpy(np.load(os.path.join(gt_3dir,  frame_idx + '_f.npy'))).to(device))
feature_map = torch.cat(feature_map, dim = 0)
feature_map_dim3 = torch.cat(feature_map_dim3, dim = 0)

bar = 0.6
if not os.path.exists('./my_result/{}_{}_heatmap/'.format(dataname, feature_level)):
    os.makedirs('./my_result/{}_{}_heatmap/'.format(dataname, feature_level))
from torchvision import transforms
just_render = False
if not just_render:
    for idx, cam in tqdm(enumerate(s)):
        R = cam[:3, :3].transpose()
        T = cam[:3, 3]
        # view = DummyCamera(cam.R, cam.T, cam.FovX, cam.FovY, 988, 730)
        # w, h = 988, 730
        view = DummyCamera(R, T, fovx, fovy, w, h)
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
        output = render(view, gaussians, pipeline, background, args, camera_pose=camera_pose)
        # render0 = output['render']
        semantic = output["language_feature_image"]
        getsem = semantic.permute(1, 2, 0).reshape(-1, 3)
        img0 = output["render"].permute(1, 2, 0)
        image_feature_get = torch.matmul(getsem, feature_map_dim3.T)
        _, indices = torch.max(image_feature_get, dim=1)

        outputs = feature_map[indices].reshape(h, w, -1)
        relevancy_map = outputs @ text_features.T # [N1,N2]
        now_map = relevancy_map[:, :, 0]
        norm_data = now_map
        norm_data = (norm_data - norm_data.min()) / (norm_data.max() - norm_data.min())
        scale = 30
        kernel = np.ones((scale,scale)) / (scale**2)
        avg_filtered = cv2.filter2D(norm_data.detach().cpu().numpy(), -1, kernel)
        avg_filtered = torch.from_numpy(avg_filtered).to(device)
        relev_norm = 0.5 * (avg_filtered + norm_data).unsqueeze(-1)
        p_i = torch.clip(relev_norm - 0.5, 0, 1)
        valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        mask0 = (relev_norm < bar).squeeze()
        valid_composited[mask0, :] = img0[mask0, :] * 0.3
        transforms.ToPILImage()(valid_composited.permute(2, 0, 1)).save('./my_result/{}_{}_heatmap/'.format(dataname, feature_level) + "{0:05d}".format(idx) + ".png")
        # print(semantic)
        # print(semantic.shape)
        torchvision.utils.save_image(
                semantic, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
            )
    
def encode_images_to_video(image_folder, video_name, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # make sure that the images are in order
    
    # read the first image to get the shape
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    # print(frame, image_folder)
    height, width, _ = frame.shape
    # create the video writer
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()

image_folder = './my_result/{}_{}'.format(dataname, feature_level)
video_path = './my_result/{}_{}.avi'.format(dataname, feature_level)
heatmap_folder = './my_result/{}_{}_heatmap'.format(dataname, feature_level)
heatvideo_path = './my_result/{}_{}_heatmap.avi'.format(dataname, feature_level)
encode_images_to_video(image_folder, video_path, 15)
encode_images_to_video(heatmap_folder, heatvideo_path, 15)
