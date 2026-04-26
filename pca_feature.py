import numpy as np
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import cv2
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

parser = ArgumentParser(description="prompt any label")
parser.add_argument('--data_name', type=str, default=None)
args = parser.parse_args()
pca = PCA(n_components=3)
data_name = args.data_name
n_view = 3
if data_name in ['teatime', 'ramen', 'figurines', 'waldo_kitchen']:
    n_view = 4
datapath_512 = './data/{}/dust3r_{}_views/language_features'.format(data_name, n_view)
datapath = './data/{}/dust3r_{}_views/language_features_dim3'.format(data_name, n_view)
if not os.path.exists(datapath):
    os.makedirs(datapath, exist_ok=True)
dir0 = os.listdir(datapath_512)
seg_name, fea_name = [], []
for d in dir0:
    if '_f' in d:
        fea_name.append(d)
    elif '_s' in d:
        seg_name.append(d)
seg_name.sort()
fea_name.sort()

total_fea = []
need_point = [0]
ind_seg = np.zeros((3, 3, 2), dtype=int)
for i, (fea, seg) in enumerate(zip(fea_name, seg_name)):
    fea_path_512 = os.path.join(datapath_512, fea)
    feature_map = torch.from_numpy(np.load(fea_path_512))
    seg_map = np.load(os.path.join(datapath_512, seg))
    for le in [1, 2, 3]:
        se = np.unique(seg_map[le].flatten()).astype(int)
        ind_seg[i][le-1][0], ind_seg[i][le-1][1] = se[1], se[-1]
        # print(se[1], se[-1])
    x, _ = feature_map.shape
    need_point.append(need_point[-1] + x)
    total_fea.append(feature_map)
np.save(os.path.join(datapath, 'ind_seg.npy'), ind_seg)

total_fea = torch.cat(total_fea, dim=0)
pca.fit(total_fea)
pca_feature = pca.transform(total_fea).astype(np.float32)
for i, fea in enumerate(fea_name):
    fea_path = os.path.join(datapath, fea)
    save_fea = pca_feature[need_point[i]:need_point[i+1], :]
    np.save(fea_path, save_fea)