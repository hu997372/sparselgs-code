import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from utils.image_utils import psnr
import cv2

# data_name = 'teatime'
data_name = 'waldo_kitchen'
feature_level = 2
img_dir = '/data1/hj/sparselgs/data/{}/dust3r_4_views/images'.format(data_name)
render_gt_path = './output/{}/4_views_{}/train/ours_1000/gt_npy'.format(data_name, feature_level)
render_path = './output/{}/4_views_{}/train/ours_1000/renders_npy'.format(data_name, feature_level)
gt3_path = './data/{}/dust3r_4_views/language_features_dim3'.format(data_name)

# img_path = '../instant_official/output/{}/4_views_{}/train/ours_1000/renders'.format(data_name, feature_level)
# gt_path = '../instant_official/output/{}/4_views_{}/train/ours_1000/gt'.format(data_name, feature_level)

imglist = sorted(os.listdir(render_path))
name_lis = sorted(os.listdir(img_dir))
img_test = cv2.imread(os.path.join(img_dir, name_lis[0]))
print(imglist, name_lis)
psnr_test = 0.
nn_test = 0.
image_shape = img_test.shape[:2]
for ind, imgname in enumerate(imglist):
    imagepath = os.path.join(render_path, imgname)  # render semantic map
    gt_segpath = os.path.join(gt3_path, name_lis[ind][:-4] + '_s.npy')
    gt_seg = torch.from_numpy(np.load(gt_segpath, allow_pickle=True)).to('cuda').to(torch.long).reshape(4, -1)
    index = gt_seg[feature_level]
    nope = (index != -1)
    
    gt_img = torch.from_numpy(np.load(os.path.join(render_gt_path, imgname), allow_pickle=True)).to('cuda').reshape(-1, 3)[nope]
    img = torch.from_numpy(np.load(imagepath, allow_pickle=True)).to('cuda').reshape(-1, 3)[nope]
    img_psnr, gt_psnr = (img + 1.) / 2., (gt_img + 1.) / 2.
    psnr_test += psnr(img_psnr, gt_psnr).mean()
    nn = torch.sum(img * gt_img, dim = -1)
    nn_test += nn.mean()
    print(nn.mean(), nn.max(), nn.min(), nn.shape)
    
print(psnr_test / len(imglist))
print(nn_test / len(imglist))