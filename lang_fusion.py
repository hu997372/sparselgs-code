import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as pl
import sys
sys.path.append("./dust3r")
# sys.path.append('./Roma')
from romatch.utils.utils import tensor_to_pil

from romatch import roma_outdoor, roma_indoor
import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from collections import Counter
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
# device = 'cpu'

def move_folder(old_feature_path, feature_path):
    if not os.path.exists(old_feature_path):
        # os.mkdir(old_feature_path)
        shutil.copytree(feature_path, old_feature_path)

def get_language_feature_old(language_feature_dir, feature_level, height, width, image_name):
    language_feature_name = os.path.join(language_feature_dir, image_name)
    seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy'))
    feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy'))
    y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    seg = seg_map[:, y, x].squeeze(-1).long()
    mask = seg != -1
    # print(seg[0:1].max(), seg[1:2].max(), seg[2:3].max(), seg[3:4].max())
    # print(seg[0:1].min(), seg[1:2].min(), seg[2:3].min(), seg[3:4].min())
    if feature_level == 0: # default
        point_feature1 = feature_map[seg[0:1]].squeeze(0)
        mask = mask[0:1].reshape(1, height, width)
    elif feature_level == 1: # s
        point_feature1 = feature_map[seg[1:2]].squeeze(0)
        mask = mask[1:2].reshape(1, height, width)
    elif feature_level == 2: # m
        point_feature1 = feature_map[seg[2:3]].squeeze(0)
        mask = mask[2:3].reshape(1, height, width)
    elif feature_level == 3: # l
        point_feature1 = feature_map[seg[3:4]].squeeze(0)
        mask = mask[3:4].reshape(1, height, width)
    else:
        raise ValueError("feature_level=", feature_level)
    point_feature = point_feature1.reshape(height, width, -1)
    if point_feature.shape[2] == 3:
        img = cv2.cvtColor(point_feature.cpu().detach().numpy(), cv2.COLOR_RGB2BGR)
        save_lan_path = os.path.join(npy_dir_look, image_name + '.jpg')
        if not os.path.exists(npy_dir_look):
            os.makedirs(npy_dir_look)
        cv2.imwrite(save_lan_path, img*255)
    
    return point_feature.cuda(), mask.cuda() 

def  get_language_feature(language_feature_dir, feature_level, height, width, image_name):
    language_feature_name = os.path.join(language_feature_dir, image_name)
    seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy'))
    feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy'))
    # print(seg_map[feature_level,:].shape)
    
    y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    seg = seg_map[feature_level, y, x].squeeze(-1).long()
    return feature_map.to(device), seg.reshape(height, width).to(device)

def vis_language_feature_new(language_feature_dir, image_name, height, width, npy_dir_new):
    language_feature_name = os.path.join(language_feature_dir, image_name)
    seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy'))
    feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy'))
    if not os.path.exists(npy_dir_new):
        os.makedirs(npy_dir_new)
    
    y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    # print(seg_map.shape, seg_map)
    seg = seg_map[feature_level, y, x].squeeze(-1).long()
    # seg = seg_map[y, x].squeeze(-1).long()
    point_feature = feature_map[seg].squeeze(0).reshape(height, width, -1)
    if point_feature.shape[2] == 3:
        img = cv2.cvtColor(point_feature.cpu().detach().numpy(), cv2.COLOR_RGB2BGR)
        save_lan_path = os.path.join(npy_dir_new, image_name + '.jpg')
        cv2.imwrite(save_lan_path, img*255)

def visual_match(img1_path, img2_path, warp, certainty):
    # visualization
    # im1 = Image.open(img1_path).resize((W, H))
    im2 = Image.open(img2_path).resize((W, H))
    # x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)
    im2_transfer_rgb = F.grid_sample(x2[None], warp[:, :, 2:][None], mode="bilinear", align_corners=False)[0]
    warp_im = im2_transfer_rgb
    white_im = torch.ones((H, W), device=device)
    vis_im = certainty * warp_im + (1 - certainty) * white_im
    tensor_to_pil(vis_im, unnormalize=False).save(save_path)

def vis_mask(seg_mask, output_filename):
    array = seg_mask.reshape(H, W).detach().cpu().numpy()
    unique_values = np.unique(array)
    color_map = {}
    for value in unique_values:
        color_map[value] = np.random.randint(0, 256, size=(3,))

    h, w = array.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    for value in unique_values:
        # print(value)
        if value != -1:
            continue
        color_image[array == value] = color_map[value]

    cv2.imwrite(output_filename, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

def vis_match_single(img1_path, img2_path, map1, map2, k1):
    im1 = Image.open(img1_path).resize((W, H))
    im2 = Image.open(img2_path).resize((W, H))
    x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)
    black_im1 = torch.zeros((3, H, W), device=device)
    black_im2 = torch.zeros((3, H, W), device=device)
    black_im1[:, map1[:, 1], map1[:, 0]] = x1[:, map1[:, 1], map1[:, 0]]
    black_im2[:, map2[:, 1], map2[:, 0]] = x2[:, map2[:, 1], map2[:, 0]]
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    tensor_to_pil(black_im1, unnormalize=False).save(os.path.join(args.save_path, 'visual1_{}.jpg'.format(k1)))
    tensor_to_pil(black_im2, unnormalize=False).save(os.path.join(args.save_path, 'visual2_{}.jpg'.format(k1)))

def cos_loss(x, y):
    return torch.sum(x * y) / (x.norm() * y.norm())

def fusion_delete(now_view):
    key_to_delete = []
    now_collision = big_mask_collision[now_view]
    for key in now_collision:
        if len(now_collision[key]) <= 1:
            key_to_delete.append(key)
        else:
            dict_count = dict()
            for index, num in enumerate(now_collision[key]):
                if num[1] not in dict_count:
                    dict_count[num[1]] = [1, num]
                else:
                    dict_count[num[1]] = [100, index]
            flag = False
            for key0 in dict_count:
                if dict_count[key0][0] > 1:
                    flag = True
                else:
                    now_collision[key].remove(dict_count[key0][1])
            if flag == False:
                key_to_delete.append(key)
    for key in key_to_delete:
        del now_collision[key]
    if look_feature:
        file = open('output.txt', mode='wt', encoding='utf-8')
        file.write(str(big_mask_collision[now_view]) + '\n \n')

# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    sample_without_b = sampled_depth_src > 0
    # exit()
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    K_xyz_reprojected[2:3][K_xyz_reprojected[2:3]==0] += 0.00001
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src, sample_without_b

def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    # x2d_src, y2d_src  ref视角投影到src上的结果
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src, ref_without_b = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                                                                 depth_src, intrinsics_src, extrinsics_src)
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref 

    mask = None
    masks = []
    for i in range(s, 11):
        mask = np.logical_and(dist < i * dist_base, relative_depth_diff < i * rel_diff_base)
        masks.append(mask)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected, x2d_src, y2d_src, ref_without_b
    
def fusion(data_dir):
    depth_dir = os.path.join(data_dir, 'depths.pt')
    pose_dir = os.path.join(data_dir, 'poses.pt')
    intri_dir = os.path.join(data_dir, 'intrinsics.pt')
    intrinsics, poses = torch.load(intri_dir).detach().cpu().numpy(), torch.load(pose_dir).detach().cpu().numpy()
    depths = torch.load(depth_dir)
    
    for select_view in tqdm(range(num_views)):
        ref_intrinsics, ref_extrinsics = intrinsics[select_view], np.linalg.inv(poses[select_view])
        ref_depth_est = depths[select_view].detach().cpu().numpy()
        src_views = [v for v in range(num_views) if v!=select_view]
        img_ref_path = os.path.join(img_path, image_name_list[select_view] + '.' + image_lat)
        
        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []
        ref_without_other = []

        # compute the geometric mask
        geo_mask_sum = 0
        dy_range = num_views
        geo_mask_sums = [0] * (dy_range - s)
        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = intrinsics[src_view], np.linalg.inv(poses[src_view])
            img_src_path = os.path.join(img_path, image_name_list[src_view] + '.' + image_lat)
            # the estimated depth of the source view
            src_depth_est = depths[src_view].detach().cpu().numpy()
            masks, geo_mask, depth_reprojected, x2d_src, y2d_src, ref_without = check_geometric_consistency(ref_depth_est, ref_intrinsics,
                                                                                            ref_extrinsics, src_depth_est,
                                                                                            src_intrinsics, src_extrinsics)
            geo_mask_sum += geo_mask.astype(np.int32)
            for i in range(s, dy_range):
                geo_mask_sums[i - s] += masks[i - s].astype(np.int32)

            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)
            ref_without_other.append(ref_without)

        without_out = np.zeros(ref_without.shape, dtype = bool)
        for mask_out in ref_without_other:
            without_out = np.logical_or(mask_out, without_out)
        without_out = ~without_out
        if without_out.sum() > 20000:
            without_out = np.zeros(ref_without.shape, dtype = bool)
        
        geo_mask = geo_mask_sum >= dy_range
        for i in range(s, dy_range):
            geo_mask = np.logical_or(geo_mask, geo_mask_sums[i - s] >= i)

        final_mask = torch.from_numpy(np.logical_or(geo_mask, without_out))
        height_in, width_in = final_mask.shape[:2]
        ratio_height, ratio_width = height / height_in, width / width_in
        x, y = torch.meshgrid(torch.arange(0, height_in, device=device), torch.arange(0, width_in, device=device))
        # print(x.shape, x)
        now_x, now_y = x * ratio_height, y * ratio_width
        now_x, now_y = now_x.to(torch.long), now_y.to(torch.long)
        ref_seg, ref_feature, ref_feature_512 = seg_map_list[select_view], language_feature_list[select_view], language_feature_list_512[select_view]

        for ind, src_view in enumerate(src_views):
            src_mask = all_srcview_geomask[ind]
            src_view_x, src_view_y = torch.from_numpy(all_srcview_x[ind][src_mask]) * ratio_width, torch.from_numpy(all_srcview_y[ind][src_mask]) * ratio_height
            src_view_x, src_view_y = src_view_x.to(torch.long).to(device), src_view_y.to(torch.long).to(device)

            if src_view_x.max() > width or src_view_x.min() < 0 or src_view_y.max() > height or src_view_y.min() < 0:
                # out_region_mask = torch.where(torch.logical_or(src_view_x > width, src_view_x < 0), False, True)
                out_region_mask_x = torch.where(torch.logical_or(src_view_x > width, src_view_x < 0), False, True)
                out_region_mask_y = torch.where(torch.logical_or(src_view_y > height, src_view_y < 0), False, True)
                out_region_mask = torch.logical_and(out_region_mask_x, out_region_mask_y)
                src_view_x, src_view_y = src_view_x[out_region_mask], src_view_y[out_region_mask]
                ref_view_x, ref_view_y = now_x[src_mask][out_region_mask], now_y[src_mask][out_region_mask]
            else:
                ref_view_x, ref_view_y = now_x[src_mask], now_y[src_mask]
            src_seg, src_feature, src_feature_512 = seg_map_list[src_view], language_feature_list[src_view], language_feature_list_512[src_view]
            count_src_ori = Counter(src_seg.reshape(-1).detach().cpu().numpy())
            count_src_ori.pop(-1)

            # print(src_view_x.max(), src_view_x.min())
            src_ind, ref_ind = src_seg[src_view_y, src_view_x], ref_seg[ref_view_x, ref_view_y]
            count_ref = Counter(ref_ind.detach().cpu().numpy())
            count_ref.pop(-1)
            # print(count_ref)
            for k_ref, v_ref in count_ref.items():
                ref_bool = (ref_ind == k_ref)
                ref_f, ref_f_512 = ref_feature[k_ref], ref_feature_512[k_ref]   # reference feature code
                count_src = Counter(src_ind[ref_bool].detach().cpu().numpy())
                map_src = torch.cat([src_view_x[:, None], src_view_y[:, None]], dim=1)
                map_ref = torch.cat([ref_view_y[:, None], ref_view_x[:, None]], dim=1)
                if -1 in count_src:
                    count_src.pop(-1)
                if not bool(count_src):
                    continue
                # print(get_2_index, map2)
                max_key = max(count_src, key=lambda k: count_src[k])
                if count_src[max_key] < fusion_bar:
                    continue
                if look_match:
                    # print(ref_view_y.shape, src_view_y.shape)
                    # 这里整个取坐标的过程是反的
                    # print(ref_bool.device, src_view_x.device, map_ref.device, map_src.device)
                    vis_match_single(img_ref_path, img_src_path, map_ref[ref_bool], map_src[ref_bool], k_ref)
                
                whole_pixel_src = count_src_ori[max_key]   # 考察投影的 patch 在整个原来分割 mask 中的占比多少 
                now_pixel_src = count_src[max_key]
                pixel_score = now_pixel_src / whole_pixel_src
                # feature_score = cos_loss(src_feature[max_key], ref_f)
                feature_score = cos_loss(src_feature_512[max_key], ref_f_512)
                total_score = lambda_pixel * pixel_score + lambda_feature * feature_score

                if max_key not in fusion_score_list[src_view]:
                    if total_score > fusion_score_bar:
                        fusion_score_list[src_view][max_key] = [total_score, select_view, k_ref]
                        src_feature[max_key] = ref_f
                        language_feature_list_512[src_view][max_key] = ref_f_512
                        # print(view_2)
                elif total_score > fusion_score_list[src_view][max_key][0]:
                    fusion_score_list[src_view][max_key] = [total_score, select_view, k_ref]
                    src_feature[max_key] = ref_f
                    language_feature_list_512[src_view][max_key] = ref_f_512

def roma_match():
    for view_1, image_name_1 in tqdm(enumerate(image_name_list)):
        # print('reference_image: {}'.format(image_name_1))
        language_feature_1, seg_map1, language_feature_1_512 = language_feature_list[view_1], seg_map_list[view_1], language_feature_list_512[view_1]
        # dino_pca_feature_1 = dino_pca[view_1]
        # vis_mask(seg_mask=seg_map1, output_filename='seg_map1.png')
        img1_path = os.path.join(img_path, image_name_1 + '.' + image_lat)
        count1_ori = Counter(seg_map1.reshape(-1).detach().cpu().numpy())
        count1_ori.pop(-1)

        # for view_2 in range(view_1 + 1, num_views):
        for view_2 in range(num_views):
            if view_2 == view_1:
                continue
            image_name_2 = image_name_list[view_2]
            # print('now_image: {}'.format(image_name_2))
            language_feature_2, seg_map2, language_feature_2_512 = language_feature_list[view_2], seg_map_list[view_2], language_feature_list_512[view_2]
            # dino_pca_feature_2 = dino_pca[view_2]
            # vis_mask(seg_mask=seg_map2, output_filename='seg_map2.png')

            img2_path = os.path.join(img_path, image_name_2 + '.' + image_lat)
            count2_ori = Counter(seg_map2.reshape(-1).detach().cpu().numpy())
            count2_ori.pop(-1)  # a dict describe the number of values in seg_map

            # Match
            warp, certainty = roma_model.match(img1_path, img2_path, device='cuda')
            certain_map = (certainty > certain_min)
            # visual_match(img1_path, img2_path, warp, certainty)

            # get key points that matches img1 and img2
            # then use certain_map to filter out these low certain value pixels
            kpts1, kpts2 = roma_model.to_pixel_coordinates(warp, H, W, H, W)  
            # print(kpts1, dino_pca_feature_2.shape)
            # exit()
            kpts1, kpts2 = kpts1[certain_map].to(torch.long), kpts2[certain_map].to(torch.long)
            # a dict describe the number of values in seg_map
            if for_debug:
                certain_map = certain_map.to('cpu')
                kpts1 = kpts1.to('cpu')
                kpts2 = kpts2.to('cpu')
                seg_map2 = seg_map2.to('cpu')
            # bool 数组作为 index 作用返回得是一维向量
            count1, count2 = Counter(seg_map1[certain_map].detach().cpu().numpy()), Counter(seg_map2[certain_map].detach().cpu().numpy())
            if -1 in count1:
                count1.pop(-1)
            if -1 in count2:
                count2.pop(-1)
            
            # let feature map 1 be a standard, project feature map 2's corresponding feature to feature map 1
            lens = 0
            for k1, v1 in count1.items():
                lens += 1
                # print(lens, k1, v1)
                now_bool = (seg_map1[certain_map] == k1)  # 有问题，seg_map2 针对地是全图坐标，这里地 kpts2 是匹配点地所有位置，根本不对，要改成 segmap1
                map1, map2 = kpts1[now_bool], kpts2[now_bool]
                # dino1, dino2 = dino_pca_feature_1[map1[:, 1], map1[:, 0], :], dino_pca_feature_2[map2[:, 1], map2[:, 0], :]
                # TODO: 接下来需要将 dino2 的映射到的区域取出来，然后将 dino1 和 dino2 的对应区域做 loss 得到新衡量指标
                nowfeature_1, nowfeature_1_512 = language_feature_1[k1], language_feature_1_512[k1]
                # get_1_index = seg_map1.reshape(H, W)[map1[:, 0], map1[:, 1]]
                get_2_index = seg_map2[map2[:, 1], map2[:, 0]]
                count_now_2 = Counter(get_2_index.detach().cpu().numpy())
                if -1 in count_now_2:
                    count_now_2.pop(-1)
                if not count_now_2:
                    continue
                # print(get_2_index, map2)
                max_key = max(count_now_2, key=lambda k: count_now_2[k])
                bool_max2 = (get_2_index == max_key)
                # dino1, dino2 = dino1[bool_max2], dino2[bool_max2]
                # print(dino1.shape, max(torch.norm(abs(dino2 - dino1), dim=1)), abs(dino2 - dino1).shape, count_now_2[max_key])
                if count_now_2[max_key] < patch_bar:
                    continue
                if look_match:
                    vis_match_single(img1_path, img2_path, map1, map2, k1)
                
                whole_pixel_2 = count2_ori[max_key]   # 考察投影的 patch 在整个原来分割 mask 中的占比多少 
                now_pixel_2 = count2[max_key]
                pixel_score = now_pixel_2 / whole_pixel_2
                # dino_score = - torch.sum(torch.norm(dino2 - dino1, dim=1)) / int(dino1.shape[0])
                # print(dino_score, now_pixel_2, whole_pixel_2, dino1.shape, dino2.shape)
                feature_score = cos_loss(language_feature_2[max_key], nowfeature_1)
                # feature_score = cos_loss(language_feature_2_512[max_key], nowfeature_1_512)
                total_score = lambda_pixel * pixel_score + lambda_feature * feature_score
                # total_score = lambda_pixel * pixel_score + lambda_feature * feature_score
                # if match_loss > match_bar_cos:
                #     # seg_map2 = torch.where(seg_map2 == max_key, k1, seg_map2)
                #     language_feature_2[max_key] = nowfeature_1

                if max_key not in dict_score_list[view_2]:
                    if total_score > total_score_bar:
                        dict_score_list[view_2][max_key] = [total_score, view_1, k1]
                        big_mask_collision[view_2][max_key] = []
                        language_feature_2[max_key] = nowfeature_1
                        language_feature_list_512[view_2][max_key] = nowfeature_1_512
                        # print(view_2)
                elif total_score > dict_score_list[view_2][max_key][0]:
                    dict_score_list[view_2][max_key] = [total_score, view_1, k1]
                    language_feature_2[max_key] = nowfeature_1
                    language_feature_list_512[view_2][max_key] = nowfeature_1_512
                if total_score > 2 * total_score_bar:
                    if pixel_score > area_bar:
                        big_mask_collision[view_2][max_key].append([total_score, view_1, k1])
            # exit()

def mask_fusion(num_views, big_mask_collision):
    for view in range(num_views):
        now_collision = big_mask_collision[view]
        feature_a = language_feature_list[view]
        for key in now_collision:
            for _, view_1, key1 in now_collision[key]:
                language_feature_list[view_1][key1] = feature_a[key]


from argparse import ArgumentParser
parser = ArgumentParser(description="prompt any label")
parser.add_argument('--dataname', type=str, default=None)
parser.add_argument('--feature_level', type=int, default=2)
args = parser.parse_args()
dataname = args.dataname
feature_level = args.feature_level

n_view = 4
if dataname not in ['teatime', 'ramen', 'waldo_kitchen', 'figurines']:
    n_view = 3
img_path = './data/{}/dust3r_{}_views/images'.format(dataname, n_view)
camera_path = './data/{}/dust3r_{}_views/sparse/0/'.format(dataname, n_view)
# dino_path = './data/{}/dino_feature/fit_3d'.format(dataname)
imgs = os.listdir(img_path)
img0 = cv2.imread(os.path.join(img_path, imgs[0]))
ori_height, ori_width = img0.shape[:2]
# exit()
WARNED = False
if ori_height > 1080:
    if not WARNED:
        print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
        WARNED = True
    global_down = ori_height / 1080
else:
    global_down = 1

height, width = int(ori_height / global_down), int(ori_width / global_down)
# test_views = [1, 4, 6]
test_views = []
image_name_list = sorted([name.split('.', 1)[0] for name in imgs])
image_name_list = [name for idx, name in enumerate(image_name_list) if idx not in test_views]
image_lat = imgs[0].split('.', 1)[1]

lf_path_512 = './data/{}/dust3r_{}_views/language_features'.format(dataname, n_view)
lf_path = './data/{}/dust3r_{}_views/language_features_dim3'.format(dataname, n_view)
npy_dir_look = './data/{}/dust3r_{}_views/language_features_look'.format(dataname,n_view)
npy_dir_new = './data/{}/dust3r_{}_views/language_feature_renew_look'.format(dataname,n_view)

num_views = len(image_name_list)
# Create model
# roma_model = roma_outdoor(device=device, coarse_res=560, upsample_res=(730, 988))
roma_model = roma_indoor(device='cuda', coarse_res=560, upsample_res=(height, width))
H, W = roma_model.get_output_resolution()
for_debug = False
look_match = False
look_feature = True
# big_mask_fusion = True
big_mask_fusion = False

# # for 3d_ovs
# area_bar = 0.1
# total_score_bar = 0.15
# fusion_score_bar = 0.1
# certain_min = 0.1
# lambda_pixel = 0.7
# lambda_feature = 0.3
# patch_bar = 150
# fusion_bar = 80
# s = 1
# times = 3
# dist_base = 1/8 * times
# rel_diff_base = 1/10 * times

# # for lerf_ovs 
# feature_level = 2
# area_bar = 0.15
# total_score_bar = 0.2
# fusion_score_bar = 0.25
# certain_min = 0.20
# lambda_pixel = 0.7
# # lambda_dino = 0
# lambda_feature = 0.3
# patch_bar = 150
# fusion_bar = 100
# s = 1
# times = 2
# dist_base = 1/8 * times
# rel_diff_base = 1/10 * times

# teatime feature_level = 3
# area_bar = 0.15
# total_score_bar = 0.2
# fusion_score_bar = 0.3
# certain_min = 0.2
# lambda_pixel = 0.7
# # lambda_dino = 0.1
# lambda_feature = 0.3
# patch_bar = 180
# fusion_bar = 180
# s = 1
# times = 1.5
# dist_base = 1/8 * times
# rel_diff_base = 1/10 * times

# feature_level = 2
# area_bar = 0.15
# total_score_bar = 0.2
# fusion_score_bar = 0.25
# certain_min = 0.2
# lambda_pixel = 0.7
# # lambda_dino = 0
# lambda_feature = 0.3
# patch_bar = 170
# fusion_bar = 120
# s = 1
# times = 1.5
# dist_base = 1/8 * times
# rel_diff_base = 1/10 * times

# # figurines feature_level = 3
# area_bar = 0.15
# total_score_bar = 0.15
# fusion_score_bar = 0.2
# certain_min = 0.17
# lambda_pixel = 0.7
# lambda_feature = 0.3
# patch_bar = 120
# fusion_bar = 170
# s = 1
# times = 4
# dist_base = 1/8 * times
# rel_diff_base = 1/10 * times

# figurines feature_level = 1
feature_level = 2
area_bar = 0.3
total_score_bar = 0.5
fusion_score_bar = 0.5
certain_min = 0.35
lambda_pixel = 0.7
lambda_feature = 0.3
patch_bar = 150
fusion_bar = 150
s = 1
times = 1.5
dist_base = 1/8 * times
rel_diff_base = 1/10 * times

# feature_level = 3
# area_bar = 0.3
# total_score_bar = 0.5
# fusion_score_bar = 0.5
# certain_min = 0.4
# lambda_pixel = 0.7
# lambda_feature = 0.3
# patch_bar = 150
# fusion_bar = 150
# s = 1
# times = 1.5
# dist_base = 1/8 * times
# rel_diff_base = 1/10 * times

# FOR 3D-OVS
# feature_level = 2
# area_bar = 0.3
# total_score_bar = 0.5
# fusion_score_bar = 0.5
# certain_min = 0.3
# lambda_pixel = 0.7
# lambda_feature = 0.3
# patch_bar = 150
# fusion_bar = 150
# s = 1
# times = 1.5
# dist_base = 1/8 * times
# rel_diff_base = 1/10 * times

if feature_level in [2, 3]:
    big_mask_fusion = True
# big_mask_fusion = False

parser = ArgumentParser()
# img_path = '/home/hu997372/code/InstantSplat/data/images'
parser.add_argument("--save_path", default="./data/match_results", type=str)

args, _ = parser.parse_known_args()
save_path = os.path.join(args.save_path, 'match.jpg')

language_feature_list, language_feature_list_512, seg_map_list = [], [], []
# dino_pca = []
dict_score_list = []
fusion_score_list = []
big_mask_collision = []

origin_path = ['./data/{}/dust3r_{}_views/language_features_origin'.format(dataname, n_view), './data/{}/dust3r_{}_views/language_features_origin_dim3'.format(dataname, n_view)]
move_folder(origin_path[0], lf_path_512)
move_folder(origin_path[1], lf_path)

for image_name in image_name_list:
    language_feature, seg_map = get_language_feature(origin_path[1], feature_level, height, width, image_name)
    language_feature_512, _ = get_language_feature(origin_path[0], feature_level, height, width, image_name)
    language_feature_list.append(language_feature)
    language_feature_list_512.append(language_feature_512)
    # dino_pca.append(torch.from_numpy(cv2.imread(os.path.join(dino_path, image_name + '.' + image_lat))))
    # im = Image.open(os.path.join(dino_path, image_name + '.' + image_lat)).resize((W, H))
    # dino_pca.append((torch.tensor(np.array(im)) / 255).to(device))
    seg_map_list.append(seg_map)
    dict_score_list.append(dict())
    fusion_score_list.append(dict())
    big_mask_collision.append(dict())

print('Roma feature matching process')
roma_match()

if look_feature:
    for now_view, image_name in enumerate(image_name_list):
        language_feature, seg_map = language_feature_list[now_view], seg_map_list[now_view]
        get_language_feature_old(origin_path[1], feature_level, height, width, image_name)
        np.save(os.path.join(lf_path, image_name + '_f.npy'), language_feature.detach().cpu().numpy())
        vis_language_feature_new(lf_path, image_name, height, width, os.path.join(npy_dir_new, 'before_fusion'))
        if big_mask_fusion:
            fusion_delete(now_view)
        if big_mask_collision[now_view]:
            for key in big_mask_collision[now_view]:
                now_msk = big_mask_collision[now_view][key]


print('big mask fusion process')
if big_mask_fusion:
    mask_fusion(num_views, big_mask_collision)
    if look_feature:
        for now_view, image_name in enumerate(image_name_list):
            language_feature, seg_map = language_feature_list[now_view], seg_map_list[now_view]
            np.save(os.path.join(lf_path, image_name + '_f.npy'), language_feature.detach().cpu().numpy())
            vis_language_feature_new(lf_path, image_name, height, width, os.path.join(npy_dir_new, 'after_big_maskfusion'))

print('depth feature fusion process')
fusion(camera_path)

if look_feature:
    for now_view, image_name in enumerate(image_name_list):
        language_feature, seg_map = language_feature_list[now_view], seg_map_list[now_view]
        np.save(os.path.join(lf_path, image_name + '_f.npy'), language_feature.detach().cpu().numpy())
        vis_language_feature_new(lf_path, image_name, height, width, os.path.join(npy_dir_new, 'after_fusion'))

for now_view, image_name in enumerate(image_name_list):
    language_feature = language_feature_list[now_view]
    language_feature_512 = language_feature_list_512[now_view]
    np.save(os.path.join(lf_path, image_name + '_f.npy'), language_feature.detach().cpu().numpy())
    np.save(os.path.join(lf_path_512, image_name + '_f.npy'), language_feature_512.detach().cpu().numpy())