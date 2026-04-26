#!/usr/bin/env python
from __future__ import annotations

import json
import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm

import sys
sys.path.append("..")
import colormaps
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork
from utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler) # type: ignore

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict: # type: ignore
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.jpg')))
    gt_ann = {}
    # print(gt_json_paths)
    # print(json_folder)
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        h, w = gt_data['info']['height'], gt_data['info']['width']
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1 
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            
            # # save for visulsization
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths


def activate_stream(sem_map, 
                    image, 
                    clip_model, 
                    image_name: Path = None,
                    img_ann: Dict = None, 
                    thresh : float = 0.5, 
                    colormap_options = None):
    valid_map = clip_model.get_max_across(sem_map)                 # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape
    # print(valid_map)
    # positive prompts
    chosen_iou_list, chosen_lvl_list, chosen_pixel_list, chosen_gt_list = [], [], [], []
    for k in range(n_prompt):
        iou_lvl = np.zeros(n_head)
        pixel_lvl, gt_lvl = np.zeros(n_head), np.zeros(n_head)
        mask_lvl = np.zeros((n_head, h, w))
        for i in range(n_head):
            # NOTE 加滤波结果后的激活值图中找最大值点
            scale = 30
            kernel = np.ones((scale,scale)) / (scale**2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])
            
            output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                            output_path_relev)
            
            # NOTE 与lerf一致，激活值低于0.5的认为是背景
            p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask = (valid_map[i][k] < 0.5).squeeze()
            valid_composited[mask, :] = image[mask, :] * 0.3
            output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_composited, colormap_options, output_path_compo)
            
            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)

            mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
            mask_pred = smooth(mask_pred)
            mask_lvl[i] = mask_pred
            mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)
            
            # calculate iou
            intersection = np.sum(np.logical_and(mask_gt, mask_pred))
            union = np.sum(np.logical_or(mask_gt, mask_pred))
            gt0 = np.sum(np.sum(mask_gt))
            pixel0 = np.sum(intersection)
            pixel_lvl[i], gt_lvl[i] = pixel0, gt0
            iou = np.sum(intersection) / np.sum(union)
            iou_lvl[i] = iou

        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)
        
        chosen_iou_list.append(iou_lvl[chosen_lvl])
        chosen_pixel_list.append(pixel_lvl[chosen_lvl])
        chosen_gt_list.append(gt_lvl[chosen_lvl])
        chosen_lvl_list.append(chosen_lvl.cpu().numpy())
        
        # save for visulsization
        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        vis_mask_save(mask_lvl[chosen_lvl], save_path)

    return chosen_iou_list, chosen_lvl_list, chosen_pixel_list, chosen_gt_list


def lerf_localization(sem_map, image, clip_model, image_name, img_ann):
    output_path_loca = image_name / 'localization'
    output_path_loca.mkdir(exist_ok=True, parents=True)

    valid_map = clip_model.get_max_across(sem_map)                 # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape
    
    # positive prompts
    acc_num = 0
    positives = list(img_ann.keys())
    for k in range(len(positives)):
        select_output = valid_map[:, k]
        # print(select_output.shape, valid_map.shape)
        
        # NOTE 平滑后的激活值图中找最大值点
        scale = 30
        kernel = np.ones((scale,scale)) / (scale**2)
        np_relev = select_output.cpu().numpy()
        # avg_filtered = cv2.filter2D(np_relev.transpose(1,2,0), -1, kernel)[:, None]
        avg_filtered = cv2.filter2D(np_relev.transpose(1,2,0), -1, kernel)
        
        score_lvl = np.zeros((n_head,))
        coord_lvl = []
        for i in range(n_head):
            score = avg_filtered[..., i].max()
            coord = np.nonzero(avg_filtered[..., i] == score)
            score_lvl[i] = score
            coord_lvl.append(np.asarray(coord).transpose(1,0)[..., ::-1])

        selec_head = np.argmax(score_lvl)
        coord_final = coord_lvl[selec_head]
        # print(coord_final.shape, coord_final, selec_head, coord_lvl)
        # print(n_head, avg_filtered, coord, avg_filtered.shape, coord.shape)
        # print(n_head, avg_filtered, coord, avg_filtered.shape)
        
        for box in img_ann[positives[k]]['bboxes'].reshape(-1, 4):
            flag = 0
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for cord_list in coord_final:
                if (cord_list[0] >= x_min and cord_list[0] <= x_max and 
                    cord_list[1] >= y_min and cord_list[1] <= y_max):
                    acc_num += 1
                    flag = 1
                    break
            if flag != 0:
                break
        
        # NOTE 将平均后的结果与原结果相加，抑制噪声并保持激活边界清晰
        avg_filtered = torch.from_numpy(avg_filtered[..., selec_head]).unsqueeze(-1).to(select_output.device)
        torch_relev = 0.5 * (avg_filtered + select_output[selec_head].unsqueeze(-1))
        p_i = torch.clip(torch_relev - 0.5, 0, 1)
        valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        mask = (torch_relev < 0.5).squeeze()
        valid_composited[mask, :] = image[mask, :] * 0.3
        
        save_path = output_path_loca / f"{positives[k]}.png"
        show_result(valid_composited.cpu().numpy(), coord_final,
                    img_ann[positives[k]]['bboxes'], save_path)
    return acc_num


def evaluate(feat_dir, output_path, json_folder, mask_thresh, logger, gt_3dir, gt_512dir, gt_imgdir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    eval_index_list = []
    ann_inedx_list = []
    gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(json_folder), Path(output_path))
    gt_index = list(map(int, gt_ann.keys()))
    need_index = sorted(os.listdir(gt_imgdir))
    for i in range(len(need_index)):
        need_ind = int(need_index[i][8:11]) - 1
        if need_ind in gt_index:
            eval_index_list.append(i)
            ann_inedx_list.append(need_ind)
    # print(gt_index, need_index, eval_index_list)
    # exit()
    # eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    # eval_index_list = [1]
    compressed_sem_feats = np.zeros((len(feat_dir), len(eval_index_list), *image_shape, 3), dtype=np.float32)
    compressed_corr_feat512 = np.zeros((len(feat_dir), len(eval_index_list), *image_shape, 512), dtype=np.float32)
    for i in range(len(feat_dir)):
        feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
                               key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
        print(feat_paths_lvl, feat_dir)
        print(compressed_sem_feats.shape, eval_index_list)
        for j, idx in enumerate(eval_index_list):
            # compressed_sem_feats[i][j] = np.load(feat_paths_lvl[idx])
            compressed_sem_feats[i][j] = np.load(feat_paths_lvl[idx])
            name_img = f"{ann_inedx_list[j] + 1:05d}"
            feature_map = np.load(os.path.join(gt_512dir, 'frame_' + name_img + '_f.npy'))
            feature_map_dim3 = np.load(os.path.join(gt_3dir, 'frame_' + name_img + '_f.npy'))
            # need_get = compressed_sem_feats[i][j].reshape(-1, 3)
            # diff = need_get[:, None, :] - feature_map_dim3[None, :, :]
            # dist_matrix = np.linalg.norm(diff, axis=2)

            # indices = np.argmin(dist_matrix, axis=1)
            image_feature_get = np.matmul(compressed_sem_feats[i][j], feature_map_dim3.T)
            indices = np.argmax(image_feature_get, axis=2)
            compressed_corr_feat512[i][j] = feature_map[indices].reshape(*image_shape, -1)

    # instantiate autoencoder and openclip
    clip_model = OpenCLIPNetwork(device)

    chosen_iou_all, chosen_lvl_list, chosen_pixel_list, chosen_gt_list = [], [], [], []
    acc_num = 0
    total_bboxes = 0
    for j, idx in enumerate(tqdm(ann_inedx_list)):
        image_name = Path(output_path) / f'{idx+1:0>5}'
        image_name.mkdir(exist_ok=True, parents=True)
        
        sem_feat = compressed_sem_feats[:, j, ...]
        sem_feat512 = compressed_corr_feat512[:, j, ...]
        sem_feat = torch.from_numpy(sem_feat).float().to(device)
        restored_feat = torch.from_numpy(sem_feat512).float().to(device)
        img_ind = gt_index.index(idx)
        rgb_img = cv2.imread(image_paths[img_ind])[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)
        
        img_ann = gt_ann[f'{idx}']
        clip_model.set_positives(list(img_ann.keys()))
        total_bboxes += len(list(img_ann.keys()))
        
        c_iou_list, c_lvl, c_pixel, c_gt = activate_stream(restored_feat, rgb_img, clip_model, image_name, img_ann,
                                            thresh=mask_thresh, colormap_options=colormap_options)
        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)
        chosen_pixel_list.extend(c_pixel)
        chosen_gt_list.append(c_gt)

        # print(restored_feat, rgb_img, clip_model, image_name, img_ann)
        acc_num_img = lerf_localization(restored_feat, rgb_img, clip_model, image_name, img_ann)
        acc_num += acc_num_img

    # # iou
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    get_total_pixel = sum(chosen_pixel_list) / sum(chosen_gt_list)
    logger.info(f'trunc thresh: {mask_thresh}')
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    logger.info(f"iou2 chosen: {get_total_pixel:.4f}")
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")

    # localization acc
    # for img_ann in gt_ann.values():
    #     total_bboxes += len(list(img_ann.keys()))
    acc = acc_num / total_bboxes
    logger.info("Localization accuracy: " + f'{acc:.4f}')


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)
    
    parser = ArgumentParser(description="prompt any label")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument('--feat_dir', type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--json_folder", type=str, default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument("--total_iters", type=int, default=1000)
    parser.add_argument("--n_views", type=int, default=4)
    args = parser.parse_args()

    # NOTE config setting
    dataset_name = args.dataset_name
    mask_thresh = args.mask_thresh
    n_view = args.n_views
    feature_level = [1, 2, 3]
    # feature_level = [2, 3]
    # feature_level = [1, 2]
    # feat_dir = [os.path.join(args.feat_dir, dataset_name+f"_{i}", "train/ours_None/renders_npy") for i in range(1,4)]
    feat_dir = [os.path.join(args.feat_dir, '{}_views'.format(n_view)+f"_{i}", "train/ours_{}/renders_npy".format(args.total_iters)) for i in feature_level]
    # feat_dir = [os.path.join(args.feat_dir, dataset_name+f"_{i}", "train/ours_{}/renders_npy".format(args.total_iters)) for i in feature_level]
    gt_3dir = '../data/{}/dust3r_{}_views/language_features_dim3'.format(dataset_name, n_view)
    gt_512dir = '../data/{}/dust3r_{}_views/language_features'.format(dataset_name, n_view)
    gt_imgdir = '../data/{}/images'.format(dataset_name)
    output_path = os.path.join(args.output_dir, dataset_name)
    json_folder = os.path.join(args.json_folder, dataset_name) 

    # NOTE logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, f'{timestamp}.log')
    logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)

    # print(dataset_name)
    # print(args.feat_dir)
    # print(feat_dir)
    # print(output_path)
    # print(json_folder)
    evaluate(feat_dir, output_path, json_folder, mask_thresh, logger, gt_3dir, gt_512dir, gt_imgdir)