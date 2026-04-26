import cv2
import numpy as np
import os
from tqdm import tqdm

# load image and mask
dataset_name = 'figurines'
# dataset_name = 'waldo_kitchen'
# dataset_name = 'teatime'
# dataset_name = 'ramen'
# select_frame = 'frame_00006'
select_frame = 'frame_00105'
img = cv2.imread("./data/{}/images/".format(dataset_name) + select_frame + '.jpg')
# foldlist = ['LEGaussians', 'sparselgs', 'instant_official', 'langsplat']
foldlist = ['LEGaussians', 'sparselgs', 'instant_official']
# foldlist = ['langsplat']
mask_folder = 'eval_result/{}/'.format(dataset_name) + select_frame[-5:]
# , 'green apple'
texts = ['pumpkin', 'red apple', 'rubber duck with buoy', 'spatula', 'old camera', 'pink ice cream']
# texts = ['knife', 'ottolenghi', 'pour-over vessel', 'Stainless steel pots', 'toaster', 'yellow desk']
# texts = ['bag of cookies', 'tea in a glass', 'stuffed bear', 'coffee', 'three cookies', 'plate']
# texts = ['bowl', 'chopsticks', 'egg', 'napkin', 'plate', 'sake cup']

color_list = [[255, 0, 0],     # 红色
              [0, 255, 0],     # 绿色
              [0, 0, 255],     # 蓝色
              [255, 255, 0],   # 黄色
              [255, 0, 255],   # 紫色
              [255, 165, 0],   # 橙色
              [0, 255, 255],]   # 青色
              
masks = [text+'.png' for text in texts]

for method in foldlist:
    get_folder = '/data1/hj/{}/'.format(method) + mask_folder
    image_with_masks = img.copy() * 0.5
    for i, mask in tqdm(enumerate(masks)):
        # print(mask, color_list[i % len(color_list)])
        mask_path = os.path.join(get_folder, 'chosen_'+mask)
        mask = cv2.imread(mask_path)[..., 0] > 10
        color = np.array(color_list[i % len(color_list)], dtype=np.uint8)
        color[0], color[1], color[2] = color[2], color[1], color[0]
        # print(color.shape)
        color_array = np.ones(img.shape, dtype=np.uint8) * np.array(color, dtype=np.uint8)[np.newaxis, np.newaxis, :]
        image_with_masks[mask] = cv2.addWeighted(img, 0.5, np.array(color_array, dtype=np.uint8), 0.5, 0)[mask]
    cv2.imwrite('./out_mask_{}.jpg'.format(method), image_with_masks)

gt_mask_folder = 'eval_result/{}/gt/frame_'.format(dataset_name) + select_frame[-5:]
image_with_masks = img.copy() * 0.5
for i, text in tqdm(enumerate(texts)):
    mask_path = os.path.join(gt_mask_folder, text+'.jpg')
    mask = cv2.imread(mask_path)[..., 0] > 10
    # print(mask.shape)
    color = np.array(color_list[i % len(color_list)], dtype=np.uint8)
    color[0], color[1], color[2] = color[2], color[1], color[0]
    # print(color.shape)
    color_array = np.ones(img.shape, dtype=np.uint8) * np.array(color, dtype=np.uint8)[np.newaxis, np.newaxis, :]
    image_with_masks[mask] = cv2.addWeighted(img, 0.5, np.array(color_array, dtype=np.uint8), 0.5, 0)[mask]
cv2.imwrite('./out_mask_gt.jpg', image_with_masks)