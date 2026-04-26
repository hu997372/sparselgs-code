import torch, os
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cv2
import glob
from dataclasses import dataclass, field
import logging
try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"
from typing import Tuple, Type
from torch import nn
import torchvision

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger

@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            # precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]
    
    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
                ).to(self.neg_embeds.device)
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
    
    def set_semantics(self, text_list):
        self.semantic_labels = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.semantic_labels]).to("cuda")
            self.semantic_embeds = self.model.encode_text(tok_phrases)
        self.semantic_embeds /= self.semantic_embeds.norm(dim=-1, keepdim=True)

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)
    
    def encode_text(self, input):
        processed_input = input
        return self.model.encode_text(processed_input)
    
    def get_semantic_map(self, sem_map: torch.Tensor) -> torch.Tensor:
        # embed: 3xhxwx512
        n_levels, h, w, c = sem_map.shape
        pos_num = self.semantic_embeds.shape[0]
        phrases_embeds = torch.cat([self.semantic_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(sem_map.dtype)
        sem_pred = torch.zeros(n_levels, h, w)
        for i in range(n_levels):
            output = torch.mm(sem_map[i].view(-1, c), p.T)
            softmax = torch.softmax(10 * output, dim=-1)
            sem_pred[i] = torch.argmax(softmax, dim=-1).view(h, w)
            sem_pred[i][sem_pred[i] >= pos_num] = -1
        return sem_pred.long()

    def get_max_across(self, sem_map):
        n_phrases = len(self.positives)
        n_phrases_sims = [None for _ in range(n_phrases)]
        
        n_levels, h, w, _ = sem_map.shape
        clip_output = sem_map.permute(1, 2, 0, 3).flatten(0, 1)

        n_levels_sims = [None for _ in range(n_levels)]
        for i in range(n_levels):
            for j in range(n_phrases):
                probs = self.get_relevancy(clip_output[..., i, :], j)
                pos_prob = probs[..., 0:1]
                n_phrases_sims[j] = pos_prob
            n_levels_sims[i] = torch.stack(n_phrases_sims)
        
        relev_map = torch.stack(n_levels_sims).view(n_levels, n_phrases, h, w)
        return relev_map

# 加载 CLIP 模型（假设使用的是 ViT-B-32 模型）
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
tokenizer = model.tokenizer

def hex_to_rgb(x):
    return [int(x[i:i + 2], 16) / 255 for i in (1, 3, 5)]

from argparse import ArgumentParser

parser = ArgumentParser(description="prompt any label")
parser.add_argument('--dataset_name', type=str, default=None)
parser.add_argument('--bar', type=float, default=0.85)
args = parser.parse_args()
dataset_name = args.dataset_name
logfolder = './eval_result/{}'.format(dataset_name)
img_path = './data/{}/images'.format(dataset_name)
imglist = sorted(os.listdir(img_path))
imglist = [i0[:2] for i0 in imglist]
img0 = cv2.imread(os.path.join(img_path, imglist[0] + '.jpg'))
ori_height, ori_width = img0.shape[:2]
WARNED = False
if ori_height > 1080:
    if not WARNED:
        print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
        WARNED = True
    global_down = ori_height / 1080
else:
    global_down = 1

H, W = int(ori_height / global_down), int(ori_width / global_down)

gt_idxfolder = './download/3d_ovs/{}/segmentations'.format(dataset_name)
txt_filepath = os.path.join(gt_idxfolder, 'classes.txt')
with open(txt_filepath, 'r') as f:
    categories = f.read().splitlines()

maskfolders = sorted([f for f in os.listdir(gt_idxfolder) if f in imglist])
whole_mask = []
for subfolder in maskfolders:
    masks = os.path.join(gt_idxfolder, subfolder)

    mask_images = []
    for category in categories:
        mask_filepath = os.path.join(masks, f"{category}.png")
        if os.path.exists(mask_filepath):
            mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
            if mask is None:
                raise ValueError(f"读取失败: {mask_filepath}")
            mask_images.append(mask // 255)

    combined_mask = np.stack(mask_images, axis=-1)
    whole_mask.append(combined_mask)

image_shape = (H, W)
# print(image_shape)
classes = categories
device = "cuda" if torch.cuda.is_available() else "cpu"
model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
tokenizer = model.tokenizer
with torch.no_grad():
    text_inputs = tokenizer(classes).to(device)
    text_features = model.encode_text(text_inputs)
    text_features = F.normalize(text_features, dim=1)

n_view = 3
# feature_level = [1, 2, 3]
feature_level = [2, 3]
feat_dir = './output/{}'.format(dataset_name)
gt_512dir = './data/{}/dust3r_{}_views/language_features'.format(dataset_name, n_view)
gt_3dir = './data/{}/dust3r_{}_views/language_features_dim3'.format(dataset_name, n_view)
feat_dir = [os.path.join(feat_dir, '{}_views'.format(n_view)+f"_{i}", "train/ours_{}/renders_npy".format(1000)) for i in feature_level]

get_mask = np.zeros((len(feat_dir), n_view, len(classes), *image_shape), dtype=bool)
n_iou = np.zeros((len(feat_dir), n_view, len(classes)), dtype=np.float32)
record_max = np.zeros((len(feat_dir), n_view, len(classes)), dtype=np.float32)
record_area = np.zeros((len(feat_dir), n_view, len(classes)), dtype=np.float32)
import time
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
if not os.path.exists(logfolder):
    os.makedirs(logfolder, exist_ok=True)
log_file = os.path.join(logfolder, f'{timestamp}.log')
logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)
# bar = 0.95 # bed
# bar = 0.8 # bench
# bar = 0.85 # room  office_desk
# bar = 0.75 # sofa
bar = args.bar
select_area_bar = 3000
print(classes)
# ind_seg = np.load(os.path.join(gt_3dir, 'ind_seg.npy'))
with torch.no_grad():
    for k in tqdm(range(len(feat_dir))):
        feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[k], '*.npy')),
                                key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
        print(feat_paths_lvl)
        savePath = f'{logfolder}'
        for i, frame_idx in enumerate(maskfolders):
            gt_seg = whole_mask[i] # [H*W=N1, n_classes]
            # low, high = ind_seg[i, feature_level[k]-1, :]
            # get feature map
            pred_feature = torch.from_numpy(np.load(feat_paths_lvl[i])).reshape(-1, 3).to(device)
            feature_map = torch.from_numpy(np.load(os.path.join(gt_512dir, frame_idx + '_f.npy'))).to(device)
            feature_map_dim3 = torch.from_numpy(np.load(os.path.join(gt_3dir,  frame_idx + '_f.npy'))).to(device)
            # feature_map = torch.from_numpy(np.load(os.path.join(gt_512dir, frame_idx + '_f.npy')))[low:high+1, :].to(device)
            # feature_map_dim3 = torch.from_numpy(np.load(os.path.join(gt_3dir,  frame_idx + '_f.npy')))[low:high+1, :].to(device)
            image_feature_get = torch.matmul(pred_feature, feature_map_dim3.T)
            _, indices = torch.max(image_feature_get, dim=1)

            # diff = pred_feature[:, None, :] - feature_map_dim3[None, :, :]
            # dist_matrix = diff.norm(p=2, dim=2, keepdim=False)
            # _, indices = torch.min(dist_matrix, dim=1)
            outputs = feature_map[indices].reshape(H, W, -1)
            relevancy_map = outputs @ text_features.T # [N1,N2]
            for ind, class0 in enumerate(classes):
                gt_class = gt_seg[:, :, ind]
                rel_now = relevancy_map[:, :, ind]
                nowmax, nowmin = rel_now.max(), rel_now.min()
                # print(nowmax, nowmin)
                pred_class = rel_now > bar * nowmax
                record_area[k][i][ind] = torch.sum(pred_class).detach().to('cpu').numpy()
                pred_score = torch.mean(rel_now[pred_class]).detach().to('cpu').numpy()
                pred_class = pred_class.detach().to('cpu').numpy()
                intersection = np.sum(np.logical_and(gt_class, pred_class))
                union = np.sum(np.logical_or(gt_class, pred_class))
                if np.sum(pred_class) < select_area_bar:
                    nowmax = 0
                    pred_score = 0
                record_max[k][i][ind] = pred_score
                iou = round(intersection / union, 4)
                n_iou[k][i][ind] = iou
                get_mask[k][i][ind] = pred_class
                if not os.path.exists(f'{savePath}/{k}/{frame_idx}'):
                    os.makedirs(f'{savePath}/{k}/{frame_idx}', exist_ok=True)
                cv2.imwrite(f'{savePath}/{k}/{frame_idx}/{class0}.png', get_mask[k][i][ind] * 255.)
                cv2.imwrite(f'{savePath}/{k}/{frame_idx}/{class0}_gt.png', gt_class * 255.)
    # max_indices = np.argmax(record_max, axis=0)
    max_indices = np.argmax(record_area, axis=0)
    result_iou, result_mask = np.zeros((n_view, len(classes)), dtype=np.float32), np.zeros((n_view, len(classes), *image_shape), dtype=bool)
    print(n_iou)
    for i in range(n_view):
        for j in range(len(classes)):
            id0 = max_indices[i][j]
            result_iou[i, j] = n_iou[id0, i, j]
            result_mask[i, j, :] = get_mask[id0, i, j, :]
    print(result_iou, result_iou.shape)
    logger.info(f'area thresh: {select_area_bar}')
    logger.info(f'trunc thresh: {bar}')
    mean_iou = np.mean(result_iou)
    logger.info(f"iou chosen: {mean_iou:.4f}")

select_view = 2
color_list = [[255, 0, 0],     # 红色
              [0, 255, 0],     # 绿色
              [0, 0, 255],     # 蓝色
              [255, 255, 0],   # 黄色
              [255, 0, 255],   # 紫色
              [255, 165, 0],   # 橙色
              [0, 255, 255],]   # 青色
mask_list = result_mask[select_view]
print(mask_list.shape)
# exit()
img = cv2.imread(os.path.join(img_path, imglist[select_view] + '.jpg'))
img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
print(img.shape)
image_with_masks = img.copy() * 0.5
gt_seg = whole_mask[select_view]
for i, mask in tqdm(enumerate(mask_list)):
    # print(mask, color_list[i % len(color_list)])
    color = np.array(color_list[i % len(color_list)], dtype=np.uint8)
    color[0], color[1], color[2] = color[2], color[1], color[0]
    color_array = np.ones(img.shape, dtype=np.uint8) * np.array(color, dtype=np.uint8)[np.newaxis, np.newaxis, :]
    image_with_masks[mask] = cv2.addWeighted(img, 0.5, np.array(color_array, dtype=np.uint8), 0.5, 0)[mask]
cv2.imwrite('./3do_mask_{}.jpg'.format(dataset_name), image_with_masks)

for i, mask in tqdm(enumerate(mask_list)):
    # print(mask, color_list[i % len(color_list)])
    gt_mask = gt_seg[..., i] > 0.5
    color = np.array(color_list[i % len(color_list)], dtype=np.uint8)
    color[0], color[1], color[2] = color[2], color[1], color[0]
    color_array = np.ones(img.shape, dtype=np.uint8) * np.array(color, dtype=np.uint8)[np.newaxis, np.newaxis, :]
    image_with_masks[gt_mask] = cv2.addWeighted(img, 0.5, np.array(color_array, dtype=np.uint8), 0.5, 0)[gt_mask]
cv2.imwrite('./3do_mask_{}_gt.jpg'.format(dataset_name), image_with_masks)
