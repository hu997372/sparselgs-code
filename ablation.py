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
    
    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
                ).to(self.neg_embeds.device)
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)
    
    def encode_text(self, input):
        processed_input = input
        return self.model.encode_text(processed_input)
    
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

gt_idxfolder = '/home/hu997372/code/langsplat/data/3D-OVS/{}/segmentations'.format(dataset_name)
txt_filepath = os.path.join(gt_idxfolder, 'classes.txt')
with open(txt_filepath, 'r') as f:
    categories = f.read().splitlines()

encoder_hidden_dims = [256, 128, 64, 32, 3]
decoder_hidden_dims = [16, 32, 64, 128, 256, 256, 512]
ckpt_path = f"./autoencoder/ckpt/{dataset_name}/best_ckpt.pth"
from autoencoder.model import Autoencoder
model_ae = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
checkpoint = torch.load(ckpt_path)
model_ae.load_state_dict(checkpoint)
model_ae.eval()

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
record_area_ab = np.zeros((len(feat_dir), n_view, len(classes)), dtype=np.float32)
n_iou_ab = np.zeros((len(feat_dir), n_view, len(classes)), dtype=np.float32)
get_mask_ab = np.zeros((len(feat_dir), n_view, len(classes), *image_shape), dtype=bool)
import time
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
if not os.path.exists(logfolder):
    os.makedirs(logfolder, exist_ok=True)
log_file = os.path.join(logfolder, f'{timestamp}.log')
logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)

bar = args.bar
select_area_bar = 3000
print(classes)
with torch.no_grad():
    for k in tqdm(range(len(feat_dir))):
        feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[k], '*.npy')),
                                key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
        print(feat_paths_lvl)
        savePath = f'{logfolder}'
        for i, frame_idx in enumerate(maskfolders):
            gt_seg = whole_mask[i] # [H*W=N1, n_classes]

            # get feature map
            pred_feature = torch.from_numpy(np.load(feat_paths_lvl[i])).reshape(-1, 3).to(device)
            feature_map = torch.from_numpy(np.load(os.path.join(gt_512dir, frame_idx + '_f.npy'))).to(device)
            feature_map_dim3 = torch.from_numpy(np.load(os.path.join(gt_3dir,  frame_idx + '_f.npy'))).to(device)
            image_feature_get = torch.matmul(pred_feature, feature_map_dim3.T)
            _, indices = torch.max(image_feature_get, dim=1)
            outputs = feature_map[indices].reshape(H, W, -1)

            outputs_ablation = model_ae.decode(pred_feature).reshape(H, W, 512)
            relevancy_map = outputs @ text_features.T # [N1,N2]
            relevancy_map_ab = outputs_ablation @ text_features.T # [N1,N2]
            for ind, class0 in enumerate(classes):
                gt_class = gt_seg[:, :, ind]
                rel_now = relevancy_map[:, :, ind]
                nowmax, nowmin = rel_now.max(), rel_now.min()
                # print(nowmax, nowmin)
                pred_class = rel_now > bar * nowmax
                record_area[k][i][ind] = torch.sum(pred_class).detach().to('cpu').numpy()
                pred_class = pred_class.detach().to('cpu').numpy()
                intersection = np.sum(np.logical_and(gt_class, pred_class))
                union = np.sum(np.logical_or(gt_class, pred_class))
                if np.sum(pred_class) < select_area_bar:
                    nowmax = 0
                record_max[k][i][ind] = nowmax
                iou = round(intersection / union, 4)
                n_iou[k][i][ind] = iou
                get_mask[k][i][ind] = pred_class
                if not os.path.exists(f'{savePath}/{k}/{frame_idx}'):
                    os.makedirs(f'{savePath}/{k}/{frame_idx}', exist_ok=True)
                cv2.imwrite(f'{savePath}/{k}/{frame_idx}/{class0}.png', get_mask[k][i][ind] * 255.)
                cv2.imwrite(f'{savePath}/{k}/{frame_idx}/{class0}_gt.png', gt_class * 255.)

            for ind, class0 in enumerate(classes):
                gt_class = gt_seg[:, :, ind]
                rel_now = relevancy_map_ab[:, :, ind]
                nowmax, nowmin = rel_now.max(), rel_now.min()
                # print(nowmax, nowmin)
                pred_class = rel_now > bar * nowmax
                record_area_ab[k][i][ind] = torch.sum(pred_class).detach().to('cpu').numpy()
                pred_class = pred_class.detach().to('cpu').numpy()
                intersection = np.sum(np.logical_and(gt_class, pred_class))
                union = np.sum(np.logical_or(gt_class, pred_class))
                if np.sum(pred_class) < select_area_bar:
                    nowmax = 0
                # record_max[k][i][ind] = nowmax
                iou = round(intersection / union, 4)
                n_iou_ab[k][i][ind] = iou
                get_mask_ab[k][i][ind] = pred_class
                if not os.path.exists(f'{savePath}/{k}/{frame_idx}'):
                    os.makedirs(f'{savePath}/{k}/{frame_idx}', exist_ok=True)
                cv2.imwrite(f'{savePath}/{k}/{frame_idx}/{class0}_ab.png', get_mask_ab[k][i][ind] * 255.)
    # max_indices = np.argmax(record_max, axis=0)
    max_indices = np.argmax(record_area, axis=0)
    max_indices_ab= np.argmax(record_area_ab, axis=0)
    result_iou, result_mask = np.zeros((n_view, len(classes)), dtype=np.float32), np.zeros((n_view, len(classes), *image_shape), dtype=bool)
    result_iou_ab, result_mask_ab = np.zeros((n_view, len(classes)), dtype=np.float32), np.zeros((n_view, len(classes), *image_shape), dtype=bool)
    print(n_iou_ab)
    print(n_iou)
    for i in range(n_view):
        for j in range(len(classes)):
            id0 = max_indices[i][j]
            result_iou[i, j] = n_iou[id0, i, j]
            result_mask[i, j, :] = get_mask[id0, i, j, :]
            
            id1 = max_indices_ab[i][j]
            result_iou_ab[i, j] = n_iou_ab[id1, i, j]
            result_mask_ab[i, j, :] = get_mask_ab[id0, i, j, :]
    print(result_iou, result_iou.shape)
    logger.info(f'area thresh: {select_area_bar}')
    logger.info(f'trunc thresh: {bar}')
    mean_iou = np.mean(result_iou)
    logger.info(f"iou chosen: {mean_iou:.4f}")

    mean_iou_ab = np.mean(result_iou_ab)
    logger.info(f"iou chosen ablation: {mean_iou_ab:.4f}")
    print(result_iou_ab)

# bar = 0.95 # bed
# bar = 0.7 # bench
# bar = 0.85 # room  office_desk
# bar = 0.75 # sofa