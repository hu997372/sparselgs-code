import torch
from dataclasses import dataclass, field
import sys
try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"
from typing import Tuple, Type

import torch
import torchvision
from torch import nn
import random
import argparse
import shutil
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
sys.path.append("./gaussian-splatting")
from autoencoder.dataset import Autoencoder_dataset
from autoencoder.model import Autoencoder

parser = argparse.ArgumentParser()
parser.add_argument('--encoder_dims',
                nargs = '+',
                type=int,
                default=[256, 128, 64, 32, 3],
                )
parser.add_argument('--decoder_dims',
                nargs = '+',
                type=int,
                default=[16, 32, 64, 128, 256, 256, 512],
                )
args = parser.parse_args()

encoder_hidden_dims = args.encoder_dims
decoder_hidden_dims = args.decoder_dims

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

# model.to(device)

# 定义要转换的文本
# texts = ["a photo of a dog", "a photo of a cat", "a photo of a bird"]
# texts = ['Pikachu', 'a stack of UNO cards', 'a red Nintendo Switch joy-con controller', 'Gundam', 'Xbox wireless controller', 'grey sofa']
# texts = ['Gundam']
# texts = ['sheep']
# texts = ['paper napkin', 'toy sheep', 'toy bear', 'cookie', 'desk']
# texts = ['red bag', 'black leather shoe', 'banana', 'hand', 'camera', 'white sheet']
# texts = ["wood wall", "shrilling chicken", "weaving basket", "rabbit", "dinosaur", "baseball"]
# texts = ['pumpkin', 'red apple', 'rubber duck', 'spatula', 'tesla door handle', 'old camera', 'green apple']
texts = ['the book of The Unbearable Lightness of Being', 'a can of red bull drink', 'a white keyboard', 'a pack of pocket tissues', 'desktop', 'blue partition'] #, 
# texts = ['desktop'] #, 'blue partition'
num_query = len(texts)

# 对文本进行编码
text_inputs = tokenizer(texts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)

import torch.nn.functional as F
# 对文本特征进行归一化
# text_features /= text_features.norm(dim=-1, keepdim=True)
text_features = F.normalize(text_features, dim=1)


import os 
import numpy as np
import cv2

# dataname = 'teatime_before'
# dataname = 'figurines'
# dataname = 'room'
# dataname = 'sofa'
dataname = 'office_desk'
n_view = 3
if dataname in ['teatime', 'waldo_kitchen', 'figurines', 'ramen']:
    n_view = 4

needi = 1
img_path = './data/{}/dust3r_{}_views/images'.format(dataname, n_view)
imgs = sorted(os.listdir(img_path))
img0 = cv2.imread(os.path.join(img_path, imgs[needi]))
ori_height, ori_width = img0.shape[:2]

feature_path = ['./data/{}/dust3r_{}_views/language_features'.format(dataname, n_view), './data/{}/dust3r_{}_views/language_features_dim3'.format(dataname, n_view)]
language_feature_name = os.path.join(feature_path[0], imgs[needi].split('.', 1)[0])
seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy')).to('cuda')
feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy'))
feature_map_dim3 = torch.from_numpy(np.load(os.path.join(feature_path[1], imgs[needi].split('.', 1)[0]) + '_f.npy')).to(device).to(torch.float32)
image_features = feature_map.to(device)

feature_level = 3
ind_seg = np.load(os.path.join(feature_path[1], 'ind_seg.npy'))[needi, feature_level-1, :]

# 计算余弦相似度
similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
# similarity = (100.0 * text_features @ image_features[ind_seg[0]:ind_seg[1]+1, :].T).softmax(dim=-1)

# 打印最相似的文本及其置信度
k_top = 1
values, indices = similarity.topk(k_top)

print(values.squeeze(), indices.squeeze())
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
img0 = Image.open(os.path.join(img_path, imgs[needi])).resize((width, height))
img0 = np.array(img0) #.transpose(2, 0, 1)
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

# print(np.load(os.path.join(feature_path[1], 'ind_seg.npy')))
y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
bool_seg_list = []
for now_ind in indices.squeeze(0):
    for i in range(1, 4):
        if now_ind in seg_map[i]:
            bool_seg_list.append(seg_map[i] == now_ind)
            break
    # seg = seg_map[1, y, x].squeeze(-1).long()
# mask = (seg == indices.to('cpu')).reshape(height, width, -1)*255.
from collections import Counter
max_key_list, mask_list = [], []
for i, bool_seg in enumerate(bool_seg_list):
    seg_0 = seg_map[feature_level][bool_seg]
    count = Counter(seg_0.detach().cpu().numpy())
    max_key_list.append(max(count, key=lambda k: count[k]))
    # print(mask, mask.shape)
for max_key in max_key_list:
    mask_list.append((seg_map[feature_level, y, x].detach().cpu().numpy() == max_key).reshape(height, width))

num_masks = len(mask_list)
# 生成随机颜色映射
colors = []
for _ in range(num_masks):
    color = [random.randint(0, 255) for _ in range(3)]
    colors.append(color)
image_with_masks = img0.copy()
# print(img0.shape, mask_list[0].shape)

for i, (mask, color) in enumerate(zip(mask_list, colors)):
    color_array = np.ones(img0.shape, dtype=np.uint8) * np.array(color, dtype=np.uint8)[np.newaxis, np.newaxis, :]
    image_with_masks[mask] = cv2.addWeighted(image_with_masks, 0.5, np.array(color_array, dtype=np.uint8), 0.5, 0)[mask]
    
cv2.imwrite("output_image_with_masks_gt.jpg", image_with_masks)

ckpt_path = f"./autoencoder/ckpt/{dataname}/best_ckpt.pth"

feature_dir = feature_path[1]
# model_ae = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

# checkpoint = torch.load(ckpt_path)
# model_ae.load_state_dict(checkpoint)
# model_ae.eval()

y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
seg = seg_map[:, y, x].squeeze(-1).long()
# data = feature_list[seg[feature_level, :].detach().cpu().numpy()]

# data = np.load('/home/hu997372/code/InstantSplat/gaussian-splatting/output/3do_2/train/ours_1000/renders_npy/00000.npy', allow_pickle=True)
data = np.load('./output/{}/{}_views_{}/train/ours_1000/renders_npy/0000{}.npy'.format(dataname, n_view, feature_level, needi), allow_pickle=True)
data = torch.from_numpy(data).to(device).reshape(-1, 3)
# outputs = model_ae.decode(data)
# index_dict = Counter(outputs)
# total_key = index_dict.keys(), index_dict.values()

# print(feature_map_dim3[ind_seg[0]:ind_seg[1]+1, :] @ feature_map_dim3[ind_seg[0]:ind_seg[1]+1, :].T)
# image_feature_get = torch.matmul(data, feature_map_dim3[ind_seg[0]:ind_seg[1]+1, :].T)
# _, indices = torch.max(image_feature_get, dim=1)
diff = data[:, None, :] - feature_map_dim3[None, :, :]
# diff = feature_map_dim3[:, None, :] - feature_map_dim3[None, :, :]
# print(diff.shape)
dist_matrix = diff.norm(p=2, dim=2, keepdim=False)

_, indices = torch.min(dist_matrix, dim=1)
print(indices.shape)
outputs = image_features[indices].reshape(height, width, -1)
# print(indices, image_feature_get.shape)
# exit()

# print(outputs.reshape(-1, 512).unique(dim=0))
print(image_features[ind_seg[0]:ind_seg[1]+1, :])
corr_map = outputs @ text_features.T
# similarity = (100.0 * text_features @ outputs.T).softmax(dim=-1)
# print(text_features.shape, outputs.shape)
corr_map = corr_map.detach().cpu().numpy()
# print(corr_map.shape)
bar = 0.6
from torchvision import transforms
import utils.color_utils as colormaps
for ind in range(num_query):
    print(texts[ind])
    img_now = img0.copy()
    now_map = corr_map[:, :, ind]
    # norm_data = ((now_map - now_map.min()) / (now_map.max() - now_map.min()) * 255).astype(np.uint8)
    norm_data = torch.from_numpy(now_map)
    norm_data = (norm_data - norm_data.min()) / (norm_data.max() - norm_data.min())
    scale = 30
    kernel = np.ones((scale,scale)) / (scale**2)
    avg_filtered = cv2.filter2D(norm_data.detach().cpu().numpy(), -1, kernel)
    avg_filtered = torch.from_numpy(avg_filtered)
    relev_norm = 0.5 * (avg_filtered + norm_data).unsqueeze(-1)
    p_i = torch.clip(relev_norm - 0.5, 0, 1)
    valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
    mask0 = (relev_norm < bar).squeeze()
    ii = torch.from_numpy(img0) / 255.
    print(valid_composited.shape, ii.shape)
    valid_composited[mask0, :] = ii[mask0, :] * 0.3

    # 保存图像
    transforms.ToPILImage()(valid_composited.permute(2, 0, 1)).save('heatmap_{}.jpg'.format(ind))

    # print(now_map.min(), now_map.max())
    mask = now_map > now_map.max() * bar
    # color_map = cv2.applyColorMap(norm_data, cv2.COLORMAP_JET)
    # overlay = cv2.addWeighted(img_now, 0.7, color_map, 0.3, 0)
    color_array = np.ones(img0.shape, dtype=np.uint8) * np.array(colors[ind], dtype=np.uint8)[np.newaxis, np.newaxis, :]
    img_now[mask] = cv2.addWeighted(img_now, 0.5, np.array(color_array, dtype=np.uint8), 0.5, 0)[mask]
    cv2.imwrite("output_{}.png".format(ind), img_now)

# print(index_dict.values())