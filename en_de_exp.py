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

# dataname = 'teatime'
# dataname = 'figurines'
# dataname = 'waldo_kitchen'
dataname = 'ramen'
# dataname = 'room'
# dataname = 'sofa'
# dataname = 'office_desk'
n_view = 3
if dataname in ['teatime', 'waldo_kitchen', 'figurines', 'ramen']:
    n_view = 4

needi = 0
img_path = './data/{}/dust3r_{}_views/images'.format(dataname, n_view)
imgs = sorted(os.listdir(img_path))
img0 = cv2.imread(os.path.join(img_path, imgs[needi]))
ori_height, ori_width = img0.shape[:2]

feature_path = ['./data/{}/dust3r_{}_views/language_features'.format(dataname, n_view), './data/{}/dust3r_{}_views/language_features_dim3'.format(dataname, n_view)]
language_feature_name = os.path.join(feature_path[0], imgs[needi].split('.', 1)[0])
feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy')).to('cuda')
feature_map_dim3 = torch.from_numpy(np.load(os.path.join(feature_path[1], imgs[needi].split('.', 1)[0]) + '_f.npy')).to(device).to(torch.float32)
# print(feature_map.shape)
# exit()
ckpt_path = f"./autoencoder/ckpt/{dataname}/best_ckpt.pth"
model_ae = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

checkpoint = torch.load(ckpt_path)
model_ae.load_state_dict(checkpoint)
model_ae.eval()

# data = np.load('/home/hu997372/code/InstantSplat/gaussian-splatting/output/3do_2/train/ours_1000/renders_npy/00000.npy', allow_pickle=True)
middle = model_ae.encode(feature_map)
get = model_ae.decode(middle)
cc = get * feature_map
cc = torch.sum(cc, dim=1)
ave = torch.mean(cc)
print(cc, ave)