from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import torch
import torchvision
from torch import nn

try:
    import open_clip
except ImportError:
    raise ImportError("open_clip is not installed, install it with `pip install open-clip-torch`")


@dataclass
class OpenCLIPNetworkConfig:
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str, ...] = ("object", "things", "stuff", "texture")
    positives: Tuple[str, ...] = ("",)
    device: str = "cuda"
    precision: str = "fp16"
    _target: object = field(default=None, repr=False)


class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig | type | str | torch.device | None = None, device: str | torch.device | None = None):
        super().__init__()
        if config is None:
            config = OpenCLIPNetworkConfig()
        elif isinstance(config, type):
            config = config()
        elif isinstance(config, (str, torch.device)):
            device = config
            config = OpenCLIPNetworkConfig()

        self.config = config
        self.device = torch.device(device or config.device)
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
            config.clip_model_type,
            pretrained=config.clip_model_pretrained,
            precision=config.precision,
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(config.clip_model_type)
        self.model = model.to(self.device)
        self.clip_n_dims = config.clip_n_dims

        self.negatives = tuple(config.negatives)
        self.positives = tuple(config.positives)
        self._set_negative_embeddings()
        self.set_positives(self.positives)

        assert self.pos_embeds.shape[1] == self.neg_embeds.shape[1], "Positive and negative embeddings must have the same dimensionality"
        assert self.pos_embeds.shape[1] == self.clip_n_dims, "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def _tokenize(self, text_list: Sequence[str]) -> torch.Tensor:
        return torch.cat([self.tokenizer(phrase) for phrase in text_list]).to(self.device)

    def _set_negative_embeddings(self) -> None:
        with torch.no_grad():
            tok_phrases = self._tokenize(self.negatives)
            self.neg_embeds = self.model.encode_text(tok_phrases)
        self.neg_embeds = self.neg_embeds / self.neg_embeds.norm(dim=-1, keepdim=True)

    def gui_cb(self, element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list: Sequence[str]) -> None:
        self.positives = tuple(text_list)
        with torch.no_grad():
            tok_phrases = self._tokenize(self.positives)
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds = self.pos_embeds / self.pos_embeds.norm(dim=-1, keepdim=True)

    def set_semantics(self, text_list: Sequence[str]) -> None:
        self.semantic_labels = tuple(text_list)
        with torch.no_grad():
            tok_phrases = self._tokenize(self.semantic_labels)
            self.semantic_embeds = self.model.encode_text(tok_phrases)
        self.semantic_embeds = self.semantic_embeds / self.semantic_embeds.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T)
        positive_vals = output[..., positive_id : positive_id + 1]
        negative_vals = output[..., len(self.positives) :]
        repeated_pos = positive_vals.repeat(1, len(self.negatives))

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        best_id = softmax[..., 0].argmin(dim=1)
        return torch.gather(
            softmax,
            1,
            best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2),
        )[:, 0, :]

    def encode_image(self, input_tensor: torch.Tensor, mask=None) -> torch.Tensor:
        processed_input = self.process(input_tensor).half()
        if mask is not None:
            return self.model.encode_image(processed_input, mask=mask)
        return self.model.encode_image(processed_input)

    def encode_text(self, input_text, device: str | torch.device | None = None) -> torch.Tensor:
        if isinstance(input_text, torch.Tensor):
            text = input_text.to(device or self.device)
        else:
            text = self.tokenizer(input_text).to(device or self.device)
        return self.model.encode_text(text)

    def get_semantic_map(self, sem_map: torch.Tensor) -> torch.Tensor:
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

    def get_max_across(self, sem_map: torch.Tensor) -> torch.Tensor:
        n_phrases = len(self.positives)
        n_levels, h, w, _ = sem_map.shape
        clip_output = sem_map.permute(1, 2, 0, 3).flatten(0, 1)

        n_levels_sims = []
        for i in range(n_levels):
            n_phrases_sims = []
            for j in range(n_phrases):
                probs = self.get_relevancy(clip_output[..., i, :], j)
                n_phrases_sims.append(probs[..., 0:1])
            n_levels_sims.append(torch.stack(n_phrases_sims))

        return torch.stack(n_levels_sims).view(n_levels, n_phrases, h, w)
