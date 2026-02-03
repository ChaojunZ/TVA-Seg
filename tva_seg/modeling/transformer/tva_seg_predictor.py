import fvcore.nn.weight_init as weight_init
import open_clip.eva_clip
import torch

from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .model import Aggregator
from tva_seg.third_party import clip
from tva_seg.third_party import imagenet_templates

import numpy as np
import open_clip
import open_clip.eva_clip as eva_clip
import json

from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import Tuple

class CF(nn.Module):
    def __init__(self, device, feature_dim=512, memory_size=25, reduction=4, frozen_text_embedding=None, alpha=0.2,
                 momentum=0.8):
        super().__init__()
        self.device = device
        self.scale = 1.68
        self.memory_size = int(memory_size*self.scale)
        self.feature_dim = feature_dim
        self.text_fine_cache = F.normalize(torch.rand(self.memory_size, feature_dim), dim=-1)
        self.text_fine_cache = self.text_fine_cache.to(self.device)
        self.text_fine_cache = self.text_fine_cache.cuda()
        self.alpha = alpha
        self.momentum = momentum
        if frozen_text_embedding is not None:
            self.frozen_text_embedding = frozen_text_embedding

        self.extractor = nn.Linear(2 * feature_dim, feature_dim, bias=False)
        self.extractor = self.extractor.to(self.device)
        self.extractor = self.extractor.cuda()

        self.writeTF = lambda x: x.clone()

    def forward(self, text_token=None,image_token4=None,image_token8=None,image_token12=None):
        fine_feature = self.read(text_token)
        text_fine_feature = torch.cat((text_token, fine_feature), dim=-1)
        text_fine_feature = self.alpha * self.extractor(text_fine_feature) + text_token
        if self.training:
            _ = self.write(image_token4, image_token8, image_token12)
            normalized_text_features = F.normalize(text_fine_feature, dim=-1)
            loss = F.l1_loss(normalized_text_features, text_token, reduction='mean')
        else:
            loss = 0.0

        return text_fine_feature, loss

    def get_score(self, query, mem):
        score = query @ mem.t()
        score_query = F.softmax(score, dim=0)
        score_mem = F.softmax(score, dim=1) # Determine which image regions are most relevant to the given text
        return score_query, score_mem

    def read(self, x):
        base_features = F.normalize(x, dim=-1)
        C, d = x.shape
        if self.training:
            self.text_fine_cache = self.text_fine_cache.detach()
        _, softmax_score_cache = self.get_score(base_features, self.text_fine_cache)
        fine_feature = softmax_score_cache @ self.text_fine_cache  # (N, d)

        return fine_feature

    def write(self, image_token4,image_token8,image_token12):
        """
        Update the text based on the three images
        """
        m, d = self.text_fine_cache.shape

        # Process features for each image

        base_features = image_token4.clone()

        base_features8 = image_token8.clone()

        base_features12 = image_token12.clone()


        base_features = base_features.reshape(-1, d)  # (B * P, d)
        base_features = F.normalize(base_features, dim=-1)
        softmax_score_query, softmax_score_cache = self.get_score(base_features, self.text_fine_cache)  # (B*P, 50)
        _, updating_indices = torch.topk(softmax_score_cache, 1, dim=1)
        updated_cache1 = self.text_fine_cache.clone().detach()
        # Update the cache
        for i in range(m):
            idx = torch.nonzero(updating_indices.squeeze(1) == i)
            a, _ = idx.size()
            if a != 0:
                score = (softmax_score_query[idx, i] / torch.max(softmax_score_query[:, i]))
                updated_cache1[i] = self.momentum * self.text_fine_cache[i] + (1 - self.momentum) * torch.sum(
                    score * base_features[idx.squeeze(1)], dim=0)
        updated_cache1 = F.normalize(updated_cache1, dim=-1)

        base_features8 = base_features8.reshape(-1, d)  # (B * P, d)
        base_features8 = F.normalize(base_features8, dim=-1)
        softmax_score_query, softmax_score_cache = self.get_score(base_features8, self.text_fine_cache)  # (B*P, 50)
        _, updating_indices = torch.topk(softmax_score_cache, 1, dim=1)
        updated_cache2 = self.text_fine_cache.clone().detach()
        # Update the cache
        for i in range(m):
            idx = torch.nonzero(updating_indices.squeeze(1) == i)
            a, _ = idx.size()
            if a != 0:
                score = (softmax_score_query[idx, i] / torch.max(softmax_score_query[:, i]))
                updated_cache2[i] = self.momentum * self.text_fine_cache[i] + (1 - self.momentum) * torch.sum(
                    score * base_features8[idx.squeeze(1)], dim=0)
        updated_cache2 = F.normalize(updated_cache2, dim=-1)

        base_features12 = base_features12.reshape(-1, d)  # (B * P, d)
        base_features12 = F.normalize(base_features12, dim=-1)
        softmax_score_query, softmax_score_cache = self.get_score(base_features12, self.text_fine_cache)  # (B*P, 50)
        _, updating_indices = torch.topk(softmax_score_cache, 1, dim=1)
        updated_cache3 = self.text_fine_cache.clone().detach()
        # Update the cache
        for i in range(m):
            idx = torch.nonzero(updating_indices.squeeze(1) == i)
            a, _ = idx.size()
            if a != 0:
                score = (softmax_score_query[idx, i] / torch.max(softmax_score_query[:, i]))
                updated_cache3[i] = self.momentum * self.text_fine_cache[i] + (1 - self.momentum) * torch.sum(
                    score * base_features12[idx.squeeze(1)], dim=0)
        updated_cache3 = F.normalize(updated_cache3, dim=-1)
        updated_cache= (updated_cache1+updated_cache2+updated_cache3)/3

        loss = 0.0
        self.text_fine_cache = updated_cache.to(self.device)
        return loss

class TvaSegPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        train_class_json: str,
        test_class_json: str,
        clip_pretrained: str,
        cache_dir: str,
        prompt_ensemble_type: str,
        text_guidance_dim: int,
        text_guidance_proj_dim: int,
        appearance_guidance_dim: int,
        appearance_guidance_proj_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        num_heads: int,
        num_layers: tuple,
        hidden_dims: tuple,
        pooling_sizes: tuple,
        feature_resolution: tuple,
        window_sizes: tuple,
        attention_type: str,
    ):
        """
        Args:
            
        """
        super().__init__()
        # use class_texts in train_forward, and test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        if clip_pretrained == "ViT-G" or clip_pretrained == "ViT-H":
            # for OpenCLIP models
            name, pretrain = ('ViT-H-14', 'laion2b_s32b_b79k') if clip_pretrained == 'ViT-H' else ('ViT-bigG-14', 'laion2b_s39b_b160k')
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                name, 
                pretrained=pretrain, 
                device=device, 
                force_image_size=336,)
        
            self.tokenizer = open_clip.get_tokenizer(name)
        elif clip_pretrained=="EVA02-CLIP-B-16" or clip_pretrained=="EVA02-CLIP-L-14-336":
            clip_model = eva_clip.create_model(model_name=clip_pretrained,
                                                pretrained=cache_dir, 
                                                force_custom_clip=True,
                                                precision="amp",
                                                device=device)
            self.tokenizer = open_clip.get_tokenizer(clip_pretrained)
            clip_preprocess=None
        else:
            # for OpenAI models
            clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
        self.prompt_ensemble_type = prompt_ensemble_type   
        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene',] # we chose the fixed template combined with LLM approach
            # prompt_templates = ['{}',]
        else:
            raise NotImplementedError
        
        self.prompt_templates = prompt_templates
        self.text_features = self.class_embeddings(self.class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        self.text_features_test = self.class_embeddings(self.test_class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        
        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess
        transformer = Aggregator(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=num_heads, 
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            window_size=window_sizes,
            attention_type=attention_type,
            prompt_channel=len(prompt_templates),
            )
        self.transformer = transformer
        self.tokens = None
        self.cache = None

        self.memory = Memory(device=device, feature_dim=768, memory_size=42, alpha=0.08)

    @classmethod
    def from_config(cls, cfg):#, in_channels, mask_classification):
        ret = {}

        ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
        ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["cache_dir"] = cfg.MODEL.SEM_SEG_HEAD.CACHE_DIR
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE

        # Aggregator parameters:
        ret["text_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM
        ret["text_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM
        ret["appearance_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM
        ret["appearance_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM

        ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
        ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
        ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS

        ret["prompt_depth"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
        ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

        ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["hidden_dims"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
        ret["pooling_sizes"] = cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES
        ret["feature_resolution"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION
        ret["window_sizes"] = cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES
        ret["attention_type"] = cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE

        return ret
    
    def hierarchical_clustering(self, text_embeddings):
        text_embeddings = text_embeddings.detach().cpu().numpy()
        distance_matrix = cosine_distances(text_embeddings)
        condensed_matrix = squareform(distance_matrix)
        Z = linkage(condensed_matrix, method='ward')
        clusters = fcluster(Z, t=0.1, criterion='distance') - 1
        return clusters
    
    def _get_cluster_indices(self, cluster_id: int, cluster_info) -> np.ndarray:
        return np.where(cluster_info == cluster_id)[0]
    
    def _get_most_similar_class(self, similarities: np.ndarray, indices: np.ndarray) -> int:

        similarities_in_cluster = similarities[indices]
        idx_max_similarity = np.argmax(similarities_in_cluster)
        return int(indices[idx_max_similarity])
    
    def _compute_topk_similarities(
        self, 
        image_features: torch.Tensor, 
        text_embedding: torch.Tensor, 
        k: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        similarities = torch.matmul(image_features, text_embedding.t()).squeeze(1)
        k = min(k, image_features.size(0))
        return torch.topk(similarities, k)
    
    def _compute_adjusted_embedding(
        self, 
        text_embedding: torch.Tensor, 
        mean_vector: torch.Tensor, 
        alpha: float
    ) -> torch.Tensor:
        adjusted = (1 - alpha) * text_embedding + alpha * mean_vector
        return F.normalize(adjusted, dim=0)
    
    def hierarchical_prompt(
        self, 
        image_features: torch.Tensor, 
        image_class_similarities: torch.Tensor,
        cluster_info,
        query_features
    ) -> torch.Tensor:
        B,_ = image_class_similarities.shape
        text1 = []
        for b in range(B):
            image_class_similarities_np = image_class_similarities[b,:].squeeze(0).detach().cpu().numpy()
            image_features1 = image_features[b,:,:].squeeze(0).to(query_features.device)
            text_features = query_features.clone().to(query_features.device)
            adjusted_text_features = text_features.clone()
            
            num_clusters = len(set(cluster_info))
            for cluster_id in range(num_clusters):

                indices = self._get_cluster_indices(cluster_id, cluster_info)
                class_i = self._get_most_similar_class(image_class_similarities_np, indices)
                
                text_embedding_i = F.normalize(text_features[class_i].unsqueeze(0), dim=1)
                image_features_norm = F.normalize(image_features1, dim=1)
                
                _, topk_indices = self._compute_topk_similarities(
                    image_features_norm, 
                    text_embedding_i
                )
                topk_image_features = image_features1[topk_indices]
                
                mean_vector = F.normalize(topk_image_features.mean(dim=0), dim=0)
                adjusted_embedding = self._compute_adjusted_embedding(
                    text_embedding_i.squeeze(0),
                    mean_vector,
                    0.01,
                )
                adjusted_text_features[class_i] = adjusted_embedding
            text1.append(adjusted_text_features)
        stacked = torch.stack(text1, dim=0)
        text_feature = stacked.mean(dim=0)
        return text_feature

    def forward(self, x,clip_features, all_feat, dino_featss, all_features, vfm_feats, filtered_ids=None, prompt=None, gt_cls=None):
        dino_featss = [dino_featss[k] for k in dino_featss.keys()][::-1]
        clip_features = [clip_features[k] for k in clip_features.keys()][::-1]
        text = self.class_texts if self.training else self.test_class_texts
        text = [text[c] for c in gt_cls] if gt_cls is not None else text
        text = self.get_text_embeds(text, self.prompt_templates, self.clip_model, prompt)
        text = text.squeeze(1)
        cluster_info = self.hierarchical_clustering(text)
        global_clip_sim = all_feat[:, 0, :] @ text.T
        text = self.hierarchical_prompt(all_feat[:, 1:, :], global_clip_sim, cluster_info, text)
        text, loss1 = self.memory(text, all_features[0], all_features[1], all_feat)
        text = text.unsqueeze(1)
        text = text.repeat(x.shape[0], 1, 1, 1)
        out = self.transformer(x,clip_features, text, dino_featss, vfm_feats, filtered_ids)
        return out, loss1

    @torch.no_grad()
    def class_embeddings(self, classnames, templates, clip_model):
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]  # format with class
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).cuda()
            else: 
                texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights
    
    def get_text_embeds(self, classnames, templates, clip_model, prompt=None):
        if self.cache is not None and not self.training:
            return self.cache
        
        if self.tokens is None or prompt is not None:
            tokens = []
            for classname in classnames:
                if ', ' in classname:
                    classname_splits = classname.split(', ')
                    texts = [template.format(classname_splits[0]) for template in templates]
                else:
                    texts = [template.format(classname) for template in templates]  # format with class
                if self.tokenizer is not None:
                    texts = self.tokenizer(texts).cuda()
                else: 
                    texts = clip.tokenize(texts).cuda()
                tokens.append(texts)
            tokens = torch.stack(tokens, dim=0).squeeze(1)
            if prompt is None:
                self.tokens = tokens
        elif self.tokens is not None and prompt is None:
            tokens = self.tokens

        class_embeddings = clip_model.encode_text(tokens, prompt)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        
        
        class_embeddings = class_embeddings.unsqueeze(1)
        
        if not self.training:
            self.cache = class_embeddings
            
        return class_embeddings