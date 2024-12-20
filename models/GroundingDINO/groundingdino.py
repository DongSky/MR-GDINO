# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn,Tensor
from torchvision.ops.boxes import nms
# from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.visualizer import COCOVisualizer
from groundingdino.util.vl_utils import create_positive_map_from_span

from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss
from torch.nn.common_types import _size_2_t
from .matcher import build_matcher

import torch.nn.functional as F


from timm.models.layers import trunc_normal_


lan_scale = 0.1
vis_scale = 0.1



class RepZeroLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.scaling = nn.parameter.Parameter(torch.ones(1) * lan_scale)
        nn.init.constant_(self.weight, val=zero_value)
        self.freeze_linear = nn.Linear(in_features, out_features, bias, device, dtype)
        nn.init.constant_(self.freeze_linear.weight, val=0.0)
        if self.bias is not None:
            nn.init.constant_(self.freeze_linear.bias, val=0.0) 
        
        # self.zero_inter_loss = torch.nn.L1Loss(reduction='mean')
        # self.zero_inter_loss = torch.nn.MSELoss(reduction='mean')
        self.zero_inter_loss = torch.nn.SmoothL1Loss(reduction='mean')

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            branch_output = self.scaling * super().forward(input)
            output = branch_output + self.freeze_linear(input)
            return output, \
                self.zero_inter_loss(branch_output, torch.zeros_like(branch_output)) + \
                    self.zero_inter_loss(output, torch.zeros_like(output))
        else:
            return self.freeze_linear(input), torch.zeros(1).to(input)

    def __rep__(self):
        self.freeze_linear.weight.data = self.weight.data  * self.scaling + self.freeze_linear.weight.data
        self.freeze_linear.bias.data = self.bias.data  * self.scaling + self.freeze_linear.bias.data
        self.scaling = nn.parameter.Parameter(torch.ones(1).to(self.weight.data) * lan_scale)
        nn.init.constant_(self.weight, val=zero_value)
        if self.bias is not None:
            nn.init.constant_(self.bias, val=zero_value)
            
            


def shift_columns(tensor, m, k):
    """
    将前 m 列整体向右移动 k 列，并保持剩下的列为 -inf
    参数:
    - tensor: 需要操作的二维 tensor
    - m: 前 m 列是非-inf 的部分
    - k: 向右移动的列数
    """
    # 获取 tensor 的行数和列数
    num_rows, num_cols = tensor.shape

    # 创建新的 tensor，初始化为 -inf
    new_tensor = torch.full((num_rows, num_cols), float('-inf'))

    # 计算移动后有效列的起始索引和终止索引
    start_idx = k
    end_idx = min(k + m, num_cols)  # 确保不会超出边界

    # 将原 tensor 的前 m 列复制到新的 tensor 的 [k:k+m] 位置
    new_tensor[:, start_idx:end_idx] = tensor[:, :end_idx - k]

    return new_tensor


def find_inf_boundary(t):
    # 创建一个布尔掩码，检查每个元素是否为 -inf
    is_inf = t == float('-inf')
    
    # 对每一列求和，看看是否整列都为 -inf
    # 使用 all(dim=0) 来检查每一列是否都是 True (-inf)
    inf_cols = is_inf.all(dim=0)
    
    # 返回第一个全为 -inf 的列的索引
    inf_boundary_index = torch.nonzero(inf_cols).min().item()
    
    return inf_boundary_index


class CoOpModule(nn.Module):
    def __init__(self, prompt_length, prompt_channel, use_prompt=False, prompt=None) -> None:
        super().__init__()
        self.prompt_length = prompt_length
        self.prompt_channel = prompt_channel
        if use_prompt:
            self.coop_prompt = prompt
        else:
            self.coop_prompt = nn.Parameter(torch.zeros(1, self.prompt_length, self.prompt_channel))
            trunc_normal_(self.coop_prompt, std=0.02)
    
    def forward(self, x):
        return x



class Prompt(nn.Module):
    def __init__(self, length=4, embed_dim=768, embed_dim_key=768, embedding_key='mean', prompt_init='uniform', prompt_pool=True, 
                 prompt_key=True, pool_size=10, top_k=4, batchwise_prompt=True, prompt_key_init='uniform',):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.embed_dim_key = embed_dim_key
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
        
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim_key)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            
            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask # B, top_k

            batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            # out['prompt_idx'] = idx

            # # Debugging, return sim as well
            # out['prompt_norm'] = prompt_norm
            # out['x_embed_norm'] = x_embed_norm
            # out['similarity'] = similarity

            # # Put pull_constraint loss calculation inside
            # batched_key_norm = prompt_norm[idx] # B, top_k, C
            # out['selected_key'] = batched_key_norm
            # x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            # sim = batched_key_norm * x_embed_norm # B, top_k, C
            # reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

            # out['reduce_sim'] = reduce_sim

        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        # out['total_prompt_len'] = batched_prompt.shape[1]
        # out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return batched_prompt

zero_value = 1e-8
class RepZeroConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding = 0,
                 dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, zero_value=zero_value) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.scaling = nn.parameter.Parameter(torch.ones(1) * vis_scale)
        nn.init.constant_(self.weight, val=zero_value)
        if self.bias is not None:
            nn.init.constant_(self.bias, val=zero_value)
        
        self.freeze_conv = nn.Conv2d(in_channels, out_channels,
                                     kernel_size, stride, padding,
                                     dilation, groups, bias,
                                     padding_mode, device, dtype)
        nn.init.constant_(self.freeze_conv.weight, val=0.0)
        if self.bias is not None:
            nn.init.constant_(self.freeze_conv.bias, val=0.0)
        
        # self.zero_inter_loss = torch.nn.L1Loss(reduction='mean')
        # self.zero_inter_loss = torch.nn.MSELoss(reduction='mean')
        self.zero_inter_loss = torch.nn.SmoothL1Loss(reduction='mean')

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            branch_output = self.scaling * super().forward(input)
            output = branch_output + self.freeze_conv(input)
            return output, \
                self.zero_inter_loss(branch_output, torch.zeros_like(branch_output)) + \
                    self.zero_inter_loss(output, torch.zeros_like(output))
        else:
            return self.freeze_conv(input), torch.zeros(1).to(input)

    def __rep__(self):
        self.freeze_conv.weight.data = self.weight.data  * self.scaling + self.freeze_conv.weight.data
        self.freeze_conv.bias.data = self.bias.data  * self.scaling + self.freeze_conv.bias.data
        self.scaling = nn.parameter.Parameter(torch.ones(1).to(self.weight.data) *vis_scale)
        nn.init.constant_(self.weight, val=zero_value)
        if self.bias is not None:
            nn.init.constant_(self.bias, val=zero_value)
            
            

class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=4500,
        prompt_length=16, 
        prompt_channel=768,
        use_coop = False,
        use_prompt = False,
        use_zira = False,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = 4500
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        self.use_coop = use_coop
        self.use_prompt = use_prompt
        self.use_zira = use_zira
        if self.use_coop:
            self.prompt_length = prompt_length
            self.prompt_channel = prompt_channel
            self.coop = CoOpModule(prompt_length, prompt_channel)
        else:
            self.coop = None
            
            if use_prompt:
                self.prompt = Prompt()
                
    
        if self.use_zira:
            self.rep_linear_adapter = RepZeroLinear(in_features=self.bert.config.hidden_size,
                                            out_features=self.hidden_dim)
        
        
            
        

        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        # freeze

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        if use_zira:
            if num_feature_levels > 1:
                num_backbone_outs = len(backbone.num_channels)
                input_proj_list = []
                for _ in range(num_backbone_outs):
                    in_channels = backbone.num_channels[_]
                    input_proj_list.append(RepZeroConv2d(in_channels, hidden_dim, kernel_size=1))
                for _ in range(num_feature_levels - num_backbone_outs):
                    input_proj_list.append(RepZeroConv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1))
                    in_channels = hidden_dim
                self.input_proj_conv_adapter = nn.ModuleList(input_proj_list)
            else:
                assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
                self.input_proj_conv_adapter = nn.ModuleList([RepZeroConv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1)])



        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def forward(self, samples: NestedTensor, targets: List = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]
            

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
            
            
        # if self.use_prompt:
        #     with torch.no_grad():
        #         img_features = self.backbone[0](samples)
        #         print(img_features)
                
        #         exit()
                
        # else:    
        features, poss = self.backbone(samples)
        
        
        if self.use_prompt:
            img_features = features[-1].tensors.detach()
            
            img_features = img_features.flatten(2).transpose(1, 2)
            
            l2p_prompt = self.prompt(img_features)
            
            self.coop = CoOpModule(16, 768, use_prompt=True, prompt = l2p_prompt).to(samples.device)
            
    

        
        if "subtasks_prompts_list" in kw:
            
            mean_feature = torch.mean(features[-1].tensors, dim=(2,3)).cpu()
            subtasks_prompts_list = kw["subtasks_prompts_list"]
            subtasks_lora_list = kw["subtasks_lora_data"]
            subtasks_mean_feat = kw["subtasks_mean_feat"]
            
            
            max_key = -1
            max_sim = -10000
            for key in subtasks_mean_feat:
                sim = F.cosine_similarity(subtasks_mean_feat[key], mean_feature.cpu())
    
                if sim > max_sim:
                    max_sim = sim
                    max_key = key
                    
            # print(max_key, max_sim)
            
            # print(max)
            
            # print(self.use_coop)
                    
            if max_sim<kw['tau']:
                self.coop = None
                self.transformer.encoder.open_lora = False
                # print(1)
            else:
                if self.use_coop:
                    if self.coop == None:
                        self.coop = CoOpModule(self.prompt_length, self.prompt_channel).to(samples.device)
                    select_prompt = subtasks_prompts_list[max_key]
                    log = self.coop.load_state_dict(select_prompt, strict=False)
                    # print(log)
                    
                self.transformer.encoder.open_lora = True
                
                select_lora = subtasks_lora_list[max_key]
                
                log = self.transformer.load_state_dict(select_lora, strict = False)
                
                # print(log)
            
    
        # encoder texts
        # for i in range(len(captions[0])):
        if type(captions[0]) == list:
            
            final_logits = []
            
            final_boxes = []
            
            sum_pre = 0
            
            for seg_idx in range(len(captions[0])):
                tokenized = self.tokenizer([captions[0][seg_idx]], padding="longest", return_tensors="pt").to(
                    samples.device
                )
                
                seg_token_num = tokenized['input_ids'].shape[1]

            


                one_hot_token = tokenized

                (
                    text_self_attention_masks,
                    position_ids,
                    cate_to_token_mask_list,
                ) = generate_masks_with_special_tokens_and_transfer_map(
                    tokenized, self.specical_tokens, self.tokenizer, self.coop
                )

                if text_self_attention_masks.shape[1] > self.max_text_len:
                    text_self_attention_masks = text_self_attention_masks[
                        :, : self.max_text_len, : self.max_text_len
                    ]
                    position_ids = position_ids[:, : self.max_text_len]
                    tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
                    tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
                    tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

                # extract text embeddings
                if self.sub_sentence_present:
                    tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
                    tokenized_for_encoder["attention_mask"] = text_self_attention_masks
                    tokenized_for_encoder["position_ids"] = position_ids
                else:
                    tokenized_for_encoder = tokenized
                    
                tokenized_for_encoder["coop"] = self.coop
                
            
            

                bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

                encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
                
                if self.use_zira:
                    rep_linear_out, loss_linear_adapter = self.rep_linear_adapter(bert_output["last_hidden_state"])
                    encoded_text = rep_linear_out + encoded_text
                
            
                if self.coop:
                    encoded_text = torch.cat((encoded_text[:,:1], encoded_text[:,17:]), dim=1)

                text_token_mask = tokenized.attention_mask.bool()  # bs, 195
                # print(encoded_text.shape)
                # print(text_token_mask.shape)
                if self.coop:
                    position_ids = torch.cat((position_ids[:,:1], position_ids[:,17:]), dim=1)
                    
                    text_self_attention_masks = torch.cat((text_self_attention_masks[:,:1], text_self_attention_masks[:,17:]),dim=1)
                    
                    text_self_attention_masks = torch.cat((text_self_attention_masks[:,:,:1], text_self_attention_masks[:,:,17:]),dim=2)

                # text_token_mask: True for nomask, False for mask
                # text_self_attention_masks: True for nomask, False for mask

                if encoded_text.shape[1] > self.max_text_len:
                    encoded_text = encoded_text[:, : self.max_text_len, :]
                    text_token_mask = text_token_mask[:, : self.max_text_len]
                    position_ids = position_ids[:, : self.max_text_len]
                    text_self_attention_masks = text_self_attention_masks[
                        :, : self.max_text_len, : self.max_text_len
                    ]

                text_dict = {
                    "encoded_text": encoded_text,  # bs, 195, d_model
                    "text_token_mask": text_token_mask,  # bs, 195
                    "position_ids": position_ids,  # bs, 195
                    "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
                }


                # if isinstance(samples, (list, torch.Tensor)):
                #     samples = nested_tensor_from_tensor_list(samples)
                # features, poss = self.backbone(samples)
        
                # print(features.shape)
                # exit()
                if "extract_feat" in kw and kw["extract_feat"] == True:
                    mean_feature = torch.mean(features[-1].tensors, dim=(2,3)).cpu()

                    return mean_feature
                
                srcs = []
                masks = []
                # loss_conv_adapter = None
                loss_conv_adapter = None
                for l, feat in enumerate(features):
                    src, mask = feat.decompose()
                    if not self.use_zira:
                        srcs.append(self.input_proj[l](src))
                    else:
                        conv_adapter_output, zero_loss = self.input_proj_conv_adapter[l](src)
                        srcs.append(self.input_proj[l][1](self.input_proj[l][0](src) + conv_adapter_output))
                        if loss_conv_adapter is None:
                            loss_conv_adapter = zero_loss
                        else:
                            loss_conv_adapter = loss_conv_adapter + zero_loss
                    masks.append(mask)
                    assert mask is not None
                    
                    
                if self.num_feature_levels > len(srcs):
                    _len_srcs = len(srcs)
                    for l in range(_len_srcs, self.num_feature_levels):
                        if l == _len_srcs:
                            if not self.use_zira:
                                src = self.input_proj[l](features[-1].tensors)
                            else:
                                conv_adapter_output, zero_loss = self.input_proj_conv_adapter[l](features[-1].tensors)
                                src = self.input_proj[l][1](self.input_proj[l][0](features[-1].tensors) + conv_adapter_output)
                                if loss_conv_adapter is None:
                                    loss_conv_adapter = zero_loss
                                else:
                                    loss_conv_adapter = loss_conv_adapter + zero_loss
                        else:
                            if not self.use_zira:
                                src = self.input_proj[l](srcs[-1])
                            else:
                                conv_adapter_output, zero_loss = self.input_proj_conv_adapter[l](srcs[-1])
                                src = self.input_proj[l][1](self.input_proj[l][0](srcs[-1]) + conv_adapter_output)
                                if loss_conv_adapter is None:
                                    loss_conv_adapter = zero_loss
                                else:
                                    loss_conv_adapter = loss_conv_adapter + zero_loss
                                    
                                    
                        m = samples.mask
                        mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                        pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                        srcs.append(src)
                        masks.append(mask)
                        poss.append(pos_l)

                input_query_bbox = input_query_label = attn_mask = dn_meta = None
                hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
                    srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
                )

                
                # deformable-detr-like anchor update
                outputs_coord_list = []
                for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                    zip(reference[:-1], self.bbox_embed, hs)
                ):
                    layer_delta_unsig = layer_bbox_embed(layer_hs)
                    layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
                    layer_outputs_unsig = layer_outputs_unsig.sigmoid()
                    outputs_coord_list.append(layer_outputs_unsig)
                outputs_coord_list = torch.stack(outputs_coord_list)


                outputs_class = torch.stack(
                    [
                        layer_cls_embed(layer_hs, text_dict)
                        for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
                    ]
                )
                
                # print(outputs_class[-1])
                
                seg_outputs_class = outputs_class[-1][0].cpu()
                seg_outputs_coord_list = outputs_coord_list[-1][0].cpu()
                
                
                
                if seg_idx > 0:
                    index = find_inf_boundary(seg_outputs_class)
                    seg_outputs_class = shift_columns(seg_outputs_class, index, sum_pre-2*seg_idx)
                    
                    
                    
                final_logits.append(seg_outputs_class)
                final_boxes.append(seg_outputs_coord_list)
                    
                    
                sum_pre+=seg_token_num
                    
            
            final_logits = torch.cat(final_logits, dim=0).unsqueeze(0)
            final_boxes = torch.cat(final_boxes, dim=0).unsqueeze(0)
            
            # print(final_logits.shape)
            



            out = {"pred_logits": final_logits.cuda(), "pred_boxes": final_boxes.cuda()}
            
            if self.use_zira:
                
                return out, loss_linear_adapter, loss_conv_adapter
            
            return out
            
            # index = find_inf_boundary(out["pred_logits"][0])
            

            
        else:
            tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(
                samples.device
            )
            
                # print(tokenized["input_ids"].shape)
            # exit()


            one_hot_token = tokenized

            (
                text_self_attention_masks,
                position_ids,
                cate_to_token_mask_list,
            ) = generate_masks_with_special_tokens_and_transfer_map(
                tokenized, self.specical_tokens, self.tokenizer, self.coop
            )

            if text_self_attention_masks.shape[1] > self.max_text_len:
                text_self_attention_masks = text_self_attention_masks[
                    :, : self.max_text_len, : self.max_text_len
                ]
                position_ids = position_ids[:, : self.max_text_len]
                tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
                tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
                tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

            # extract text embeddings
            if self.sub_sentence_present:
                tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
                tokenized_for_encoder["attention_mask"] = text_self_attention_masks
                tokenized_for_encoder["position_ids"] = position_ids
            else:
                tokenized_for_encoder = tokenized
                
            tokenized_for_encoder["coop"] = self.coop
            
            

            
        

            bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

            encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
            
            if self.use_zira:
                rep_linear_out, loss_linear_adapter = self.rep_linear_adapter(bert_output["last_hidden_state"])
                encoded_text = rep_linear_out + encoded_text
            
        
            if self.coop:
                encoded_text = torch.cat((encoded_text[:,:1], encoded_text[:,17:]), dim=1)

            text_token_mask = tokenized.attention_mask.bool()  # bs, 195
            # print(encoded_text.shape)
            # print(text_token_mask.shape)
            if self.coop:
                position_ids = torch.cat((position_ids[:,:1], position_ids[:,17:]), dim=1)
                
                text_self_attention_masks = torch.cat((text_self_attention_masks[:,:1], text_self_attention_masks[:,17:]),dim=1)
                
                text_self_attention_masks = torch.cat((text_self_attention_masks[:,:,:1], text_self_attention_masks[:,:,17:]),dim=2)

            # text_token_mask: True for nomask, False for mask
            # text_self_attention_masks: True for nomask, False for mask

            if encoded_text.shape[1] > self.max_text_len:
                encoded_text = encoded_text[:, : self.max_text_len, :]
                text_token_mask = text_token_mask[:, : self.max_text_len]
                position_ids = position_ids[:, : self.max_text_len]
                text_self_attention_masks = text_self_attention_masks[
                    :, : self.max_text_len, : self.max_text_len
                ]

            text_dict = {
                "encoded_text": encoded_text,  # bs, 195, d_model
                "text_token_mask": text_token_mask,  # bs, 195
                "position_ids": position_ids,  # bs, 195
                "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
            }


            # if isinstance(samples, (list, torch.Tensor)):
            #     samples = nested_tensor_from_tensor_list(samples)
            # features, poss = self.backbone(samples)
    
            # print(features.shape)
            # exit()
            if "extract_feat" in kw and kw["extract_feat"] == True:
                mean_feature = torch.mean(features[-1].tensors, dim=(2,3)).cpu()

                return mean_feature
            srcs = []
            masks = []
            loss_conv_adapter = None
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                if not self.use_zira:
                    srcs.append(self.input_proj[l](src))
                else:
                    conv_adapter_output, zero_loss = self.input_proj_conv_adapter[l](src)
                    srcs.append(self.input_proj[l][1](self.input_proj[l][0](src) + conv_adapter_output))
                    if loss_conv_adapter is None:
                        loss_conv_adapter = zero_loss
                    else:
                        loss_conv_adapter = loss_conv_adapter + zero_loss
                masks.append(mask)
                assert mask is not None
                
                
            if self.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        if not self.use_zira:
                            src = self.input_proj[l](features[-1].tensors)
                        else:
                            conv_adapter_output, zero_loss = self.input_proj_conv_adapter[l](features[-1].tensors)
                            src = self.input_proj[l][1](self.input_proj[l][0](features[-1].tensors) + conv_adapter_output)
                            if loss_conv_adapter is None:
                                loss_conv_adapter = zero_loss
                            else:
                                loss_conv_adapter = loss_conv_adapter + zero_loss
                    else:
                        if not self.use_zira:
                            src = self.input_proj[l](srcs[-1])
                        else:
                            conv_adapter_output, zero_loss = self.input_proj_conv_adapter[l](srcs[-1])
                            src = self.input_proj[l][1](self.input_proj[l][0](srcs[-1]) + conv_adapter_output)
                            if loss_conv_adapter is None:
                                loss_conv_adapter = zero_loss
                            else:
                                loss_conv_adapter = loss_conv_adapter + zero_loss
                                
                                
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    poss.append(pos_l)

            input_query_bbox = input_query_label = attn_mask = dn_meta = None
            hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
                srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
            )

            
            # deformable-detr-like anchor update
            outputs_coord_list = []
            for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, hs)
            ):
                layer_delta_unsig = layer_bbox_embed(layer_hs)
                layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
                layer_outputs_unsig = layer_outputs_unsig.sigmoid()
                outputs_coord_list.append(layer_outputs_unsig)
            outputs_coord_list = torch.stack(outputs_coord_list)


            outputs_class = torch.stack(
                [
                    layer_cls_embed(layer_hs, text_dict)
                    for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
                ]
            )
            


            out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}
            

            # Used to calculate losses
            bs, len_td = text_dict['text_token_mask'].shape
            out['text_mask']=torch.zeros(bs, self.max_text_len, dtype=torch.bool).to(
                samples.device
            )
            for b in range(bs):
                for j in range(len_td):
                    if text_dict['text_token_mask'][b][j] == True:
                        out['text_mask'][b][j] = True

            # for intermediate outputs
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)
            out['token']=one_hot_token
            # # for encoder output
            if hs_enc is not None:
                # prepare intermediate outputs
                interm_coord = ref_enc[-1]
                interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
                out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
                out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}
   
            if self.use_zira:
                
                return out, loss_linear_adapter, loss_conv_adapter

            return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]




class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, focal_alpha,focal_gamma, losses):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma= focal_gamma

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """

        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes


        return losses


    def token_sigmoid_binary_focal_loss(self, outputs, targets, indices, num_boxes):
        pred_logits=outputs['pred_logits']
        new_targets=outputs['one_hot'].to(pred_logits.device)
        text_mask=outputs['text_mask']

        assert (new_targets.dim() == 3)
        assert (pred_logits.dim() == 3)  # batch x from x to
        
        bs, n, _ = pred_logits.shape
        alpha=self.focal_alpha
        gamma=self.focal_gamma
        if text_mask is not None:
            # ODVG: each sample has different mask 
            text_mask = text_mask.repeat(1, pred_logits.size(1)).view(outputs['text_mask'].shape[0],-1,outputs['text_mask'].shape[1])
            pred_logits = torch.masked_select(pred_logits, text_mask)
            new_targets = torch.masked_select(new_targets, text_mask)

        new_targets=new_targets.float()
        p = torch.sigmoid(pred_logits)
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, new_targets, reduction="none")
        p_t = p * new_targets + (1 - p) * (1 - new_targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * new_targets + (1 - alpha) * (1 - new_targets)
            loss = alpha_t * loss

        total_num_pos=0
        for batch_indices in indices:
            total_num_pos += len(batch_indices[0])
        num_pos_avg_per_gpu = max(total_num_pos , 1.0)
        loss=loss.sum()/num_pos_avg_per_gpu
        
        losses = {'loss_ce': loss}
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.token_sigmoid_binary_focal_loss,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, cat_list, caption, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
        """
        device=next(iter(outputs.values())).device
        one_hot = torch.zeros(outputs['pred_logits'].size(),dtype=torch.int64) # torch.Size([bs, 900, 256])
        token = outputs['token'] 
        
        label_map_list = []
        indices = []
        for j in range(len(cat_list)): # bs
            label_map=[]
            for i in range(len(cat_list[j])):
                label_id=torch.tensor([i])
                per_label=create_positive_map(token[j], label_id, cat_list[j], caption[j])
                label_map.append(per_label)
            label_map=torch.stack(label_map,dim=0).squeeze(1)
            label_map_list.append(label_map)
        for j in range(len(cat_list)): # bs
            for_match = {
                "pred_logits" : outputs['pred_logits'][j].unsqueeze(0),
                "pred_boxes" : outputs['pred_boxes'][j].unsqueeze(0)
            }
            inds = self.matcher(for_match, [targets[j]], label_map_list[j])
            indices.extend(inds)
        # indices : A list of size batch_size, containing tuples of (index_i, index_j) where:
        # - index_i is the indices of the selected predictions (in order)
        # - index_j is the indices of the corresponding selected targets (in order)

        # import pdb; pdb.set_trace()
        tgt_ids = [v["labels"].cpu() for v in targets]
        # len(tgt_ids) == bs
        for i in range(len(indices)):
            tgt_ids[i]=tgt_ids[i][indices[i][1]]
            one_hot[i,indices[i][0]] = label_map_list[i][tgt_ids[i]].to(torch.long)
        outputs['one_hot'] = one_hot
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_list = [len(t["labels"]) for t in targets]
        num_boxes = sum(num_boxes_list)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = []
                for j in range(len(cat_list)): # bs
                    aux_output_single = {
                        'pred_logits' : aux_outputs['pred_logits'][j].unsqueeze(0),
                        'pred_boxes': aux_outputs['pred_boxes'][j].unsqueeze(0)
                    }
                    inds = self.matcher(aux_output_single, [targets[j]], label_map_list[j])
                    indices.extend(inds)
                one_hot_aux = torch.zeros(outputs['pred_logits'].size(),dtype=torch.int64)
                tgt_ids = [v["labels"].cpu() for v in targets]
                for i in range(len(indices)):
                    tgt_ids[i]=tgt_ids[i][indices[i][1]]
                    one_hot_aux[i,indices[i][0]] = label_map_list[i][tgt_ids[i]].to(torch.long)
                aux_outputs['one_hot'] = one_hot_aux
                aux_outputs['text_mask'] = outputs['text_mask']
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)                
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = []
            for j in range(len(cat_list)): # bs
                interm_output_single = {
                    'pred_logits' : interm_outputs['pred_logits'][j].unsqueeze(0),
                    'pred_boxes': interm_outputs['pred_boxes'][j].unsqueeze(0)
                }
                inds = self.matcher(interm_output_single, [targets[j]], label_map_list[j])
                indices.extend(inds)
            one_hot_aux = torch.zeros(outputs['pred_logits'].size(),dtype=torch.int64)
            tgt_ids = [v["labels"].cpu() for v in targets]
            for i in range(len(indices)):
                tgt_ids[i]=tgt_ids[i][indices[i][1]]
                one_hot_aux[i,indices[i][0]] = label_map_list[i][tgt_ids[i]].to(torch.long)
            interm_outputs['one_hot'] = one_hot_aux
            interm_outputs['text_mask'] = outputs['text_mask']
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                kwargs = {}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100,text_encoder_type='text_encoder_type', nms_iou_threshold=-1,use_coco_eval=False,args=None) -> None:
        super().__init__()
        self.num_select = num_select
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        if args.use_coco_eval:
            from pycocotools.coco import COCO
            coco = COCO(args.coco_val_path)
            category_dict = coco.loadCats(coco.getCatIds())
            cat_list = [item['name'] for item in category_dict]
        else:
            cat_list=args.label_list
        caption = " . ".join(cat_list) + ' .'

        tokenized = self.tokenizer(caption, padding="longest", return_tensors="pt")
        label_list = torch.arange(len(cat_list))
        
    
        # exit()
        pos_map=create_positive_map(tokenized,label_list,cat_list,caption)
 
        # build a mapping from label_id to pos_map
        if args.use_coco_eval and 'coco' in args.datasets:
            id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                    41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
            new_pos_map = torch.zeros((91, 4500))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map=new_pos_map
        elif args.use_coco_eval and 'odinw_aerial' in args.datasets:
            id_map = {0:1, 1:2, 2:3, 3:4, 4:5}
            # new_pos_map = torch.zeros((91, 256))
            max_real_id = 0
            for key in id_map.keys():
                max_real_id = max(max_real_id, id_map[key])
            new_pos_map = torch.zeros((max_real_id+1, 4500))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map=new_pos_map
        elif args.use_coco_eval and 'odinw_aqua' in args.datasets:
            id_map = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7}
            # new_pos_map = torch.zeros((91, 256))
            max_real_id = 0
            for key in id_map.keys():
                max_real_id = max(max_real_id, id_map[key])
            new_pos_map = torch.zeros((max_real_id+1, 4500))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map=new_pos_map
        elif args.use_coco_eval and 'odinw_cotton' in args.datasets:
            id_map = {0:1}
            # new_pos_map = torch.zeros((91, 256))
            max_real_id = 0
            for key in id_map.keys():
                max_real_id = max(max_real_id, id_map[key])
            new_pos_map = torch.zeros((max_real_id+1, 4500))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map=new_pos_map
        elif args.use_coco_eval and 'odinw_egohand' in args.datasets:
            id_map = {0:1}
            # new_pos_map = torch.zeros((91, 256))
            max_real_id = 0
            for key in id_map.keys():
                max_real_id = max(max_real_id, id_map[key])
            new_pos_map = torch.zeros((max_real_id+1, 4500))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map=new_pos_map
        elif args.use_coco_eval and 'odinw_mushroom' in args.datasets:
            id_map = {0:1, 1:2}
            # new_pos_map = torch.zeros((91, 256))
            max_real_id = 0
            for key in id_map.keys():
                max_real_id = max(max_real_id, id_map[key])
            new_pos_map = torch.zeros((max_real_id+1, 4500))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map=new_pos_map
        elif args.use_coco_eval and 'odinw_package' in args.datasets:
            id_map = {0:1}
            # new_pos_map = torch.zeros((91, 256))
            max_real_id = 0
            for key in id_map.keys():
                max_real_id = max(max_real_id, id_map[key])
            new_pos_map = torch.zeros((max_real_id+1, 4500))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map=new_pos_map
        elif args.use_coco_eval and 'odinw_pascalvoc' in args.datasets:
            id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20}
            # new_pos_map = torch.zeros((91, 256))
            max_real_id = 0
            for key in id_map.keys():
                max_real_id = max(max_real_id, id_map[key])
            new_pos_map = torch.zeros((max_real_id+1, 4500))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map=new_pos_map
        elif args.use_coco_eval and 'odinw_pistol' in args.datasets:
            id_map = {0:1}
            # new_pos_map = torch.zeros((91, 256))
            max_real_id = 0
            for key in id_map.keys():
                max_real_id = max(max_real_id, id_map[key])
            new_pos_map = torch.zeros((max_real_id+1, 4500))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map=new_pos_map
        elif args.use_coco_eval and 'odinw_pothole' in args.datasets:
            id_map = {0:1}
            # new_pos_map = torch.zeros((91, 256))
            max_real_id = 0
            for key in id_map.keys():
                max_real_id = max(max_real_id, id_map[key])
            new_pos_map = torch.zeros((max_real_id+1, 4500))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map=new_pos_map
        elif args.use_coco_eval and 'odinw_raccoon' in args.datasets:
            id_map = {0:1}
            # new_pos_map = torch.zeros((91, 256))
            max_real_id = 0
            for key in id_map.keys():
                max_real_id = max(max_real_id, id_map[key])
            new_pos_map = torch.zeros((max_real_id+1, 4500))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map=new_pos_map
        elif args.use_coco_eval and 'odinw_shellfish' in args.datasets:
            id_map = {0:1, 1:2, 2:3}
            # new_pos_map = torch.zeros((91, 256))
            max_real_id = 0
            for key in id_map.keys():
                max_real_id = max(max_real_id, id_map[key])
            new_pos_map = torch.zeros((max_real_id+1, 4500))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map=new_pos_map
        elif args.use_coco_eval and 'odinw_thermal' in args.datasets:
            id_map = {0:1, 1:2}
            # new_pos_map = torch.zeros((91, 256))
            max_real_id = 0
            for key in id_map.keys():
                max_real_id = max(max_real_id, id_map[key])
            new_pos_map = torch.zeros((max_real_id+1, 4500))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map=new_pos_map
        elif args.use_coco_eval and 'odinw_vehicle' in args.datasets:
            id_map = {0:1, 1:2, 2:3, 3:4, 4:5}
            # new_pos_map = torch.zeros((91, 256))
            max_real_id = 0
            for key in id_map.keys():
                max_real_id = max(max_real_id, id_map[key])
            new_pos_map = torch.zeros((max_real_id+1, 4500))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map=new_pos_map

        self.nms_iou_threshold=nms_iou_threshold
        self.positive_map = pos_map

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']


        prob_to_token = out_logits.sigmoid()
        pos_maps = self.positive_map.to(prob_to_token.device)
        for label_ind in range(len(pos_maps)):
            if pos_maps[label_ind].sum() != 0:
                pos_maps[label_ind]=pos_maps[label_ind]/pos_maps[label_ind].sum()

        prob_to_label = prob_to_token @ pos_maps.T

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(prob.view(prob.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, prob.shape[2], rounding_mode='trunc')
        labels = topk_indexes % prob.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # if test:
        #     assert not not_to_xyxy
        #     boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]

            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


@MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingdino")
def build_groundingdino(args):
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present
    


    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
        use_coop=args.use_coop,
        use_prompt = args.use_prompt,
        use_zira = args.use_zira
    )



    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    

    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    # losses = ['labels', 'boxes', 'cardinality']
    losses = ['labels', 'boxes']

    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,losses=losses
                             )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select  , text_encoder_type=args.text_encoder_type,nms_iou_threshold=args.nms_iou_threshold,args=args)}

    return model, criterion, postprocessors

def create_positive_map(tokenized, tokens_positive,cat_list,caption):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 4500), dtype=torch.float)

    for j,label in enumerate(tokens_positive):

        start_ind = caption.find(cat_list[label])
        end_ind = start_ind + len(cat_list[label]) - 1
        beg_pos = tokenized.char_to_token(start_ind)
        try:
            end_pos = tokenized.char_to_token(end_ind)
        except:
            end_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end_ind - 1)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end_ind - 2)
            except:
                end_pos = None
        # except Exception as e:
        #     print("beg:", beg, "end:", end)
        #     print("token_positive:", tokens_positive)
        #     # print("beg_pos:", beg_pos, "end_pos:", end_pos)
        #     raise e
        # if beg_pos is None:
        #     try:
        #         beg_pos = tokenized.char_to_token(beg + 1)
        #         if beg_pos is None:
        #             beg_pos = tokenized.char_to_token(beg + 2)
        #     except:
        #         beg_pos = None
        # if end_pos is None:
        #     try:
        #         end_pos = tokenized.char_to_token(end - 2)
        #         if end_pos is None:
        #             end_pos = tokenized.char_to_token(end - 3)
        #     except:
        #         end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        if beg_pos < 0 or end_pos < 0:
            continue
        if beg_pos > end_pos:
            continue
        # assert beg_pos is not None and end_pos is not None
        positive_map[j,beg_pos: end_pos + 1].fill_(1)
    return positive_map 


