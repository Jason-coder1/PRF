from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.transformer_blocks import Block
from .layers.time_decoder import TimeDecoder
from .layers.retrodictive_decoder import RetrodictiveDecoder
from .layers.residual_distillation import REDistill
from .layers.mamba.vim_mamba import init_weights, create_block
from functools import partial
from timm.models.layers import DropPath, to_2tuple
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
torch.cuda.empty_cache()
import os
import numpy as np

# only 'DeMo'
class ModelForecast(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps: int = 60,
    ) -> None:
        super().__init__()

        assert future_steps == 60, "Future_steps should be 30 for Argoverse1 dataset!"

        self.hist_embed_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

        # Agent Encoding Mamba
        self.hist_embed_mamba = nn.ModuleList(  
            [
                create_block(  
                    d_model=embed_dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=False,  
                    rms_norm=True,  
                )
                for i in range(4)
            ]
        )

        self.norm_f10 = RMSNorm(embed_dim, eps=1e-5)
        self.norm_f20 = RMSNorm(embed_dim, eps=1e-5)
        self.norm_f30 = RMSNorm(embed_dim, eps=1e-5)
        self.norm_f40 = RMSNorm(embed_dim, eps=1e-5)
        self.norm_f50 = RMSNorm(embed_dim, eps=1e-5)

        self.drop_path = DropPath(drop_path)

        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Scene Context Transformer
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=0.2,
            )
            for i in range(5)
        )
        
        self.norm_l10 = nn.LayerNorm(embed_dim)
        self.norm_l20 = nn.LayerNorm(embed_dim)
        self.norm_l30 = nn.LayerNorm(embed_dim)
        self.norm_l40 = nn.LayerNorm(embed_dim)
        self.norm_l50 = nn.LayerNorm(embed_dim)

        self.norm10 = nn.LayerNorm(embed_dim)
        self.norm20 = nn.LayerNorm(embed_dim)
        self.norm30 = nn.LayerNorm(embed_dim)
        self.norm40 = nn.LayerNorm(embed_dim)
        self.norm50 = nn.LayerNorm(embed_dim)

        self.res_dist10 = REDistill(embed_dim)
        self.res_dist20 = REDistill(embed_dim)
        self.res_dist30 = REDistill(embed_dim)
        self.res_dist40 = REDistill(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(3, embed_dim))

        self.dense_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, future_steps * 2)
        )

        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.GELU(), nn.Linear(64, embed_dim)
        )

        self.time_decoder = TimeDecoder()

        self.retrodictive_decoder = RetrodictiveDecoder(retrodictive_len=10)

        self.teacher_feats = {}
        self.student_feats = {}

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.") :]: v for k, v in ckpt.items() if k.startswith("net.")
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, data_list):

        assert isinstance(data_list, list), "Input data should be a list of dictionaries."

        # collecting hist_len, hist_start, hist_end
        hist_lens = []
        hist_starts = []
        hist_ends = []
        for data in data_list:

            hist_len = data.get('hist_len')[0].item()
            hist_start = data.get('hist_start')[0].item()
            hist_end = data.get('hist_end')[0].item()
            hist_lens.append(hist_len)
            hist_starts.append(hist_start)
            hist_ends.append(hist_end)

        if len(set(hist_ends)) != 1:   # 如果去重后hist_end不止一个
            raise ValueError("hist_end不止一个!")
        
        ###### Scene context encoding ###### 
        # agent encoding
        hist_feat_all = []
        hist_key_valid_mask_all = []
        for data in data_list:
            hist_valid_mask = data["x_valid_mask"]
            hist_key_valid_mask = hist_valid_mask.any(-1)
            hist_feat = torch.cat(
                [
                    data["x_positions_diff"],
                    data["x_velocity_diff"][..., None],
                    hist_valid_mask[..., None],
                ],
                dim=-1,
            )
            hist_feat_all.append(hist_feat)
            hist_key_valid_mask_all.append(hist_key_valid_mask)

        # 拼接所有 data 成一个大 batch
        hist_feat_all = torch.cat(hist_feat_all, dim=0)
        hist_key_valid_mask_all = torch.cat(hist_key_valid_mask_all)

        B, N, L, D = hist_feat_all.shape

        # reshape to (B * N, L, D)
        hist_feat_all = hist_feat_all.view(B * N, L, D)
        hist_feat_key_valid_all = hist_key_valid_mask_all.view(B * N)

        # unidirectional mamba
        actor_feat = self.hist_embed_mlp(hist_feat_all[hist_feat_key_valid_all].contiguous())
        residual = None
        for blk_mamba in self.hist_embed_mamba:
            actor_feat, residual = blk_mamba(actor_feat, residual)

        in_batch = len(data_list)
        bs = B // in_batch
        batch_idx = [0]
        for i in range(in_batch):
            idx = torch.sum(hist_feat_key_valid_all[:(i+1)*bs*N])
            batch_idx.append(idx.item())

        actor_feat_all = []
        for i, hist_len in enumerate(hist_lens):
            norm_f = getattr(self, f'norm_f{hist_len}')

            fused_add_norm_fn = rms_norm_fn if isinstance(norm_f, RMSNorm) else layer_norm_fn
            start, end = batch_idx[i], batch_idx[i + 1]

            actor_feat_mini = fused_add_norm_fn(
                self.drop_path(actor_feat[start: end]),
                norm_f.weight,
                norm_f.bias,
                eps=norm_f.eps,
                residual=residual[start: end],
                prenorm=False,
                residual_in_fp32=True
            )

            actor_feat_mini = actor_feat_mini[:, -1]
            actor_feat_tmp = torch.zeros(
                bs * N, actor_feat_mini.shape[-1], device=actor_feat_mini.device, dtype=actor_feat_mini.dtype
            )
            actor_feat_tmp[hist_feat_key_valid_all[i*bs*N: (i+1)*bs*N]] = actor_feat_mini
            actor_feat_mini = actor_feat_tmp.view(bs, N, actor_feat_mini.shape[-1])
            actor_feat_all.append(actor_feat_mini)
        actor_feat_all = torch.cat(actor_feat_all, dim=0)

        # map encoding, do not need to be processed by multiple times as lane information are the same for different batch
        data_tmp = data_list[0]
        lane_valid_mask = data_tmp["lane_valid_mask"]
        lane_normalized = data_tmp["lane_positions"] - data_tmp["lane_centers"].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, lane_valid_mask[..., None]], dim=-1
        )
        sB, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(sB, M, -1)
        lane_feat_all = lane_feat.repeat(in_batch, 1, 1)

        # position embedding
        pos_feat_all = []
        for data in data_list:
            x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
            angles = torch.cat([data["x_angles"][:, :, -1], data["lane_angles"]], dim=1)
            x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
            pos_feat = torch.cat([x_centers, x_angles], dim=-1)
            pos_feat_all.append(pos_feat)
        pos_feat_all = torch.cat(pos_feat_all, dim=0)  
        pos_embed_all = self.pos_embed(pos_feat_all)

        # type embedding, do not need to be processed by multiple times for the same reason presented below
        actor_type_embed = self.actor_type_embed[data_tmp["x_attr"][..., 2].long()]
        lane_type_embed = self.lane_type_embed[data_tmp["lane_attr"][..., 0].long()]
        actor_type_embed_all = actor_type_embed.repeat(in_batch, 1, 1)
        lane_type_embed_all = lane_type_embed.repeat(in_batch, 1, 1)
        actor_feat_all += actor_type_embed_all
        lane_feat_all += lane_type_embed_all

        # scene context features, do not need to be processed by multiple times for the same reason presented below
        x_encoder = torch.cat([actor_feat_all, lane_feat_all], dim=1)
        len_key_valid_mask = data_tmp["lane_key_valid_mask"]
        lane_key_valid_mask_all = len_key_valid_mask.repeat(in_batch, 1)
        key_valid_mask = torch.cat(
            [hist_key_valid_mask_all, lane_key_valid_mask_all], dim=1
        )
        x_encoder = x_encoder + pos_embed_all

        #  intra-interaction learning for scene context features
        x_encoder_list = []
        for i, hist_len in enumerate(hist_lens):
            norm = getattr(self, f'norm_l{hist_len}')
            x_encoder_mini = x_encoder[i*bs: (i+1)*bs]
            for blk in self.blocks:
                x_encoder_mini = blk(x_encoder_mini, key_padding_mask=~key_valid_mask[i*bs: (i+1)*bs], norm_layer=norm)
            x_encoder_list.append(x_encoder_mini)
        x_encoder = torch.cat(x_encoder_list, dim=0)

        # layer norm for different history features
        x_encoder_list = []
        ret_feat_list = []
        full_hist_lens = list(range(10, hist_end, 10)) #full_hist_lens = [10, 20, 30, 40] #, 50]
        for i, (idx, hist_len) in enumerate(sorted(enumerate(hist_lens), key=lambda x: x[1])):
            norm = getattr(self, f'norm{hist_len}') # from short to long: 10, 20, 30, 40, 50
            if not hist_len == hist_end: # 50->hist_end
                res_dist = getattr(self, f'res_dist{hist_len}')
            hist_start, hist_end = hist_starts[idx], hist_ends[idx]
            x_encoder_mini = x_encoder[idx*bs: (idx+1)*bs]
            x_encoder_mini = norm(x_encoder_mini)

            if not hist_len == 10:
                self.teacher_feats[(hist_start, hist_end)] = x_encoder_mini[:, :N]
            if not hist_len == hist_end: # 50 -> hist_end
                x_encoder_mini = res_dist(x_encoder_mini, N)
                self.student_feats[(hist_start, hist_end)] = x_encoder_mini[:, :N]

            # the code below collecting features for retrodictive prediction
            if self.training and hist_len < hist_end: #50->hist_end # delete self.training if validating bi-query interaction
                ret_feat_list.append(x_encoder_mini)

            # The loop below will not be executed during evaluation as hist_lens contains only one element
            for j in range(i + 1, len(hist_lens)-1):
                next_idx, next_len = sorted(enumerate(hist_lens), key=lambda x: x[1])[j]
                resdist_next = getattr(self, f'res_dist{next_len}')
                x_encoder_mini = resdist_next(x_encoder_mini, N)
            
            # For evaluation, we use the remaining norm layer further norm the features
            if not self.training:
                fur_norm_lens = [l for l in full_hist_lens if l > hist_len]
                for fur_len in fur_norm_lens:
                    fur_resdist = getattr(self, f'res_dist{fur_len}')
                    x_encoder_mini = fur_resdist(x_encoder_mini, N)
                    #ret_feat_list.append(x_encoder_mini) # for validation of bi-query interaciton (intermediate)

            x_encoder_list.append(x_encoder_mini)
        x_encoder = torch.cat(x_encoder_list, dim=0) # shape: (5B, N, D) 其中N表示行人和车道线数量和

        ###### Retrodictive decoding with decoupled queries ######   # restrodictive prediction module
        #mode_hist = hist_others = hist_predict = goal_predict = hist_x_hat = hist_pi = hist_scal = None
        
        if self.training and len(ret_feat_list) != 0:
            ret_feat = torch.cat(ret_feat_list[::-1], dim=0) # reverse: 0-10, 10-20, 20-30, 30-40
            x_mask = key_valid_mask[bs:, :]

            mode_hist, hist_others, hist_predict, goal_predict, hist_x_hat , hist_pi , hist_scal = \
                self.retrodictive_decoder(ret_feat, N, mask=~x_mask)
        #elif hist_len < hist_end: # TODO: the third step, interaction between ret query and encoding feature
        #    ret_feat = torch.cat(ret_feat_list[::-1], dim=0)
        #    mB = ret_feat.shape[0] // key_valid_mask.shape[0]
        #    x_mask = torch.cat([key_valid_mask]*mB, dim=0)

        #    mode_hist, hist_others, hist_predict, goal_predict, hist_x_hat , hist_pi , hist_scal = \
        #        self.retrodictive_decoder(ret_feat, N, mask=~x_mask)
        else:
            mode_hist = hist_others = hist_predict = goal_predict = hist_x_hat = hist_pi = hist_scal = None
            
        ###### Trajectory decoding with decoupled queries ######
        new_y_hat = None
        new_pi = None
        dense_predict = None
        mode = None

        # outputs of other agents
        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(B, x_others.size(1), -1, 2)

        # state query initialization
        time = torch.arange(60).long().to(x_encoder.device)
        time = time * 0.1 + 0.1
        time = time.unsqueeze(-1)
        mode = self.time_embedding_mlp(time)
        mode = mode.repeat(x_encoder.size(0), 1, 1) # shape: (B, T, dim)

        # decoder module with decoupled queries
        dense_predict, y_hat, pi, x_mode, new_y_hat, new_pi, mode_dense, scal, scal_new, \
        y_hat_f, pi_f, scal_f = self.time_decoder(mode, x_encoder, mode_hist=mode_hist, mask=~key_valid_mask) # input:mode_dense, outputs: pi_f, scal_f

        ret_dict = {
            "y_hat": y_hat,  # trajectory output from mode query
            "pi": pi,  # probability output from mode query
            "scal": scal,  # output for Laplace loss from mode query

            "dense_predict": dense_predict,  # trajectory output from state query

            "y_hat_others": y_hat_others,  # trajectory of other agents

            "new_y_hat": new_y_hat,  # final trajectory output
            "new_pi": new_pi,  # final probability output     
            "scal_new": scal_new,  # final output for Laplace loss

            "hist_others": hist_others,  # retrodictive decoding output for other agents
            "hist_predict": hist_predict,  # retrodictive decoding output
            "goal_predict": goal_predict,  # retrodictive decoding output for goal prediction
            
            "hist_x_hat": hist_x_hat, # hist trajectory output
            "hist_pi": hist_pi,  # hist probability output 
            "hist_scal": hist_scal,  # hist output for Laplace loss
            
            "final_y_hat": y_hat_f, # final trajectory output
            "final_pi": pi_f, # final probability output
            "final_scal": scal_f, # final output for Laplace loss
        }

        return ret_dict