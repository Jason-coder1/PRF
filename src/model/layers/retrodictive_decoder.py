import torch
import torch.nn as nn
from .transformer_blocks import Cross_Block, Block
import torch.nn.functional as F
from .mamba.vim_mamba import init_weights, create_block
from functools import partial
from timm.models.layers import DropPath, to_2tuple
from .time_decoder import GMMPredictor
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class GMMPredictor_dense(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super(GMMPredictor_dense, self).__init__()
        self._future_len = future_len
        self.gaussian = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 2)
        )
        self.score = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )
        self.scale = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 2)
        )
    
    def forward(self, input):
        res = self.gaussian(input)
        scal = F.elu_(self.scale(input), alpha=1.0) + 1.0 + 0.0001
        input = input.max(dim=2)[0]
        score = self.score(input).squeeze(-1)

        return res, score, scal


class RetrodictiveDecoder(nn.Module):
    def __init__(self, retrodictive_len=10, dim=128):
        super(RetrodictiveDecoder, self).__init__()
        ##### hist other predictor #####
        self.hist_predictor = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, retrodictive_len * 2)
        )
        self.layerNorm = nn.LayerNorm(dim)


        ###### State Consistency Module ######
        # state cross attention
        self.cross_block_time = nn.ModuleList(
            Cross_Block()
            for i in range(3) # 2
        )

        # state bidirectional mamba
        self.timequery_embed_mamba = nn.ModuleList(
            [
                create_block(  
                    d_model=dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(3) # 2
            ]
        )
        self.timequery_norm_f = RMSNorm(dim, eps=1e-5)
        self.timequery_drop_path = DropPath(0.2)

        # state query initialization
        self.register_buffer('state', torch.arange(retrodictive_len).long())
        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, dim),
        )

        # MLP for state query
        self.dense_predict = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, dim),
            nn.GELU(),
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        ###### Mode Localization Module ######
        # mode cross attention
        self.cross_block_mode = nn.ModuleList(
            Cross_Block()
            for i in range(3)
        )

        # state bidirectional mamba
        self.self_block_mode = nn.ModuleList(
            Block()
            for i in range(3)
        )

        # mode query initialization
        self.multi_modal_query_embedding = nn.Embedding(6, dim)
        self.register_buffer('modal', torch.arange(6).long())

        self.mode_predictor = GMMPredictor(retrodictive_len)


        ###### Hybrid Coupling Module ######

        self.mode_embedding_mlp = nn.Sequential(
            nn.Linear(2*retrodictive_len, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, dim),
        )

        self.cross_block_branch = nn.ModuleList(
            Cross_Block()
            for i in range(3)
        )

        self.mixquery_embed_mamba = nn.ModuleList(  
            [
                create_block(  
                    d_model=dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(3)
            ]
        )
        self.mixquery_norm_f = RMSNorm(dim, eps=1e-5)
        self.mixquery_drop_path = DropPath(0.2)

        # MLP for final output -> from single modality to multi-modality
        self.predictor_dense = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, retrodictive_len*2)
        )


    def forward(self, encoding, agent_n, mask=None):
        # output for other agents
        x_others = encoding[:, 1:agent_n]  # shape: (B, T, dim)
        y_hat_others = self.hist_predictor(x_others).view(x_others.size(0), x_others.size(1), -1, 2) # shape: (B, T, 2)

        # state query initialization
        mode = self.state*0.1 + 0.1 
        mode = mode.unsqueeze(-1)  # shape: (1, T)
        mode = self.time_embedding_mlp(mode) # shape: (1, T, dim)
        mode = mode.repeat(encoding.size(0), 1, 1)  # shape: (B, T, dim)

        # Dynamic state consistency
        for blk in self.cross_block_time:
            mode = blk(mode, encoding, key_padding_mask=mask, norm_layer=self.layerNorm)

        residual = None
        for blk_mamba in self.timequery_embed_mamba:
            mode, residual = blk_mamba(mode, residual)
        fused_add_norm_fn = rms_norm_fn if isinstance(self.timequery_norm_f, RMSNorm) else layer_norm_fn
        mode = fused_add_norm_fn(
            self.timequery_drop_path(mode),
            self.timequery_norm_f.weight,
            self.timequery_norm_f.bias,
            eps=self.timequery_norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True
        )
        
        dense_pred = self.dense_predict(mode)

        mode_tmp = mode

        # Directional intention localization
        multi_modal_query = self.multi_modal_query_embedding(self.modal) # M x dim
        mode_query = encoding[:, 0] # shape: (B, dim)

        mode = mode_query[:, None] + multi_modal_query # shape: (B, M, dim)

        for blk in self.cross_block_mode:
            mode = blk(mode, encoding, key_padding_mask=mask, norm_layer=self.layerNorm)

        for blk in self.self_block_mode:
            mode = blk(mode, norm_layer=self.layerNorm) # shape: (B, M, dim)

        x_hat_hist, pi_hist, scal_hist = self.mode_predictor(mode)

        B, M, T, _ = x_hat_hist.shape
        mode_embedding = self.mode_embedding_mlp(x_hat_hist.reshape(B, M, T*2))
        mode_dense = mode_tmp
        for blk in self.cross_block_branch:
            mode_dense = blk(mode_dense, mode_embedding, norm_layer=self.layerNorm)

        residual = None
        for blk_mamba in self.mixquery_embed_mamba:
            mode_dense, residual = blk_mamba(mode_dense, residual)
        fused_add_norm_fn = rms_norm_fn if isinstance(self.mixquery_norm_f, RMSNorm) else layer_norm_fn
        mode_dense = fused_add_norm_fn(
            self.mixquery_drop_path(mode_dense),
            self.mixquery_norm_f.weight,
            self.mixquery_norm_f.bias,
            eps=self.mixquery_norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True
        )

        mode_pred = self.predictor_dense(mode_dense)

        return mode_dense, y_hat_others, dense_pred, mode_pred, x_hat_hist, pi_hist, scal_hist


class BANStyleCrossAttention(nn.Module):
    def __init__(self, dim, proj_dim=None, use_layernorm=True, activation='gelu'):
        super().__init__()
        self.dim = dim
        self.proj_dim = proj_dim or dim // 2
        self.use_layernorm = use_layernorm

        # Linear projections for bilinear fusion
        self.x1_proj = nn.Linear(dim, self.proj_dim)
        self.x2_proj = nn.Linear(dim, self.proj_dim)

        # Optional fusion projection
        self.out_proj = nn.Linear(self.proj_dim, dim)

        # Attention weight map from x1 and x2
        self.attn_q = nn.Linear(dim, dim)
        self.attn_k = nn.Linear(dim, dim)

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")

        if use_layernorm:
            self.norm = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        """
        x1: B x T x C
        x2: B x M x C
        return: B x T x M x C
        """
        B, T, C = x1.shape
        M = x2.size(1)

        # 1. Attention map (BAN 原文使用多个 glimpse，这里简化为 1)
        q = self.attn_q(x1)  # B x T x C
        k = self.attn_k(x2)  # B x M x C
        attn_logits = torch.matmul(q, k.transpose(1, 2)) / (C ** 0.5)  # B x T x M
        attn = torch.sigmoid(attn_logits)  # Optional: use sigmoid as in BAN

        # 2. Project x1 and x2 to lower-rank
        x1_proj = self.act(self.x1_proj(x1))  # B x T x d
        x2_proj = self.act(self.x2_proj(x2))  # B x M x d

        # 3. Create bilinear interactions: outer fusion
        # Expand for pairwise (T, M)
        x1_exp = x1_proj.unsqueeze(2)         # B x T x 1 x d
        x2_exp = x2_proj.unsqueeze(1)         # B x 1 x M x d
        fused = x1_exp * x2_exp               # B x T x M x d (low-rank bilinear)

        # 4. Apply attention weight
        attn = attn.unsqueeze(-1)             # B x T x M x 1
        fused = fused * attn                  # B x T x M x d

        # 5. Project back to original dim (optional)
        out = self.out_proj(fused)            # B x T x M x C

        # Residual: expand x1 to match (B x T x M x C)
        x1_exp = x1.unsqueeze(2).expand(-1, -1, M, -1)  # B x T x M x C
        out = x1_exp + out                        # B x T x M x C

        if self.use_layernorm:
            out = self.norm(out)

        return out
