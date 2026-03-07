import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_blocks import Cross_Block, Block

class REDistill(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # 交互编码模块
        self.cross_block_agent = nn.ModuleList(
            Cross_Block()
            for i in range(3)
        )

        # Logit门控模块（通道注意力）
        self.self_block_logit = nn.ModuleList(
            Block()
            for i in range(3)
        )
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid()
        )

        # 残差编码器
        self.self_block_residual = nn.ModuleList(
            Block()
            for i in range(3)
        )
        self.encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU6(inplace=True)
        )

        # 归一化层
        self.norm = nn.LayerNorm(dim)
        self.norm_cross = nn.LayerNorm(dim)
        self.norm_logit = nn.LayerNorm(dim)
        self.norm_residual = nn.LayerNorm(dim)

    def forward(self, f_s, agent_n):
        """
        f_s: 学生特征 [B, M, D]
        """

        B, M, D = f_s.shape

        f_a = f_s[:, :agent_n]
        f_l = f_s[:, agent_n:]
        for blk in self.cross_block_agent:
            f_a = blk(f_a, f_l, norm_layer=self.norm_cross)
        f_al = f_a.clone()
        f_ar = f_a.clone()
        for blk in self.self_block_logit:
            f_al = blk(f_al, norm_layer=self.norm_logit)
        for blk in self.self_block_residual:
            f_ar = blk(f_ar, norm_layer=self.norm_residual)

        f_sl = torch.cat([f_al, f_l], dim=1)  # [B, M, D]
        f_sr = torch.cat([f_ar, f_l], dim=1)  # [B, M, D]

        # reshape for BN 和全连接层：-> [B*M, D]
        f_sl_flat = f_sl.view(-1, D)
        f_sr_flat = f_sr.view(-1, D)
        f_s_flat = f_s.view(-1, D)

        # 残差路径
        f_r = self.encoder(f_sr_flat)  # [B*M, D]

        # 门控路径
        gate = self.gate(f_sl_flat)    # [B*M, D]
        f_s_weighted = gate * f_s_flat

        # 蒸馏输出
        f_d = f_r + f_s_weighted      # [B*M, D]
        f_d = self.norm(f_d)          # 归一化
        f_d = f_d.view(B, M, D)       # reshape 回原始尺寸

        return f_d
    
'''
f_s_flat = f_s.view(-1, D)  # [B*M, D]

# 残差路径
f_r = self.encoder(f_s_flat)  # [B*M, D]

# 门控路径
gate = self.gate(f_s_flat)    # [B*M, D]
f_s_weighted = gate * f_s_flat

# 蒸馏输出
f_d = f_r + f_s_weighted      # [B*M, D]
f_d = self.norm(f_d)          # 归一化
f_d = f_d.view(B, M, D)       # reshape 回原始尺寸
'''