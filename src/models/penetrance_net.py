import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .explicit_implicit import ExplicitImplicitFusion


class SequenceFeatureExtractor(nn.Module):
    """特征投影网络：支持 [B, L, D]"""

    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim)  # 线性变换
        )

    def forward(self, x):
        B, L, D = x.shape
        return self.proj(x)  # 直接投影


class FusionBlock(nn.Module):
    """融合块：fusion + MLP + 残差，支持序列输入"""

    def __init__(self, dim, mlp_ratio=4, dropout_rate=0.1, activation='silu', res_scale_init_value=1e-6):
        super().__init__()
        self.fusion = ExplicitImplicitFusion(
            input_dim=dim,
            output_dim=dim,
            dropout_rate=dropout_rate,
            activation=activation
        )
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        
        # Residual scaling parameters
        self.res_scale1 = nn.Parameter(res_scale_init_value * torch.ones(dim), requires_grad=True)
        self.res_scale2 = nn.Parameter(res_scale_init_value * torch.ones(dim), requires_grad=True)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, L, D]
        # Fusion block with residual scaling
        fused = self.fusion(self.norm1(x))
        x = x + self.res_scale1 * fused

        # MLP block with residual scaling  
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.res_scale2 * mlp_out

        return x


class PenetranceNet(pl.LightningModule):
    """外显率预测主模型"""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg, logger=False)
        self._has_setup = False
        self.setup()

    def setup(self, stage=None):
        if self._has_setup:
            return
        self._has_setup = True

        # annotation 投影
        self.annotation_net = SequenceFeatureExtractor(
            input_dim=len(self.hparams.annotation_columns),
            output_dim=self.hparams.annotation_dim,
            dropout_rate=self.hparams.dropout_rate
        )

        # gene 投影
        self.gene_net = SequenceFeatureExtractor(
            input_dim=len(self.hparams.gene_columns),
            output_dim=self.hparams.gene_dim,
            dropout_rate=self.hparams.dropout_rate
        )

        # sequence 投影
        if hasattr(self.hparams, 'hyena_output_dim') and self.hparams.hyena_output_dim > 0:
            self.seq_net = SequenceFeatureExtractor(
                input_dim=self.hparams.hyena_output_dim,
                output_dim=self.hparams.hyena_output_dim,
                dropout_rate=self.hparams.dropout_rate
            )
        else:
            self.seq_net = nn.Identity()

        # fusion 输入维度
        fusion_input_dim = (getattr(self.hparams, 'hyena_output_dim', 0) +
                            self.hparams.annotation_dim +
                            self.hparams.gene_dim)

        self.input_proj = nn.Linear(fusion_input_dim, self.hparams.fusion_dim)

        # fusion blocks
        fusion_activation = getattr(self.hparams, 'fusion_activation', 'silu')
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(
                dim=self.hparams.fusion_dim,
                mlp_ratio=getattr(self.hparams, 'mlp_ratio', 4),
                dropout_rate=self.hparams.dropout_rate,
                activation=fusion_activation,
                res_scale_init_value=getattr(self.hparams, 'res_scale_init_value', 1e-6)
            )
            for _ in range(self.hparams.fusion_blocks)
        ])

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(self.hparams.fusion_dim, max(8, self.hparams.fusion_dim // 2)),
            nn.GELU(),
            nn.Dropout(self.hparams.dropout_rate),
            nn.Linear(max(8, self.hparams.fusion_dim // 2), self.hparams.output_dim)
        )

    def forward(self, batch):
        seq_emb = batch['embedding']
        seq_emb = self.seq_net(seq_emb)

        ann_emb = self.annotation_net(batch['annotation_features'])
        gene_emb = self.gene_net(batch['gene_features'])

        # [B, L, D_seq]
        B, L, _ = seq_emb.shape
        ann_broadcast = ann_emb.expand(B, L, -1)
        gene_broadcast = gene_emb.expand(B, L, -1)
        combined = torch.cat([seq_emb, ann_broadcast, gene_broadcast], dim=-1)
        x = self.input_proj(combined)
        for block in self.fusion_blocks:
            x = block(x)
        x_pooled = x.mean(dim=1)  # [B, fusion_dim]

        prediction = self.regressor(x_pooled)
        return prediction, None