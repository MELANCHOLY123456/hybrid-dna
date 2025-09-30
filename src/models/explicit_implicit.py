import torch
import torch.nn as nn
import torch.nn.functional as F


class StarReLU(nn.Module):
    """StarReLU 激活：s * relu(x) ** 2 + b"""

    def __init__(self, scale_value=1.0, bias_value=0.0, scale_learnable=True, bias_learnable=True):
        super().__init__()
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * F.relu(x) ** 2 + self.bias


class ExplicitImplicitFusion(nn.Module):
    """
    基于 Explicit + Implicit 的融合模块。
    输入： x shape = [B, L, D_in]  （沿序列维度做 depthwise conv）
    输出： [B, L, D_out]
    """

    def __init__(self, input_dim, output_dim, dropout_rate=0.1, activation='silu',
                 explicit_kernel_size=3):
        super().__init__()

        self.output_dim = output_dim
        adjusted_dim = int(output_dim * 4 / 3)

        # A 路径
        self.linear_A = nn.Linear(input_dim, adjusted_dim)

        # depthwise conv 用于在序列维度捕捉局部相邻关系
        self.dw_conv_A = nn.Conv1d(
            in_channels=adjusted_dim,
            out_channels=adjusted_dim,
            kernel_size=explicit_kernel_size,
            # padding=explicit_kernel_size // 2, #same
            padding="same",
            groups=adjusted_dim,
            bias=True
        )

        # V 路径：从原始输入直接生成 gating 向量
        self.linear_V = nn.Linear(input_dim, output_dim)

        # 输出投影
        self.out_proj = nn.Linear(adjusted_dim, output_dim)

        # 激活、归一化、dropout
        self.activation = self._get_activation(activation)

        self._init_weights()

    def _get_activation(self, activation):
        if activation == 'tanh':
            return nn.Tanh()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'silu':
            return nn.SiLU()
        else:
            return nn.SiLU()

    def _init_weights(self):
        for module in [self.linear_A, self.linear_V, self.out_proj]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.dw_conv_A.weight)
        if self.dw_conv_A.bias is not None:
            nn.init.zeros_(self.dw_conv_A.bias)

    def forward(self, x):
        """
        x: [B, L, D_in]
        returns: [B, L, D_out]
        """

        # sequence mode
        B, L, D_in = x.shape
        A = self.linear_A(x)  # [B, L, C]

        A = A.permute(0, 2, 1)  # [B, C, L]
        A = self.dw_conv_A(A)  # [B, C, L]
        A = A.permute(0, 2, 1)  # [B, L, C]
        A_act = self.activation(A)  # [B, L, C]

        A = A + torch.tanh(A_act)  # [B, L, C]

        V = self.linear_V(x)  # [B, L, output_dim]
        A_proj = self.out_proj(A)  # [B, L, output_dim]
        x = V * A_proj  # [B, L, output_dim]

        return x