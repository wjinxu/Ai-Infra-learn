"""
mHC: Manifold-Constrained Hyper-Connections (DeepSeek-AI, 2025.12)
论文: https://arxiv.org/abs/2512.24880
"""

import torch
import torch.nn as nn


def sinkhorn_knopp(logits: torch.Tensor, n_iters: int = 20, eps: float = 1e-8) -> torch.Tensor:
    """
    Sinkhorn-Knopp 迭代：把任意实数矩阵投影到"双随机矩阵"集合 (Birkhoff polytope) 上。

    Args:
        logits: (..., N, N)，未经约束的原始矩阵 (对应论文里的 H~_res)
        n_iters: 迭代次数。论文取 20 作为精度/效率的折中，不是精确解，是近似解
        eps: 防止除0

    Returns:
        (..., N, N) 矩阵，每一行、每一列的和都 ≈ 1，所有元素 >= 0
    """
    # 第一步：exp 保证所有元素为正数，对应论文 Eq.(9): M^(0) = exp(H~_res)
    m = torch.exp(logits)
    for _ in range(n_iters):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)  # 行归一化：每行和为1
        m = m / (m.sum(dim=-2, keepdim=True) + eps)  # 列归一化：每列和为1
    # 交替进行足够多次之后，行和列会同时逼近1(严格意义上要迭代到无穷次才精确收敛)
    return m


class ManifoldHyperConnection(nn.Module):
    """
    h: hyper hidden matrix (B,L,N,D):
        B: batch size
        L: sequence length
        N: number of hyper connections (等价于论文里的 expansion rate n)
        D: hidden dimension
    """

    def __init__(self, dim, rate, layer_id, dynamic, sinkhorn_iters=20, device=None):
        super().__init__()
        self.dim = dim
        self.rate = rate  # 并行的残差流条数 n
        self.dynamic = dynamic  # 是否使用输入相关的动态权重
        self.sinkhorn_iters = sinkhorn_iters  # Sinkhorn-Knopp 迭代次数，论文取20

        # ============ H_post (beta)：静态/动态分解方式与 hc.py 完全一样 ============
        # 区别只在于最后要过 2*sigmoid，所以这里初始化为0，2*sigmoid(0)=1，
        # 表示初始化时 H_post 恰好把 F 的输出"原样"写回残差流(等价于标准残差连接)
        self.static_beta = nn.Parameter(torch.zeros(rate, device=device))

        # ============ H_pre 与 H_res 的合并 logits ============
        # 和 hc.py 的 static_alpha 一样打包成一个 (rate, 1+rate) 矩阵：
        #   第 0 列  -> H_pre 的 logits，之后过 sigmoid 变成非负的聚合权重
        #   第 1: 列 -> H_res 的 logits，之后过 Sinkhorn-Knopp 变成双随机矩阵
        # 用较大的 +/-scale 而不是 hc.py 里的 0/1，是为了让 sigmoid / Sinkhorn
        # 变换之后依然近似"只读取本层残差流" + "残差流原样保留"的恒等初始化，
        # 呼应标准残差连接 x_{l+1}=x_l+F(x_l) 的初始行为。
        scale = 6.0
        init_pre = torch.full((rate, 1), -scale, device=device)
        init_pre[layer_id % rate, 0] = scale  # 初始化时只从第 layer_id % rate 条流读入，同 hc.py
        init_res = torch.eye(rate, device=device) * (2 * scale) - scale  # 对角+scale，非对角-scale，近似单位矩阵
        self.static_alpha = nn.Parameter(torch.cat([init_pre, init_res], dim=1))  # (rate, 1+rate)

        if dynamic:
            # Wmr = cat(Wm, Wr)，与 hc.py 完全一致
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, rate + 1, device=device))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)

            # WB
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim, device=device))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)

            # hc.py 里 self.layer_norm 被用到但没有定义，这里补上：
            # RMSNorm 作用在最后一维 D 上(逐条残差流独立归一化)
            self.rms_weight = nn.Parameter(torch.ones(dim, device=device))

    def _rms_norm(self, h, eps=1e-6):
        h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
        return h * self.rms_weight

    def width_connection(self, h):
        # get alpha and beta
        if self.dynamic:  # 如果动态，则需要进行层归一化
            norm_h = self._rms_norm(h)

        # Note: 求 H_pre 和 H_res 的原始 logits (还没加约束)
        if self.dynamic:
            wc_weight = norm_h @ self.dynamic_alpha_fn  # (B,L,N,1+rate)
            wc_weight = torch.tanh(wc_weight)
            dynamic_alpha_logits = wc_weight * self.dynamic_alpha_scale  # 缩放因子
            alpha_logits = dynamic_alpha_logits + self.static_alpha[None, None, ...]
        else:
            alpha_logits = self.static_alpha[None, None, ...]

        # Note: 求 H_post 的原始 logits
        if self.dynamic:
            dc_weight = norm_h @ self.dynamic_beta_fn  # (B,L,N)
            dc_weight = torch.tanh(dc_weight)
            dynamic_beta_logits = dc_weight * self.dynamic_beta_scale
            beta_logits = dynamic_beta_logits + self.static_beta[None, None, ...]
        else:
            beta_logits = self.static_beta[None, None, ...]

        # =================== mHC 与 HC 唯一的核心差异，就在这几行 ===================
        # hc.py 里是直接把 alpha_logits / beta_logits 当成 alpha / beta 使用(无约束)。
        # mHC 在这里插入了"manifold projection"：
        pre_logits, res_logits = alpha_logits[..., :1], alpha_logits[..., 1:]
        pre = torch.sigmoid(pre_logits)                          # H_pre: 非负约束 (论文 Eq.8)
        res = sinkhorn_knopp(res_logits, self.sinkhorn_iters)     # H_res: 双随机矩阵约束 (Birkhoff polytope)
        alpha = torch.cat([pre, res], dim=-1)                     # 拼回 (B,L,N,1+rate)，复用同一套乘法逻辑

        beta = 2 * torch.sigmoid(beta_logits)                     # H_post: 非负约束 (论文 Eq.8)
        # ============================================================================

        # Note: 缩放因子对输入进行缩放
        # alpha 融合了 H_pre 和 H_res(已分别做过 sigmoid / Sinkhorn-Knopp 投影)，
        # mix_h 的第 0 个分量喂给层函数 F，剩下 rate 个分量就是新的、满足双随机
        # 约束的残差流
        mix_h = alpha.transpose(-1, -2) @ h
        return mix_h, beta

    def depth_connection(self, mix_h, h_o, beta):
        # Note: beta(H_post) 缩放因子处理变换分支，再与残差分支 mix_h[...,1:,:] (H_res @ h) 相加
        h = torch.einsum("blh,bln->blnh", h_o, beta) + mix_h[..., 1:, :]
        return h

