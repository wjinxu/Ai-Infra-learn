import torch
class HyperConnection(nn.Module):
"""
 h: hyper hidden matrix(B,L,N,D) :
   B: batch size
   L: sequence length
   N: number of hyper connections
   D: hidden dimension
"""
    def __init__(self, dim, rate,layer_id, dynamic, device = None):
        super().__init__()
        self.dim = dim
        self.rate = rate # 并行的条数
        self.dynamic = dynamic # 是否动态
        # B1 and B2
        self.static_beta = nn.Parameter(torch.ones(rate, )) # 有两个静态beta，一个用于B1，一个用于B2

        init_alpha0 = torch.zeros((rate, 1),) # Am的静态部分
        init_alpha0[layer_id % rate, 0] = 1. # 初始化时，让第layer_id % rate条为1 就是单流动态的alpha
        # torch.eye(rate) ： 创建一个单位矩阵 这个代表Ar的静态部分
        self.static_alpha = nn.Parameter(torch.cat([init_alpha0, torch.eye((rate), )], dim=1)) # 把这两个合并成一个矩阵
        if dynamic:
            ## Wmr = cat(Wm, Wr)
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, rate+1))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01) 
            
            ## WB
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)
    def width_connection(self, h):
        # get alpha and beta
        if self.dynamic: # 如果动态，则需要进行层归一化
            norm_h = self.layer_norm(h)

        # Note: 求 Am(H) 和 Ar(H)
        if self.dynamic: # 如果动态，则需要进行权重计算
            wc_weight = norm_h @ self.dynamic_alpha_fn # 计算动态alpha的权重
            wc_weight = F.tanh(wc_weight) 
            dynamic_alpha = wc_weight  * self.dynamic_alpha_scale # 缩放因子
            alpha = dynamic_alpha + self.static_alpha[None, None, ...]
        else:
            alpha = self.static_alpha[None, None, ...]

        # Note: 求 B(H)
        if self.dynamic:
            dc_weight = norm_h @ self.dynamic_beta_fn #H, W_beta
            dc_weight = F.tanh(dc_weight)
            dynamic_beta = dc_weight * self.dynamic_beta_scale # S_beta
            beta = dynamic_beta + self.static_beta[None, None, ...] # + B
        else:
            beta = self.static_beta[None, None, ...]
        # 缩放因子计算完毕

        # Note: 缩放因子对输入进行缩放
        # alpha 因子是融合了 Am和Ar, mix_h 是包含残差分支和多条变换分支 
        mix_h = alpha.transpose(-1, -2)  @  h
        return mix_h, beta
    def depth_connection(self, mix_h, h_o, beta):
        # Note: beta 缩放因子处理残差分支, 再与变换分支 `mix_h` 进行相加
        h = torch.einsum("blh,bln->blnh", h_o, beta) + mix_h[..., 1:, :]