import time
import torch
from types import SimpleNamespace
from typing import Callable, Optional
from typing_extensions import Unpack
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache

# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------
# 自动选择 target的那些层需要被使用
def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]

# 抽取特征， 用于后续的注意力计算
def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)

# 采样
def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def dflash_generate(
    model: "DFlashDraftModel", # draft 模型
    target: nn.Module, # target 模型
    input_ids: torch.LongTensor, # 输入tokens
    max_new_tokens: int, # 最大新添加tokens
    stop_token_ids: Optional[list[int]], # 停止token
    temperature: float, # 温度
    block_size: Optional[int] = None, # 块大小
    mask_token_id: Optional[int] = None, # 掩码token
    return_stats: bool = False, # 返回统计信息
):
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size
    mask_token_id = model.mask_token_id if mask_token_id is None else mask_token_id

    output_ids = torch.full( # 预分配的输出缓冲 多留一个block防止越界 初始全填 mask_token_id
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=target.device,
    )
    # 位置编码
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    # 创建动态Cache 用于存储过去的key和value
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    # 记录prefill的开始时间
    prefill_start = _cuda_time() if return_stats else None
    # 调用target模型进行prefill
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens], # 位置编码
        past_key_values=past_key_values_target, # 传入空的target cache
        use_cache=True,
        logits_to_keep=1, # 只关心最后一个输出
        output_hidden_states=block_size > 1, # 做推测解码需要中间的hidden_states
    )

    output_ids[:, :num_input_tokens] = input_ids # 把prompt的tokens放入输出缓冲
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature) # 第一个新的token采样
    if block_size > 1: # 提取target的hidden_states 用于后续的注意力计算
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None # 计算第一个token prefill的时间

    decode_start = _cuda_time() if return_stats else None # decode开始时间
    acceptance_lengths = [] # 记录每轮接受了多少token，用于计算加速比
    start = num_input_tokens # 当前已经确定token的末尾位置
    draft_prefill = True # 标记draft模型是否需要prefill

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone() # 第一次只有一个token，其他都是mask
        block_position_ids = position_ids[:, start : start + block_size] # 对应的block位置编码
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(model( # 先调用draft model 模型然后调用target模型得到logits
                target_hidden=target_hidden, # 需要注入的hidden_states
                noise_embedding=noise_embedding, # 第一个是对应的token，其他都是mask
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size], # 位置编码
                past_key_values=past_key_values_draft, # kv cache
                use_cache=True, # 使用 cache
                is_causal=False, # 非自回归
            )[:, 1 - block_size :, :]) # 取出 block_size - 1个位置
            past_key_values_draft.crop(start) # 因为draft模型是非自回归的，所以每次都需要裁剪掉已经使用过的cache
            block_output_ids[:, 1:] = sample(draft_logits) # draft 使用贪心采样
            if draft_prefill and return_stats: # 结束prefill
                draft_prefill = False
                decode_start = _cuda_time()

        output = target( # 调用target模型得到最终的logits
            block_output_ids, # 已经存进去猜测的tokens
            position_ids=block_position_ids, # 位置编码
            past_key_values=past_key_values_target, # KV cache
            use_cache=True, # 使用 cache
            output_hidden_states=block_size > 1, # 输出hidden_states
        )

        posterior = sample(output.logits, temperature) # 采样得到最终的token
        # 总结
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item() # 输出连续猜对的前缀
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1 # 游标推进
        past_key_values_target.crop(start) # 砍掉被拒绝的部分KV
        acceptance_lengths.append(acceptance_length + 1) # 几率本轮战绩

        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]
        # 提取对应的target_hidden 用于下一轮的注意力计算
        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :min(start + 1, max_length)]
    # 截断到实际长度
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]
    # 判断是否到了停止运算符
    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = _cuda_time() - decode_start
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / num_output_tokens,
        acceptance_lengths=acceptance_lengths,
    )
    # 返回统计

# ---------------------------------------------------------------------------
# DFlash model
# ---------------------------------------------------------------------------
# 旋转位置编码
# q只有当前块但是k包含历史上下文
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# 搭建DFlash Attention 模块
class Qwen3DFlashAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx # 第几层 KV cache按层存取要用
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads) # 每个注意力头的维度
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads # GQA 分组查询注意力
        self.scaling = self.head_dim**-0.5 # 缩放因子
        self.attention_dropout = config.attention_dropout
        self.is_causal = False # 非因果，最重要差别
        self.q_proj = nn.Linear( # q_proj
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear( # k_proj
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear( # v_proj
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear( # o_proj
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # 滑动窗口注意力
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor, # 当前块 (noise，要预测的那块)
        target_hidden: torch.Tensor, # 上下文（target 的 hidden states）
        position_embeddings: tuple[torch.Tensor, torch.Tensor], # 位置编码
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None, # kv cache
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1] # 当前块的长度
        ctx_len = target_hidden.shape[1] # 上下文长度
        q = self.q_proj(hidden_states) # q_proj q 投影
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden) # 上下文投影
        k_noise = self.k_proj(hidden_states) # 噪音投影
        v_ctx = self.v_proj(target_hidden) # 上下文投影
        v_noise = self.v_proj(hidden_states) # 噪音投影
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim) # 拼接 上下文 + 当前块
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin) # 使用位置编码
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs) # 更新kv 其实就是 上下文 + 自己， 但是自己之后会drop 其实就是上下文
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn( # 不加任何屏蔽的attention 非因果注意力
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        # hidden_size : 维度
        self.hidden_size = config.hidden_size
        # Dflash的注意力机制
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        # MLP层
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
    # 没啥特别的这里
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

# 没啥特别的
class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = self.config.dflash_config.get(
            "target_layer_ids", build_target_layer_ids(config.num_target_layers, config.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = self.config.dflash_config.get("mask_token_id", None)
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
    ):
        self.eval()
        return dflash_generate(
            self,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )
