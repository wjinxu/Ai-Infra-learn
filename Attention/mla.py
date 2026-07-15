class MultiHeadLatentAttention(nn.Module) :
    def __init__(self, hidden_dim, num_head, q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, max_position, base):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.softmax_scale = math.sqrt(qk_nope_head_dim + qk_rope_head_dim)

        self.q_down_proj = nn.Linear(hidden_dim, q_lora_rank, bias=False)
        self.q_nope_proj = nn.Linear(
            q_lora_rank, num_head * qk_nope_head_dim, bias=False
        )
        self.q_rope_proj = nn.Linear(
            q_lora_rank, num_head * qk_rope_head_dim, bias=False
        )

        self.kv_down_proj = nn.Linear(hidden_dim, kv_lora_rank, bias=False)
        self.k_up_proj = nn.Linear(
            kv_lora_rank, num_head * qk_nope_head_dim, bias=False
        )
        self.v_up_proj = nn.Linear(
            kv_lora_rank, num_head * v_head_dim, bias=False
        )
        self.k_rope_proj = nn.Linear(hidden_dim, qk_rope_head_dim, bias=False)

        self.o_proj = nn.Linear(num_head * v_head_dim, hidden_dim, bias=False)
        self.rope = RotaryEmbeding(max_position, base, qk_rope_head_dim)

    def forward(self, input, positions, attention_mask=None):
        batch_size, seq_len, _ = input.shape

        q_latent = self.q_down_proj(input)
        q_nope = self.q_nope_proj(q_latent).view(
            batch_size, seq_len, self.num_head, self.qk_nope_head_dim
        ).transpose(1, 2)
        q_rope = self.q_rope_proj(q_latent).view(
            batch_size, seq_len, self.num_head, self.qk_rope_head_dim
        ).transpose(1, 2)

        kv_latent = self.kv_down_proj(input)
        absorbed_k = self.k_up_proj.weight.view(
            self.num_head, self.qk_nope_head_dim, self.kv_lora_rank
        )
        score = torch.einsum(
            "bhtd,hdr,bsr->bhts", q_nope, absorbed_k, kv_latent
        )

        k_rope = self.k_rope_proj(input).unsqueeze(1)
        q_rope, k_rope = self.rope(q_rope, k_rope, positions)
        score = score + torch.matmul(q_rope, k_rope.transpose(-1, -2))
        score = score / self.softmax_scale

        if attention_mask is not None :
            score = score.masked_fill(
                attention_mask == 0, torch.finfo(score.dtype).min
            )

        attention = torch.softmax(score, dim=-1)
        latent_context = torch.einsum(
            "bhts,bsr->bhtr", attention, kv_latent
        )
        absorbed_v = self.v_up_proj.weight.view(
            self.num_head, self.v_head_dim, self.kv_lora_rank
        )
        context = torch.einsum(
            "bhtr,hdr->bhtd", latent_context, absorbed_v
        )
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(context)
