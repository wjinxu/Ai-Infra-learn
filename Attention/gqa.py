class GroupQueryAttention(nn.Module) :
    def __init__(self, hidden_dim, nums_head, nums_kv_head, max_positions, base):
        super().__init__()
        self.nums_head = nums_head
        self.head_dim = hidden_dim // nums_head
        self.nums_kv_head = nums_kv_head
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim + self.head_dim * self.nums_kv_head * 2)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rope = RotaryEmbeding(max_positions, base, hidden_dim)
    def forward(self, input, positions, attention_mask):
        batch_size, seqs, _ = input.shape()
        qkv_proj = self.qkv_proj(input)
        # [batch seqs hidden_dim]
        q, k, v = qkv_proj.chunk(3, dim=-1)
        # [batch seqs hidden_dim] -> [batch, nums_head, seqs, head_dim]
        q = q.view(batch_size, seqs, self.nums_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seqs, self.nums_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seqs, self.nums_kv_head, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.nums_head // self.nums_kv_head, dim = 1)
        v = v.repeat_interleave(self.nums_head // self.nums_kv_head, dim = 1)
        # Rope
        q, k = self.rope(q, k, positions)
        # q @ k^T [batch, num_head, seqs, head_dim] @ [batch, num_head, head_dim, seqs] -> [batch, num_head, seqs, seqs]
        score = q @ (k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None :
            score = score.masked_fill(
                attention_mask == 0, -1e20
            )
        # score @ v
        score = torch.softmax(score, dim = -1)
        # [batch, num_head, seqs, seqs] @ [batch, num_head, seqs, head_dim] -> [batch, num_head, seqs, head_dim]
        score = score @ v
        score = score.transpose(1, 2).contiguous()
        return self.o_proj(score.transpose(batch_size, seqs, -1))