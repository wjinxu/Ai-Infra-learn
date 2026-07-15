class MultiHeadAttention (nn.Module):
    def __init__(self, hidden_dim, num_head, max_position, base):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.head_dim = hidden_dim // num_head
        # [q, k]
        self.qkv_proj = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rope = RotaryEmbeding(max_position, base, hidden_dim)
    # input [batch, seqs, hidden_dim]
    def forward(self, input, attention_mask, positions):
        batch_size, seq, _ = input.size()
        qkv_proj = self.qkv_proj(input)
        q, k, v = qkv_proj.chunk(3, dim=-1)
        # 分出 q k v
        # [batch, seqs, hidden_dim] -> [batch, num_head, seqs, head_dim]
        q = q.view(batch_size, seq, self.num_head, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq, self.num_head, self.head_dim).transpose(1,2)
        v = v.view(batch_size, seq, self.num_head, self.head_dim).transpose(1,2)
        # Rope
        q, k = self.rope(q, k, positions)
        # q @ k^T [batch, num_head, seqs, head_dim] @ [batch, num_head, head_dim, seqs] -> [batch, num_head, seqs, seqs]
        score = q @ (k.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
        if attention_mask is not None :
            score = score.masked_fill(
                attention_mask == 0, -1e20
            )
        # score @ v
        score = torch.softmax(score, dim = -1)
        # [batch, num_head, seqs, seqs] @ [batch, num_head, seqs, head_dim] -> [batch, num_head, seqs, head_dim]
        score = score @ v
        score = score.transpose(1, 2).contiguous()
        return self.o_proj(score.view(batch_size, seq, -1))