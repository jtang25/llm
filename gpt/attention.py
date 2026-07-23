import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads=None, max_seq_len=512,
                 dropout=0.0, rope_base=10000.0):
        super().__init__()
        assert d_model % num_heads == 0
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim % 2 == 0

        self.q_dim = num_heads * self.head_dim
        self.kv_dim = num_kv_heads * self.head_dim
        self.W_qkv = nn.Linear(d_model, self.q_dim + 2 * self.kv_dim)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Precompute RoPE cos/sin tables of shape (max_seq_len, head_dim).
        positions = torch.arange(max_seq_len)
        pair_indices = torch.arange(self.head_dim // 2)
        thetas = rope_base ** (-2 * pair_indices / self.head_dim)
        angles = positions.unsqueeze(1) * thetas.unsqueeze(0)
        self.register_buffer("cos", torch.cos(angles).repeat_interleave(2, dim=1))
        self.register_buffer("sin", torch.sin(angles).repeat_interleave(2, dim=1))

    def _split(self, x, num_heads):
        B, L, _ = x.size()
        return x.view(B, L, num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d)

    def combine_heads(self, x):
        B, H, L, Hd = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, H * Hd)

    def rotate_half(self, x):
        t1, t2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([-t2, t1], dim=-1).flatten(start_dim=-2)

    def repeat_kv(self, x):
        # (B, num_kv_heads, L, d) -> (B, num_heads, L, d) by repeating each kv head.
        if self.n_rep == 1:
            return x
        B, G, L, Hd = x.shape
        return x[:, :, None].expand(B, G, self.n_rep, L, Hd).reshape(B, G * self.n_rep, L, Hd)

    def project(self, x, start_pos=0):
        """x: (B, L, d_model) -> Q,K,V heads with RoPE applied to Q and K.

        start_pos is the absolute position of x[:, 0] in the full sequence, so
        RoPE stays correct when we feed one token at a time during decoding.
        """
        B, L, _ = x.size()
        qkv = self.W_qkv(x)
        Q, K, V = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        Q = self._split(Q, self.num_heads)
        K = self._split(K, self.num_kv_heads)
        V = self._split(V, self.num_kv_heads)

        cos = self.cos[start_pos:start_pos + L].unsqueeze(0).unsqueeze(0)  # (1,1,L,d)
        sin = self.sin[start_pos:start_pos + L].unsqueeze(0).unsqueeze(0)
        Q = Q * cos + self.rotate_half(Q) * sin
        K = K * cos + self.rotate_half(K) * sin
        return Q, K, V

    def forward(self, x, mask=None, kv_cache=None, start_pos=0):
        """Dense path: KV cache is a single contiguous (past_k, past_v) tuple."""
        Q, K, V = self.project(x, start_pos)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                K = torch.cat([past_k, K], dim=2)
                V = torch.cat([past_v, V], dim=2)
            new_cache = (K, V)
        else:
            new_cache = None

        K = self.repeat_kv(K)
        V = self.repeat_kv(V)
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = self.dropout(self.W_o(self.combine_heads(out)))
        return out, new_cache
