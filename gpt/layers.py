import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(in_dtype) * self.weight


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.silu(self.w_gate(x)) * self.w_up(x)   # gated activation
        return self.w_down(self.dropout(x))


class MoEFeedForward(nn.Module):
    """Top-k mixture of experts. Each token is routed to top_k of num_experts
    SwiGLU experts; outputs are weighted by the (renormalized) router probs.
    Returns (output, aux_loss); aux_loss encourages balanced expert usage.
    """
    def __init__(self, d_model, d_ff, num_experts, num_experts_per_tok=2, dropout=0.1):
        super().__init__()
        assert 1 <= num_experts_per_tok <= num_experts
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUFeedForward(d_model, d_ff, dropout) for _ in range(num_experts)]
        )

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)
        N = x_flat.size(0)

        probs = F.softmax(self.router(x_flat), dim=-1)          # (N, E)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)   # (N, k)
        topk_probs = topk_probs / topk_probs.sum(-1, keepdim=True)

        # Load-balancing aux loss (Switch Transformer): fraction of tokens per
        # expert * mean router prob per expert; == 1.0 at perfect balance.
        one_hot = F.one_hot(topk_idx, self.num_experts).float()  # (N, k, E)
        P = probs.mean(dim=0)
        f = one_hot.sum(dim=(0, 1)) / (N * self.top_k)
        aux_loss = self.num_experts * torch.sum(f * P)

        y = torch.zeros_like(x_flat)
        mask = one_hot.permute(2, 1, 0).bool()   # (E, k, N)
        for i in range(self.num_experts):
            slot, tok = torch.where(mask[i])
            if tok.numel() == 0:
                continue
            h_i = self.experts[i](x_flat[tok])
            g_i = topk_probs[tok, slot].to(h_i.dtype).unsqueeze(-1)
            y.index_add_(0, tok, (h_i * g_i).to(y.dtype))

        return y.reshape(B, L, D), aux_loss
