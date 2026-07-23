from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 128
    num_heads: int = 4
    num_kv_heads: int = 4
    n_layer: int = 2
    d_ff: int = 512
    dropout: float = 0.0
    max_seq_len: int = 256
    num_experts: Optional[int] = 8      # None or 1 -> dense SwiGLU instead of MoE
    num_experts_per_tok: int = 2
    aux_coef: float = 0.01              # weight on the MoE load-balancing loss

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


def checkpoint_config(vocab_size: int) -> GPTConfig:
    """The exact config used to train gpt_model.pth (vocab comes from the tokenizer)."""
    return GPTConfig(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_kv_heads=4,
        n_layer=2,
        d_ff=512,
        dropout=0.0,
        max_seq_len=256,
        num_experts=8,
        num_experts_per_tok=2,
    )
