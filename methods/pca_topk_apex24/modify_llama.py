from typing import List, Optional, Tuple, Union
import math
import warnings
import torch
from torch import nn

import sys
sys.path.append('/ephemeral/purva_exp/loki/apex')

import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv, apply_rotary_pos_emb
from transformers.cache_utils import Cache
from methods.common.utils import mask_attn_top_k
import methods

# Optional AxoNN support for tensor saving
try:
    from axonn import axonn as ax
    from axonn.intra_layer import gather
    AXONN_AVAILABLE = True
except ImportError:
    AXONN_AVAILABLE = False

# Apex ASP imports for 2:4 structured sparsity
from apex.contrib.sparsity import ASP
from torch.cuda.amp import autocast
ASP_INITIALIZED = False

class SparseAttentionProj(nn.Module):
    """
    Applies 2:4 structured-sparsity pruning to Q/K/V projections via Apex ASP.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

    def apply_asp(self):
        global ASP_INITIALIZED
        if ASP_INITIALIZED:
            return
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        ASP.prune_trained_model(self, optimizer)
        ASP_INITIALIZED = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Mixed-precision inference
        with autocast():
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        return q, k, v


def get_top_k_forward(args):
    """
    Returns a modified forward that:
      - Applies 2:4 ASP sparsity on Q/K/V via SparseAttentionProj
      - Computes sparse attention scores
      - Masks to top-K entries
      - Softmax & dropout, then matmul
    """
    def modified_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated; use `attention_mask` instead."
            )

        bsz, q_len, _ = hidden_states.size()

        # Compute rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Lazy init 2:4 sparse proj per layer
        if not hasattr(self, "_asp_proj"):
            self._asp_proj = SparseAttentionProj(self.hidden_size).to(hidden_states.device)
            self._asp_proj.apply_asp()

        # 1) Project raw inputs -> sparse Q/K/V
        # import ipdb; ipdb.set_trace()

        q_sp, k_sp, v_sp = self._asp_proj(hidden_states)
       
        # 2) Reshape into heads
        q_sp = q_sp.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_sp = k_sp.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v_sp = v_sp.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 3) Apply rotary on sparse projections
        q_sp, k_sp = apply_rotary_pos_emb(q_sp, k_sp, cos, sin)

        # 4) Repeat KV heads to match num_heads
        k_sp = repeat_kv(k_sp, self.num_key_value_groups)
        v_sp = repeat_kv(v_sp, self.num_key_value_groups)

        # 5) Compute sparse attention scores
        attn_weights = torch.matmul(q_sp, k_sp.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : k_sp.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # 6) Mask to top-K
        if args.top_k <= 1:
            topk = int(args.top_k * attn_weights.shape[-1])
        else:
            topk = int(args.top_k)
        attn_weights = mask_attn_top_k(attn_weights, topk, dim=-1)

        # 7) Softmax & dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_sp.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # 8) Final matmul with sparse values
        attn_output = torch.matmul(attn_weights, v_sp)

        # Sanity check
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be {(bsz, self.num_heads, q_len, self.head_dim)}, but got {attn_output.size()}"
            )

        # 9) Reshape & output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        if self.config.pretraining_tp > 1:
            chunks = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            out_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum(F.linear(chunks[i], out_slices[i]) for i in range(len(chunks)))
        else:
            attn_output = self.o_proj(attn_output)

        # Hide attentions if not requested
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return modified_forward


def make_llama_attention_pca_topk_apex24(args):
    print("Modifying Llama Attention -> TopK + 2:4-ASP Sparse Attention")
    if args.top_k <= 1:
        print(f"TopK% - {args.top_k}")
    else:
        print(f"TopK - {args.top_k}")
    LlamaAttention.forward = get_top_k_forward(args)
