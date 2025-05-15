import math
import sys
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from typing import Any, Dict, List, Optional, Tuple
 

import methods.pca_topk.kernel.pca_topk as G
from methods.common.timers import Timers

# Force use of CUTLASS kernels for semi-structured sparsity (if available)
SparseSemiStructuredTensor._FORCE_CUTLASS = True


# -----------------------------------------------------------------------------
# Cache class (unchanged)
# -----------------------------------------------------------------------------
class PcaTopKCache:
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    @torch.no_grad()
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        layer_idx: int,
        topk: bool = True,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        max_length = self.get_max_length()
        prev = self.get_seq_length(layer_idx)
        if max_length is not None and prev + new_seq_length > max_length:
            return max_length - new_seq_length
        return prev

    def reset(self):
        self.key_cache = []
        self.value_cache = []


# -----------------------------------------------------------------------------
# Attention projection with semi-structured 2:4 sparsity
# -----------------------------------------------------------------------------
class SparseAttentionProj(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        # no bias for simplicity
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._sparsity_initialized = False

    def apply_semi_structured_sparsity(self):
        if self._sparsity_initialized:
            return

        # Build a 2:4 mask: keep 2 of every 4 columns in each row
        h, w = self.q_proj.weight.shape
        assert w % 4 == 0, "hidden_dim must be divisible by 4 for 2:4 sparsity"
        # pattern [1,1,0,0] repeated to fill each row
        base = torch.tensor([1,1,0,0], dtype=torch.bool, device=self.q_proj.weight.device)
        mask = base.repeat(w // 4).unsqueeze(0).expand(h, -1)

        for proj in (self.q_proj, self.k_proj, self.v_proj):
            # zero out pruned weights
            proj.weight.data.mul_(mask)
            # convert dense weight to a sparse-semi-structured tensor
            proj.weight = nn.Parameter(to_sparse_semi_structured(proj.weight.data))

        self._sparsity_initialized = True

    def forward(self, x):
        # uses sparse kernels under the hood once converted
        with torch.inference_mode():
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        return q, k, v


# -----------------------------------------------------------------------------
# PCA-topk micro-benchmark using sparse projections
# -----------------------------------------------------------------------------
def micro_benchmark_pca_topk(
    cache: PcaTopKCache,
    prompt_keys: List[torch.Tensor],
    top_r: int,
    top_k: int,
    num_layers: int,
    timers: Timers,
    num_gen_steps: int = 2000,
    use_optimised_gather: bool = False
):
    torch.set_float32_matmul_precision("highest")
    head_dim = prompt_keys[0].shape[-1]
    bs, num_heads = prompt_keys[0].shape[0], prompt_keys[0].shape[1]
    dtype = prompt_keys[0].dtype

    sparse_proj = SparseAttentionProj(head_dim).to("cuda")
    sparse_proj.apply_semi_structured_sparsity()
    print(f"PCA-TOPK with semi-structured sparsity, dtype={dtype}")

    # pre-allocate PCA projection
    pca_proj = torch.randn(num_heads, head_dim, head_dim, dtype=dtype, device='cuda')

    timers.start('total')
    for _ in range(num_gen_steps):
        for layer in range(num_layers):
            timers.start('qk-gen')
            inp = torch.rand(bs, num_heads, 1, head_dim, device='cuda', dtype=dtype)
            q, k, _ = sparse_proj(inp)
            timers.stop('qk-gen')

            timers.start('project')
            k = k.squeeze().transpose(0,1).bmm(pca_proj).unsqueeze(2)
            q = q.squeeze().transpose(0,1).bmm(pca_proj).unsqueeze(2)
            timers.stop('project')

            timers.start('cache-update')
            keys, vals = cache.update(k, k, q, layer, False)
            timers.stop('cache-update')

            timers.start('qk-matmul-1')
            nh, bs_new, s, r = keys.shape
            attn_w = G.topr_bmv_optimized(
                A=q.view(nh*bs_new,1,r),
                B=keys.view(nh*bs_new,s,r).transpose(-1,-2),
                r=top_r
            ).view(nh, bs_new, 1, s)
            timers.stop('qk-matmul-1')

            timers.start('top-k')
            idx = torch.argsort(attn_w, dim=-1, descending=True)[..., :top_k]
            timers.stop('top-k')

            timers.start('reshape-0')
            idx = idx.reshape(-1, idx.shape[-1])
            timers.stop('reshape-0')

            timers.start('reshape-1')
            k_flat = keys.view(-1, keys.shape[-2], keys.shape[-1])
            v_flat = vals.view(-1, vals.shape[-2], vals.shape[-1])
            timers.stop('reshape-1')

            timers.start('qk-matmul-2')
            attn_w = G.gather_outer_bmv_optimized(
                q.reshape(-1,1,head_dim),
                k_flat.transpose(-1,-2),
                idx
            ) / math.sqrt(head_dim)
            timers.stop('qk-matmul-2')

            timers.start('softmax')
            attn_w = torch.softmax(attn_w.float(), dim=-1).to(dtype)
            timers.stop('softmax')

            timers.start('sv-matmul')
            out = G.gather_inner_matrix_only_bmv_optimized(attn_w, v_flat, idx)
            timers.stop('sv-matmul')

            timers.start('reshape-output')
            out = out.view(num_heads, bs, 1, head_dim).transpose(0,1).transpose(1,2).contiguous()
            timers.stop('reshape-output')

    timers.stop('total')


# -----------------------------------------------------------------------------
# Vanilla attention micro-benchmark (for comparison)
# -----------------------------------------------------------------------------
def micro_bench_actual_attention(
    cache: PcaTopKCache,
    prompt_keys: List[torch.Tensor],
    num_layers: int,
    timers: Timers,
    num_gen_steps: int = 2000
):
    torch.set_float32_matmul_precision("highest")
    head_dim = prompt_keys[0].shape[-1]
    bs, num_heads = prompt_keys[0].shape[0], prompt_keys[0].shape[1]
    dtype = prompt_keys[0].dtype

    sparse_proj = SparseAttentionProj(head_dim).to("cuda")
    sparse_proj.apply_semi_structured_sparsity()

    timers.start('total')
    for _ in range(num_gen_steps):
        for layer in range(num_layers):
            timers.start('qk-gen')
            inp = torch.rand(bs, num_heads, 1, head_dim, device='cuda', dtype=dtype)
            q, k, _ = sparse_proj(inp)
            timers.stop('qk-gen')

            timers.start('cache-update')
            keys, vals = cache.update(k, k, q, layer, False)
            timers.stop('cache-update')

            timers.start('qk-matmul-1')
            attn_w = torch.matmul(q, keys.transpose(2,3)) / math.sqrt(head_dim)
            timers.stop('qk-matmul-1')

            timers.start('softmax')
            attn_w = torch.softmax(attn_w.float(), dim=-1).to(dtype)
            timers.stop('softmax')

            timers.start('sv-matmul')
            out = torch.matmul(attn_w, vals)
            timers.stop('sv-matmul')

            timers.start('reshape-output')
            out = out.transpose(1,2).contiguous()
            timers.stop('reshape-output')

    timers.stop('total')


# -----------------------------------------------------------------------------
# Benchmark entry point
# -----------------------------------------------------------------------------
@torch.no_grad()
def benchmark_attention(
    batch_size: int = 1,
    num_heads: int = 32,
    num_gen_steps: int = 128,
    prompt_length: int = 3072,
    topk: int = 256,
    topr: int = 32,
    num_layers: int = 32,
    dtype: torch.dtype = torch.float16,
    vanilla: bool = True,
    pcatopk: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    head_dim = 128
    prompt_keys = [
        torch.rand(batch_size, num_heads, prompt_length, head_dim, device='cuda', dtype=dtype)
        for _ in range(num_layers)
    ]

    times_pca = None
    if pcatopk:
        print("=== PCA-TOPK SEMI-STRUCTURED SPARSE BENCHMARK ===")
        for _ in range(10):
            cache = PcaTopKCache()
            for i in range(num_layers):
                cache.update(
                    prompt_keys[i].transpose(0,1).contiguous(),
                    prompt_keys[i].transpose(0,1).contiguous(),
                    prompt_keys[i].transpose(0,1).contiguous(),
                    i
                )
            timers = Timers()
            micro_benchmark_pca_topk(
                cache, prompt_keys, topr, topk, num_layers, timers,
                num_gen_steps=num_gen_steps, use_optimised_gather=True
            )
            times_pca = timers.get_times()
            del cache
        print("PCA-TOPK times:", times_pca)

    times_vanilla = None
    if vanilla:
        print("=== VANILLA ATTENTION BENCHMARK ===")
        for _ in range(10):
            cache = PcaTopKCache()
            for i in range(num_layers):
                cache.update(
                    prompt_keys[i], prompt_keys[i], prompt_keys[i], i
                )
            timers = Timers()
            micro_bench_actual_attention(
                cache, prompt_keys, num_layers, timers, num_gen_steps=num_gen_steps
            )
            times_vanilla = timers.get_times()
            del cache
        print("Vanilla times:", times_vanilla)

    return times_pca, times_vanilla


# -----------------------------------------------------------------------------
# Example run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    benchmark_attention()
