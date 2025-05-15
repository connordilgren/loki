import math
import time
import torch
import torch.nn as nn
import sys 
sys.path.append('/ephemeral/purva_exp/loki/apex')
from apex.contrib.sparsity import ASP  # make sure Apex is installed

# -----------------------------
# Dense Attention Module
# -----------------------------
class DenseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(DenseAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Define the query, key, value, and output projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        # Linear projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention: split embed_dim into heads
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, V)
        
        # Concatenate heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        return self.out_proj(attn_output)

# -----------------------------
# Sparse Attention Projection Module
# -----------------------------
class SparseAttentionProj(nn.Module):
    """
    This module contains linear projections for queries, keys, and values.  
    The apply_asp() method uses NVIDIA Apex to apply 2:4 sparsity on these projections.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self._asp_initialized = False

    def apply_asp(self):
        if self._asp_initialized:
            return
        # Dummy optimizer required by ASP
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        ASP.prune_trained_model(self, optimizer)
        self._asp_initialized = True

    def forward(self, x):
        # x shape: (batch_size, seq_length, hidden_dim)
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

# -----------------------------
# Sparse Attention Module
# -----------------------------
class SparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SparseAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Use the sparse projection module for Q, K, V
        self.sparse_proj = SparseAttentionProj(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._asp_initialized = False

    def apply_asp(self):
        if not self._asp_initialized:
            self.sparse_proj.apply_asp()
            self._asp_initialized = True

    def forward(self, x):
        # Optionally initialize sparsity if not already done
        if not self._asp_initialized:
            self.apply_asp()
        
        batch_size, seq_length, embed_dim = x.size()
        # Get sparse projections for Q, K, V
        Q, K, V = self.sparse_proj(x)
        
        # Reshape to multi-head format
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        
        # Scaled dot-product attention computation
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, V)
        
        # Concatenate heads and run through output projection
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, seq_length, embed_dim)
        return self.out_proj(attn_output)

# -----------------------------
# Benchmarking Function
# -----------------------------
def benchmark_model(model, input_tensor, num_iterations=100):
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model(input_tensor)
        torch.cuda.synchronize()
    return (time.time() - start_time) / num_iterations

# -----------------------------
# Main Benchmark Script
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Benchmark parameters
    batch_size = 64
    seq_length = 128  # sequence length (number of tokens)
    embed_dim = 512   # embedding dimension
    num_heads = 8     # number of attention heads
    
    # Create a random input tensor
    x = torch.randn(batch_size, seq_length, embed_dim, device=device)
    
    # Instantiate Dense and Sparse Attention models
    dense_attn = DenseAttention(embed_dim, num_heads).to(device)
    sparse_attn = SparseAttention(embed_dim, num_heads).to(device)
    
    # Warmup iterations
    for _ in range(10):
        dense_attn(x)
        sparse_attn(x)
    
    # Benchmark the dense attention model
    dense_time = benchmark_model(dense_attn, x, num_iterations=100)
    print(f"Dense Attention: {dense_time*1000:.3f} ms per iteration")
    
    # Benchmark the sparse attention model
    sparse_time = benchmark_model(sparse_attn, x, num_iterations=100)
    print(f"Sparse Attention (with 2:4 sparsity): {sparse_time*1000:.3f} ms per iteration")
    
    speedup = dense_time / sparse_time if sparse_time > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")
