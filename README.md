# Sparse(r) Loki

## Compute Evaluation

Compute Benchmark for loki, sparse transformers + loki,
```bash
python evaluate_compute_sparse_transformers.py
```

Compute Benchmark for unstructured QK sparsity
```bash
python evaluate_compute_naive_sparse.py
```
 
Compute Benchmark for 2:4 structured sparsity
```bash
python evaluate_compute_apex24.py
```

## Downstream Performance
```bash
python evaluate_tasks.py
```
