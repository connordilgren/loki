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
To run the downstream task on apex 2:4, setup lm-eval-harness version 0.4.8 and run 
```bash
bash run_lm_eval.sh
```
Our experiment runs are saved in `compute_files_apex24'`, compute_files_sparse_transformers, compute_files_naive_sparse and compute_files_loki. 
