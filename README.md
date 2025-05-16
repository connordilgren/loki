# Sparse(r) Loki

Note: This repo is a fork of Loki (https://github.com/hpcgroup/loki)


# Setup

On Zaratan, load the following modules:
```bash
module load cuda/12.3.0/gcc/11.3.0/x86_64
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-x86_64
```

Create a virtual env and activate it:
```bash
python -m venv .venv
source .venv/bin/activate
```

The 2:4 Fine-Grained Structured Sparsity Method requires the apex library. To install:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
```

Then, comment out "check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)" in apex/setup.py
```bash
⁠python setup.py install --cpp_ext --cuda_ext
```

Install other requirements:
```bash
pip install -r requirements.txt
```

To setup the PCA transforms and tensors, please copy `/afs/shell.umd.edu/project/cmsc828/shared/pchiniya/cache/pca` to directory under loki `pca` and
`/afs/shell.umd.edu/project/cmsc828/shared/pchiniya/cache/saved_tensors` to  `saved_tensors`.


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
Our experiment runs are saved in `compute_files_apex24`, `compute_files_sparse_transformers`, `compute_files_naive_sparse` and `compute_files_loki`. 
