import json
import torch
import os
import gc
import sys
import traceback

import sys

from methods.pca_topk.attention_benchmark import benchmark_attention
from methods.pca_topk.attention_benchmark_apex import benchmark_attention_apex

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

os.environ["OMP_NUM_THREADS"] = "1"  # Set OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"  # Set MKL threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # Set numexpr threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Set OpenBLAS threads

def free_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / (1024**3):.2f} GB used, "
          f"{torch.cuda.memory_reserved() / (1024**3):.2f} GB reserved")

if __name__ == "__main__":
    os.makedirs("compute_files", exist_ok=True)

    with torch.no_grad():
        # for prompt_length in [512, 1024, 2048]:
        for prompt_length in [512]:
            # for num_gen_steps in [64, 128, 256]:
            for num_gen_steps in [64]:
                # Vanilla attention
                # does not depend on topk, topr, stride
                free_gpu_memory()
                _, times_vanilla, _ = benchmark_attention(prompt_length=prompt_length, num_gen_steps=num_gen_steps, batch_size=16, pcatopk=False, sparse_transformer=False)
                with open(f"compute_files/vanilla_prompt_{prompt_length}_gen_{num_gen_steps}.json", "w") as f:
                    json.dump(times_vanilla, f, indent=2)

                # for topk in [4, 8]:
                for topk in [4]:
                    for topr in [4]:
                        # Loki
                        # does not depend on stride
                        print(f"prompt length = {prompt_length}, gen length = {num_gen_steps}, batch_size=16, topk={topk} and topr={topr}")
                        free_gpu_memory()
                        times_pca_topk, _, _ = benchmark_attention(prompt_length=prompt_length, num_gen_steps=num_gen_steps, batch_size=16, topk=prompt_length // topk, topr=128 // topr, vanilla=False, sparse_transformer=False)
                        with open(f"compute_files/loki_prompt_{prompt_length}_gen_{num_gen_steps}_topk_{topk}_topr_{topr}.json", "w") as f:
                            json.dump(times_pca_topk, f, indent=2)
                        
                        # Loki with attention-query-key sparsity
                        # print(f"\nRunning Loki with Attention-Query-Key Sparsity benchmark...")
                        free_gpu_memory()
                        times_sparsity, _ = benchmark_attention_apex(
                            prompt_length=prompt_length, 
                            num_gen_steps=num_gen_steps, 
                            batch_size=16, 
                            topk=prompt_length // topk, 
                            topr=128 // topr, 
                            vanilla=False,
                            pcatopk=True,
                            sparsity_type="attention-query-key",
                            dtype=torch.float16
                        )
                        sparsity_filename = f"compute_files/attention_query_key_prompt_{prompt_length}_gen_{num_gen_steps}_topk_{topk}_topr_{topr}.json"
                        print(f"Saving to {sparsity_filename}")
                        with open(sparsity_filename, "w") as f:
                            json.dump(times_sparsity, f, indent=2)
                        print("Loki with Attention-Query-Key Sparsity Times:")
                        for key, value in times_sparsity.items():
                            print(f"  {key}: {value:.6f} s")
                        print(f"Net time (minus cache updates): {times_sparsity.get('total', 0) - times_sparsity.get('cache-update', 0):.6f} s")

                        # for stride in [128, 512]:
                        for stride in [128]:
                            print(f"prompt length = {prompt_length}, gen length = {num_gen_steps}, batch_size={16}, topk={topk} and topr={topr}, stride={stride}")
                            free_gpu_memory()
                            _, _, times_sparse_pca_topk = benchmark_attention(prompt_length=prompt_length, num_gen_steps=num_gen_steps, batch_size=16, topk=prompt_length // topk, topr=128 // topr, stride=stride, pcatopk=False, vanilla=False)
                            with open(f"compute_files/sparse_transformer_loki_prompt_{prompt_length}_gen_{num_gen_steps}_topk_{topk}_topr_{topr}_stride_{stride}.json", "w") as f:
                                json.dump(times_sparse_pca_topk, f, indent=2)
