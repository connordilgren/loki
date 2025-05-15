from methods.pca_topk.attention_benchmark_torch24 import benchmark_attention
import json
import torch
import os 

import sys
sys.path.append('/ephemeral/purva_exp/loki/apex')
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


if __name__ == "__main__":
    os.makedirs("compute_files_apex24", exist_ok=True)
    
    with torch.no_grad():
        for prompt_length in [512, 1024, 2048]:
            for num_gen_steps in [64, 128, 256]:
                # does not depend on topk, topr, stride
                _, times_vanilla = benchmark_attention(prompt_length=prompt_length, num_gen_steps=num_gen_steps, batch_size=16, pcatopk=False)
                with open(f"compute_files_apex24/vanilla_prompt_{prompt_length}_gen_{num_gen_steps}.json", "w") as f:
                    json.dump(times_vanilla, f, indent=2)

                for topk in [4, 8]:
                    for topr in [4]:
                        times_pca_topk, _ = benchmark_attention(prompt_length=prompt_length, num_gen_steps=num_gen_steps, batch_size=16, topk=prompt_length // topk, topr=128 // topr, vanilla=False)
                        with open(f"compute_files_apex24/24_transformer_loki_prompt_{prompt_length}_gen_{num_gen_steps}_topk_{topk}_topr_{topr}.json", "w") as f:
                            json.dump(times_pca_topk, f, indent=2)