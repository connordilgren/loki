from methods.pca_topk.attention_benchmark import benchmark_attention
import json
import torch

import sys
sys.path.append('/ephemeral/purva_exp/loki/apex')
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

if __name__ == "__main__":
    with torch.no_grad():
        for prompt_length in [512]:
        # for prompt_length in [2048, 3072]:
            # for num_gen_steps in [128, 256, 512]:
            for num_gen_steps in [128]:
                # does not depend on topk, topr, stride
                _, times_vanilla, _ = benchmark_attention(prompt_length=prompt_length, num_gen_steps=num_gen_steps, batch_size=16, pcatopk=False, sparse_transformer=False)
                # with open(f"compute_files/vanilla_prompt_{prompt_length}_gen_{num_gen_steps}.json", "w") as f:
                #     json.dump(times_vanilla, f, indent=2)

                # for topk in [4, 8]:
                # # for topk in [8]:
                #     for topr in [4]:
                #         # does not depend on stride
                #         print(f"prompt length = {prompt_length}, gen length = {num_gen_steps}, batch_size={16}, topk={topk} and topr={topr}")
                #         times_pca_topk, _, _ = benchmark_attention(prompt_length=prompt_length, num_gen_steps=num_gen_steps, batch_size=16, topk=prompt_length // topk, topr=128 // topr, vanilla=False, sparse_transformer=False)
                #         with open(f"compute_files/loki_prompt_{prompt_length}_gen_{num_gen_steps}_topk_{topk}_topr_{topr}.json", "w") as f:
                #             json.dump(times_pca_topk, f, indent=2)

                #         for stride in [512]:
                #             print(f"prompt length = {prompt_length}, gen length = {num_gen_steps}, batch_size={16}, topk={topk} and topr={topr}, stride={stride}")
                #             _, _, times_sparse_pca_topk = benchmark_attention(prompt_length=prompt_length, num_gen_steps=num_gen_steps, batch_size=16, topk=prompt_length // topk, topr=128 // topr, stride=stride, pcatopk=False, vanilla=False)
                #             with open(f"compute_files/sparse_transformer_loki_prompt_{prompt_length}_gen_{num_gen_steps}_topk_{topk}_topr_{topr}_stride_{stride}.json", "w") as f:
                #                 json.dump(times_sparse_pca_topk, f, indent=2)
