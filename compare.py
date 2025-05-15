import json
import sys

def get_pred(item):
    # try both possible fields
    txt = item.get("outputs", item.get("resp", ""))
    return txt.splitlines()[0].strip().lower()

def get_tgt(item):
    return item["target"][0].strip().lower()

def compare_exact_match(file1, file2):
    with open(file1) as f1, open(file2) as f2:
        run1 = json.load(f1)
        run2 = json.load(f2)
    count = 0 
    res = []

    for i in range(len(run1["gsm8k"])):
        
        if run1["gsm8k"][i]["exact_match"] ==1 and  run2["gsm8k"][i]["exact_match"]==0:
            count +=1
            res.append((run1["gsm8k"][i]["doc"], run1["gsm8k"][i]["resps"], run2["gsm8k"][i]["resps"]))
    print(count)
    import ipdb; ipdb.set_trace()
        
if __name__ == "__main__":
    
    compare_exact_match("/ephemeral/purva_exp/loki/output_loki_sparse.json", "/ephemeral/purva_exp/loki/output_loki.json")
   
