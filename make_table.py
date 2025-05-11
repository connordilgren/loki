import os
import re
import json
import pandas as pd

# Directory containing the JSON files
COMPUTE_DIR = 'compute_files'

# Regex patterns for extracting properties from filenames
PATTERNS = {
    'loki': re.compile(r"loki_prompt_(\d+)_gen_(\d+)_topk_(\d+)_topr_(\d+)\.json"),
    'sparse_transformer_loki': re.compile(r"sparse_transformer_loki_prompt_(\d+)_gen_(\d+)_topk_(\d+)_topr_(\d+)_stride_(\d+)\.json"),
    'vanilla': re.compile(r"vanilla_prompt_(\d+)_gen_(\d+)\.json"),
}

# Collect all JSON files in the directory (not subdirectories)
files = [f for f in os.listdir(COMPUTE_DIR) if f.endswith('.json') and os.path.isfile(os.path.join(COMPUTE_DIR, f))]

# Store all data for DataFrame construction
data = {}
columns = []
all_keys = set()

for fname in files:
    run_type = None
    props = {}
    if fname.startswith('loki_'):
        m = PATTERNS['loki'].match(fname)
        if m:
            run_type = 'loki'
            props = {
                'run_type': run_type,
                'prompt_length': int(m.group(1)),
                'gen_steps': int(m.group(2)),
                'topk': int(m.group(3)),
                'topr': int(m.group(4)),
                'stride': None
            }
    elif fname.startswith('sparse_transformer_loki_'):
        m = PATTERNS['sparse_transformer_loki'].match(fname)
        if m:
            run_type = 'sparse_transformer_loki'
            props = {
                'run_type': run_type,
                'prompt_length': int(m.group(1)),
                'gen_steps': int(m.group(2)),
                'topk': int(m.group(3)),
                'topr': int(m.group(4)),
                'stride': int(m.group(5))
            }
    elif fname.startswith('vanilla_'):
        m = PATTERNS['vanilla'].match(fname)
        if m:
            run_type = 'vanilla'
            props = {
                'run_type': run_type,
                'prompt_length': int(m.group(1)),
                'gen_steps': int(m.group(2)),
                'topk': None,
                'topr': None,
                'stride': None
            }
    if not run_type:
        continue  # skip files that don't match
    # Read the JSON file
    with open(os.path.join(COMPUTE_DIR, fname), 'r') as f:
        jdata = json.load(f)
    # Add all keys to the set
    all_keys.update(jdata.keys())
    # Store the data with a tuple of properties as the column key
    col = (
        props['run_type'],
        props['prompt_length'],
        props['gen_steps'],
        props['topk'],
        props['topr'],
        props['stride']
    )
    columns.append(col)
    data[col] = jdata

# Sort columns for nice display
columns = sorted(set(columns))
all_keys = sorted(all_keys)

# Build the DataFrame
rows = []
for key in all_keys:
    row = []
    for col in columns:
        val = data.get(col, {}).get(key, None)
        row.append(val)
    rows.append(row)

multi_index = pd.MultiIndex.from_tuples(
    columns,
    names=['run_type', 'prompt_length', 'gen_steps', 'topk', 'topr', 'stride']
)
df = pd.DataFrame(rows, index=all_keys, columns=multi_index)

# Print and save the DataFrame
print(df)
df.to_csv('comparison_table.csv')
