import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# --- Stacked Bar Chart Generation ---

def get_label(run_type):
    if run_type == 'vanilla':
        return 'V'
    elif run_type == 'loki':
        return 'L'
    elif run_type == 'sparse_transformer_loki':
        return 'L+ST'
    else:
        return run_type


cols_compare_all = [
    ('sparse_transformer_loki',  512,  64, 4.0, 4.0, 128.0),
    ('sparse_transformer_loki',  512,  64, 8.0, 4.0, 128.0),
    ('sparse_transformer_loki', 1024, 128, 4.0, 4.0, 512.0),
    ('sparse_transformer_loki', 1024, 128, 8.0, 4.0, 512.0),
    ('sparse_transformer_loki', 2048, 256, 4.0, 4.0, 512.0),
    ('sparse_transformer_loki', 2048, 256, 8.0, 4.0, 512.0),
]

# Find all groups (each group: one loki, one sparse_transformer_loki, one vanilla)
groups = {}
for col in df.columns:
    run_type, prompt_length, gen_steps, topk, topr, stride = col
    if run_type == 'sparse_transformer_loki':
        group_key = (prompt_length, gen_steps, topk, topr, stride)
        if col not in cols_compare_all:
            continue        
        groups.setdefault(group_key, {})['sparse_transformer_loki'] = col
        # Find matching loki (ignore stride)
        for c in df.columns:
            if c[0] == 'loki' and c[1] == prompt_length and c[2] == gen_steps and c[3] == topk and c[4] == topr:
                groups[group_key]['loki'] = c
        # Find matching vanilla (ignore topk, topr, stride)
        for c in df.columns:
            if c[0] == 'vanilla' and c[1] == prompt_length and c[2] == gen_steps:
                groups[group_key]['vanilla'] = c

# Define the attention breakdown keys (excluding KV-cache update and total)
attn_keys = ['qk-gen', 'project', 'fixed-keys', 'qk-matmul-1', 'top-k', 'qk-matmul-2', 'softmax', 'sv-matmul']

# For legend and color/hatch
color_map = {
    'fixed-keys': '#D55E00',
    'project': '#009E73',
    'qk-gen': '#0072B2',
    'qk-matmul-1': '#CC79A7',
    'qk-matmul-2': '#000000',
    'softmax': '#F0E442',
    'sv-matmul': '#56B4E9',
    'top-k': '#009E73',
}
hatch_map = {
    'fixed-keys': 'xxx',
    'project': '////',
    'qk-gen': '...',
    'qk-matmul-1': '++',
    'qk-matmul-2': 'xx',
    'softmax': '**',
    'sv-matmul': '||',
    'top-k': '...',
}

# --- Combined Group Plot ---

# Gather all group data for plotting
plot_groups = [g for g in groups.values() if all(k in g for k in ['loki', 'sparse_transformer_loki', 'vanilla'])]

bar_data = {k: [] for k in attn_keys}
labels = []
settings_labels = []
group_positions = []
bar_width = 0.22
intra_group_gap = 0.03
inter_group_gap = 0.35
x_pos = 0

for group in plot_groups:
    group_key = group['sparse_transformer_loki'][1:]
    # group_key: (prompt_length, gen_steps, topk, topr, stride)
    group_settings = f"{group_key[0]}, {group_key[1]}, {group_key[2]}, {group_key[3]}, {group_key[4]}"
    for i, run_type in enumerate(['vanilla', 'loki', 'sparse_transformer_loki']):
        col = group[run_type]
        labels.append(get_label(run_type))
        settings_labels.append(group_settings)
        group_positions.append(x_pos)
        for k in attn_keys:
            v = df[col][k]
            bar_data[k].append(v if not np.isnan(v) else 0)
        x_pos += bar_width + intra_group_gap
    # Add extra space after each group
    x_pos += inter_group_gap - intra_group_gap

fig, ax = plt.subplots(figsize=(max(8, 0.5*len(labels)), 6), constrained_layout=True)
bottom = np.zeros(len(labels))

for k in attn_keys:
    color = color_map.get(k, None)
    hatch = hatch_map.get(k, None)
    ax.bar(group_positions, bar_data[k], width=bar_width, label=k, bottom=bottom, color=color, hatch=hatch, edgecolor='black', align='center')
    bottom += np.array(bar_data[k])

# X-tick labels: only V/L/L+ST for each bar
xtick_labels = labels
ax.set_xticks(group_positions)
ax.set_xticklabels(xtick_labels, rotation=0, fontsize=10, ha='center')

# Add value labels
for i, total in enumerate(bottom):
    ax.text(group_positions[i], total, f'{total:.2f}', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Time per Layer (s)')
ax.set_xlabel('')
ax.set_title('Attention Breakdown (No KV-cache update)')
# Move legend outside to the right using fig.legend and constrained_layout
# fig.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
ax.legend(loc='upper left')
plt.savefig('attention_breakdown_all_groups.png')
plt.close()

# --- Loki Only Stacked Bar Chart ---

# Select all columns where run_type == 'loki'
loki_cols = [col for col in df.columns if col[0] == 'loki']

# Sort for consistent display
loki_cols = sorted(loki_cols)

# Remove 'fixed-keys' from attention keys for loki plot
loki_attn_keys = [k for k in attn_keys if k != 'fixed-keys']

loki_bar_data = {k: [] for k in loki_attn_keys}
loki_labels = []
loki_positions = []
loki_bar_width = 0.35
loki_gap = 0.15
loki_x_pos = 0

for col in loki_cols:
    # Only include prompt_length, gen_steps, 1/topk%, 1/topr% in the label
    topk = col[3]
    topr = col[4]
    topk_pct = f"{100/int(topk):.0f}%" if topk and int(topk) != 0 else "-"
    topr_pct = f"{100/int(topr):.0f}%" if topr and int(topr) != 0 else "-"
    loki_labels.append(f"{col[1]}, {col[2]}, {topk_pct}, {topr_pct}")
    loki_positions.append(loki_x_pos)
    for k in loki_attn_keys:
        v = df[col][k]
        loki_bar_data[k].append(v if not np.isnan(v) else 0)
    loki_x_pos += loki_bar_width + loki_gap

fig, ax = plt.subplots(figsize=(max(8, 0.5*len(loki_labels)), 6), constrained_layout=True)
loki_bottom = np.zeros(len(loki_labels))

for k in loki_attn_keys:
    color = color_map.get(k, None)
    hatch = hatch_map.get(k, None)
    ax.bar(loki_positions, loki_bar_data[k], width=loki_bar_width, label=k, bottom=loki_bottom, color=color, hatch=hatch, edgecolor='black', align='center')
    loki_bottom += np.array(loki_bar_data[k])

ax.set_xticks(loki_positions)
ax.set_xticklabels(loki_labels, rotation=30, fontsize=10, ha='right')

for i, total in enumerate(loki_bottom):
    ax.text(loki_positions[i], total, f'{total:.2f}', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Time per Layer (s)')
ax.set_xlabel('')
ax.set_title('Attention Breakdown (Loki Only, No KV-cache update)')
ax.legend(loc='upper left')
plt.savefig('attention_breakdown_loki_only.png')
plt.close()
