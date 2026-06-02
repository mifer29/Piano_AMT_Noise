'''
Check token imbalance in the training set, especially for time tokens. 
This is important because if some time bins are never used, the model might not learn to use them, which could lead to worse generalization on unseen data with different timing patterns.
'''


from collections import Counter
from torch.utils.data import DataLoader
from maestro_dataset import MAESTROSeq2SeqDataset
import os
from dotenv import load_dotenv
load_dotenv()

# Dataset
dataset = MAESTROSeq2SeqDataset(
    hdf5_path=os.getenv("hdf5_path"),
    split="train",
    max_output_tokens=1024,
    logger=None,
    use_noisy=True,
    p_clean=0.5
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# Count tokens
counter = Counter()
N = 2000  # enough

for i, batch in enumerate(loader):
    tokens = batch["target_ids"][0].tolist()
    tokens = [t for t in tokens if t != dataset.pad_id]  # remove PAD
    counter.update(tokens)

    if i >= N:
        break

# Reverse vocab
id_to_token = {v: k for k, v in dataset.token_to_id.items()}

# Extract time bins
time_bins_used = set()

for token_id in counter.keys():
    token_str = id_to_token[token_id]

    if token_str.startswith("time_"):
        t = int(token_str.split("_")[1])
        time_bins_used.add(t)

# Stats
time_bins_used = sorted(time_bins_used)

print("TIME TOKEN USAGE")
print("Min bin:", min(time_bins_used))
print("Max bin:", max(time_bins_used))
print("Used bins:", len(time_bins_used))
print("Total bins:", dataset.max_time_bins)
print("Unused ratio:", (dataset.max_time_bins - len(time_bins_used)) / dataset.max_time_bins)