import os
import pickle
import time
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict


cell_line_path = "path/to/filtered_data_CVCL_xxx.pkl"


def fast_pickle_load(file_path, buffer_size=65536 * 4):  
    with open(file_path, 'rb', buffering=buffer_size) as f:
        filtered_dict = pickle.load(f)
    return filtered_dict


data_dict = fast_pickle_load(cell_line_path)

sample_metadata = load_dataset("tahoebio/Tahoe-100M","sample_metadata", revision="affe86a848ac896240aa75fe1a2b568051f3b850", split="train")

sample_to_metadata = {}
for meta_record in tqdm(sample_metadata, desc="indexing"):
    sample_id = meta_record['sample']
    sample_to_metadata[sample_id] = meta_record


perturbation_dict = defaultdict(int)

missing_samples = []

for idx in tqdm(list(data_dict.keys()), desc="frequency"):
    sample = data_dict[idx]
    sample_id = sample['sample']
    
    if sample_id in sample_to_metadata:
        drugname_drugconc = sample_to_metadata[sample_id]['drugname_drugconc']
        perturbation_dict[drugname_drugconc] += 1
    else:
        missing_samples.append(sample_id)

sorted_perturbations = sorted(perturbation_dict.items(), key=lambda x: x[1], reverse=True)

for drugname_drugconc, count in sorted_perturbations:
    print(f"{drugname_drugconc}: {count}")
