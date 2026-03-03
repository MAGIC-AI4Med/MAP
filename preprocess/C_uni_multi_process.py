import pickle
import time
import random
import re
import os
from collections import defaultdict
import json
from datasets import load_dataset
from tqdm import tqdm
import ast
from multiprocessing import Pool, cpu_count

def fast_pickle_load(file_path, buffer_size=65536 * 4):
    with open(file_path, 'rb', buffering=buffer_size) as f:
        filtered_dict = pickle.load(f)

    return filtered_dict

def parse_drug_combinations(txt_file):
    combinations = []
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                match = re.search(r"\('([^']+)', ([\d.]+), '([^']+)'\)]: (\d+)", line)
                if match:
                    drug_name = match.group(1)
                    concentration = float(match.group(2))
                    unit = match.group(3)
                    frequency = int(match.group(4))
                    
                    if drug_name != 'DMSO_TF' and frequency >= 500:
                        combinations.append((drug_name, concentration, unit, frequency))
    
    return combinations

def split_train_test_sets(pkl_file, txt_file, cell_line):
    PREDEFINED_EXTERNAL_DRUGS = {
        "BI-78D3",
        "Balsalazide (sodium hydrate)",
        "Bergenin",
        "Bortezomib",
        "Brivudine",
        "CP21R7",
        "Carbidopa (monohydrate)",
        "Ciclopirox",
        "Drospirenone",
        "ERK5-IN-2",
        "Estrone sulfate (potassium)",
        "Idarubicin (hydrochloride)",
        "Lidocaine (hydrochloride)",
        "Nafamostat (mesylate)",
        "Ralimetinib dimesylate",
        "Sildenafil",
        "Sivelestat (sodium tetrahydrate)",
        "ULK-101"
    }
    
    combinations = parse_drug_combinations(txt_file)
    
    unique_drugs = list(set([drug for drug, conc, unit, freq in combinations]))
    
    external_drugs = PREDEFINED_EXTERNAL_DRUGS & set(unique_drugs)
    internal_drugs = set(unique_drugs) - external_drugs
    
    missing_drugs = PREDEFINED_EXTERNAL_DRUGS - set(unique_drugs)
    
    external_combinations = [(drug, conc, unit, freq) for drug, conc, unit, freq in combinations if drug in external_drugs]
    internal_combinations = [(drug, conc, unit, freq) for drug, conc, unit, freq in combinations if drug in internal_drugs]
    
    external_set = {(drug, conc, unit) for drug, conc, unit, _ in external_combinations}
    internal_set = {(drug, conc, unit) for drug, conc, unit, _ in internal_combinations}
    
    data_dict = fast_pickle_load(pkl_file)
    
    sample_metadata = load_dataset("tahoebio/Tahoe-100M", "sample_metadata", 
                                  revision="", split="train")
    
    sample_to_metadata = {}
    for meta_record in tqdm(sample_metadata, desc="indexing"):
        sample_id = meta_record['sample']
        sample_to_metadata[sample_id] = meta_record
    
    external_combination_cells = defaultdict(list)
    internal_combination_cells = defaultdict(list)
    control_cells = []
    
    for idx, cell_data in tqdm(data_dict.items(), desc="grouping cells"):
        sample_id = cell_data['sample']
        
        if sample_id in sample_to_metadata:
            drugname_drugconc_str = sample_to_metadata[sample_id]['drugname_drugconc']
            drugname_drugconc = ast.literal_eval(drugname_drugconc_str)
            
            assert len(drugname_drugconc) == 1
            drug_tuple = drugname_drugconc[0]
            
            assert len(drug_tuple) == 3
            drug_name = drug_tuple[0]
            concentration = drug_tuple[1]
            unit = drug_tuple[2]
            combo = (drug_name, concentration, unit)
            
            if drug_name == 'DMSO_TF':
                control_cells.append(idx)
            elif combo in external_set:
                external_combination_cells[combo].append(idx)
            elif combo in internal_set:
                internal_combination_cells[combo].append(idx)

    
    external_test_indices = []
    train_indices = []
    internal_test_indices = []
    
    for combo, cell_indices in external_combination_cells.items():
        if len(cell_indices) > 0:
            sampled_cells = random.sample(cell_indices, min(5000, len(cell_indices)))
            external_test_indices.extend(sampled_cells)
    
    for combo, cell_indices in internal_combination_cells.items():
        if len(cell_indices) > 0:
            sampled_cells = random.sample(cell_indices, min(5000, len(cell_indices)))
            split_point = int(len(sampled_cells) * 0.8)
            train_indices.extend(sampled_cells[:split_point])
            internal_test_indices.extend(sampled_cells[split_point:])
    
    sampled_control = random.sample(control_cells, min(5000, len(control_cells)))
    
    save_dir = f"path/to/preprocessed/{cell_line}/"
    os.makedirs(save_dir, exist_ok=True)

    with open(save_dir+'train_indices.json', 'w') as f:
        json.dump(train_indices, f)
    
    with open(save_dir+'internal_test_indices.json', 'w') as f:
        json.dump(internal_test_indices, f)
    
    with open(save_dir+'external_test_indices.json', 'w') as f:
        json.dump(external_test_indices, f)
    
    with open(save_dir+'control_indices.json', 'w') as f:
        json.dump(sampled_control, f)
    
    split_info = {
        'train_size': len(train_indices),
        'internal_test_size': len(internal_test_indices),
        'external_test_size': len(external_test_indices),
        'control_size': len(sampled_control),
        'external_drugs': sorted(list(external_drugs)),
        'internal_drugs': sorted(list(internal_drugs)),
        'external_combinations': [(drug, conc, unit) for drug, conc, unit, _ in external_combinations],
        'internal_combinations': [(drug, conc, unit) for drug, conc, unit, _ in internal_combinations],
        'total_drugs': len(unique_drugs),
        'total_combinations': len(combinations),
        'predefined_external_drugs': sorted(list(PREDEFINED_EXTERNAL_DRUGS)),
        'missing_external_drugs': sorted(list(missing_drugs))
    }
    
    with open(save_dir+'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    return cell_line


def process_cell_line(cell_line):
    seed = 42 + hash(cell_line) % 1000
    random.seed(seed)
    
    try:
        pkl_file = f"path/to/filtered_data_{cell_line}.pkl"
        txt_file = f"path/to/B_{cell_line.split('_')[-1]}.txt"
        
        result = split_train_test_sets(pkl_file, txt_file, cell_line)
        return (True, result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (False, cell_line)


if __name__ == "__main__":
    random.seed(42)
    
    CELL_LINES = ["CVCL_1098", "CVCL_1056", "CVCL_0131", "CVCL_0069", "CVCL_0218"]
    
    num_processes = min(len(CELL_LINES), cpu_count())
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_cell_line, CELL_LINES)