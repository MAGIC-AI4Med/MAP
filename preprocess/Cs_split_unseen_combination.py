import pickle
import time
import random
import re
import os
from collections import defaultdict
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import ast

def fast_pickle_load(file_path, buffer_size=65536 * 4):
    with open(file_path, 'rb', buffering=buffer_size) as f:
        data = pickle.load(f)
    return data

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


def collect_all_drugs_per_cell_line(cell_lines):
    cell_line_drugs = {}
    
    for cell_line in cell_lines:
        txt_file = f"path/to/B_{cell_line.split('_')[-1]}.txt"
        
        combinations = parse_drug_combinations(txt_file)
        unique_drugs = set([drug for drug, conc, unit, freq in combinations])
        
        cell_line_drugs[cell_line] = unique_drugs
    
    return cell_line_drugs


def allocate_unseen_drugs(cell_line_drugs, unseen_ratio=0.05, seed=42):
    random.seed(seed)
    
    sorted_cell_lines = sorted(cell_line_drugs.items(), key=lambda x: len(x[1]))
    
    allocated_unseen = {}
    global_used_drugs = set()
    
    for cell_line, drugs in sorted_cell_lines:
        available_drugs = drugs - global_used_drugs
        
        num_to_select = max(1, int(len(drugs) * unseen_ratio))
        
        if len(available_drugs) < num_to_select:
            num_to_select = len(available_drugs)
        
        selected_drugs = set(random.sample(list(available_drugs), num_to_select))
        allocated_unseen[cell_line] = selected_drugs
        
        global_used_drugs.update(selected_drugs)
    
    all_pairs = [(cl1, cl2) for i, cl1 in enumerate(allocated_unseen.keys()) 
                 for cl2 in list(allocated_unseen.keys())[i+1:]]
    
    has_intersection = False
    for cl1, cl2 in all_pairs:
        intersection = allocated_unseen[cl1] & allocated_unseen[cl2]
        if intersection:
            has_intersection = True
    
    if has_intersection:
        raise ValueError("Unseen drugs分配存在交集，请检查！")
    
    return allocated_unseen

def split_train_test_sets(meta_file, txt_file, cell_line, external_drugs):
    combinations = parse_drug_combinations(txt_file)
    
    unique_drugs = list(set([drug for drug, conc, unit, freq in combinations]))
    
    external_drugs_set = external_drugs & set(unique_drugs)
    internal_drugs = set(unique_drugs) - external_drugs_set

    external_combinations = [(drug, conc, unit, freq) for drug, conc, unit, freq in combinations if drug in external_drugs_set]
    internal_combinations = [(drug, conc, unit, freq) for drug, conc, unit, freq in combinations if drug in internal_drugs]
    
    external_set = {(drug, conc, unit) for drug, conc, unit, _ in external_combinations}
    internal_set = {(drug, conc, unit) for drug, conc, unit, _ in internal_combinations}
    
    meta_data = fast_pickle_load(meta_file)
    
    external_combination_cells = defaultdict(list)
    internal_combination_cells = defaultdict(list)
    control_cells = []
    
    ds_level_indices = meta_data['ds_level_index']
    
    for idx, drug_conc_str in enumerate(tqdm(meta_data['drug_conc'], desc="grouping cells")):
        try:
            drug_conc_list = ast.literal_eval(drug_conc_str)
        except:
            continue
        
        if not isinstance(drug_conc_list, list) or len(drug_conc_list) == 0:
            continue
        
        drug_tuple = drug_conc_list[0]
        
        if not isinstance(drug_tuple, tuple) or len(drug_tuple) != 3:
            continue
        
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
    
    train_ds_indices = [ds_level_indices[i] for i in train_indices]
    internal_test_ds_indices = [ds_level_indices[i] for i in internal_test_indices]
    external_test_ds_indices = [ds_level_indices[i] for i in external_test_indices]
    control_ds_indices = [ds_level_indices[i] for i in sampled_control]

    save_dir = f"path/to/preprocessed/{cell_line}/"
    os.makedirs(save_dir, exist_ok=True)

    with open(save_dir+'train_indices_UC.json', 'w') as f:
        json.dump(train_ds_indices, f)
    
    with open(save_dir+'internal_test_indices_UC.json', 'w') as f:
        json.dump(internal_test_ds_indices, f)
    
    with open(save_dir+'external_test_indices_UC.json', 'w') as f:
        json.dump(external_test_ds_indices, f)
    
    with open(save_dir+'control_indices_UC.json', 'w') as f:
        json.dump(control_ds_indices, f)
    
    split_info = {
        'train_size': len(train_ds_indices),
        'internal_test_size': len(internal_test_ds_indices),
        'external_test_size': len(external_test_ds_indices),
        'control_size': len(control_ds_indices),
        'external_drugs': sorted(list(external_drugs_set)),
        'internal_drugs': sorted(list(internal_drugs)),
        'external_combinations': [(drug, conc, unit) for drug, conc, unit, _ in external_combinations],
        'internal_combinations': [(drug, conc, unit) for drug, conc, unit, _ in internal_combinations],
        'total_drugs': len(unique_drugs),
        'total_combinations': len(combinations),
        'unseen_drug_ratio': len(external_drugs_set) / len(unique_drugs) if len(unique_drugs) > 0 else 0,
        'index_type': 'ds_level_index',
        'note': 'All indices are ds_level_index from meta.pkl, not row indices'
    }
    
    with open(save_dir+'split_info_UC.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    return cell_line

def process_cell_line(args):
    cell_line, external_drugs = args
    
    seed = 42 + hash(cell_line) % 1000
    random.seed(seed)
    
    try:
        meta_file = f"path/to/preprocessed/{cell_line}/{cell_line}_meta.pkl"
        txt_file = f"path/to/preprocess_pipeline/logs/B_{cell_line.split('_')[-1]}.txt"
        
        result = split_train_test_sets(meta_file, txt_file, cell_line, external_drugs)
        return (True, result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (False, cell_line)


if __name__ == "__main__":
    random.seed(42)
    CELL_LINES = ["CVCL_1098", "CVCL_1056", "CVCL_0131", "CVCL_0069", "CVCL_0023", "CVCL_0480"]
    
    cell_line_drugs = collect_all_drugs_per_cell_line(CELL_LINES)
    
    allocated_unseen_drugs = allocate_unseen_drugs(cell_line_drugs, unseen_ratio=0.05, seed=42)
    
    allocation_summary = {
        cell_line: {
            'total_drugs': len(cell_line_drugs[cell_line]),
            'unseen_drugs': sorted(list(allocated_unseen_drugs[cell_line])),
            'unseen_count': len(allocated_unseen_drugs[cell_line]),
            'unseen_ratio': len(allocated_unseen_drugs[cell_line]) / len(cell_line_drugs[cell_line])
        }
        for cell_line in CELL_LINES
    }
    
    os.makedirs("path/to/preprocessed/", exist_ok=True)
    with open("path/to/preprocessed/unseen_drugs_allocation.json", 'w') as f:
        json.dump(allocation_summary, f, indent=2)
    
    num_processes = min(len(CELL_LINES), cpu_count())
    
    process_args = [(cell_line, allocated_unseen_drugs[cell_line]) for cell_line in CELL_LINES]
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_cell_line, process_args)