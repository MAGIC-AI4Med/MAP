import os
import math
import pickle
from multiprocessing import Pool
from data.ds_tahoe_se import Tahoe100mDataset
from tqdm import tqdm


global_data = None

def init_worker(data):
    global global_data
    global_data = data

def process_chunk(args):
    start_index, end_index, target_value = args
    result = {}
    
    for i in tqdm(range(start_index, end_index)):
        if i < len(global_data):
            item = global_data[i]
            if item.get('cell_line_id') == target_value:
                result[i] = item
    return result

def parallel_filter(large_list, target_value, num_processes=4):
    chunk_size = math.ceil(len(large_list) / num_processes)
    chunks = []
    
    for i in range(0, len(large_list), chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, len(large_list))
        chunks.append((start_idx, end_idx, target_value))
    
    with Pool(num_processes, initializer=init_worker, initargs=(large_list,)) as pool:
        results = pool.map(process_chunk, chunks)
    
    filtered_dict = {}
    for result in results:
        filtered_dict.update(result)

    with open(f'A/filtered_data_{target_value}.pkl', 'wb') as f:
        pickle.dump(filtered_dict, f)

def main():
    cpu_count = os.cpu_count()

    ds = Tahoe100mDataset()
    target_cell_line = 'chosen cell line id'

    num_processes = 26
    parallel_filter(ds.d_lst, target_cell_line, num_processes=num_processes)

if __name__ == '__main__':
    main()