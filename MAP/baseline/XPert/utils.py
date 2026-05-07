import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import scanpy as sc
import tqdm
from unimol_tools import UniMolRepr

from metrics import get_metrics_new, get_metrics_infer
from datasets.MyDataset import MyDataset



atom2id = {'N': 2,
         'Se': 3,
         'I': 4,
         'As': 5,
         'Pt': 6,
         'S': 7,
         'Sn': 8,
         'Mg': 9,
         'P': 10,
         'Cl': 11,
         'Au': 12,
         'F': 13,
         'Co': 14,
         'Br': 15,
         'Si': 16,
         'O': 17,
         'H': 18,
         'B': 19,
         'C': 20,
         'Ca': 21,
         'Hg': 22,
         'Li':23,
         "[UNK]":24,
         "Na":25,
         'hg_molecule': 0,
         'molecule': 1}

def generate_unimol_feat(smiles_dict, max_atoms=122):
    
    smiles_list = list(smiles_dict.values())
    clf = UniMolRepr(data_type='molecule', remove_hs=False)
    unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)

    assert len(unimol_repr['cls_repr']) == len(smiles_list), f"only {len(unimol_repr['cls_repr'])}/{len(smiles_list)} molecule's unimol repr is returned!"

    unimol_feat = {}
    idx = 0 
    for k in tqdm(list(smiles_dict.keys()), desc="Wrapping UniMol features"):
        molecule_feat = unimol_repr['cls_repr'][idx]
        atom_feat = unimol_repr['atomic_reprs'][idx]
        atom_symbols = unimol_repr['atomic_symbol'][idx]
        
        atom_feat = np.vstack((np.array([molecule_feat,molecule_feat]), atom_feat))
        atom_symbols = [*['hg_molecule','molecule'],*atom_symbols]
        atom_symbols_ids = [atom2id[atom] for atom in atom_symbols]

        l = len(atom_feat) 
        
        if l < max_atoms:
            atom_feat_pad = np.pad(atom_feat, ((0, max_atoms - l),(0,0)), 'constant', constant_values=0)
            atom_symbols_ids_pad = np.pad(atom_symbols_ids, (0, max_atoms - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (max_atoms - l))
            
        else:
            atom_feat_pad = atom_feat[:max_atoms]
            atom_symbols_ids_pad = atom_symbols_ids[:max_atoms]
            input_mask = [1] * max_atoms
          
        tmp = np.hstack((np.array(atom_symbols_ids_pad)[:,None], atom_feat_pad))
        unimol_feat[k] = np.hstack((np.array(input_mask)[:,None], tmp))
        idx += 1 

    return unimol_feat


def load_dataloader(args,config,logger,nfold, return_rawdata=False):

    batch_size = config['train']['batch_size']

    if args.dataset == 'tahoe' or args.dataset == 'nips' or args.dataset == 'sciplex':
        # Special handling - we don't read the anndata here
        data = None
        # Drug loading handled later when we create the adapter
    elif args.dataset == 'l1000_sdst':
        data = sc.read_h5ad(config['dataset']['l1000_sdst_data_root'])
    elif args.dataset == 'l1000_mdmt':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_data_root'])
    elif args.dataset == 'l1000_mdmt_pretrain':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_pretrain_data_root'])
    elif args.dataset == 'l1000_mdmt_full':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_full_data_root'])
    elif args.dataset == 'panacea_mdmt':
        data = sc.read_h5ad(config['dataset']['PANACEA_data_root'])
    elif args.dataset == 'cdsdb_mdmt':
        data = sc.read_h5ad(config['dataset']['CDSDB_data_root'])
    else:
        data = sc.read_h5ad(f'processed_data/{args.dataset}.h5ad')
        # assert False, 'Dataset not found!'


    if args.dataset == 'tahoe' or args.dataset == 'nips' or args.dataset == 'sciplex':
        # Already has drug_unimol_path in config, so just load it
        if args.drug_feat == 'unimol':
            if config['dataset']['drug_unimol_path'] is not None:
                drug_feat_loaded = np.load(config['dataset']['drug_unimol_path'], allow_pickle=True)
                # When np.save saves a dict, it becomes 0-d ndarray, need .item() to get dict
                if isinstance(drug_feat_loaded, np.ndarray) and drug_feat_loaded.size == 1:
                    drug_feat = drug_feat_loaded.item()
                else:
                    drug_feat = drug_feat_loaded
                max_atom_size = config['dataset']['max_atom_size']
                # If shape is (num_drugs, ...) convert to dict format
                if isinstance(drug_feat, np.ndarray) and len(drug_feat.shape) == 3:
                    temp_dict = {}
                    for idx in range(drug_feat.shape[0]):
                        temp_dict[idx] = drug_feat[idx]
                    drug_feat = temp_dict
                    if max_atom_size < 122:
                        for k in drug_feat:
                            drug_feat[k] = drug_feat[k][:, :max_atom_size, :]
                print(f'Using {max_atom_size} atom, total {len(drug_feat)} drugs')
        else:
            assert False, 'Only unimol drug_feat supported for this dataset'
    else:
        # Original drug loading for non-tahoe datasets
        if args.drug_feat == 'unimol':
            if config['dataset']['drug_unimol_path'] is not None:
                drug_feat = np.load(config['dataset']['drug_unimol_path'], allow_pickle=True)
                max_atom_size = config['dataset']['max_atom_size']
                if max_atom_size < 122:
                    drug_feat = drug_feat[:, :max_atom_size, :]
                    print(f'Using {max_atom_size} atom')
            else:
                drug_smis =  np.load(config['dataset']['drug_smi_path'], allow_pickle=True).item()
                drug_smis_remain = {k:v for k,v in drug_smis.items() if k in data.obs.pert_idx.unique()}
                print(f"Saved UniMol Feature File is not provide! Generating UniMol feature for {len(drug_smis_remain)}/{len(drug_smis)} molecules....")
                drug_feat = generate_unimol_feat(drug_smis_remain)
        elif args.drug_feat == 'KPGT':
            drug_feat = np.load(config['dataset']['drug_KPGT_path'], allow_pickle=True).item()
        elif args.drug_feat == 'smi':
            drug_feat = np.load(config['dataset']['drug_smi_path'], allow_pickle=True).item()
        elif args.drug_feat == 'morgan':
            drug_feat = np.load(config['dataset']['drug_morgan_path'], allow_pickle=True).item()
        else:
            assert False, 'Drug feature not found!'


        # Only split data for non-tahoe datasets that use anndata
        tr_data = data[data.obs[nfold] == 'train']
        val_data = data[data.obs[nfold] == 'valid']
        test_data = data[data.obs[nfold] == 'test']


        # for five-fold cross-validation
        if val_data.n_obs == 0:
            val_data = test_data

        print(f'train data:{tr_data.n_obs}')
        print(f'Valid data:{val_data.n_obs}')
        print(f'Test data:{test_data.n_obs}')

    if args.dataset == 'tahoe':
        # Add repo root to path so we can find ours_6
        import sys
        repo_root = '/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell'
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        # Use our adapter to match TahoePerturbDatasetSE to XPert interface
        from ours_6.data.ds_multi_cell_lora_se import TahoePerturbDatasetSE
        from dataset_adapter_tahoe_claude import TahoeXPertDatasetAdapter
        # We don't use the original XPert MyDataset
        # Our adapter wraps the existing Tahoe dataset and converts to XPert format
        # base_dir and se_inputs_base_dir need to be set correctly
        base_dir = '/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/ours_3/preprocessed/'
        se_inputs_base_dir = '/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/ours_4/preprocessed_se_inputs_memmap/'
        # Tahoe has 6 cell lines - list them all
        cell_lines = sorted(['CVCL_0023', 'CVCL_0069', 'CVCL_0131', 'CVCL_0480', 'CVCL_1056', 'CVCL_1098'])
        # Map XPert split names to original Tahoe split names:
        # XPert: train → original: train
        # XPert: valid → original: internal_test
        # XPert: test → original: external_test
        tahoe_tr = TahoePerturbDatasetSE(
            cell_lines=cell_lines,
            split='train',
            base_dir=base_dir,
            se_inputs_base_dir=se_inputs_base_dir,
            set_size=24,
            is_train=True,
            UC=args.UC,
        )
        Dataset = TahoeXPertDatasetAdapter
        max_value = config['dataset']['max_value']
        min_value = config['dataset']['min_value']
        print(f'For dataset {args.dataset}, wrapped by TahoeXPertDatasetAdapter')
        print(f'Train samples in original: {len(tahoe_tr)}, n_cells_lines: {len(cell_lines)}')
        tr_dataset = Dataset(tahoe_tr, drug_feat, n_bins=config['dataset']['n_bins'],
                               max_value=max_value, min_value=min_value)
        # validation - use internal_test for validation
        tahoe_val = TahoePerturbDatasetSE(
            cell_lines=cell_lines,
            split='internal_test',
            base_dir=base_dir,
            se_inputs_base_dir=se_inputs_base_dir,
            set_size=24,
            is_train=False,
            UC=args.UC,
        )
        val_dataset = Dataset(tahoe_val, drug_feat, n_bins=config['dataset']['n_bins'],
                               max_value=max_value, min_value=min_value)
        # test - use external_test
        tahoe_test = TahoePerturbDatasetSE(
            cell_lines=cell_lines,
            split='external_test',
            base_dir=base_dir,
            se_inputs_base_dir=se_inputs_base_dir,
            set_size=24,
            is_train=False,
            UC=args.UC,
        )
        test_dataset = Dataset(tahoe_test, drug_feat, n_bins=config['dataset']['n_bins'],
                                max_value=max_value, min_value=min_value)
    elif args.dataset in ['nips', 'sciplex']:
        # Add repo root to path so we can find ours_3
        import sys
        repo_root = '/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell'
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        # ours_3 is where the original datasets are
        ours_root = '/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/ours_3'
        if ours_root not in sys.path:
            sys.path.insert(0, ours_root)
        # Use our adapter to match NIPS/Sciplex to XPert interface
        from data.ds_nips import NIPSPerturbDataset
        from data.ds_sciplex import SciplexPerturbDataset
        from dataset_adapter_nips_sciplex_claude import NipsSciplexXPertDatasetAdapter
        # Get cell types based on dataset
        if args.dataset == 'nips':
            cell_types = sorted(['B cells', 'Myeloid cells', 'NK cells', 'T cells CD4+', 'T cells CD8+', 'T regulatory cells'])
            # NIPS: use train split for XPert training
            # Note: UC=False means we use the standard split (not unknown cell split)
            nips_tr = NIPSPerturbDataset(
                cell_types=cell_types,
                split="train",
                sequential=False,
                set_size=24,
                is_train=True,
                UC=False,
            )
            # Use test split for validation
            nips_val = NIPSPerturbDataset(
                cell_types=cell_types,
                split="test",
                sequential=False,
                set_size=24,
                is_train=False,
            )
            # Test dataset same as validation
            nips_test = NIPSPerturbDataset(
                cell_types=cell_types,
                split="test",
                sequential=False,
                set_size=24,
                is_train=False,
            )
        else:  # sciplex
            cell_types = sorted(['A549', 'K562', 'MCF7'])
            # Sciplex: use train split for XPert training
            sciplex_tr = SciplexPerturbDataset(
                cell_types=cell_types,
                split="train",
                sequential=False,
                set_size=24,
                is_train=True,
            )
            # Use test split for validation
            sciplex_val = SciplexPerturbDataset(
                cell_types=cell_types,
                split="test",
                sequential=False,
                set_size=24,
                is_train=False,
            )
            # Test dataset same as validation
            sciplex_test = SciplexPerturbDataset(
                cell_types=cell_types,
                split="test",
                sequential=False,
                set_size=24,
                is_train=False,
            )
        Dataset = NipsSciplexXPertDatasetAdapter
        max_value = config['dataset']['max_value']
        min_value = config['dataset']['min_value']
        print(f'For dataset {args.dataset}, wrapped by NipsSciplexXPertDatasetAdapter')
        if args.dataset == 'nips':
            print(f'Train samples in original: {len(nips_tr)}, n_cell_types: {len(cell_types)}')
            tr_dataset = Dataset(nips_tr, drug_feat, n_bins=config['dataset']['n_bins'],
                                   max_value=max_value, min_value=min_value)
            val_dataset = Dataset(nips_val, drug_feat, n_bins=config['dataset']['n_bins'],
                                   max_value=max_value, min_value=min_value)
            test_dataset = Dataset(nips_test, drug_feat, n_bins=config['dataset']['n_bins'],
                                    max_value=max_value, min_value=min_value)
        else:
            print(f'Train samples in original: {len(sciplex_tr)}, n_cell_types: {len(cell_types)}')
            tr_dataset = Dataset(sciplex_tr, drug_feat, n_bins=config['dataset']['n_bins'],
                                   max_value=max_value, min_value=min_value)
            val_dataset = Dataset(sciplex_val, drug_feat, n_bins=config['dataset']['n_bins'],
                                   max_value=max_value, min_value=min_value)
            test_dataset = Dataset(sciplex_test, drug_feat, n_bins=config['dataset']['n_bins'],
                                    max_value=max_value, min_value=min_value)
    else:
        if args.model in ['XPert', 'TranSiGen', 'DeepCE', 'PRnet', 'CIGER', 'MLP']:
            Dataset = MyDataset
        else:
            assert False, 'Dataset Class is not Assigned!'

        max_value = config['dataset']['max_value']
        min_value = config['dataset']['min_value']
        print(f'For dataset {args.dataset}, max_value:{max_value}, min_value:{min_value}')

        print("Starting load train data...")
        tr_dataset = Dataset(tr_data, drug_feat, args=args, config=config, logger=logger, max_value=max_value, min_value=min_value)
        print("Starting load valid data...")
        val_dataset = Dataset(val_data, drug_feat, args=args, config=config, logger=logger, max_value=max_value, min_value=min_value)
        print("Starting load test data...")
        test_dataset = Dataset(test_data, drug_feat, args=args, config=config, logger=logger, max_value=max_value, min_value=min_value)
    
    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10, drop_last=False)

    print(f'train dataloader:{len(tr_dataloader)}')  
    print(f'Valid dataloader:{len(val_dataloader)}')
    print(f'Test dataloader:{len(test_dataloader)}')

    if return_rawdata:
        return tr_dataloader, val_dataloader, test_dataloader, data
    else:
        return tr_dataloader, val_dataloader, test_dataloader


def load_test_dataloader(args,config,logger,nfold):

    batch_size = config['train']['batch_size']

    if args.dataset == 'l1000_sdst':
        data = sc.read_h5ad(config['dataset']['l1000_sdst_data_root'])
    elif args.dataset == 'l1000_mdmt':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_data_root'])
    elif args.dataset == 'l1000_mdmt_pretrain':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_pretrain_data_root'])
    elif args.dataset == 'l1000_mdmt_full':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_full_data_root'])
    elif args.dataset == 'panacea_mdmt':
        data = sc.read_h5ad(config['dataset']['PANACEA_data_root'])
    elif args.dataset == 'cdsdb_mdmt':
        data = sc.read_h5ad(config['dataset']['CDSDB_data_root'])
    else:
        data = sc.read_h5ad(f'processed_data/{args.dataset}.h5ad')
        # assert False, 'Dataset not found!'

    if args.drug_feat == 'unimol':
        if config['dataset']['drug_unimol_path'] is not None:
            drug_feat = np.load(config['dataset']['drug_unimol_path'], allow_pickle=True)
            max_atom_size = config['dataset']['max_atom_size']
            if max_atom_size < 122:
                drug_feat = drug_feat[:, :max_atom_size, :]
                print(f'Using {max_atom_size} atom')
        else:
            drug_smis =  np.load(config['dataset']['drug_smi_path'], allow_pickle=True).item()
            drug_smis_remain = {k:v for k,v in drug_smis.items() if k in data.obs.pert_idx.unique()}
            print(f"Saved UniMol Feature File is not provide! Generating UniMol feature for {len(drug_smis_remain)}/{len(drug_smis)} molecules....")
            drug_feat = generate_unimol_feat(drug_smis_remain)
    elif args.drug_feat == 'KPGT':
        drug_feat = np.load(config['dataset']['drug_KPGT_path'], allow_pickle=True).item()
    elif args.drug_feat == 'smi':
        drug_feat = np.load(config['dataset']['drug_smi_path'], allow_pickle=True).item()
    elif args.drug_feat == 'morgan':
        drug_feat = np.load(config['dataset']['drug_morgan_path'], allow_pickle=True).item()
    else:
        assert False, 'Drug feature not found!'


    test_data = data[data.obs[nfold] == 'test']

    print(f'Test data:{test_data.n_obs}')

    if args.model in ['XPert', 'TranSiGen', 'DeepCE', 'PRnet', 'CIGER', 'MLP']:
        Dataset = MyDataset
    else:
        assert False, 'Dataset Class is not Assigned!'

    max_value = config['dataset']['max_value']
    min_value = config['dataset']['min_value']
    print(f'For dataset {args.dataset}, max_value:{max_value}, min_value:{min_value}')

    print("Starting load test data...")
    test_dataset = Dataset(test_data, drug_feat, args=args, config=config, logger=logger, max_value=max_value, min_value=min_value)
    

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10, drop_last=False)

    print(f'Test dataloader:{len(test_dataloader)}')
    
    return test_dataloader



def load_infer_dataloader(args,config,logger,nfold):

    if args.output_attention:
        batch_size = 1
    else:
        batch_size = 256
    print(f'batch_size:{batch_size}')

    if args.dataset == 'l1000_sdst':
        data = sc.read_h5ad(config['dataset']['l1000_sdst_data_root'])
    elif args.dataset == 'l1000_mdmt':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_data_root'])
    elif args.dataset == 'l1000_mdmt_pretrain':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_pretrain_data_root'])
    elif args.dataset == 'l1000_mdmt_full':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_full_data_root'])
    elif args.dataset == 'panacea_mdmt':
        data = sc.read_h5ad(config['dataset']['PANACEA_data_root'])
    elif args.dataset == 'cdsdb_mdmt':
        data = sc.read_h5ad(config['dataset']['CDSDB_data_root'])
    else:
        data = sc.read_h5ad(f'processed_data/{args.dataset}.h5ad')
        # assert False, 'Dataset not found!'


    if args.drug_feat == 'unimol':
        if config['dataset']['drug_unimol_path'] is not None:
            drug_feat = np.load(config['dataset']['drug_unimol_path'], allow_pickle=True)
            max_atom_size = config['dataset']['max_atom_size']
            if max_atom_size < 122:
                drug_feat = drug_feat[:, :max_atom_size, :]
                print(f'Using {max_atom_size} atom')
        else:
            drug_smis =  np.load(config['dataset']['drug_smi_path'], allow_pickle=True).item()
            drug_smis_remain = {k:v for k,v in drug_smis.items() if k in data.obs.pert_idx.unique()}
            print(f"Saved UniMol Feature File is not provide! Generating UniMol feature for {len(drug_smis_remain)}/{len(drug_smis)} molecules....")
            drug_feat = generate_unimol_feat(drug_smis_remain)
    elif args.drug_feat == 'KPGT':
        drug_feat = np.load(config['dataset']['drug_KPGT_path'], allow_pickle=True).item()
    elif args.drug_feat == 'smi':
        drug_feat = np.load(config['dataset']['drug_smi_path'], allow_pickle=True).item()
    elif args.drug_feat == 'morgan':
        drug_feat = np.load(config['dataset']['drug_morgan_path'], allow_pickle=True).item()
    else:
        assert False, 'Drug feature not found!'

    
    # test_data = data[data.obs[nfold] == "infer" ]
    test_data = data

    print(f'Test data:{test_data.n_obs}')

    if args.model in ['XPert', 'TranSiGen', 'DeepCE', 'PRnet', 'CIGER', 'MLP']:
        Dataset = MyDataset
    else:
        assert False, 'Dataset Class is not Assigned!'


    max_value = config['dataset']['max_value']
    min_value = config['dataset']['min_value']
    print(f'For dataset {args.dataset}, max_value:{max_value}, min_value:{min_value}')

    print("Starting load infer data...")
    test_dataset = Dataset(test_data, drug_feat, args=args, config=config, logger=logger, max_value=max_value, min_value=min_value)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10, drop_last=False)

    print(f'Test dataloader:{len(test_dataloader)}')
    
    return test_dataloader




def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



def log_nested_dict(raw_dict, logger, indent=0):
    for key, value in raw_dict.items():
        if isinstance(value, dict):
            if logger == print:
                logger(f'{"    " * indent}{key}:')
            else:
                print(f'{"    " * indent}{key}:')
            log_nested_dict(value, logger, indent + 1)
        else:
            if logger == print:
                logger(f'{"    " * indent}{key}: {value}')
            else:
                print(f'{"    " * indent}{key}: {value}')


def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)


def mse_loss_ls_sum(pred, target):
    return torch.mean((pred - target) ** 2, dim=1).sum()



def pcc_loss_sum(pred, target):
    """
    Calculate the Pearson Correlation Coefficient (PCC) loss.
    
    Args:
        pred (torch.Tensor): Model predicted values, shape (batch_size, n_features).
        target (torch.Tensor): Ground truth values, shape (batch_size, n_features).
    
    Returns:
        torch.Tensor: Computed PCC loss for the batch.
    """
    # Ensure the inputs are 2D tensors
    assert pred.ndim == 2 and target.ndim == 2, "Inputs must be 2D tensors."
    
    # Flatten along the batch dimension to compute correlation
    pred_mean = torch.mean(pred, dim=1, keepdim=True)
    target_mean = torch.mean(target, dim=1, keepdim=True)

    # Center the data
    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    # Compute numerator and denominator for PCC
    numerator = torch.sum(pred_centered * target_centered, dim=1)
    denominator = torch.sqrt(torch.sum(pred_centered ** 2, dim=1) * torch.sum(target_centered ** 2, dim=1) + 1e-6)

    # Compute PCC for each sample in the batch
    pcc = numerator / denominator

    # Loss is 1 - PCC, so maximizing PCC minimizes the loss
    loss = 1 - pcc

    # Return the mean loss across the batch
    return torch.sum(loss)




def cos_loss_sum(pred, target):
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
        
    cos = nn.CosineSimilarity(dim=-1)
    cos_loss = 1 - cos(pred, target)
    return cos_loss.sum() 




def train(model, opt, dataloader, args, config, scaler=None, epoch=0):
    model.train()
    total_loss = 0.
    total_loss1 = 0.
    total_loss2 = 0.
    total_loss3 = 0.
    total_loss4 = 0.
    num_batches = 0


    a, b = config['train']['loss_weight']
    for data in dataloader:

        model.zero_grad()

        if scaler:
            with autocast():
                output_deg, trt_raw_data, ctl_raw_data, _ = model(data)
                num_samples =  trt_raw_data.shape[0]
                num_batches += num_samples
                loss1 = mse_loss_ls_sum(output_deg, trt_raw_data-ctl_raw_data)
                loss2 = cos_loss_sum(output_deg, trt_raw_data-ctl_raw_data)
                weighted_loss = loss1 * a + loss2 * b
                batch_weighted_loss = torch.sqrt(loss1/num_samples) * a + (loss2/num_samples) * b
                total_loss += weighted_loss.item()
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()

            # scaler.scale(weighted_loss).backward()
            if epoch < 70:
                scaler.scale(batch_weighted_loss).backward()
            else:
                scaler.scale(weighted_loss).backward()
            
            scaler.step(opt)
            scaler.update()

        else:
            output_deg, trt_raw_data, ctl_raw_data, _ = model(data)
            num_samples =  trt_raw_data.shape[0]
            num_batches += num_samples
            loss1 = mse_loss_ls_sum(output_deg, trt_raw_data-ctl_raw_data)
            loss2 = cos_loss_sum(output_deg, trt_raw_data-ctl_raw_data)
            weighted_loss = loss1 * a + loss2 * b
            batch_weighted_loss = torch.sqrt(loss1/num_samples) * a + (loss2/num_samples) * b
            total_loss += weighted_loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()

            if epoch < 70:
                batch_weighted_loss.backward()
            else:
                weighted_loss.backward()
            
            opt.step()


    avg_total_loss = total_loss / num_batches
    avg_loss1 = np.sqrt(total_loss1 / num_batches) if total_loss1 else -999
    # avg_loss1 = total_loss1 / num_batches if total_loss1 else -999
    avg_loss2 = total_loss2 / num_batches if total_loss2 else -999
    avg_loss3 = total_loss3 / num_batches if total_loss3 else -999
    avg_loss4 = total_loss4 / num_batches if total_loss4 else -999

    return avg_total_loss, avg_loss1, avg_loss2, avg_loss3, avg_loss4




def validate(model, dataloader, args, config, flag=False, scaler=None):
    model.eval()  
    total_loss = 0.0
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss3 = 0.0
    total_loss4 = 0.0
    num_batches = 0
    trt_raw_data_ls = []
    output2_ls = []
    ctl_raw_data_ls = []
    

    with torch.no_grad():  
        for data in dataloader:
            if scaler:
                with autocast():
                    output_deg, trt_raw_data, ctl_raw_data, _ = model(data)
                    num_batches += trt_raw_data.shape[0] 
                    loss1 = mse_loss_ls_sum(output_deg, trt_raw_data-ctl_raw_data)
                    loss2 = cos_loss_sum(output_deg, trt_raw_data-ctl_raw_data)
                    total_loss1 += loss1.item()
                    total_loss2 += loss2.item()
            else:
                output_deg, trt_raw_data, ctl_raw_data, _  = model(data)
                num_batches += trt_raw_data.shape[0] 
                loss1 = mse_loss_ls_sum(output_deg, trt_raw_data-ctl_raw_data)
                loss2 = cos_loss_sum(output_deg, trt_raw_data-ctl_raw_data)
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()                        

            if flag:
                trt_raw_data_ls.append(trt_raw_data)
                output2_ls.append(output_deg+ctl_raw_data)
                ctl_raw_data_ls.append(ctl_raw_data)


    if flag:
        y_true = torch.cat(trt_raw_data_ls, dim=0).cpu().detach().numpy()
        y_pred = torch.cat(output2_ls, dim=0).cpu().detach().numpy()
        ctl_true = torch.cat(ctl_raw_data_ls, dim=0).cpu().detach().numpy()
        metrics = get_metrics_new(y_true, y_pred, ctl_true)
    else:
        metrics = None

    avg_total_loss = total_loss / num_batches
    avg_loss1 = np.sqrt(total_loss1 / num_batches) if total_loss1 else -999
    avg_loss2 = total_loss2 / num_batches if total_loss2 else -999
    avg_loss3 = total_loss3 / num_batches if total_loss3 else -999
    avg_loss4 = total_loss4 / num_batches if total_loss4 else -999

    return avg_total_loss, avg_loss1, avg_loss2, avg_loss3, avg_loss4, metrics





def infer(model, dataloader, args, config, flag=True, output_profile=False, scaler=None):
    model.eval()  
    trt_raw_data_ls = []
    output2_ls = []
    ctl_raw_data_ls = []
    pert_id_ls = []
    cell_id_ls = []
    pert_dose_ls = []
    pert_time_ls = []
    all_attention_dict_ls= {}


    with torch.no_grad():  

        for data in dataloader:
            trt_raw_data, ctl_raw_data, trt_raw_data_binned, ctl_raw_data_binned,drug_feat, pert_dose_idx, pert_time_idx, drug_idx, cell_idx, tissue_idx, pert_id, cell_id, pert_dose, pert_time = data
            # feat_data = trt_raw_data, ctl_raw_data, trt_raw_data_binned, ctl_raw_data_binned,drug_feat, pert_dose_idx, pert_time_idx, drug_idx, cell_idx, tissue_idx  

            if scaler:
                with autocast():
                    output_deg, _, _, attention_dict, _, _ = model((trt_raw_data, ctl_raw_data, trt_raw_data_binned, ctl_raw_data_binned,drug_feat, pert_dose_idx, pert_time_idx, drug_idx, cell_idx, tissue_idx))
            else:
                output_deg, _, _, attention_dict, _, _ = model((trt_raw_data, ctl_raw_data, trt_raw_data_binned, ctl_raw_data_binned,drug_feat, pert_dose_idx, pert_time_idx, drug_idx, cell_idx, tissue_idx))                     

            if flag:
                trt_raw_data_ls.append(trt_raw_data)
                output2_ls.append(output_deg.cpu()+ctl_raw_data)
                ctl_raw_data_ls.append(ctl_raw_data)
                if args.output_attention:
                    for key, value in attention_dict.items():
                        if key not in all_attention_dict_ls:
                            all_attention_dict_ls[key] = []
                        if value is None:
                            print(f'value of {key} is None!')
                        all_attention_dict_ls[key].append(value)

                pert_id = list(drug_idx)
                cell_id = list(cell_id)
                pert_dose = list(pert_dose)
                pert_time = list(pert_time)

                pert_id_ls.extend(pert_id)
                cell_id_ls.extend(cell_id)
                pert_dose_ls.extend(pert_dose)
                pert_time_ls.extend(pert_time)


        if flag:
            y_true = torch.cat(trt_raw_data_ls, dim=0).cpu().detach().numpy()
            y_pred = torch.cat(output2_ls, dim=0).cpu().detach().numpy()
            ctl_true = torch.cat(ctl_raw_data_ls, dim=0).cpu().detach().numpy()
            metrics, metrics_all_ls = get_metrics_infer(y_true, y_pred, ctl_true)

            metrics_all_ls['pert_id'] = np.array(pert_id_ls)
            metrics_all_ls['cell_id'] = np.array(cell_id_ls)
            metrics_all_ls['pert_dose'] = np.array(pert_dose_ls)
            metrics_all_ls['pert_time'] = np.array(pert_time_ls)
        else:
            metrics = None
            metrics_all_ls = None

        if args.output_attention:
            all_attention_dict = {}
            for key, value in all_attention_dict_ls.items():
                all_attention_dict[key] = torch.cat(value, dim=0).cpu().detach().numpy()
        else:
           all_attention_dict = None 
        
        output_pred_profile = y_pred if output_profile else None

    return metrics, metrics_all_ls, output_pred_profile, all_attention_dict





class EarlyStopping():
    """
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.
    """

    def __init__(self, mode='higher', patience=None, metric=None, n_fold=None, folder=None, logger=None):

        filepath = os.path.join(folder, '{}_fold_early_stop.pth'.format(n_fold))

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score', 'mse'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score' or 'mse', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse', 'mse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filepath = filepath
        self.best_score = None
        self.early_stop = False
        self.logger = logger

    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model, current_epoch, optimizer):
        """Update based on a new score.
        The new score is typically model performance on the validation set
        for a new epoch.
        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.
        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, current_epoch, optimizer)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model, current_epoch, optimizer)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model, current_epoch, optimizer):
        '''Saves model when the metric on the validation set gets improved.
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
        }, self.filepath)

    def load_checkpoint(self, model, optimizer):
        '''Load the latest checkpoint
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        checkpoint = torch.load(self.filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model




def save_loss_fig(tr_loss_list,val_loss_list,n_fold,folder,logger):
    x = range(0, len(tr_loss_list))
    y1 = tr_loss_list
    y2 = val_loss_list
    plt.figure(dpi=150)  
    # plt.style.use('seaborn-bright')
    plt.plot(x, y1, '.-', label='Train loss', color="b")
    plt.plot(x, y2, '.-', label='Valid loss', color="g")
    plt.legend(loc='best',edgecolor='white')
    plt.xlabel('Epoches')
    plt.ylabel('Train/Valid loss')
    # plt.xticks(np.arange(0,len(tr_loss_list),25))
    
    plt.savefig(folder + f'/{n_fold}_fold_train_loss.png')

    print('Loss figure saved!')



def set_requires_grad(model, finetune_params="all"):
    """
    Set requires_grad for model parameters.

    Args:
        model: The model containing parameters to update.
        finetune_params (list or all): List of parameter name fragments to finetune.
                                         If all, all parameters are set to requires_grad=True.
    """
    # enable all parameters
    if finetune_params == "all":
        pass
    else:
        # Otherwise, set requires_grad based on finetune_params
        for name, param in model.named_parameters():
            if any(module in name for module in finetune_params):
                param.requires_grad = True
            else:
                param.requires_grad = False
