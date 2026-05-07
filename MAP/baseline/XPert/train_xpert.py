import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
sys.path.append(os.getcwd())
os.chdir(os.getcwd())

import torch
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import argparse
import time
import os.path as osp
import logging
import yaml
import pickle
from tqdm import tqdm

from utils import EarlyStopping, save_loss_fig, set_random_seed, log_nested_dict, load_dataloader, load_test_dataloader, load_infer_dataloader
from utils import mse_loss_ls_sum, pcc_loss_sum, cos_loss_sum, set_requires_grad
from metrics import get_metrics_new, get_metrics_infer
from models.model_XPert import XPertNet


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='train or valid od test')
    parser.add_argument('--nfold', default='split', help='split, split_cold_drug, split_cold_cell')
    parser.add_argument('--drug_feat', default='unimol', help='unimol, KPGT, morgan')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model', default='XPert', help='XPert')
    parser.add_argument('--config', default='config',help='the filename of config file')
    parser.add_argument('--seed', default=2024)
    parser.add_argument('--dataset', default='transigen_sdst', help='transigen_sdst, l1000_sdst, l1000_mdmt, l1000_sdst_rna')
    parser.add_argument('--pretrained_mode', default='global', help='global, specific')
    parser.add_argument('--UC', action='store_true')
    
    # abalation study
    parser.add_argument('--include_cell_idx', type=bool, default=False)
    parser.add_argument('--wo_HG', type=bool, default=False)
    parser.add_argument('--wo_atom', type=bool, default=False)
    parser.add_argument('--wo_atom_HG', type=bool, default=False)
    parser.add_argument('--wo_unimol', type=bool, default=False)
    parser.add_argument('--wo_ppi', type=bool, default=False)
    parser.add_argument('--use_gene_pos_emed', type=bool, default=False)

    parser.add_argument('--use_gradscaler', type=bool, default=False)
    parser.add_argument('--lr_scheduler', type=bool, default=True)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--saved_model_path', type=str, default=None)
    parser.add_argument('--saved_model', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--output_profile', type=bool, default=False)
    parser.add_argument('--output_attention', type=bool, default=False)
    parser.add_argument('--output_cls_embed', type=bool, default=False)

    parser.add_argument('--weighted_loss', type=bool, default=False)
    parser.add_argument('--kl_loss', type=bool, default=False)

    # Progress monitoring options
    parser.add_argument('--valid_per_epoch', type=int, default=20,
                        help='Number of validation checks per epoch (default: 20)')
    parser.add_argument('--log_freq', type=int, default=40,
                        help='Log training loss every N steps (default: 100 = 1/100 epoch)')

    return parser.parse_args()


    return avg_total_loss, avg_loss1, avg_loss2, avg_loss3, avg_loss4





def validate(model, dataloader, args, config, flag=False, scaler=None, expt_folder=None, fold=None):
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
    deg_pred_ls = []
    
    a, b, c, d = config['train']['loss_weight']
    MultiClassLoss = torch.nn.CrossEntropyLoss(reduction='sum')

    print('Validating...')

    with torch.no_grad():  
        for i, data in enumerate(tqdm(dataloader)):
            if i >= 100:
                break
            if scaler:
                with autocast():
                    trt_output, ctl_output, deg_output, trt_raw_data, ctl_raw_data, _,cell_class_true, cell_class_predict = model(data)
                    num_samples =  trt_raw_data.shape[0]
                    num_batches += num_samples
                    loss1 = mse_loss_ls_sum(trt_output, trt_raw_data)
                    if cell_class_predict is not None:
                        cell_class_predict_1, cell_class_predict_2 = cell_class_predict
                        loss2 = MultiClassLoss(cell_class_predict_1, cell_class_true) + MultiClassLoss(cell_class_predict_2, cell_class_true)
                    else:
                        loss2 = mse_loss_ls_sum(ctl_output, ctl_raw_data)
                    loss3 = mse_loss_ls_sum(deg_output, trt_raw_data-ctl_raw_data)
                    loss4 = pcc_loss_sum(deg_output, trt_raw_data-ctl_raw_data)
                    weighted_loss = loss1 * a + loss2 * b + loss3 * c + loss4 * d
                    total_loss += weighted_loss.item()
                    total_loss1 += loss1.item()
                    total_loss2 += loss2.item()
                    total_loss3 += loss3.item()
                    total_loss4 += loss4.item()
            else:
                trt_output, ctl_output, deg_output, trt_raw_data, ctl_raw_data, _, cell_class_true, cell_class_predict = model(data)
                num_samples =  trt_raw_data.shape[0]
                num_batches += num_samples
                loss1 = mse_loss_ls_sum(trt_output, trt_raw_data)
                loss2 = mse_loss_ls_sum(ctl_output, ctl_raw_data)
                loss3 = mse_loss_ls_sum(deg_output, trt_raw_data-ctl_raw_data)
                loss4 = pcc_loss_sum(deg_output, trt_raw_data-ctl_raw_data)
                weighted_loss = loss1 * a + loss2 * b + loss3 * c + loss4 * d
                total_loss += weighted_loss.item()
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()
                total_loss3 += loss3.item()
                total_loss4 += loss4.item()                        

            if flag:
                trt_raw_data_ls.append(trt_raw_data)
                output2_ls.append(deg_output + ctl_raw_data)
                ctl_raw_data_ls.append(ctl_raw_data)
                deg_pred_ls.append(deg_output)


    if flag:
        y_true = torch.cat(trt_raw_data_ls, dim=0).cpu().detach().numpy()
        y_pred = torch.cat(output2_ls, dim=0).cpu().detach().numpy()
        ctl_true = torch.cat(ctl_raw_data_ls, dim=0).cpu().detach().numpy()
        deg_pred = torch.cat(deg_pred_ls, dim=0).cpu().detach().numpy()
        metrics = get_metrics_new(y_true, y_pred, ctl_true)
        if args.output_profile:
            print('Saving output_profile....')
            output_profile_dict = {}
            output_profile_dict['y_true'] = y_true
            output_profile_dict['y_pred'] = y_pred
            output_profile_dict['ctl_true'] = ctl_true
            output_profile_dict['deg_true'] = y_true-ctl_true
            output_profile_dict['deg_pred'] = y_pred-ctl_true
            np.save(f'{expt_folder}/{fold}_predict_profile.npy', output_profile_dict)
    else:
        metrics = None

    avg_total_loss = total_loss / num_batches
    avg_loss1 = np.sqrt(total_loss1 / num_batches) if total_loss1 else -999
    # avg_loss1 = total_loss1 / num_batches if total_loss1 else -999
    if cell_class_predict is not None:
        avg_loss2 = (total_loss2 / num_batches) if total_loss2 else -999
    else:
        avg_loss2 = np.sqrt(total_loss2 / num_batches) if total_loss2 else -999
    avg_loss3 = np.sqrt(total_loss3 / num_batches) if total_loss3 else -999
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
    trt_attention_dict_ls= {}
    ctl_attention_dict_ls= {}
    cls_embed_ls = []


    with torch.no_grad():  

        for data in dataloader:
            trt_raw_data, ctl_raw_data, trt_raw_data_binned, ctl_raw_data_binned, drug_feat, pert_dose_idx, pert_time_idx, drug_idx, cell_idx, tissue_idx, pert_id, cell_id, pert_dose, pert_time = data
            # feat_data = trt_raw_data, ctl_raw_data, trt_raw_data_binned, ctl_raw_data_binned,drug_feat, pert_dose_idx, pert_time_idx, drug_idx, cell_idx, tissue_idx  

            if scaler:
                with autocast():
                    trt_output, ctl_output, output_deg, _, _, attention_dict, _, _, cls_embed= model((trt_raw_data, ctl_raw_data, trt_raw_data_binned, ctl_raw_data_binned,drug_feat, pert_dose_idx, pert_time_idx, drug_idx, cell_idx, tissue_idx))
            else:
                trt_output, ctl_output, output_deg, _, _, attention_dict, _, _, cls_embed = model((trt_raw_data, ctl_raw_data, trt_raw_data_binned, ctl_raw_data_binned,drug_feat, pert_dose_idx, pert_time_idx, drug_idx, cell_idx, tissue_idx))                     

            if flag:
                trt_raw_data_ls.append(trt_raw_data)
                output2_ls.append(output_deg.cpu()+ctl_raw_data)
                ctl_raw_data_ls.append(ctl_raw_data)
                if args.output_attention:
                    trt_attention_dict, ctl_attention_dict = attention_dict
                    for key, value in trt_attention_dict.items():
                        if key not in trt_attention_dict_ls:
                            trt_attention_dict_ls[key] = []
                        if value is None:
                            print(f'value of {key} is None!')
                        trt_attention_dict_ls[key].append(value)
                    for key, value in ctl_attention_dict.items():
                        if key not in ctl_attention_dict_ls:
                            ctl_attention_dict_ls[key] = []
                        if value is None:
                            print(f'value of {key} is None!')
                        ctl_attention_dict_ls[key].append(value) 
                if args.output_cls_embed:
                    cls_embed_ls.append(cls_embed)


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

        all_attention_dict = None
        if args.output_attention:
            all_attention_dict = {}
            all_trt_attention_dict = {}
            all_ctl_attention_dict = {}
            for key, value in trt_attention_dict_ls.items():
                all_trt_attention_dict[key] = torch.cat(value, dim=0).cpu().detach().numpy()
            for key, value in ctl_attention_dict_ls.items():
                all_ctl_attention_dict[key] = torch.cat(value, dim=0).cpu().detach().numpy()
            all_attention_dict['trt'] = all_trt_attention_dict
            all_attention_dict['ctl'] = all_ctl_attention_dict 

        all_cls_embed = None
        if args.output_cls_embed:
            all_cls_embed = torch.cat(cls_embed_ls, dim=0).cpu().detach().numpy()

        
        if output_profile:
            output_pred_profile = {}
            output_pred_profile['y_true'] = y_true
            output_pred_profile['y_pred'] = y_pred
            output_pred_profile['ctl_true'] = ctl_true
            output_pred_profile['deg_true'] = y_true-ctl_true
            output_pred_profile['deg_pred'] = y_pred-ctl_true
        else:
            output_pred_profile = None

    return metrics, metrics_all_ls, output_pred_profile, all_attention_dict, all_cls_embed


def train(model, opt, dataloader, args, config, scaler=None, epoch=0, loss_func=None, kl_loss=None,
          val_dataloader=None, stopper=None, expt_folder=None, logger=None):
    model.train()
    total_loss = 0.
    total_loss1 = 0.
    total_loss2 = 0.
    total_loss3 = 0.
    total_loss4 = 0.
    num_batches = 0

    # print('cp3')

    mse_loss_func = mse_loss_ls_sum
    MultiClassLoss = torch.nn.CrossEntropyLoss(reduction='sum')
    a, b, c, d = config['train']['loss_weight']

    if args.pretrained_model:
        init_epoch = config['finetune']['init_epoch']
    else:
        init_epoch = config['train']['init_epoch']

    # Calculate validation frequency and logging frequency
    total_steps = len(dataloader)
    valid_interval = max(1, total_steps // args.valid_per_epoch)
    log_interval = args.log_freq  # log every N steps, controlled by command line arg

    from tqdm import tqdm
    for step, data in enumerate(tqdm(dataloader)):

        model.zero_grad()
        # print('cp4')

        if scaler:
            with autocast():
                trt_output, ctl_output, deg_output, trt_raw_data, ctl_raw_data, _, cell_class_true, cell_class_predict = model(data)
                deg_true = trt_raw_data-ctl_raw_data
                num_samples =  trt_raw_data.shape[0]
                num_batches += num_samples
                loss1 = mse_loss_ls_sum(trt_output, trt_raw_data)
                if cell_class_predict is not None:
                    cell_class_predict_1, cell_class_predict_2 = cell_class_predict
                    loss2 = MultiClassLoss(cell_class_predict_1, cell_class_true) + MultiClassLoss(cell_class_predict_2, cell_class_true)
                else:
                    loss2 = mse_loss_ls_sum(ctl_output, ctl_raw_data)

                loss3 = mse_loss_func(deg_output, deg_true)
                loss4 = pcc_loss_sum(deg_output, deg_true)

                weighted_loss = loss1 * a + loss2 * b + loss3 * c + loss4 * d
                if cell_class_predict is not None:
                    batch_weighted_loss = torch.sqrt(loss1/num_samples) * a + (loss2/num_samples) * b + torch.sqrt(loss3/num_samples) * c + (loss4/num_samples) * d
                else:
                    batch_weighted_loss = torch.sqrt(loss1/num_samples) * a + torch.sqrt(loss2/num_samples) * b + torch.sqrt(loss3/num_samples) * c + (loss4/num_samples) * d
                total_loss += weighted_loss.item()
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()
                total_loss3 += loss3.item()
                total_loss4 += loss4.item()

            if epoch < init_epoch:
                scaler.scale(batch_weighted_loss).backward()
            else:
                scaler.scale(weighted_loss).backward()

            scaler.step(opt)
            scaler.update()

        else:
            trt_output, ctl_output, deg_output, trt_raw_data, ctl_raw_data, _ = model(data)
            num_samples =  trt_raw_data.shape[0]
            num_batches += num_samples
            loss1 = mse_loss_ls_sum(trt_output, trt_raw_data)
            loss2 = mse_loss_ls_sum(ctl_output, ctl_raw_data)
            loss3 = mse_loss_ls_sum(deg_output, trt_raw_data-ctl_raw_data)
            loss4 = pcc_loss_sum(deg_output, trt_raw_data-ctl_raw_data)
            weighted_loss = loss1 * a + loss2 * b + loss3 * c + loss4 * d
            batch_weighted_loss = torch.sqrt(loss1/num_samples) * a + torch.sqrt(loss2/num_samples) * b + torch.sqrt(loss3/num_samples) * c + (loss4/num_samples) * d
            total_loss += weighted_loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()
            total_loss4 += loss4.item()

            if epoch < init_epoch:
                batch_weighted_loss.backward()  # accelerate model convergence
            else:
                weighted_loss.backward()

            opt.step()

        # Log frequent progress
        if step % log_interval == 0 and step > 0:
            running_avg_loss = total_loss / num_batches
            print(f"[Epoch {epoch}] Step {step}/{total_steps} - Avg loss: {running_avg_loss:.4f}")

        # Intermediate validation
        if val_dataloader is None:
            print('val none.')
        if stopper is None:
            print('stopper none.')
        
        if val_dataloader is not None and stopper is not None and step % valid_interval == 0 and step > 0:
            # Switch back to train mode after validation
            val_loss, val_loss1, val_loss2, val_loss3, val_loss4, _ = validate(model, val_dataloader, args, config, flag=False, scaler=scaler)
            early_stop = stopper.step(val_loss4, model, epoch, opt)
            model.train()
            print(f"[Epoch {epoch}] Step {step}/{total_steps} - Valid Total Loss: {val_loss:.3f}, Loss4: {val_loss4:.3f}")
            if early_stop:
                print(f"[Epoch {epoch}] Early stopping triggered at step {step}")
                # Return early with current stats
                return total_loss / num_batches, np.sqrt(total_loss1 / num_batches), \
                       (total_loss2 / num_batches) if cell_class_predict else np.sqrt(total_loss2 / num_batches), \
                       np.sqrt(total_loss3 / num_batches), total_loss4 / num_batches

    avg_total_loss = total_loss / num_batches
    avg_loss1 = np.sqrt(total_loss1 / num_batches) if total_loss1 else -999
    # avg_loss1 = total_loss1 / num_batches if total_loss1 else -999
    if cell_class_predict is not None:
        avg_loss2 = (total_loss2 / num_batches) if total_loss2 else -999
    else:
        avg_loss2 = np.sqrt(total_loss2 / num_batches) if total_loss2 else -999
    avg_loss3 = np.sqrt(total_loss3 / num_batches) if total_loss3 else -999
    avg_loss4 = total_loss4 / num_batches if total_loss4 else -999

    return avg_total_loss, avg_loss1, avg_loss2, avg_loss3, avg_loss4


def main():

    args = arg_parse()

    start_time = time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    if args.mode == 'infer':
        expt_folder = osp.join('experiment/infer/', f'{timestamp}')
    else:
        if 'l1000' in args.dataset:
            expt_folder = osp.join('experiment/l1000', f'{timestamp}')
        elif 'panacea' in args.dataset:
            expt_folder = osp.join('experiment/panacea/', f'{timestamp}')
        elif 'cdsdb' in args.dataset:
            expt_folder = osp.join('experiment/cdsdb/', f'{timestamp}')
        elif 'tahoe' in args.dataset:
            expt_folder = osp.join('experiment/tahoe/', f'{timestamp}')
        elif 'sciplex' in args.dataset:
            expt_folder = osp.join('experiment/sciplex/', f'{timestamp}')
        elif 'nips' in args.dataset:
            expt_folder = osp.join('experiment/nips/', f'{timestamp}')
        else:
            expt_folder = osp.join('experiment/general/', f'{timestamp}')


    if not os.path.exists(expt_folder):
        os.makedirs(expt_folder)

    # All output uses print, no logging file (user request)
    logger = None

    # loading args 
    print('\n---------args-----------')
    log_nested_dict(vars(args), print)
    print('\n')

    #set random seed
    set_random_seed(int(args.seed))

    # loading config
    with open(f'configs/{args.config}.yaml', 'r') as f:
        config = yaml.safe_load(f) 
    print('\n--------configs----------')
    log_nested_dict(config, print)

    # check cuda
    if torch.cuda.is_available():
        device = torch.device(args.device)
        torch.backends.cudnn.benchmark = True
        print("Use GPU: %s\n" % device)
    else:
        device = torch.device("cpu")
        print("Use CPU")
    

    if args.mode == 'train':
        print('Train mode:')

        nfold = [i for i in args.nfold.split(',')]

        for k in nfold:
            
            tr_dataloader, val_dataloader, test_dataloader, adata = load_dataloader(args,config,logger,nfold=k,return_rawdata=True)


            loss_func = None
            kl_loss = None


            # define model 
            if args.model =='XPert':
                model = XPertNet(args, config, device, logger)
            else:
                assert False, 'Model not found!'
            model.init_weights()
            model.to(device)
            print(model)

            # calculate the number of parameters for the model
            total_params = sum([param.nelement() for param in model.parameters()])
            print("Number of parameters: %.2fM" % (total_params/1e6))

            if loss_func is not None:
                optimizer = torch.optim.Adam(
                                [
                                    {"params": loss_func.weights_param, "lr": 1e-7},
                                    {"params": model.parameters(), "lr": config['train']['train_lr'], "weight_decay": config['train']['weight_decay']}
                                ]
                            )
            else:
                optimizer =  torch.optim.Adam(model.parameters(), lr=config['train']['train_lr'],  weight_decay=config['train']['weight_decay'])


            if args.lr_scheduler:
                # dataset=sdst/mdmt
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, 
                    lr_lambda=lambda epoch: 1.0 if epoch < 40 else 0.5
                )
                # for smaller datasers, use bigger learning rate
                # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                #     optimizer, 
                #     lr_lambda=lambda epoch: 1.0 if epoch < 100 else 0.5
                # )
            else:
                lr_scheduler = None

            if args.use_gradscaler:
                gradscaler = GradScaler()
                print('Using GradScaler to accelerate trainning..')
            else:
                gradscaler = None


            start_epoch = 0
            if args.resume_from:
                resume_path=args.resume_from
                print(f'resume_from path:{resume_path}')
                checkpoint = torch.load(resume_path, map_location=device)
                model_dict = model.state_dict()
                pretrained_dict = {k:v for k,v in checkpoint['model_state_dict'].items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f'Load previous-trained parameters sucessfully! From epoch {start_epoch} to train……')
            if args.pretrained_model:
                pretrained_model_path=args.pretrained_model
                print(f'pretrained model path:{pretrained_model_path}')
                checkpoint = torch.load(pretrained_model_path, map_location=device)
                model_dict = model.state_dict()

                exclude_keys = config['finetune']['exclude_keys']
                pretrained_dict = {
                    k: v for k, v in checkpoint['model_state_dict'].items()
                    if k in model_dict
                    and v.shape == model_dict[k].shape
                    and not any(exclude_key in k for exclude_key in exclude_keys)
                }

                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = 0
                print(f'Load previous-trained parameters sucessfully! From epoch {start_epoch} to train……')

                finetune_params=config['finetune']['finetune_params']
                print(f'Updating params: {finetune_params}')
                set_requires_grad(model, finetune_params=finetune_params)
                print("Model Parameters:")
                for name, param in model.named_parameters():
                    print(f"{name}: requires_grad={param.requires_grad}")

                optimizer = torch.optim.Adam(model.parameters(), lr=config['finetune']['finetune_lr'],  weight_decay=config['finetune']['weight_decay'])

            
            max_epoch = config['train']['num_epochs']
            tr_loss_list = []
            val_loss_list = []
            # print('cp1')
            stopper = EarlyStopping(mode='lower', metric='mse', patience=config['train']['patience'], n_fold=k, folder=expt_folder, logger=logger) 
            fixed_params = []

            # print('cp2')

            early_stop = False
            for epoch in range(start_epoch, max_epoch):
                if epoch == start_epoch:
                    epoch_start_time = time.time()
                train_loss, train_loss1, train_loss2, train_loss3, train_loss4 = train(
                    model, optimizer, tr_dataloader, args, config,
                    scaler=gradscaler, epoch=epoch, loss_func=loss_func, kl_loss=kl_loss,
                    val_dataloader=val_dataloader, stopper=stopper, expt_folder=expt_folder, logger=None
                )
                tr_loss_list.append(train_loss)
                print(f'Epoch {epoch}, Train Total Loss: {train_loss:.3f}, Loss1: {train_loss1:.3f}, Loss2: {train_loss2:.3f}, Loss3: {train_loss3:.3f}, Loss4: {train_loss4:.3f}')                       
                if epoch == start_epoch:    
                    print(f'Epoch_{epoch}_train_time:{int(time.time() - epoch_start_time)/60:.2f}min')
                    valid_start_time = time.time()
                val_loss, val_loss1, val_loss2, val_loss3, val_loss4, _ = validate(model, val_dataloader, args, config, flag=False, scaler=gradscaler)
                early_stop = stopper.step(val_loss4, model, epoch, optimizer)
                val_loss_list.append(val_loss)
                print(f'Epoch {epoch}, Valid Total Loss: {val_loss:.3f}, Loss1: {val_loss1:.3f}, Loss2: {val_loss2:.3f}, Loss3: {val_loss3:.3f}, Loss4: {val_loss4:.3f}')
                if early_stop or epoch == max_epoch-1:
                    best_model = stopper.load_checkpoint(model, optimizer)
                    break
                if lr_scheduler:
                    lr_scheduler.step()
                if epoch == start_epoch or epoch % 10 == 0:
                    current_lr = optimizer.param_groups[-1]['lr']
                    print(f'Epoch {epoch+1}, Learning Rate: {current_lr}') 
                if epoch == start_epoch:
                    print(f'Epoch_{epoch}_valid_time:{int(time.time() - valid_start_time)/60:.2f}min')

                                                                                           

            try:
                save_loss_fig(tr_loss_list,val_loss_list,k,expt_folder,logger)
            except Exception as e:
                print(f"An error occurred while saving the loss fig: {e}")

            print(f'{k}_Fold_Training is done! Training_time:{int(time.time() - start_time)/60:.2f}min')
            print('Start testing ... ')

            print('Tesing on train dataset:')
            _,_,_,_,_, tr_metrics = validate(best_model, tr_dataloader, args, config, flag=True, scaler=gradscaler)
            print('Training Metrics:')
            log_nested_dict(tr_metrics, print, indent=1)

            print('Tesing on Valid dataset:')
            _,_,_,_,_, val_metrics = validate(best_model, val_dataloader, args, config, flag=True, scaler=gradscaler)
            print('Validation Metrics:')
            log_nested_dict(val_metrics, print, indent=1)

            print('Tesing on test dataset:')
            _,_,_,_,_, test_metrics = validate(model, test_dataloader, args, config, flag=True, scaler=gradscaler)
            test_metrics = val_metrics
            print('Test Metrics:')
            log_nested_dict(test_metrics, print, indent=1)

            print('Saving result……')
            result_dict_list = [tr_metrics, val_metrics, test_metrics]
            result_df = pd.DataFrame(result_dict_list)          

            try:
                result_df.to_csv(os.path.join(expt_folder, f'{k}_train_result.csv'), index=False)
                print('Results successfully saved!!!\n\n')
            except Exception as e:
                print(f"An error occurred while saving the [{k}_train_result.csv] file: {e}")

        print(f'All folds training are done! Training time: {int(time.time() - start_time)/60:.2f} min')

            

    elif args.mode == 'test':
        print('Test mode...')
        
        nfold = [i for i in args.nfold.split(',')]

        for k in nfold:
            test_dataloader = load_test_dataloader(args, config, logger, k)

            if args.model =='XPert':
                model = XPertNet(args, config, device, logger)
            else:
                assert False, 'Model not found!'
            model.init_weights()
            model.to(device)

            if args.saved_model:
                saved_model = args.saved_model
            elif args.saved_model_path:
                saved_model = osp.join(args.saved_model_path, '{}_fold_early_stop.pth'.format(k))
            else:
                print('No trained parameters are provided!')
            checkpoint = torch.load(saved_model)
            model.load_state_dict(checkpoint['model_state_dict'])

            if args.use_gradscaler:
                gradscaler = GradScaler()
                print('Using GradScaler to accelerate trainning..')
            else:
                gradscaler = None


            print('Start testing ... ')
            start_test_time = time.time()
            print('Tesing on test dataset:')
            _,_,_,_,_, test_metrics = validate(model, test_dataloader, args, config, flag=True, scaler=gradscaler, expt_folder=expt_folder, fold=k)
            print('Test Metrics:')
            log_nested_dict(test_metrics, print, indent=1)

            print(f'Test time: {int(time.time() - start_test_time)/60:.2f} min')      

            test_metrics = test_metrics
            val_metrics = test_metrics
            tr_metrics = test_metrics

            print('Saving result……')
            result_dict_list = [tr_metrics, val_metrics, test_metrics]
            result_df = pd.DataFrame(result_dict_list)          

            try:
                result_df.to_csv(os.path.join(expt_folder, f'{k}_train_result.csv'), index=False)
                print('Results successfully saved!!!\n\n')
            except Exception as e:
                print(f"An error occurred while saving the [{k}_train_result.csv] file: {e}")

    elif args.mode == 'infer':

        print('Infer mode...')
        # 在infer的模式下，可以返回总体指标、每条item的指标、以及predicted profile的结果

        nfold = [i for i in args.nfold.split(',')]

        for k in nfold:
            test_dataloader = load_infer_dataloader(args, config, logger, k)

            # define model 
            if args.model =='XPert':
                model = XPertNet(args, config, device, logger)
            else:
                assert False, 'Model not found!'
            
            model.to(device)

            total_params = sum([param.nelement() for param in model.parameters()])
            print("Number of parameters: %.2fM" % (total_params/1e6))

            # load model
            if args.saved_model:
                saved_model = args.saved_model
            elif args.saved_model_path:
                saved_model = osp.join(args.saved_model_path, '{}_fold_early_stop.pth'.format(k))
            else:
                print('No trained parameters are provided!')
            
            checkpoint = torch.load(saved_model)

            model_dict = model.state_dict()
            pretrained_dict = {k:v for k,v in checkpoint['model_state_dict'].items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)


            print('Start testing ... ')
           
            print('Tesing on test dataset:')
            test_metrics, test_metrics_all_ls, test_output_pred_profile, test_all_attention_dict, all_cls_embed = infer(model, test_dataloader, args, config, flag=True, output_profile=args.output_profile, scaler=args.use_gradscaler)

            print('Test Metrics:')
            log_nested_dict(test_metrics, print, indent=1) 

                     
            print('Saving result……')

            result_df = pd.DataFrame.from_dict(test_metrics, orient='index').T         
            try:
                result_df.to_csv(expt_folder + f'/{k}_train_result.csv')
                print('Results successfully saved!!!\n\n')
            except Exception as e:
                print(f"An error occurred while saving the [{k}_train_result.csv] file: {e}")

            if test_metrics_all_ls is not None:
                result_ls_df = pd.DataFrame.from_dict(test_metrics_all_ls)          
                try:
                    result_ls_df.to_csv( expt_folder + f'/{args.dataset}_{k}_test_samples_result.csv' )
                    print('Results successfully saved!!!\n\n')
                except Exception as e:
                    print(f"An error occurred while saving the [{args.dataset}_{k}_test_result_all_samples.csv] file: {e}")
            
            if test_output_pred_profile is not None:
                try:
                    np.save( expt_folder + f'/{args.dataset}_{k}_test_samples_prediction_profile.npy' , test_output_pred_profile)
                    print('Profiles successfully saved!!!\n\n')
                except Exception as e:
                    print(f"An error occurred while saving the [{args.dataset}_{k}_test_samples_prediction_profile.npy] file: {e}") 

            if args.output_attention:
                try:
                    output_file_path = expt_folder + f'/{args.dataset}_{k}_test_samples_attention_scores.pkl'
                    with open(output_file_path, "wb") as f:
                        pickle.dump(test_all_attention_dict, f)
                    print('Attention scores successfully saved!!!')
                except Exception as e:
                    print(f"An error occurred while saving the [{args.dataset}_{k}_test_samples_attention_scores.npy] file: {e}") 

            if args.output_cls_embed:
                try:
                    np.save( expt_folder + f'/{args.dataset}_{k}_test_samples_cls_embed.npy' , all_cls_embed)
                    print('Profiles successfully saved!!!\n\n')
                except Exception as e:
                    print(f"An error occurred while saving the [{args.dataset}_{k}_test_samples_cls_embed.npy] file: {e}") 


if __name__ == '__main__':

    main()
