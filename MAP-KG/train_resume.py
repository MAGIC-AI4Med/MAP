import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

import random
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import logging

from data.dataload import DrugGeneKG_Dataset
from model.model import DrugGeneModel

import warnings
warnings.filterwarnings('ignore')

class TrainingConfig:
    exp = 'map_kg'
    resume_from = "./checkpoints/kg_v3/best_model.pt"
    
    drug_node_path = "data/selected_data_csvs/DRUG_merged_drugs_with_residuals.csv"
    gene_node_path = "data/selected_data_csvs/GENE_tahoe_filtered.csv"
    drug_gene_edge_path = "data/selected_data_csvs/DRUG-GENE_filtered_by_existing_drugs_and_genes.csv"
    
    bert_model_name = "dmis-lab/biobert-v1.1"
    empty_relation = "xxx"
    d_model = 1024
    d_esm = 5120
    d_smiles = 256
    mode_threshold = 0.5
    finetune_smiles_encoder = False
    finetune_bert = True
    freeze_bert_layers = 7
    
    batch_size = 128
    num_epochs = 200
    learning_rate = 1e-4
    weight_decay = 0.01
    warmup_steps = 1000

    use_amp = True
    amp_dtype = 'float16'
    
    seed = 42
    num_workers = 8
    
    save_dir = f"./checkpoints/{exp}"
    log_dir = "./logs"
    epoch_save_interval = 10
    log_interval = 200


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(config, rank=0):
    if rank != 0:
        logging.basicConfig(level=logging.ERROR)
        return logging.getLogger(__name__)
    
    os.makedirs(config.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.log_dir, f"training_{config.exp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, config, filename=None, rank=0, scaler=None):
    if rank != 0:
        return None
    
    os.makedirs(config.save_dir, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch{epoch}_step{step}.pt"
    
    checkpoint_path = os.path.join(config.save_dir, filename)
    
    model_state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
        'config': config.__dict__
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']


def collate_fn(batch):
    property_1s = []
    property_2s = []
    type_1s = []
    type_2s = []
    relations = []
    id_1s = []
    id_2s = []
    
    for sample in batch:
        property_1, property_2, type_1, type_2, relation, id_1, id_2 = sample
        property_1s.append(property_1)
        property_2s.append(property_2)
        type_1s.append(type_1)
        type_2s.append(type_2)
        relations.append(relation)
        id_1s.append(id_1)
        id_2s.append(id_2)
    
    return {
        'property_1s': property_1s,
        'property_2s': property_2s,
        'type_1s': type_1s,
        'type_2s': type_2s,
        'relations': relations,
        'id_1s': id_1s,
        'id_2s': id_2s
    }


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return total_params, trainable_params, frozen_params


def ensure_device(module, device):
    for param in module.parameters():
        if param.device != device:
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(device)
    
    for buffer_name, buffer in module.named_buffers():
        if buffer.device != device:
            module._buffers[buffer_name] = buffer.to(device)
    
    for child in module.children():
        ensure_device(child, device)


def train_epoch(model, dataloader, optimizer, scheduler, device, config, logger, epoch, global_step, writer, rank, scaler=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    if rank == 0:
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        progress_bar = dataloader
    
    for batch_idx, batch in enumerate(progress_bar):
        
        optimizer.zero_grad()

        if config.use_amp and scaler is not None:
            with autocast(dtype=torch.float16 if config.amp_dtype == 'float16' else torch.bfloat16):
                output = model(batch)
                loss = output['loss']
        else:
            output = model(batch)
            loss = output['loss']
        
        if config.use_amp and scaler is not None:
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        global_step += 1
        
        if rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            if writer is not None:
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                writer.add_scalar('Train/LearningRate', current_lr, global_step)
            
            if global_step % config.log_interval == 0:
                logger.info(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {total_loss/num_batches:.4f} | "
                    f"LR: {current_lr:.2e}"
                )
    
    avg_loss = total_loss / num_batches
    return avg_loss, global_step


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def train_worker(rank, world_size, config):
    setup_ddp(rank, world_size)
    
    set_seed(config.seed + rank)
    
    logger = setup_logging(config, rank)
    
    if rank == 0:
        logger.info("=" * 80)
        logger.info("Starting training DrugGeneModel (Multi-GPU Distributed Training)")
        logger.info(f"Using {world_size} GPUs")
        logger.info("=" * 80)

        run_name = config.exp
        tensorboard_dir = os.path.join(config.log_dir, run_name)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        logger.info(f"TensorBoard logs saved to: {tensorboard_dir}")
    else:
        writer = None
    
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        logger.info("Loading dataset...")
    
    dataset = DrugGeneKG_Dataset(
        drug_node_path=config.drug_node_path,
        gene_node_path=config.gene_node_path,
        drug_gene_edge_path=config.drug_gene_edge_path,
    )
    
    if rank == 0:
        logger.info(f"Dataset size: {len(dataset)}")
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    if rank == 0:
        logger.info(f"Batch size per GPU: {config.batch_size}")
        logger.info(f"Total batch size: {config.batch_size * world_size}")
        logger.info(f"Number of batches per epoch: {len(dataloader)}")
    
    if rank == 0:
        logger.info("Initializing model...")
    
    model = DrugGeneModel(
        bert_model_name=config.bert_model_name,
        empty_relation=config.empty_relation,
        d_model=config.d_model,
        d_esm=config.d_esm,
        d_smiles=config.d_smiles,
        mode_threshold=config.mode_threshold,
        finetune_smiles_encoder=config.finetune_smiles_encoder,
        finetune_bert=config.finetune_bert,
        freeze_bert_layers=config.freeze_bert_layers
    )
    
    model = model.to(device)

    if rank == 0:
        smiles_encoder_device = next(model.smiles_encoder.parameters()).device
        logger.info(f"SMILES encoder device: {smiles_encoder_device}")
        assert smiles_encoder_device == device, \
            f"SMILES encoder on wrong device: {smiles_encoder_device} vs {device}"
    
    if rank == 0:
        total_params, trainable_params, frozen_params = count_parameters(model)
        logger.info("=" * 80)
        logger.info("Model Parameter Statistics:")
        logger.info(f"  Total parameters:     {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        logger.info(f"  Frozen parameters:    {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
        logger.info("=" * 80)
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    if config.use_amp and config.amp_dtype == 'float16':
        scaler = GradScaler()
        if rank == 0:
            logger.info("Using mixed precision training (FP16) with GradScaler")
    elif config.use_amp and config.amp_dtype == 'bfloat16':
        scaler = None
        if rank == 0:
            logger.info("Using mixed precision training (BF16)")
    else:
        scaler = None
        if rank == 0:
            logger.info("Using full precision training (FP32)")
    
    total_steps = len(dataloader) * config.num_epochs
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, total_steps)
    
    start_epoch = 1
    global_step = 0
    best_loss = float('inf')
    
    if config.resume_from and os.path.exists(config.resume_from):
        if rank == 0:
            logger.info("=" * 80)
            logger.info(f"Resuming training from checkpoint: {config.resume_from}")
        
        loaded_epoch, loaded_step, loaded_loss = load_checkpoint(
            config.resume_from, 
            model, 
            optimizer, 
            scheduler, 
            scaler,
            device=device
        )
        
        start_epoch = loaded_epoch + 1
        global_step = loaded_step
        best_loss = loaded_loss
        
        if rank == 0:
            logger.info(f"Restored to Epoch {loaded_epoch}, Step {loaded_step}, Loss {loaded_loss:.4f}")
            logger.info(f"Will continue training from Epoch {start_epoch}")
            logger.info("=" * 80)
    elif config.resume_from and rank == 0:
        logger.warning(f"Specified checkpoint does not exist: {config.resume_from}")
        logger.warning("Will start training from scratch")
    
    if rank == 0:
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Warmup steps: {config.warmup_steps}")
        logger.info("=" * 80)
        logger.info("Starting training")
        logger.info("=" * 80)
    
    for epoch in range(start_epoch, config.num_epochs + 1):
        sampler.set_epoch(epoch)
        
        if rank == 0:
            logger.info(f"\nEpoch {epoch}/{config.num_epochs}")
        
        avg_loss, global_step = train_epoch(
            model, dataloader, optimizer, scheduler, 
            device, config, logger, epoch, global_step, writer, rank, scaler
        )
        
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()
        
        if rank == 0:
            if writer is not None:
                writer.add_scalar('Train/EpochAvgLoss', avg_loss, epoch)
            
            logger.info(f"Epoch {epoch} completed | Average Loss: {avg_loss:.4f}")
            
            if epoch % config.epoch_save_interval == 0:
                epoch_checkpoint_path = save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step,
                    avg_loss, config, filename=f"checkpoint_epoch{epoch}.pt", rank=rank, scaler=scaler
                )
                logger.info(f"Epoch checkpoint saved to {epoch_checkpoint_path}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_checkpoint_path = save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step,
                    avg_loss, config, filename="best_model.pt", rank=rank, scaler=scaler
                )
                logger.info(f"New best model! Loss: {best_loss:.4f}")
                logger.info(f"Best model saved to {best_checkpoint_path}")
        
        dist.barrier()
    
    if rank == 0:
        if writer is not None:
            writer.close()
        
        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info(f"Best Loss: {best_loss:.4f}")
        logger.info("=" * 80)
    
    cleanup_ddp()

def train_single_gpu(config: TrainingConfig):
    set_seed(config.seed)

    logger = setup_logging(config, rank=0)
    logger.info("=" * 80)
    logger.info("Starting training DrugGeneModel (Single GPU Training)")
    logger.info("=" * 80)

    run_name = config.exp + "_single"
    tensorboard_dir = os.path.join(config.log_dir, run_name)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard logs saved to: {tensorboard_dir}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    logger.info("Loading dataset...")
    dataset = DrugGeneKG_Dataset(
        drug_node_path=config.drug_node_path,
        gene_node_path=config.gene_node_path,
        drug_gene_edge_path=config.drug_gene_edge_path,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Number of batches per epoch: {len(dataloader)}")

    logger.info("Initializing model...")
    model = DrugGeneModel(
        bert_model_name=config.bert_model_name,
        empty_relation=config.empty_relation,
        d_model=config.d_model,
        d_esm=config.d_esm,
        d_smiles=config.d_smiles,
        mode_threshold=config.mode_threshold,
        finetune_smiles_encoder=config.finetune_smiles_encoder,
        finetune_bert=config.finetune_bert,
        freeze_bert_layers=config.freeze_bert_layers
    ).to(device)

    smiles_encoder_device = next(model.smiles_encoder.parameters()).device
    logger.info(f"SMILES encoder device: {smiles_encoder_device}")
    assert smiles_encoder_device == device, \
        f"SMILES encoder on wrong device: {smiles_encoder_device} vs {device}"

    total_params, trainable_params, frozen_params = count_parameters(model)
    logger.info("=" * 80)
    logger.info("Model Parameter Statistics:")
    logger.info(f"  Total parameters:     {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    logger.info(f"  Frozen parameters:    {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    logger.info("=" * 80)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    if config.use_amp and config.amp_dtype == 'float16':
        scaler = GradScaler()
        logger.info("Using mixed precision training (FP16) with GradScaler")
    elif config.use_amp and config.amp_dtype == 'bfloat16':
        scaler = None
        logger.info("Using mixed precision training (BF16)")
    else:
        scaler = None
        logger.info("Using full precision training (FP32)")
    
    total_steps = len(dataloader) * config.num_epochs
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, total_steps)

    start_epoch = 1
    global_step = 0
    best_loss = float('inf')
    
    if config.resume_from and os.path.exists(config.resume_from):
        logger.info("=" * 80)
        logger.info(f"Resuming training from checkpoint: {config.resume_from}")
        
        loaded_epoch, loaded_step, loaded_loss = load_checkpoint(
            config.resume_from, 
            model, 
            optimizer, 
            scheduler, 
            scaler,
            device=device
        )
        
        start_epoch = loaded_epoch + 1
        global_step = loaded_step
        best_loss = loaded_loss
        
        logger.info(f"Restored to Epoch {loaded_epoch}, Step {loaded_step}, Loss {loaded_loss:.4f}")
        logger.info(f"Will continue training from Epoch {start_epoch}")
        logger.info("=" * 80)
    elif config.resume_from:
        logger.warning(f"Specified checkpoint does not exist: {config.resume_from}")
        logger.warning("Will start training from scratch")

    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {config.warmup_steps}")
    logger.info("=" * 80)
    logger.info("Starting training")
    logger.info("=" * 80)

    for epoch in range(start_epoch, config.num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{config.num_epochs}")
        avg_loss, global_step = train_epoch(
            model, dataloader, optimizer, scheduler,
            device, config, logger, epoch, global_step, writer, rank=0, scaler=scaler
        )

        if writer is not None:
            writer.add_scalar('Train/EpochAvgLoss', avg_loss, epoch)

        logger.info(f"Epoch {epoch} completed | Average Loss: {avg_loss:.4f}")

        if epoch % config.epoch_save_interval == 0:
            epoch_checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                avg_loss, config, filename=f"checkpoint_epoch{epoch}.pt", rank=0, scaler=scaler
            )
            logger.info(f"Epoch checkpoint saved to {epoch_checkpoint_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                avg_loss, config, filename="best_model.pt", rank=0, scaler=scaler
            )
            logger.info(f"New best model! Loss: {best_loss:.4f}")
            logger.info(f"Best model saved to {best_checkpoint_path}")

    writer.close()
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Best Loss: {best_loss:.4f}")
    logger.info("=" * 80)

def train():
    config = TrainingConfig()
    
    world_size = torch.cuda.device_count()
    
    if world_size == 0:
        print("Error: No available GPUs!")
        return
    
    print(f"Detected {world_size} GPU(s), starting multi-GPU training...")

    if world_size == 1:
        train_single_gpu(config)
    else:
        import torch.multiprocessing as mp
        mp.spawn(
            train_worker,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    train()