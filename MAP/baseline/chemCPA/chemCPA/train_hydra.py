from pathlib import Path
import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
import numpy as np

from chemCPA.data.data import PerturbationDataModule, load_dataset_splits
from chemCPA.paths import WB_DIR
from lightning_module import ChemCPA  # your LightningModule containing ComPert usage


@hydra.main(version_base=None, config_path="../config/", config_name="main")
def main(args):
    OmegaConf.set_struct(args, False) 

    data_params = args["dataset"]
    datasets, dataset = load_dataset_splits(**data_params, return_dataset=True)
    dm = PerturbationDataModule(datasplits=datasets, train_bs=args["model"]["hparams"]["batch_size"])
    dataset_config = {
        "num_genes": datasets["training"].num_genes,
        "num_drugs": datasets["training"].num_drugs,
        "num_covariates": datasets["training"].num_covariates,
        "use_drugs_idx": dataset.use_drugs_idx,
        "canon_smiles_unique_sorted": dataset.canon_smiles_unique_sorted,
    }
    dataset.debug_print()

    # Initialize model
    model = ChemCPA(args, dataset_config)

    # 1) Check drug indices are in range
    drugs_idx = datasets["training"].drugs_idx
    print(f"drugs_idx range: {drugs_idx.min().item()} to {drugs_idx.max().item()}, total num_drugs={dataset_config['num_drugs']}")
    assert drugs_idx.min() >= 0, "Negative drug index found!"
    assert drugs_idx.max() < dataset_config["num_drugs"], "Drug index out of range!"

    # 2) After your ChemCPA model is instantiated:
    embedding_w = model.drug_embeddings.weight.data  # shape [num_drugs, embedding_dim]
    if torch.isnan(embedding_w).any():
        bad_rows = torch.where(torch.isnan(embedding_w).any(dim=1))[0]
        print(f"NaNs in embedding row(s): {bad_rows.tolist()}")
        raise ValueError("drug_embeddings contains NaNs!")

    print(f"Warning: No pretrained hash found for model")

    wandb_logger = WandbLogger(**args["wandb"], save_dir=WB_DIR)
    run_id = wandb_logger.experiment.id

    checkpoint_callback = ModelCheckpoint(dirpath=Path(args["training"]["save_dir"]) / run_id, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        accelerator="cuda",
        devices=1,
        logger=wandb_logger,
        max_epochs=args["training"]["num_epochs"],
        max_time=args["training"]["max_minutes"],
        callbacks=[checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=args["training"]["checkpoint_freq"],
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

