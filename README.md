# MAP

The official codes for "MAP: A Knowledge-driven Framework for Predicting Single-cell Responses for Unprofiled Drugs".

[BioRxiv](https://www.biorxiv.org/content/10.64898/2026.02.25.708091v1)

We present **MAP**, a framework that integrates structured pharmacological knowledge into cellular response prediction. MAP learns mechanism-aware drug and gene representations by aligning molecular structures, protein targets, and mechanistic descriptions in a unified embedding space, and then conditions a perturbation predictor on these knowledge-informed representations.

<img width="1261" height="1288" alt="57ecdf1f0b06e3bb4835b71b473b1ff4" src="https://github.com/user-attachments/assets/043387ba-0377-4475-849d-d9a0d65e74eb" />

## Setup

### Environment
Core libraries required include:

`Python 3.8`
`Torch 2.1.2`
`scanpy 1.9.8`

For complete environment requirements, we also provide `requirements.txt` as a reference, the versions of packages are not compulsory. You can run `pip install -r requirements.txt` to quick install the conda environment. The typical installation time for setting up the environment is a few minutes.

### Data Preparation

* For knowledge encoders pre-training, we provide preprocessed knowledge graph data files at [Huggingface](https://huggingface.co/datasets/RainGate/MAP-KG). Download and put them under `MAP-KG/data/selected_csvs/`.
* For MAP training, download Tahoe-100M, OP3 or SciPlex3 from official sites, and go through all scripts under `preprocess/` by alphabetical order.
* We suggest you prepare at least 4 TB storage for the above three datasets.

### Raw Data Source
* Tahoe-100M: https://huggingface.co/datasets/tahoebio/Tahoe-100M
* OP3: https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data
* SciPlex3: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150378
* Combosciplex: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE206741

## Implementation
### To pre-train knowledge encoders
After environment setup and data preparation, you should first check all the files, and replace all 'path/to/sth' into your own paths, then run:
```
MAP-KG/train_resume.sh
```
Training logs and checkpoints will be placed under `MAP-KG/logs` and `MAP-KG/checkpoints`.

### To train MAP
After environment setup and data preparation, you should first check all the files, and replace all 'path/to/sth' into your own paths, then run:
```
MAP/train.sh
```
Training logs and checkpoints will be placed under `MAP/logs` and `MAP/checkpoints`.

### Demo
We provide a demo to help you understand the expected actions of the model. Run it like this:
```                                                                        
python demo.py
  --ckpt [ckpt path]
  --cell_line CVCL_0023
  --drug_smiles "CC1=NC=C(C(=C1O)CO"
  --drug_conc 0.5 --output_dir ./demo_output
```

### Pre-trained model weights
The pretrained model weights (multi-modal knowledge encoders and perturbation prediction model) can be found in [Google Drive](https://drive.google.com/drive/folders/1cV0ZTk92PguKS2nyii6dLV0IfqoSDHsQ?usp=drive_link).

## Citation
```
@article{feng2026map,
  title={MAP: A Knowledge-driven Framework for Predicting Single-cell Responses for Unprofiled Drugs},
  author={Feng, Jinghao and Zhao, Ziheng and Zhang, Xiaoman and Liu, Mingfei and Chen, Jingyi and Quan, Xingran and Fu, Boyang and Zhang, Jian and Wang, Yanfeng and Zhang, Ya and Xie, Weidi},
  journal={bioRxiv},
  pages={2026--02},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```
