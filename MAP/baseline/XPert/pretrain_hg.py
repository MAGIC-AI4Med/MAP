import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import os.path as osp
import time
import logging
import argparse

import torch
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import numpy as np


from utils import EarlyStopping, set_random_seed
from utils_hg import log_nested_dict, generate_multi_relation_samples, \
                ContrastiveLoss, get_embeddings, get_batch_drug_subgraph



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--node_input_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--out_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--hg_path', type=str, default='HG_data/')
    parser.add_argument('--dataset_prefix', type=str, default='tahoe', help='Prefix for dataset files: tahoe/nips/sciplex')
    parser.add_argument('--include_edges', type=str, nargs='+', default=['PPI', 'DTI', 'DDS'])
    parser.add_argument('--seed', type=int, default=4242)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--num_neg_samples', type=int, default=5)
    parser.add_argument('--num_neighbors', nargs='+', type=int, default=[35, 20, 10])
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--global_saved_model', type=str, default=None)
    parser.add_argument('--num_iters', type=int, default=2)
    parser.add_argument('--output_relation_matrix', type=bool, default=False, help='Extract relation matrix')
    parser.add_argument('--infer_dti_scores', type=bool, default=False, help='Extract dti matrix')

    return parser.parse_args()




# Define the HeteroGNN model with configurable edges
class HeteroGNN(torch.nn.Module):
    def __init__(self, node_input_dim=512, hidden_dim=128, out_dim=128, num_layers=2, include_edges=None, gene_input_dim=None, drug_input_dim=None):
        super().__init__()

        # If separate dimensions not provided, use the same for both
        if gene_input_dim is None:
            gene_input_dim = node_input_dim
        if drug_input_dim is None:
            drug_input_dim = node_input_dim

        self.transforme_gene_node = torch.nn.Linear(gene_input_dim, hidden_dim)
        self.transforme_drug_node = torch.nn.Linear(drug_input_dim, hidden_dim)

        # Define all possible edge types and their corresponding convolutions
        all_conv_layers = {
            ('gene', 'PPI', 'gene'): SAGEConv((-1, -1), hidden_dim),
            ('gene', 'GSEA', 'gene'): SAGEConv((-1, -1), hidden_dim), 
            ('gene', 'PCC', 'gene'): SAGEConv((-1, -1), hidden_dim),
            ('drug', 'DDS', 'drug'): SAGEConv((-1, -1), hidden_dim),
            ('drug', 'DTI', 'gene'): SAGEConv((-1, -1), hidden_dim),
            ('gene', 'DTI', 'drug'): SAGEConv((-1, -1), hidden_dim)
        }
        
        # Filter conv_layers based on include_edges parameter
        if include_edges is not None:
            conv_layers = {}
            for edge_type, conv in all_conv_layers.items():
                if edge_type[1] in include_edges:
                    conv_layers[edge_type] = conv
                # Add reverse DTI edge if DTI is included
                elif edge_type[1] == 'DTI' and 'DTI' in include_edges:
                    conv_layers[('gene', 'DTI', 'drug')] = SAGEConv((-1, -1), hidden_dim)
        else:
            conv_layers = all_conv_layers

        self.convs = torch.nn.ModuleList([HeteroConv(conv_layers) for _ in range(num_layers)])
        
        self.relation_predictors = torch.nn.ModuleDict({
            'DTI': torch.nn.Linear(hidden_dim * 2, 1),
            'DDS': torch.nn.Linear(hidden_dim * 2, 1),
            'PPI': torch.nn.Linear(hidden_dim * 2, 1)
        })

    def forward(self, x_dict, edge_index_dict):
        # transform node embeddings
        x_dict['gene'] = self.transforme_gene_node(x_dict['gene'])
        x_dict['drug'] = self.transforme_drug_node(x_dict['drug'])
        for conv in self.convs:
            x_dict = {key: x.relu() for key, x in conv(x_dict, edge_index_dict).items()}
        return x_dict

    def predict_bipartite_relations(self, src_emb, dst_emb, relation_type):

        num_src = src_emb.size(0)
        num_dst = dst_emb.size(0)
        
        # calculate all possible node pairs
        src_expanded = src_emb.unsqueeze(1).expand(-1, num_dst, -1)  # [num_src, num_dst, hidden_dim]
        dst_expanded = dst_emb.unsqueeze(0).expand(num_src, -1, -1)  # [num_src, num_dst, hidden_dim]
        
        pair_repr = torch.cat([src_expanded, dst_expanded], dim=-1)  # [num_src, num_dst, hidden_dim*2]

        pair_repr = pair_repr.view(-1, pair_repr.size(-1))  # [num_src*num_dst, hidden_dim*2]
        
        scores = self.relation_predictors[relation_type](pair_repr)  # [num_src*num_dst, 1]
        scores = scores.view(num_src, num_dst)  # [num_src, num_dst]
        
        return scores

    def predict_pairwise_relations(self, node_emb, relation_type):

        num_nodes = node_emb.size(0)
        
        # calculate all possible node pairs
        src_expanded = node_emb.unsqueeze(1).expand(-1, num_nodes, -1)  # [num_nodes, num_nodes, hidden_dim]
        dst_expanded = node_emb.unsqueeze(0).expand(num_nodes, -1, -1)  # [num_nodes, num_nodes, hidden_dim]
        
        pair_repr = torch.cat([src_expanded, dst_expanded], dim=-1)  # [num_nodes, num_nodes, hidden_dim*2]
        
        pair_repr = pair_repr.view(-1, pair_repr.size(-1))  # [num_nodes*num_nodes, hidden_dim*2]
        
        scores = self.relation_predictors[relation_type](pair_repr)  # [num_nodes*num_nodes, 1]
        scores = scores.view(num_nodes, num_nodes)  # [num_nodes, num_nodes]
        
        return scores

    def predict_relations(self, x_dict, edge_type=None):

        # node embeddings
        drug_emb = x_dict['drug']
        gene_emb = x_dict['gene']
            
        if edge_type == 'DTI':
            scores = self.predict_bipartite_relations(drug_emb, gene_emb, 'DTI')
        elif edge_type == 'DDS':
            scores = self.predict_pairwise_relations(drug_emb, 'DDS')
        elif edge_type == 'PPI':
            scores = self.predict_pairwise_relations(gene_emb, 'PPI')
        else:
            raise ValueError(f"Invalid edge type: {edge_type}")
        
        return scores




def train_contrastive_batch(model, data, optimizer, loader, epochs=100, patience=30, num_neg_samples=5, temperature=0.5, expt_folder=None, logger=None, nfold='global'):

    model.train()
    contrastive_loss_fn = ContrastiveLoss(temperature)
    relation_criterion = torch.nn.BCEWithLogitsLoss()
    edge_attr_criterion = torch.nn.MSELoss()
    stopper = EarlyStopping(mode='lower', metric='mse', patience=patience, n_fold=nfold, folder=expt_folder, logger=logger)

    for epoch in range(epochs):
        epoch_loss = 0

        i = 0
        for batch in loader:
            optimizer.zero_grad()
            i+=1

            x_dict = model(batch.x_dict, batch.edge_index_dict)

            contrastive_loss = torch.tensor(0.0).to(batch.x_dict['drug'].device)
            pos_samples, neg_samples = generate_multi_relation_samples(batch, num_neg_samples)
            if not pos_samples or not neg_samples:
                logger.warning("Empty samples generated, skipping batch")
                continue

            valid_edges = 0
            for edge_type in pos_samples:
                try:
                    # generate postive and negtive samples
                    pos_src_idx = pos_samples[edge_type][0]
                    pos_dst_idx = pos_samples[edge_type][1]
                    neg_dst_idx = neg_samples[edge_type][1]
                    
                    # get node embeddings
                    z_i = x_dict[edge_type[0]][pos_src_idx]
                    z_j = x_dict[edge_type[2]][pos_dst_idx]
                    z_neg = x_dict[edge_type[2]][neg_dst_idx]
                    
                    edge_loss = contrastive_loss_fn(z_i, z_j, z_neg)
                    if not torch.isnan(edge_loss):
                        contrastive_loss += edge_loss
                        valid_edges += 1
                    
                except Exception as e:
                    logger.error(f"Error processing edge type {edge_type}: {str(e)}")
                    continue

            logger.info(f'batch_contrast_loss:{contrastive_loss}')

            loss = contrastive_loss
            logger.info(f'batch_loss:{loss}')

            if loss != 0.:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        logger.info(f'Epoch {epoch:03d}, '
                   f'Loss: {avg_loss:.4f}, ')

        early_stop = stopper.step(avg_loss, model, epoch, optimizer)
        if early_stop:
            logger.info(f'Early stopping at epoch {epoch}')
            break
    
    best_model = stopper.load_checkpoint(model, optimizer)
    return best_model





def compute_relation_matrix(drug_embedding, gene_embedding, similarity_metric='cosine'):

    if similarity_metric == 'cosine':

        norm_drug = np.linalg.norm(drug_embedding, axis=1, keepdims=True)
        norm_gene = np.linalg.norm(gene_embedding, axis=1, keepdims=True)
        drug_normalized = np.divide(drug_embedding, norm_drug, where=norm_drug!=0)
        gene_normalized = np.divide(gene_embedding, norm_gene, where=norm_gene!=0)
    
        # calculate cos_similarity matrix
        relation_matrix = np.dot(drug_normalized, gene_normalized.T)  # [num_drugs, num_genes]
    elif similarity_metric == 'dot':
        relation_matrix = np.dot(drug_embedding, gene_embedding.T)  # [num_drugs, num_genes]
    else:
        raise ValueError("Similarity metric must be 'cosine' or 'dot'")
    
    return relation_matrix




def predict_all_dti_scores(model, data, device, logger, batch_size=128):

    model.eval()
    
    num_drugs = data.x_dict['drug'].shape[0]
    num_genes = data.x_dict['gene'].shape[0]

    scores_matrix = np.zeros((num_drugs, num_genes))
    
    with torch.no_grad():
        for i in range(0, num_drugs, batch_size):
            
            batch_end = min(i + batch_size, num_drugs)
            drug_indices = torch.arange(i, batch_end).to(device)

            logger.info(f'Processing drugs batch from index {i} to {batch_end-1}')

            subgraph = get_batch_drug_subgraph(data, drug_indices, device)
            subgraph = subgraph.to(device)

            x_dict = model(subgraph.x_dict, subgraph.edge_index_dict)
            
            batch_scores = model.predict_relations(x_dict, edge_type='DTI')
            
            batch_scores = batch_scores.cpu().numpy()

            batch_scores = 1 / (1 + np.exp(-batch_scores))

            scores_matrix[i:batch_end] = batch_scores

            del subgraph, x_dict, batch_scores
            torch.cuda.empty_cache()

    return scores_matrix







def main():
    args = get_args()
    
    start_time = time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    expt_folder = osp.join('HG_data/experiment/', args.dataset_prefix)
    if not os.path.exists(expt_folder):
        os.makedirs(expt_folder)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(expt_folder+f'/{timestamp}.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    logger.info('\n---------args-----------')
    log_nested_dict(vars(args), logger)
    logger.info('\n')

    set_random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Gene input is 5120 (full ESM2 embedding), drug input is 512 (UniMol CLS)
    model = HeteroGNN(
        node_input_dim=args.node_input_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_layers=args.num_layers,
        include_edges=args.include_edges,
        gene_input_dim=5120,  # Full ESM2 embedding dimension
        drug_input_dim=512    # UniMol CLS embedding dimension
    ).to(device)
    logger.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 加载数据
    logger.info("Loading data...")

    # Load mappings from our generated {dataset_prefix}_mappings.json
    import json
    with open(args.hg_path + f'{args.dataset_prefix}_mappings.json', 'r') as f:
        mappings = json.load(f)

    num_genes = mappings['num_genes']
    num_drugs = mappings['num_drugs']
    stats = mappings['stats']

    # Load gene and drug features with dataset prefix
    gene_feat_file = args.hg_path + f'{args.dataset_prefix}_all_gene_node_feat_{num_genes}.npy'
    drug_feat_file = args.hg_path + f'{args.dataset_prefix}_all_drug_node_feat_{num_drugs}.npy'

    logger.info(f"Loading gene features from {gene_feat_file}...")
    gene_feat = torch.tensor(np.load(gene_feat_file), dtype=torch.float)
    logger.info(f"Loading drug features from {drug_feat_file}...")
    drug_feat = torch.tensor(np.load(drug_feat_file), dtype=torch.float)

    data = HeteroData()
    data['gene'].x = gene_feat
    data['drug'].x = drug_feat

    # Load edge data with dynamic filenames and dataset prefix
    edge_files_map = {
        'PPI': f'{args.dataset_prefix}_PPI_edge_index_all_{stats["ppi_edges"]}_pairs_weighted.npy',
        'DTI': f'{args.dataset_prefix}_DTI_edge_index_all_{stats["dti_edges"]}_pairs.npy',
        'DDS': f'{args.dataset_prefix}_DDS_edge_index_all_{stats["dds_edges"]}_pairs_weighted.npy',
    }

    for edge_type in edge_files_map:
        if edge_type in args.include_edges:
            edge_file = args.hg_path + edge_files_map[edge_type]
            edge_data = np.load(edge_file, allow_pickle=True).item()
            edge_index = torch.tensor(edge_data['edge_index'], dtype=torch.long)
            edge_attr = torch.tensor(edge_data['edge_attr'], dtype=torch.float) if 'weighted' in edge_file else None

            if edge_type == 'PPI':
                data['gene', edge_type, 'gene'].edge_index = edge_index
                if edge_attr is not None:
                    data['gene', edge_type, 'gene'].edge_attr = edge_attr
            elif edge_type == 'DTI':
                data['drug', edge_type, 'gene'].edge_index = edge_index
                data['gene', edge_type, 'drug'].edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            elif edge_type == 'DDS':  # DDS
                data['drug', edge_type, 'drug'].edge_index = edge_index
                data['drug', edge_type, 'drug'].edge_attr = edge_attr
            else:
                logger.info(f'{edge_type} edge_type is not assigned!')

    
    logger.info('Checking isolated nodes...')
    all_drug_indices = set(range(data['drug'].x.size(0)))
    all_gene_indices = set(range(data['gene'].x.size(0)))
    connected_drugs = set()
    connected_genes = set()
    
    for edge_type, edge_index_dict in data.edge_index_dict.items():
        src_type, relation, dst_type = edge_type
        edge_index = edge_index_dict
        if src_type == 'drug':
            connected_drugs.update(edge_index[0].cpu().numpy())
        elif src_type == 'gene':
            connected_genes.update(edge_index[0].cpu().numpy())
            
        if dst_type == 'drug':
            connected_drugs.update(edge_index[1].cpu().numpy())
        elif dst_type == 'gene':
            connected_genes.update(edge_index[1].cpu().numpy())
    
    isolated_drugs = all_drug_indices - connected_drugs
    isolated_genes = all_gene_indices - connected_genes
    
    logger.info(f"Total number of drug nodes: {len(all_drug_indices)}")
    logger.info(f"Number of isolated drug nodes: {len(isolated_drugs)}")
    logger.info(f"Total number of gene nodes: {len(all_gene_indices)}")
    logger.info(f"Number of isolated gene nodes: {len(isolated_genes)}")
    
    if len(isolated_drugs) > 0 or len(isolated_genes) > 0:
        np.save(f"{expt_folder}/isolated_drug_indices.npy", np.array(list(isolated_drugs)))
        np.save(f"{expt_folder}/isolated_gene_indices.npy", np.array(list(isolated_genes)))

    data = data.to(device)

    if args.global_saved_model:
        logger.info(f"Loading pretrained global model from {args.global_saved_model}")
        model.load_state_dict(torch.load(args.global_saved_model)['model_state_dict'])

        logger.info('loading Phase 1 embeddings...')
        drug_embeddings, gene_embeddings = get_embeddings(model, data)
        phase1_drug_embeddings = drug_embeddings.cpu().detach().numpy()
        phase1_gene_embeddings = gene_embeddings.cpu().detach().numpy()
    else:
        logger.info("Starting Phase 1: Global Graph Training...")
        loader = NeighborLoader(
            data,
            num_neighbors=args.num_neighbors,
            # num_neighbors={
            #     ('gene', 'PPI', 'gene'): [10, 10],
            #     ('drug', 'DTI', 'gene'): [20, 10],
            #     ('gene', 'DTI', 'drug'): [20, 10],
            #     ('drug', 'DDS', 'drug'): [10, 10],
            # },
            batch_size=args.batch_size,
            input_nodes=('drug', torch.arange(data['drug'].x.size(0)))
        )

        model = train_contrastive_batch(
            model=model,
            data=data,
            optimizer=optimizer,
            loader=loader,
            epochs=args.epochs,
            patience=args.patience,
            num_neg_samples=args.num_neg_samples,
            temperature=args.temperature,
            expt_folder=expt_folder,
            logger=logger,
            nfold='global'
        )
        
        logger.info(f'Phase 1 completed! Training time: {int(time.time() - start_time)/60:.2f}min')
        

        num_dti_edges = sum(len(edge_index[0]) for edge_type, edge_index in data.edge_index_dict.items() if edge_type[1] == 'DTI')
        logger.info(f'Number of DTI edges in data: {num_dti_edges}')

        logger.info('Saving Phase 1 embeddings...')
        drug_embeddings, gene_embeddings = get_embeddings(model, data)
        phase1_drug_embeddings = drug_embeddings.cpu().detach().numpy()
        phase1_gene_embeddings = gene_embeddings.cpu().detach().numpy() #[18498, 128]
        np.save(f"{expt_folder}/phase1_drug_embeddings.npy", phase1_drug_embeddings)
        np.save(f"{expt_folder}/phase1_gene_embeddings.npy", phase1_gene_embeddings)

    if args.output_relation_matrix:
        logger.info('Extracting relation matrix...')
        phase1_gene_embeddings_mean = np.mean(phase1_gene_embeddings, axis=0, keepdims=True)  # Shape: [1, embed_dim]
        # Concatenate with original array
        phase1_gene_embeddings_arr = np.concatenate([phase1_gene_embeddings, phase1_gene_embeddings_mean], axis=0)  # Shape: [num_genes+1, embed_dim]
        # For Tahoe dataset, we use all genes (no cross-validation split)
        # Skip loading 978_gene_node_idx.npy which is for original dataset split
        all_gene_idx = np.arange(len(phase1_gene_embeddings))
        phase1_gene_embeddings_arr_all = phase1_gene_embeddings_arr[all_gene_idx ,:]
        relation_matrix = compute_relation_matrix(phase1_drug_embeddings, phase1_gene_embeddings_arr_all, similarity_metric='cosine')
        np.save(f"{expt_folder}/phase1_relation_matrix.npy", relation_matrix)
        logger.info(f'Saved relation_matrix with shape: {relation_matrix.shape}')

    if args.infer_dti_scores:
        logger.info('Inferring DTI scores...')
        dti_scores_matrix = predict_all_dti_scores(model, data, device, logger, batch_size=64)
        logger.info(f'dti_scores_matrix.shape:{dti_scores_matrix.shape}')
        phase1_dti_scores_mean = np.mean(dti_scores_matrix, axis=-1, keepdims=True)  # Shape: [1, embed_dim]
        # Concatenate with original array
        phase1_dti_arr = np.concatenate([dti_scores_matrix, phase1_dti_scores_mean], axis=-1)  # Shape: [num_drugs, num_genes+1]
        # For Tahoe dataset, we use all genes (no cross-validation split)
        all_gene_idx = np.arange(phase1_gene_embeddings.shape[0])
        phase1_dti_arr_all = phase1_dti_arr[:, all_gene_idx]
        logger.info(f'phase1_dti_arr_all.shape:{phase1_dti_arr_all.shape}')
        np.save(f"{expt_folder}/phase1_dti_scores_matrix_all.npy", phase1_dti_arr_all)


    logger.info(f'Total training time: {int(time.time() - start_time)/60:.2f}min')    
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    main()