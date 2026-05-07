import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader


def MaxMinScalr(feat):
    diff = feat.max() - feat.min()
    return (feat - feat.min()) / diff


def log_nested_dict(raw_dict, logger, indent=0):
    for key, value in raw_dict.items():
        if isinstance(value, dict):
            logger.info(f'{"    " * indent}{key}:')
            log_nested_dict(value, logger, indent + 1)
        else:
            logger.info(f'{"    " * indent}{key}: {value}')

# Contrastive Loss (InfoNCE)
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j, z_neg):
        pos_sim = F.cosine_similarity(z_i, z_j) / self.temperature
        neg_sim = F.cosine_similarity(z_i.unsqueeze(1), z_neg, dim=-1) / self.temperature
        exp_pos = torch.exp(pos_sim)
        exp_neg = torch.exp(neg_sim).sum(dim=-1)
        loss = -torch.log(exp_pos / (exp_pos + exp_neg))
        return loss.mean()



def generate_negative_samples(num_src, num_dst, num_neg_samples, pos_edge_index=None):

    if pos_edge_index is not None:
        pos_edges = set(tuple(edge) for edge in pos_edge_index.t().tolist())
    else:
        pos_edges = set()
    
    neg_edges = []
    max_attempts = 3 * num_src * num_neg_samples
    attempts = 0
    
    while len(neg_edges) < num_src * num_neg_samples and attempts < max_attempts:

        src = torch.randint(0, num_src, (1,)).item()
        dst = torch.randint(0, num_dst, (1,)).item()
        edge = (src, dst)
        if edge not in pos_edges and edge not in neg_edges:
            neg_edges.append(edge)
        
        attempts += 1
    
    if len(neg_edges) < num_src * num_neg_samples:
        remaining = num_src * num_neg_samples - len(neg_edges)
        neg_src = torch.randint(0, num_src, (remaining,))
        neg_dst = torch.randint(0, num_dst, (remaining,))
        additional_edges = list(zip(neg_src.tolist(), neg_dst.tolist()))
        neg_edges.extend(additional_edges)
    
    neg_edges = torch.tensor(neg_edges, dtype=torch.long)
    neg_edge_index = neg_edges.t()
    
    return neg_edge_index


def generate_multi_relation_samples(data, num_neg_samples):
    pos_samples = {}
    neg_samples = {}
    
    for edge_type in data.edge_index_dict.keys():
        
        src_type, _, dst_type = edge_type
        num_src = data[src_type].x.size(0)
        num_dst = data[dst_type].x.size(0)

        if num_src > 0 and num_dst > 0:
            # print(f'num_src:{num_src}, num_dst:{num_dst}')
            # print(edge_type)
            # print(len(pos_samples[edge_type]))
            pos_samples[edge_type] = data[edge_type].edge_index
            neg_samples[edge_type] = generate_negative_samples(num_src, num_dst, num_neg_samples, pos_edge_index = pos_samples[edge_type])
    
    return pos_samples, neg_samples



def get_single_drug_subgraph(data, drug_idx, device):
    """Extract subgraph containing a single drug and all genes
    
    Args:
        data: HeteroData object containing the full graph
        drug_idx: integer index of the target drug
        
    Returns:
        subgraph: HeteroData object containing only one drug and all genes
    """
    subgraph = HeteroData()
    
    # Copy all gene nodes and features
    subgraph['gene'].x = data['gene'].x
    num_genes = data['gene'].x.size(0)
    
    # Copy only the specific drug node and its features
    subgraph['drug'].x = data['drug'].x[drug_idx:drug_idx+1] 

    has_drug_connection = False
    
    # Copy relevant edges
    for edge_type, edge_dict in data.edge_index_dict.items():
        src, rel, dst = edge_type
        edge_index = edge_dict
        
        if src == 'drug' and dst == 'gene':
            # Filter edges that start from the specific drug
            mask = (edge_index[0] == drug_idx)
            if mask.any():
                new_edge_index = edge_index[:, mask].clone()
                new_edge_index[0].fill_(0)  
                # Verify gene indices
                gene_mask = (new_edge_index[1] >= 0) & (new_edge_index[1] < num_genes)
                new_edge_index = new_edge_index[:, gene_mask]
                if new_edge_index.size(1) > 0:  # Only add if there are valid edges
                    subgraph[edge_type].edge_index = new_edge_index
                    has_drug_connection = True
                    
        elif src == 'gene' and dst == 'drug':
            # Filter edges that end at the specific drug
            mask = (edge_index[1] == drug_idx)
            if mask.any():
                new_edge_index = edge_index[:, mask].clone()
                new_edge_index[1].fill_(0)
                # Verify gene indices
                gene_mask = (new_edge_index[0] >= 0) & (new_edge_index[0] < num_genes)
                new_edge_index = new_edge_index[:, gene_mask]
                if new_edge_index.size(1) > 0:  # Only add if there are valid edges
                    subgraph[edge_type].edge_index = new_edge_index
                    has_drug_connection = True
                    
        elif src == 'gene' and dst == 'gene':
            # Copy all gene-gene edges
            # Verify gene indices
            mask = (edge_index[0] >= 0) & (edge_index[0] < num_genes) & \
                   (edge_index[1] >= 0) & (edge_index[1] < num_genes)
            if mask.any():
                subgraph[edge_type].edge_index = edge_index[:, mask]
    
    #  add virtual edges if none of edges related to the drug
    if not has_drug_connection:
        virtual_edge_index = torch.zeros((2, 1), dtype=torch.long).to(device)
        virtual_edge_index[0, 0] = 0 
        virtual_edge_index[1, 0] = 0
        subgraph[('drug', 'DTI', 'gene')].edge_index = virtual_edge_index
        
        virtual_edge_index = torch.zeros((2, 1), dtype=torch.long).to(device)
        virtual_edge_index[0, 0] = 0
        virtual_edge_index[1, 0] = 0
        subgraph[('gene', 'DTI', 'drug')].edge_index = virtual_edge_index
    
    return subgraph



def get_batch_drug_subgraph(data, drug_indices, device):
    """Extract drug-specific subgraph containing all genes and the target drugs"""
    subgraph = HeteroData()
    
    # Copy all gene nodes and features
    subgraph['gene'].x = data['gene'].x
    num_genes = data['gene'].x.size(0)
    
    # Copy only the specific drug nodes and their features
    subgraph['drug'].x = data['drug'].x[drug_indices]
    
    # Create index mapping for drugs
    drug_idx_map = {int(old_idx): new_idx for new_idx, old_idx in enumerate(drug_indices)}
    
    has_drug_connection = False
    
    # Copy relevant edges
    for edge_type, edge_dict in data.edge_index_dict.items():
        src, rel, dst = edge_type
        edge_index = edge_dict
        
        if src == 'drug' and dst == 'gene':
            # Filter edges that start from the specific drugs
            mask = torch.isin(edge_index[0], drug_indices).to(device)
            if mask.any():
                new_edge_index = edge_index[:, mask].clone()
                # Remap drug indices
                new_src = torch.tensor([drug_idx_map[int(idx)] for idx in new_edge_index[0]]).to(device)
                # Verify gene indices
                gene_mask = (new_edge_index[1] >= 0) & (new_edge_index[1] < num_genes)
                new_edge_index = torch.stack([new_src[gene_mask], new_edge_index[1][gene_mask]])
                if new_edge_index.size(1) > 0:  # Only add if there are valid edges
                    subgraph[edge_type].edge_index = new_edge_index
                    has_drug_connection = True
                    
        elif src == 'gene' and dst == 'drug':
            # Filter edges that end at the specific drugs
            mask = torch.isin(edge_index[1], drug_indices).to(device)
            if mask.any():
                new_edge_index = edge_index[:, mask].clone()
                # Remap drug indices
                new_dst = torch.tensor([drug_idx_map[int(idx)] for idx in new_edge_index[1]]).to(device)
                # Verify gene indices
                gene_mask = (new_edge_index[0] >= 0) & (new_edge_index[0] < num_genes)
                new_edge_index = torch.stack([new_edge_index[0][gene_mask], new_dst[gene_mask]])
                if new_edge_index.size(1) > 0:  # Only add if there are valid edges
                    subgraph[edge_type].edge_index = new_edge_index
                    has_drug_connection = True
                    
        elif src == 'gene' and dst == 'gene':
            # Verify gene indices
            mask = (edge_index[0] >= 0) & (edge_index[0] < num_genes) & \
                   (edge_index[1] >= 0) & (edge_index[1] < num_genes)
            if mask.any():
                subgraph[edge_type].edge_index = edge_index[:, mask]

    # add a virtual self-loop edge if no edges related to the drug
    if not has_drug_connection:

        virtual_edge_index = torch.zeros((2, 1), dtype=torch.long).to(device)
        virtual_edge_index[0, 0] = 0
        virtual_edge_index[1, 0] = 0
        subgraph[('drug', 'DDS', 'drug')].edge_index = virtual_edge_index

    return subgraph


def get_high_confidence_edges(scores, threshold=0.5, transform=False):

    if scores.dim() == 1:
        num_src = int(torch.sqrt(float(scores.size(0))))
        num_dst = num_src
        scores = scores.view(num_src, num_dst)

    if transform:
        scores = torch.sigmoid(scores)

    high_conf_mask = scores > threshold

    src_idx, dst_idx = torch.where(high_conf_mask)

    if len(src_idx) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=scores.device), \
               torch.zeros((0,), dtype=scores.dtype, device=scores.device)

    edge_scores = scores[src_idx, dst_idx]

    sorted_idx = torch.argsort(edge_scores, descending=True)
    src_idx = src_idx[sorted_idx]
    dst_idx = dst_idx[sorted_idx]
    edge_scores = edge_scores[sorted_idx]

    high_conf_edges = torch.stack([src_idx, dst_idx], dim=0)
    
    return high_conf_edges, edge_scores



def update_graph_with_predictions(data, base_model, num_drugs, device, batch_size=64, threshold=0.5, logger=None, args=None):

    updated_data = data.clone()

    loader = NeighborLoader(
        data,
        num_neighbors=[600,400,200],
        batch_size=batch_size,
        input_nodes=('drug', torch.arange(num_drugs)),
        shuffle=False
    )


    for batch in loader:
        batch = batch.to(device)
        batch_drug_indices = batch['drug'].input_id
        # logger.info(f'batch_drug_indices: {batch_drug_indices}')

        if not batch.validate():
            logger.warning(f"Invalid batch for drugs {batch_drug_indices[0]}-{batch_drug_indices[-1]}")
            continue


        with torch.no_grad():
            base_model.eval()
            x_dict = base_model(batch.x_dict, batch.edge_index_dict)
            
            dti_scores = base_model.predict_relations(x_dict, 'DTI')
            predicted_dti_edges, _ = get_high_confidence_edges(dti_scores, threshold, transform=True)

            dds_scores = base_model.predict_relations(x_dict, 'DDS')
            predicted_dds_edges, edge_scores = get_high_confidence_edges(dds_scores, threshold)
            
            # update DTI edges
            if len(predicted_dti_edges[0]) > 0:
                for edge_idx in range(len(predicted_dti_edges[0])):
                    drug_idx = predicted_dti_edges[0][edge_idx]  
                    gene_idx = predicted_dti_edges[1][edge_idx]
                    
                    if ('drug', 'DTI', 'gene') in updated_data.edge_index_dict:
                        updated_data[('drug', 'DTI', 'gene')].edge_index = torch.cat([
                            updated_data[('drug', 'DTI', 'gene')].edge_index,
                            torch.tensor([[drug_idx], [gene_idx]], device=device)
                        ], dim=1)
                        
                        updated_data[('gene', 'DTI', 'drug')].edge_index = torch.cat([
                            updated_data[('gene', 'DTI', 'drug')].edge_index,
                            torch.tensor([[gene_idx], [drug_idx]], device=device)
                        ], dim=1)
            
             # update DDI edges
            if len(predicted_dds_edges[0]) > 0:
                for edge_idx in range(len(predicted_dds_edges[0])):
                    drug1_idx = predicted_dds_edges[0][edge_idx]
                    drug2_idx = predicted_dds_edges[1][edge_idx]
                    
                    if ('drug', 'DDS', 'drug') in updated_data.edge_index_dict:
                        updated_data[('drug', 'DDS', 'drug')].edge_index = torch.cat([
                            updated_data[('drug', 'DDS', 'drug')].edge_index,
                            torch.tensor([[drug1_idx], [drug2_idx]], device=device)
                        ], dim=1)
                        updated_data[('drug', 'DDS', 'drug')].edge_attr = torch.cat([
                            updated_data[('drug', 'DDS', 'drug')].edge_attr,
                            edge_scores[edge_idx].unsqueeze(0)
                        ])
                        
                        updated_data[('drug', 'DDS', 'drug')].edge_index = torch.cat([
                            updated_data[('drug', 'DDS', 'drug')].edge_index,
                            torch.tensor([[drug2_idx], [drug1_idx]], device=device)
                        ], dim=1)
                        updated_data[('drug', 'DDS', 'drug')].edge_attr = torch.cat([
                            updated_data[('drug', 'DDS', 'drug')].edge_attr,
                            edge_scores[edge_idx].unsqueeze(0)
                        ])
        
        predicted_dds_edges = [[]]
        
        logger.info(f"Processed drugs {batch_drug_indices[0]}-{batch_drug_indices[-1]} out of {num_drugs}, "
                   f"added {len(predicted_dti_edges[0])} DTI edges and {len(predicted_dds_edges[0])} DDS edges")
            
        torch.cuda.empty_cache()

    return updated_data




def get_embeddings(model, data):
    model.eval()
    with torch.no_grad():

        x_dict = model(data.x_dict, data.edge_index_dict)
        drug_embeddings = x_dict['drug']
        gene_embeddings = x_dict['gene']
    return drug_embeddings, gene_embeddings





class DrugSpecificLoss(torch.nn.Module):

    def __init__(self, dti_weight=1.0, topology_weight=0.1):
        super().__init__()
        self.dti_criterion = torch.nn.BCEWithLogitsLoss()
        self.topology_criterion = torch.nn.MSELoss()
        self.dti_weight = dti_weight
        self.topology_weight = topology_weight
        
    def compute_topology_loss(self, x_dict, subgraph):

        topology_loss = 0.0
        

        if ('gene', 'PPI', 'gene') in subgraph.edge_index_dict and subgraph.x_dict['gene'].size(0)>1:
            edge_index = subgraph[('gene', 'PPI', 'gene')].edge_index
            
            gene_emb = x_dict['gene']
            src_emb = gene_emb[edge_index[0]]
            dst_emb = gene_emb[edge_index[1]]

            src_emb_norm = F.normalize(src_emb, p=2, dim=1)
            dst_emb_norm = F.normalize(dst_emb, p=2, dim=1)
            pred_sim = torch.sum(src_emb_norm * dst_emb_norm, dim=1)
            
            target_sim = torch.ones_like(pred_sim)
            topology_loss_ppi = self.topology_criterion(pred_sim, target_sim)
        
            if not torch.isnan(topology_loss_ppi):
                topology_loss += topology_loss_ppi
            else:
                topology_loss += 0.
            
        if ('drug', 'DTI', 'gene') in subgraph.edge_index_dict and subgraph.x_dict['gene'].size(0)>0:
            edge_index = subgraph[('drug', 'DTI', 'gene')].edge_index
            
            drug_emb = x_dict['drug']
            gene_emb = x_dict['gene']
            src_emb = drug_emb[edge_index[0]]
            dst_emb = gene_emb[edge_index[1]]
            
            src_emb_norm = F.normalize(src_emb, p=2, dim=1)
            dst_emb_norm = F.normalize(dst_emb, p=2, dim=1)
            pred_sim = torch.sum(src_emb_norm * dst_emb_norm, dim=1)
            
            target_sim = torch.ones_like(pred_sim)
            
            topology_loss_dti = self.topology_criterion(pred_sim, target_sim)
            if not torch.isnan(topology_loss_dti):
                topology_loss += topology_loss_dti
            else:
                topology_loss += 0.
            
        return topology_loss
        
    def forward(self, x_dict, subgraph, predicted_dti, known_dti):

        dti_loss = self.dti_criterion(predicted_dti, known_dti)
        # print('dti_loss:', dti_loss)
        
        topology_loss = self.compute_topology_loss(x_dict, subgraph)
        # print('topology_loss:', topology_loss)
        
        total_loss = self.dti_weight * dti_loss + self.topology_weight * topology_loss
        # print('total_loss:', total_loss)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'dti_loss': dti_loss.item(),
            'topology_loss': topology_loss.item()
        }
        
        return total_loss, loss_dict



def get_known_dti(batch):

    edge_index = batch[('drug', 'DTI', 'gene')].edge_index
    num_genes = batch['gene'].x.size(0)
    
    known_dti = torch.zeros(num_genes, device=edge_index.device)
    
    known_dti[edge_index[1]] = 1.0
    
    return known_dti