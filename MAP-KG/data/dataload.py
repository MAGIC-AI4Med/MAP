import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class DrugGeneKG_Dataset(Dataset):
    def __init__(
        self,
        drug_node_path,
        gene_node_path,
        drug_gene_edge_path,
    ):
        self.drug_nodes = pd.read_csv(drug_node_path)
        self.gene_nodes = pd.read_csv(gene_node_path)
        self.drug_gene_edges = pd.read_csv(drug_gene_edge_path)

        self.length = (
            len(self.drug_nodes)
            + len(self.gene_nodes)
            + len(self.drug_gene_edges)
        )

        self.select_node = (len(self.drug_nodes) + len(self.gene_nodes)) / self.length
        self.when_node_select_drug = len(self.drug_nodes) / (
            len(self.drug_nodes) + len(self.gene_nodes)
        )

        self.drug_id_column = self.drug_nodes.columns[0]
        self.drug_name_column = self.drug_nodes.columns[1]
        self.gene_id_column = self.gene_nodes.columns[0]
        self.gene_name_column = self.gene_nodes.columns[1]

        self.drug_other_property_columns = [
            c for c in self.drug_nodes.columns
            if c not in [self.drug_id_column, self.drug_name_column]
        ]
        self.gene_other_property_columns = [
            c for c in self.gene_nodes.columns
            if c not in [self.gene_id_column, self.gene_name_column]
        ]

        print("Drug ID column: ", self.drug_id_column)
        print("Drug Name column: ", self.drug_name_column)
        print("Drug other columns: ", self.drug_other_property_columns)

        print("Gene ID column: ", self.gene_id_column)
        print("Gene Name column: ", self.gene_name_column)
        print("Gene other columns: ", self.gene_other_property_columns)

    def __len__(self):
        return self.length

    @staticmethod
    def _is_valid_value(v):
        return pd.notna(v) and str(v).strip() != ""

    def _sample_one_non_empty_from_columns(self, row, candidate_columns):
        non_empty_cols = [
            c for c in candidate_columns
            if self._is_valid_value(row[c])
        ]
        if len(non_empty_cols) == 0:
            print("ERROR: no non-empty columns in candidate set for row:\n", row)
            exit()
        col = np.random.choice(non_empty_cols)
        return row[col], col

    def _get_type_from_column(self, column_name, node_type):
        if node_type == "drug":
            if "smiles" in column_name.lower():
                return "smiles"
            else:
                return "text"
        elif node_type == "gene":
            if column_name == "ESM":
                return "esm"
            else:
                return "text"
        else:
            print(f"ERROR: invalid node_type: {node_type}")
            exit()

    def _sample_drug_node(self):
        idx = np.random.randint(0, len(self.drug_nodes))
        row = self.drug_nodes.iloc[idx]

        drug_id = row[self.drug_id_column]
        name_value = row[self.drug_name_column]

        if not self._is_valid_value(name_value):
            print(f"ERROR: Drug name column is empty for row:\n{row}")
            exit()

        # 从其它列中采样一个属性
        if len(self.drug_other_property_columns) == 0:
            print("ERROR: no other drug properties (other than id and name) to sample from.")
            exit()

        property_2, col_2 = self._sample_one_non_empty_from_columns(
            row,
            self.drug_other_property_columns
        )

        property_1 = name_value
        col_1 = self.drug_name_column

        type_1 = self._get_type_from_column(col_1, "drug")
        type_2 = self._get_type_from_column(col_2, "drug")

        relation = "xxx"
        return property_1, property_2, type_1, type_2, relation, drug_id, drug_id

    def _sample_gene_node(self):
        idx = np.random.randint(0, len(self.gene_nodes))
        row = self.gene_nodes.iloc[idx]

        gene_id = row[self.gene_id_column]
        name_value = row[self.gene_name_column]

        if not self._is_valid_value(name_value):
            print(f"ERROR: Gene name column is empty for row:\n{row}")
            exit()

        if len(self.gene_other_property_columns) == 0:
            print("ERROR: no other gene properties (other than id and name) to sample from.")
            exit()

        property_2, col_2 = self._sample_one_non_empty_from_columns(
            row,
            self.gene_other_property_columns
        )

        property_1 = name_value
        col_1 = self.gene_name_column

        type_1 = self._get_type_from_column(col_1, "gene")
        type_2 = self._get_type_from_column(col_2, "gene")

        relation = "xxx"
        return property_1, property_2, type_1, type_2, relation, gene_id, gene_id

    def _sample_drug_gene_edge(self):
        idx = np.random.randint(0, len(self.drug_gene_edges))
        edge = self.drug_gene_edges.iloc[idx]

        if "PubChem CID" not in edge or "Gene stable ID" not in edge or "relation" not in edge:
            print("ERROR: drug-gene edge row missing required columns:", edge)
            exit()

        drug_id = edge["PubChem CID"]
        gene_id = edge["Gene stable ID"]
        relation = edge["relation"]

        drug_rows = self.drug_nodes[self.drug_nodes[self.drug_id_column] == drug_id]
        if len(drug_rows) == 0:
            print("ERROR: [drug-gene] cannot find DRUG node for edge:\n", edge)
            exit()
        drug_row = drug_rows.iloc[0]

        gene_rows = self.gene_nodes[self.gene_nodes[self.gene_id_column] == gene_id]
        if len(gene_rows) == 0:
            print("ERROR: [drug-gene] cannot find GENE node for edge:\n", edge)
            exit()
        gene_row = gene_rows.iloc[0]

        drug_name = drug_row[self.drug_name_column]
        gene_name = gene_row[self.gene_name_column]

        if not self._is_valid_value(drug_name):
            print("ERROR: [drug-gene] DRUG name is empty for edge:\n", edge)
            exit()
        if not self._is_valid_value(gene_name):
            print("ERROR: [drug-gene] GENE name is empty for edge:\n", edge)
            exit()

        property_1 = drug_name
        property_2 = gene_name

        type_1 = self._get_type_from_column(self.drug_name_column, "drug")
        type_2 = self._get_type_from_column(self.gene_name_column, "gene")

        return property_1, property_2, type_1, type_2, relation, drug_id, gene_id

    def __getitem__(self, idx):
        is_node = np.random.random() < self.select_node

        if is_node:
            is_drug_node = np.random.random() < self.when_node_select_drug
            if is_drug_node:
                return self._sample_drug_node()
            else:
                return self._sample_gene_node()
        else:
            return self._sample_drug_gene_edge()