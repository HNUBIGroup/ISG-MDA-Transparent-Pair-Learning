"""Multimodal ISG-MDA model definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelOptions:
    mirna_feature_dim: int = 128
    drug_input_dim: int = 69
    drug_hidden_dim: int = 128
    drug_feature_dim: int = 128
    gat_layers: int = 3
    gat_heads: int = 2
    rna_fm_model: str = "rna_fm_t12"
    chemberta_model: str = "DeepChem/ChemBERTa-77M-MLM"
    local_files_only: bool = False


class CNNFeatureExtractor(nn.Module):
    def __init__(self, output_dim: int, vocab_size: int = 5, embed_dim: int = 128,
                 num_filters: int = 64, filter_sizes: Sequence[int] = (2, 3, 4)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=size) for size in filter_sizes]
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)

    def forward(self, encoded_sequence: torch.Tensor) -> torch.Tensor:
        x = self.embedding(encoded_sequence).transpose(1, 2)
        pooled = []
        for convolution in self.convs:
            activated = F.relu(convolution(x))
            pooled.append(F.max_pool1d(activated, activated.size(2)).squeeze(2))
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class RNAFMFeatureExtractor(nn.Module):
    def __init__(self, output_dim: int, model_name: str = "rna_fm_t12", model_layer: int = 12):
        super().__init__()
        import fm

        if model_name != "rna_fm_t12":
            raise ValueError(f"Unsupported RNA-FM model: {model_name}")
        self.model, alphabet = fm.pretrained.rna_fm_t12()
        self.batch_converter = alphabet.get_batch_converter()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model_layer = model_layer
        self.projection = nn.Linear(640, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, sequences: Sequence[str]) -> torch.Tensor:
        records = [(f"sequence_{index}", sequence) for index, sequence in enumerate(sequences)]
        _, _, tokens = self.batch_converter(records)
        device = next(self.projection.parameters()).device
        with torch.no_grad():
            representations = self.model(tokens.to(device), repr_layers=[self.model_layer])
        cls_embeddings = representations["representations"][self.model_layer][:, 0, :]
        return self.dropout(self.projection(cls_embeddings))


class GATFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, heads: int = 2):
        super().__init__()
        from torch_geometric.nn import GATConv

        if num_layers < 2:
            raise ValueError("GAT requires at least two layers")
        self.convs = nn.ModuleList([GATConv(input_dim, hidden_dim // heads, heads=heads)])
        self.convs.extend(GATConv(hidden_dim, hidden_dim // heads, heads=heads) for _ in range(num_layers - 2))
        self.convs.append(GATConv(hidden_dim, output_dim, heads=1, concat=False))
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)] + [nn.BatchNorm1d(output_dim)]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
                return_attention: bool = False):
        from torch_geometric.nn import global_max_pool, global_mean_pool

        attention = []
        for index, convolution in enumerate(self.convs):
            if return_attention:
                x, weights = convolution(x, edge_index, return_attention_weights=True)
                attention.append(weights)
            else:
                x = convolution(x, edge_index)
            x = self.batch_norm[index](x)
            if index < len(self.convs) - 1:
                x = self.dropout(F.relu(x))
        embedding = global_mean_pool(x, batch) + global_max_pool(x, batch)
        return (embedding, attention) if return_attention else embedding


class ChemBERTaFeatureExtractor(nn.Module):
    def __init__(self, output_dim: int, model_name_or_path: str, local_files_only: bool = False):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.projection = nn.Linear(self.model.config.hidden_size, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, smiles: Sequence[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            list(smiles), padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
        return self.dropout(self.projection(outputs.last_hidden_state[:, 0, :]))


class CrossAttention(nn.Module):
    def __init__(self, feature_dim: int, heads: int = 2):
        super().__init__()
        if feature_dim % heads:
            raise ValueError("feature_dim must be divisible by attention heads")
        self.feature_dim = feature_dim
        self.heads = heads
        self.head_dim = feature_dim // heads
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size = query.size(0)
        q = self.query(query).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(key).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(value).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        weights = self.dropout(F.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim), dim=-1))
        context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)
        return self.out_proj(context)


class FeatureFusion(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.cross_attention = CrossAttention(feature_dim, heads=2)
        self.layer_norm = nn.LayerNorm(feature_dim)
        # Retained for state-dict compatibility with the training implementation.
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first = first.unsqueeze(1)
        second = second.unsqueeze(1)
        attended_first = self.cross_attention(first, second, second)
        attended_second = self.cross_attention(second, first, first)
        first_out = self.layer_norm(first + attended_first)
        second_out = self.layer_norm(second + attended_second)
        return (first_out + second_out).squeeze(1)


class InteractionPredictor(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        input_dim = feature_dim * 4
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(input_dim // 2, 1), nn.Sigmoid(),
        )

    def pair_features(self, mirna: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        if mirna.shape != drug.shape:
            raise ValueError(f"Entity feature shapes differ: {mirna.shape} versus {drug.shape}")
        return torch.cat([mirna, drug, torch.abs(mirna - drug), mirna * drug], dim=1)

    def forward(self, mirna: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.pair_features(mirna, drug)).squeeze(-1)


class ISGMDA(nn.Module):
    """ISG-MDA with injectable semantic encoders for offline testing."""

    def __init__(self, options: ModelOptions = ModelOptions(),
                 mirna_semantic_encoder: nn.Module | None = None,
                 drug_semantic_encoder: nn.Module | None = None):
        super().__init__()
        if options.mirna_feature_dim != options.drug_feature_dim:
            raise ValueError("miRNA and drug output dimensions must match")
        self.options = options
        self.mirna_cnn = CNNFeatureExtractor(options.mirna_feature_dim)
        self.drug_gat = GATFeatureExtractor(
            options.drug_input_dim, options.drug_hidden_dim, options.drug_feature_dim,
            options.gat_layers, options.gat_heads,
        )
        self.mirna_rnafm = mirna_semantic_encoder or RNAFMFeatureExtractor(
            options.mirna_feature_dim, options.rna_fm_model
        )
        self.drug_chembert = drug_semantic_encoder or ChemBERTaFeatureExtractor(
            options.drug_feature_dim, options.chemberta_model, options.local_files_only
        )
        self.mirna_fusion = FeatureFusion(options.mirna_feature_dim)
        self.drug_fusion = FeatureFusion(options.drug_feature_dim)
        self.interaction_predictor = InteractionPredictor(options.mirna_feature_dim)
        first_layer = self.interaction_predictor.predictor[0]
        if first_layer.in_features != 512 or first_layer.out_features != 256:
            raise RuntimeError("Unexpected prediction-head dimensions")

    @staticmethod
    def _batch_graphs(graphs, device: torch.device):
        from torch_geometric.data import Batch

        batch = Batch.from_data_list(graphs) if isinstance(graphs, list) else graphs
        return batch.to(device)

    def encode_entities(self, drug_graphs, drug_smiles: Sequence[str],
                        mirna_encoded: torch.Tensor, mirna_sequences: Sequence[str]):
        device = next(self.parameters()).device
        graph_batch = self._batch_graphs(drug_graphs, device)
        mirna_topology = self.mirna_cnn(mirna_encoded.to(device))
        mirna_semantic = self.mirna_rnafm(mirna_sequences)
        drug_topology = self.drug_gat(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
        drug_semantic = self.drug_chembert(drug_smiles)
        return self.mirna_fusion(mirna_topology, mirna_semantic), self.drug_fusion(drug_topology, drug_semantic)

    def forward(self, drug_graphs, drug_smiles: Sequence[str],
                mirna_encoded: torch.Tensor, mirna_sequences: Sequence[str]) -> torch.Tensor:
        mirna, drug = self.encode_entities(drug_graphs, drug_smiles, mirna_encoded, mirna_sequences)
        return self.interaction_predictor(mirna, drug)
