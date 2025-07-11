import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class SimpleGATv2Net(nn.Module):
    def __init__(self, float_feat_dim, hidden_dim, out_dim, heads=1):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=float_feat_dim + 1,  # +1 for binary input
            out_channels=out_dim,
            heads=heads,
            concat=False  # If True, output dim = out_dim * heads
        )

    def forward(self, x_float, x_binary, edge_index):
        """
        Parameters
        ----------
        x_float : Tensor of shape [num_nodes, float_feat_dim]
        x_binary : Tensor of shape [num_nodes] or [num_nodes, 1], binary 0/1
        edge_index : Tensor of shape [2, num_edges], COO format
        """
        if x_binary.ndim == 1:
            x_binary = x_binary.unsqueeze(1)  # [num_nodes, 1]
        x = torch.cat([x_float, x_binary], dim=1)  # [num_nodes, float_feat_dim + 1]
        out = self.gat(x, edge_index)
        return out

class GATv2NBNet(nn.Module):
    def __init__(self, float_feat_dim, hidden_dim):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=float_feat_dim + 1,
            out_channels=hidden_dim,
            heads=1,
            concat=True
        )

        # Output NB parameters per feature dimension
        self.to_logit = nn.Linear(hidden_dim, float_feat_dim)
        self.dispersion = nn.Parameter(torch.zeros(float_feat_dim))


    def forward(self, x_float, x_binary, edge_index):
        if x_binary.ndim == 1:
            x_binary = x_binary.unsqueeze(1)

        x = torch.cat([x_float, x_binary], dim=1)         # [num_nodes, float_feat_dim + 1]
        h = F.elu(self.gat(x, edge_index))                # [num_nodes, hidden_dim]

        logits = self.to_logit(h)                         # [num_nodes, float_feat_dim]
        dispersion = F.softplus(self.dispersion)    # [num_nodes, float_feat_dim]

        return logits, dispersion

class JointEmbeddingGATv2NBNet(nn.Module):
    def __init__(self, n_cell_types, embed_dim, hidden_dim, out_dim):
        super().__init__()

        # Instead of nn.Embedding, you take the one-hot cell type + lipid droplet scalar as input
        # and embed jointly with a linear layer.
        self.joint_embed = nn.Linear(n_cell_types + 1, embed_dim)

        self.gat = GATv2Conv(
            in_channels=embed_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=True
        )

        self.to_logit = nn.Linear(hidden_dim, out_dim)
        self.dispersion = nn.Parameter(torch.zeros(out_dim, n_cell_types))

    def forward(self, cell_type_onehot, lipid_droplet_scalar, edge_index):
        if lipid_droplet_scalar.ndim == 1:
            lipid_droplet_scalar = lipid_droplet_scalar.unsqueeze(1)  # [n_nodes,1]

        # Concatenate one-hot cell type and lipid droplet scalar: [n_nodes, n_cell_types + 1]
        combined = torch.cat([cell_type_onehot.float(), lipid_droplet_scalar.float()], dim=1)

        # Joint embedding: [n_nodes, embed_dim]
        x_embedding = F.elu(self.joint_embed(combined))

        h = F.elu(self.gat(x_embedding, edge_index))

        logits = self.to_logit(h)

        # Use cell_type_onehot to index dispersion per node (same as before)
        cell_type_indices = torch.argmax(cell_type_onehot, dim=1)
        dispersion = F.softplus(self.dispersion[:, cell_type_indices]).T  # [n_nodes, out_dim]

        return logits, dispersion
