import torch
from torch_geometric.data import Data

import numpy as np
from scipy.spatial import cKDTree

def build_edge_index_from_anndata(adata, k=6):
    """
    Build a torch_geometric edge_index from AnnData using k-NN over x_centroid and y_centroid.

    Parameters
    ----------
    adata : AnnData
        AnnData object with 'x_centroid' and 'y_centroid' in `adata.obs`.
    k : int
        Number of nearest neighbors to connect.

    Returns
    -------
    edge_index : torch.LongTensor
        Tensor of shape [2, num_edges] with COO-format edge indices.
    """
    coords = adata.obs[["x_centroid", "y_centroid"]].to_numpy()
    tree = cKDTree(coords)

    # Query k+1 because the closest point is the point itself
    distances, neighbors = tree.query(coords, k=k + 1)

    # Build COO-format edge_index
    row_indices = []
    col_indices = []

    for i, nbrs in enumerate(neighbors):
        for j in nbrs[1:]:  # skip self-loop (first neighbor is the point itself)
            row_indices.append(i)
            col_indices.append(j)

    edge_index = torch.tensor([row_indices, col_indices], dtype=torch.long)
    return edge_index

def generate_torch_geometric_data(adata, k=10):
    x = torch.tensor(adata.X)
    edge_index = build_edge_index_from_anndata(adata, k=k)
    return Data(x=x, edge_index=edge_index)

def counts_to_size_factors(counts, method="mean"):
    """
    Convert a vector of counts to size factors.
    Args:
        counts (array-like or torch.Tensor): Vector of counts per cell.
        method (str): Method for normalization ('median' or 'mean').
    Returns:
        torch.Tensor: Size factors for each cell.
    """

    counts = torch.as_tensor(counts, dtype=torch.float32).flatten()
    if method == "median":
        size_factor = counts / torch.median(counts)
    elif method == "mean":
        size_factor = counts / torch.mean(counts)
    else:
        raise ValueError("method must be 'median' or 'mean'")
    return torch.log(size_factor)