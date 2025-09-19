import numpy as np
import torch
#import numba

# Could enable numba for performance optimization if needed
# @numba.jit(nopython=True, parallel=True, cache=True)
def compute_expert_co_occurrence_matrix(history_data, num_experts):
    """Compute expert co-occurrence matrix from history data."""
    history_data = history_data.cpu().numpy()
    num_samples, num_layers, top_k = history_data.shape
    expert_co_occurrence = np.zeros((num_layers, num_experts, num_experts), dtype=np.int64)
    
    for sample_idx in range(num_samples):
        for layer_idx in range(num_layers):
            experts = history_data[sample_idx, layer_idx]
            if (-1 in experts) or (len(set(experts)) < top_k):
                continue
            for i in range(top_k):
                for j in range(i+1, top_k):
                    expert_i = experts[i]
                    expert_j = experts[j]
                    
                    if expert_i < num_experts and expert_j < num_experts:
                        expert_co_occurrence[layer_idx, expert_i, expert_j] += 1
                        expert_co_occurrence[layer_idx, expert_j, expert_i] += 1
    co_occurrence = torch.tensor(expert_co_occurrence, dtype=torch.int64)

    return co_occurrence

def generate_comm_matrix(history_data, num_experts):
    """
    Process input tensor to compute expert co-occurrence matrix and generate communication matrix
    """
    
    if history_data.numel() == 0:
        return None
    co_occurrence = compute_expert_co_occurrence_matrix(history_data, num_experts)
    comm_matrix = co_occurrence.float()
    comm_matrix = comm_matrix / comm_matrix.max()  # Normalize to [0,1]

    return comm_matrix
