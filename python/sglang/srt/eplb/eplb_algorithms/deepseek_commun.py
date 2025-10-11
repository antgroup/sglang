from typing import Tuple
import time
import torch

def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly n/m objects and the weights of all packs
    are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs
    
    Returns: 
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1), dtype=torch.int64, device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device='cpu')
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min((i for i in range(num_packs) if pack_items[i] < groups_per_pack), 
                       key=pack_weights.__getitem__)
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(weight: torch.Tensor, num_phy: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication
    
    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def optimize_group_placement(pphy2log, comm_matrix, num_nodes, num_gpus, group_size):
    """
    Optimize the placement of expert groups to minimize inter-node communication cost.
    
    Parameters:
        pphy2log: [num_layers, num_physical_experts], physical to logical expert mapping
        comm_matrix: [num_logical_experts, num_logical_experts], communication cost between experts
        num_nodes: number of server nodes
        num_gpus: number of GPUs, must be a multiple of `num_nodes`
        group_size: number of experts in each group
        
    Returns:
        optimized_pphy2log: [num_layers, num_physical_experts], optimized physical to logical expert mapping
    """
    num_layers, num_physical_experts = pphy2log.shape
    num_groups = num_physical_experts // group_size
    groups_per_node = num_groups // num_nodes
    optimized_pphy2log = pphy2log.clone()
    
    # compute group start indices before-hand
    group_start_indices = [g * group_size for g in range(num_groups)]
    
    for layer in range(num_layers):
        # compute group to node mapping before-hand
        group_to_node = torch.zeros(num_groups, dtype=torch.int64)
        for g in range(num_groups):
            group_to_node[g] = (g * group_size) // (num_physical_experts // num_nodes)
        
        # get the leader expert of each group
        leader_experts = torch.zeros(num_groups, dtype=torch.int64)
        for g in range(num_groups):
            leader_idx = g * group_size
            leader_experts[g] = pphy2log[layer, leader_idx].item()
        
        # Only consider leader experts for inter-group communication cost
        group_comm_cost = torch.zeros((num_groups, num_groups), dtype=torch.float32)
        for g1 in range(num_groups):
            leader_expert_g1 = leader_experts[g1]
            for g2 in range(num_groups):
                if g1 != g2:
                    leader_expert_g2 = leader_experts[g2]
                    group_comm_cost[g1, g2] = comm_matrix[layer, leader_expert_g1, leader_expert_g2]
        
        # construct initial node groups
        node_groups = [[] for _ in range(num_nodes)]
        for g in range(num_groups):
            node_idx = group_to_node[g].item()
            node_groups[node_idx].append(g)
        
        # compute initial node pair costs
        node_pair_costs = {}
        for node1 in range(num_nodes):
            for node2 in range(node1 + 1, num_nodes):
                cost = 0
                for g1 in node_groups[node1]:
                    for g2 in node_groups[node2]:
                        cost += group_comm_cost[g1, g2]
                node_pair_costs[(node1, node2)] = cost
        
        # Do the optimization iterations
        improved = True
        iterations = 0
        max_iterations = 20
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Find the best swap
            best_gain = 0
            best_swap = None
            
            for node1 in range(num_nodes):
                for node2 in range(node1 + 1, num_nodes):
                    current_cost = node_pair_costs[(node1, node2)]
                    
                    # Try all pairs of groups between node1 and node2
                    for g1_idx, g1 in enumerate(node_groups[node1]):
                        for g2_idx, g2 in enumerate(node_groups[node2]):
                            gain = 0
                            
                            # Compute the gain from swapping g1 and g2
                            for other_g1 in node_groups[node1]:
                                if other_g1 != g1:
                                    gain += group_comm_cost[g1, other_g1]
                                    gain -= group_comm_cost[g2, other_g1]
                            
                            for other_g2 in node_groups[node2]:
                                if other_g2 != g2:
                                    gain += group_comm_cost[g2, other_g2]
                                    gain -= group_comm_cost[g1, other_g2]

                            if gain > best_gain:
                                best_gain = gain
                                best_swap = (node1, g1_idx, node2, g2_idx)

            if best_gain > 0 and best_swap:
                node1, g1_idx, node2, g2_idx = best_swap
                g1 = node_groups[node1][g1_idx]
                g2 = node_groups[node2][g2_idx]
                
                # update node groups
                node_groups[node1][g1_idx] = g2
                node_groups[node2][g2_idx] = g1
                
                # update node pair costs
                for n1, n2 in node_pair_costs:
                    if n1 == node1 or n1 == node2 or n2 == node1 or n2 == node2:
                        cost = 0
                        for g_n1 in node_groups[n1]:
                            for g_n2 in node_groups[n2]:
                                cost += group_comm_cost[g_n1, g_n2]
                        node_pair_costs[(n1, n2)] = cost

                # swap physical expert mapping
                for offset in range(group_size):
                    idx1 = g1 * group_size + offset
                    idx2 = g2 * group_size + offset
                    optimized_pphy2log[layer, idx1], optimized_pphy2log[layer, idx2] = \
                        optimized_pphy2log[layer, idx2].item(), optimized_pphy2log[layer, idx1].item()
                
                improved = True
                
        print(f"Layer {layer} optimized in {iterations} iterations")
    
    return optimized_pphy2log


def rebalance_experts_hierarchical(weight: torch.Tensor, num_physical_experts: int, 
                      num_groups: int, num_nodes: int, num_gpus: int, 
                      comm_matrix: torch.Tensor = None):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`
        comm_matrix: [num_logical_experts, num_logical_experts], communication cost between experts

    Returns: 
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups 
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(perm.shape))
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes) 
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) + 
                torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, num_physical_experts // num_nodes)    

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)

    
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy) # [num_layers * num_nodes, num_log_per_nodes]
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + 
                 torch.arange(0, num_logical_experts, num_logical_experts // num_nodes,
                              device=group_pack_index.device).view(1, -1, 1)).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    # Step 4: rearrange group placement according to communication cost
    start_time = time.perf_counter()
    if comm_matrix is not None:
        pphy2log = optimize_group_placement(pphy2log, comm_matrix, num_nodes, num_gpus, group_size)
    print(pphy2log)
    print(pphy2log.shape)
    end_time = time.perf_counter()
    print(f"Group placement optimization time: {end_time - start_time:.4f} seconds")
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt

def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    comm_matrix: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer with communication optimization.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all logical experts
        num_replicas: number of physical experts, must be a multiple of `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`
        comm_matrix: [num_logical_experts, num_logical_experts], communication cost between experts

    Returns: 
        physical_to_logical_map: [layers, num_replicas], the expert index of each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica indices for each expert
        expert_count: [layers, num_logical_experts], number of physical replicas for each logical expert
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()

    if comm_matrix is not None:
        comm_matrix = comm_matrix.float().cpu()
    
    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy with communication awareness
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus, comm_matrix)
    else:
        # use global load-balance policy with communication awareness
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus, comm_matrix)
    
    maxlogcnt = logcnt.max().item()
    log2phy: torch.Tensor = torch.full((num_layers, num_logical_experts, maxlogcnt), 
                                       -1, dtype=torch.int64, device=logcnt.device)
    log2phy.view(num_layers, -1).scatter_(-1, phy2log * maxlogcnt + phyrank, 
            torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(num_layers, -1))
    return phy2log, log2phy, logcnt