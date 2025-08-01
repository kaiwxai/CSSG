import torch
import numpy as np
import multiprocessing as mp
import random
import math
import scipy.sparse as sp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

mp.set_start_method('spawn', True)

class Spatial_Side_Information():
    def __init__(self, set_node_index, epsilon, c, seed):
        super(Spatial_Side_Information, self).__init__()
        # self._anchor_node = set_node_index
        # self._node = np.concatenate((set_node_index[0], set_node_index[-1]))
        self._node = set_node_index
        self._num_nodes = len(self._node)
        self._epsilon = epsilon
        self._c = c
        self._seed = seed

    def get_random_anchorset(self, node_num):
        np.random.seed(self._seed)
        c = self._c
        # c = 10
        n = node_num
        distortion = math.ceil(np.log2(n))

        sampling_rep_rounds = c
        anchorset_num = sampling_rep_rounds * distortion
        anchorset_id = [np.array([]) for _ in range(anchorset_num)]
        for i in range(distortion):
            anchor_size = int(math.ceil(n / np.exp2(i + 1)))
            for j in range(sampling_rep_rounds):
                anchorset_id[i*sampling_rep_rounds+j] = np.sort(self._node[np.random.choice(n, size=anchor_size, replace=False)])
        return anchorset_id, anchorset_num

    def nodes_dist_range(self, adj, node_range):
        dists_dict = defaultdict(dict)
        for node in node_range:
            for neighbor in self._node:
                if neighbor not in dists_dict[node]:
                    dists_dict[node][neighbor] = 0
                diff = abs(adj[node, self._node] - adj[neighbor, self._node])
                if isinstance(adj, sp.coo_matrix):
                    dists_dict[node][neighbor] = sp.coo_matrix.sum(diff)
                elif isinstance(adj, np.ndarray):
                    dists_dict[node][neighbor] = np.sum(diff)
                if adj[node, neighbor] > 0:
                    dists_dict[node][neighbor] += 1
        return dists_dict

    def merge_dicts(self, dicts):
        result = {}
        for dictionary in dicts:
            result.update(dictionary)
        return result

    def all_pairs_dist_parallel(self, adj, num_workers=16):
        nodes = self._node
        if len(nodes) < 200:
            num_workers = int(num_workers/4)
        elif len(nodes) < 800:
            num_workers = int(num_workers/2)
        elif len(nodes) < 3000:
            num_workers = int(num_workers)
        slices = np.array_split(nodes, num_workers)
        pool = mp.Pool(processes = num_workers)
        results = [pool.apply_async(self.nodes_dist_range, args=(adj, slices[i], )) for i in range(num_workers)]
        output = [p.get() for p in results]
        dists_dict = self.merge_dicts(output)

        pool.close()
        pool.join()
        return dists_dict

    # 计算hamming距离dict
    def precompute_dist_data(self, adj):
        dists_array = np.zeros((self._num_nodes, self._num_nodes))
        # 并行或者不并行
        dists_dict = self.all_pairs_dist_parallel(adj)
        # dists_dict = self.nodes_hamming_dist_range(self._node)
        for i, node in enumerate(self._node):
            #dists_array[i] = list(dists_dict[node].values())
            dists_array[i] = np.array(list(dists_dict[node].values()))
        return dists_array

    def get_dist_min(self, dist, node_num):
        anchorset_id, anchorset_num = self.get_random_anchorset(node_num)
        # print(len(anchorset_id))
        dist_max = torch.zeros((dist.shape[0],len(anchorset_id)))
        dist_argmax = torch.zeros((dist.shape[0],len(anchorset_id))).long()
        coefficient = torch.ones(self._num_nodes, anchorset_num)
        for i in range(len(anchorset_id)):
            temp_id = torch.as_tensor(anchorset_id[i], dtype=torch.long)
            dist_temp = dist[:, temp_id]
            dist_max_temp, dist_argmax_temp = torch.min(dist_temp, dim=-1)
            dist_max[:,i] = dist_max_temp
            dist_argmax[:,i] = temp_id[dist_argmax_temp]
        return dist_max, dist_argmax, coefficient

    def spatial_emb_matrix(self, dist=None):
        node_num = dist.shape[0]
        spatial_emb_matrix, dist_argmin, coefficient = self.get_dist_min(dist, node_num)
        return spatial_emb_matrix, dist_argmin
    