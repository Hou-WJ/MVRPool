import os
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import TUDataset
from typing import Literal
from typing import Callable

from torch_geometric.utils import to_undirected

import torch
import numpy as np
import warnings
from typing import List, Optional, Dict, Any, Tuple
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

try:
    from torch_cluster import knn_graph

    TORCH_CLUSTER_AVAILABLE = True
except ImportError:
    TORCH_CLUSTER_AVAILABLE = False

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def is_connected_union_find(edge_index, num_nodes):
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    parent = list(range(num_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        union(u, v)

    root = find(0)
    for i in range(1, num_nodes):
        if find(i) != root:
            return False
    return True

class AdjacencyRewiringAugmentation:
    def __init__(self, device: Optional[str] = None, dtype: torch.dtype = torch.long):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

    def random_edge_dropout(self, edge_index: torch.Tensor, dropout_rate: float = 0.1) -> torch.Tensor:
        if dropout_rate == 0:
            return edge_index
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges, device=edge_index.device) > dropout_rate
        return edge_index[:, mask]

    def build_fully_connected(self, num_nodes: int) -> torch.Tensor:
        rows_upper, cols_upper = torch.triu_indices(num_nodes, num_nodes, offset=1)
        rows = torch.cat([rows_upper, cols_upper])
        cols = torch.cat([cols_upper, rows_upper])
        edge_index = torch.stack([rows, cols], dim=0)
        return edge_index

    def build_higher_order_graphs(
            self,
            edge_index: torch.Tensor,
            num_nodes: int,
            order: int = 2,
            include_self_loops: bool = False,
            normalize: bool = True,
            threshold: Optional[float] = None
    ) -> torch.Tensor:
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device, dtype=torch.float)
        adj[edge_index[0], edge_index[1]] = 1.0
        if not include_self_loops:
            adj.fill_diagonal_(0)

        higher_adj = torch.matrix_power(adj, order)

        if normalize:
            row_sum = higher_adj.sum(dim=1, keepdim=True)
            row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
            higher_adj = higher_adj / row_sum

        higher_adj_max = torch.max(higher_adj)
        if threshold is not None and higher_adj_max > threshold:
            higher_adj = (higher_adj > threshold).float()

        if not include_self_loops:
            higher_adj.fill_diagonal_(0)

        nonzero = higher_adj.nonzero(as_tuple=False)
        if nonzero.size(0) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        else:
            edge_index = nonzero.t().contiguous()
            edge_index = edge_index.to(torch.long)
        return edge_index

    def build_using_spectral_embedding(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            num_nodes: int,
            k: int = 1,
            include_original: bool = True
    ) -> torch.Tensor:
        adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float, device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        adj = adj.cpu().numpy()
        embedding = self._compute_spectral_embedding(adj, dim=2)
        if embedding is None:
            return edge_index
        from scipy.spatial.distance import cdist
        distances = cdist(embedding, embedding, metric='euclidean')
        augmented_edges = []
        for i in range(num_nodes):
            d_i = distances[i].copy()
            d_i[i] = -1
            farthest = np.argsort(-d_i)[:k]
            for j in farthest:
                if d_i[j] > 0:
                    augmented_edges.append([i, j])
                    augmented_edges.append([j, i])
        if include_original:
            all_edges = edge_index.t().tolist() + augmented_edges
        else:
            all_edges = augmented_edges
        if all_edges:
            unique_edges = list(set(tuple(edge) for edge in all_edges))
            edge_index = torch.tensor(unique_edges, device=edge_index.device).t()
        else:
            edge_index = torch.empty((2, 0), dtype=self.dtype, device=edge_index.device)
        return edge_index

    def _floyd_warshall(self, adj: np.ndarray) -> np.ndarray:
        """Floyd-Warshall 算法计算最短路径距离"""
        n = adj.shape[0]
        dist = adj.copy().astype(float)
        dist[dist == 0] = np.inf
        np.fill_diagonal(dist, 0)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, j] > dist[i, k] + dist[k, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        dist[dist == np.inf] = -1
        return dist

    def _compute_spectral_embedding(self, adj: np.ndarray, dim: int = 2) -> Optional[np.ndarray]:
        """计算图的谱嵌入（拉普拉斯矩阵最小非零特征值对应的特征向量）"""
        try:
            n_nodes = adj.shape[0]
            # 对称化
            adj_sym = (adj + adj.T) / 2
            deg = adj_sym.sum(axis=1)
            deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-10))
            laplacian = np.eye(n_nodes) - deg_inv_sqrt @ adj_sym @ deg_inv_sqrt
            # 计算特征值特征向量
            if n_nodes <= 100:
                from scipy.linalg import eigh
                eigvals, eigvecs = eigh(laplacian)
            else:
                from scipy.sparse.linalg import eigs
                eigvals, eigvecs = eigs(laplacian, k=min(dim + 1, n_nodes - 1), which='SM', maxiter=10000)
                eigvals = np.real(eigvals)
                eigvecs = np.real(eigvecs)
            idx = np.argsort(eigvals)
            eigvecs = eigvecs[:, idx]
            # 取最小的非零特征向量（跳过第一个）
            embedding = eigvecs[:, 1:dim + 1]
            return embedding
        except Exception as e:
            warnings.warn(f"Spectral embedding failed: {e}")
            return None

    def build_acyclic_view(
            self,
            edge_index: torch.Tensor,
            num_nodes: int,
            method: str = 'spanning_tree',
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return edge_index
        if method == 'spanning_tree':
            return self._build_spanning_tree(edge_index, num_nodes)
        elif method == 'dfs_removal':
            return self._build_dfs_acyclic(edge_index, num_nodes)
        else:
            raise ValueError(f"Unknown acyclic method: {method}")

    def _build_spanning_tree(self, edge_index, num_nodes):
        adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float, device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        adj = (adj + adj.t()).clamp(max=1)
        adj_np = adj.cpu().numpy()
        adj_sparse = csr_matrix(adj_np)
        mst = minimum_spanning_tree(adj_sparse)  # 返回稀疏矩阵
        mst_coo = mst.tocoo()
        if mst_coo.nnz == 0:
            return torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)
        rows = torch.tensor(mst_coo.row, device=edge_index.device, dtype=torch.long)
        cols = torch.tensor(mst_coo.col, device=edge_index.device, dtype=torch.long)
        mst_edges = torch.stack([torch.cat([rows, cols]), torch.cat([cols, rows])], dim=0)
        mst_edges = torch.unique(mst_edges, dim=1)
        return mst_edges

    def _build_dfs_acyclic(self, edge_index, num_nodes):
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            adj_list[u].append((v, i))
        visited = [0] * num_nodes
        keep = set()

        def dfs(u):
            visited[u] = 1
            for v, idx in adj_list[u]:
                if visited[v] == 0:
                    keep.add(idx)
                    dfs(v)
            visited[u] = 2

        for start in range(num_nodes):
            if visited[start] == 0:
                dfs(start)
        if keep:
            kept_edges = edge_index[:, list(keep)]
        else:
            kept_edges = torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)
        return kept_edges


    def build_curvature_based_augmentation(
            self,
            edge_index: torch.Tensor,
            num_nodes: int,
            curvature_type: Literal[
                'ollivier_ricci', 'balanced_forman', 'combinatorial_forman'] = 'combinatorial_forman',
            dynamic_update_fraction: float = 0.5,
            use_long_range: bool = True,
            use_curvature_max: bool = True
    ) -> torch.Tensor:
        device = edge_index.device
        adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)
        adj[edge_index[0], edge_index[1]] = True
        deg = adj.sum(dim=1).long()

        if curvature_type == 'ollivier_ricci':
            curvatures = self._compute_ollivier_ricci_approx(adj, edge_index, deg)
        elif curvature_type == 'balanced_forman':
            curvatures = self._compute_balanced_forman_curvature(adj, edge_index, deg)
        else:
            curvatures = self._compute_balanced_forman_curvature(adj, edge_index, deg)

        neg_mask = curvatures < 0
        neg_edges = edge_index[:, neg_mask]
        neg_curv = curvatures[neg_mask]
        sort_idx = torch.argsort(neg_curv)
        neg_edges = neg_edges[:, sort_idx]
        neg_curv = neg_curv[sort_idx]

        if neg_edges.size(1) == 0:
            return edge_index

        adj_np = adj.cpu().numpy().astype(float)
        if hasattr(self, '_floyd_warshall'):
            dist = self._floyd_warshall(adj_np)
        else:
            from scipy.sparse.csgraph import floyd_warshall
            dist = floyd_warshall(adj_np, directed=False, unweighted=True)
            dist[dist == np.inf] = -1

        long_candidates = [[] for _ in range(num_nodes)]
        for u in range(num_nodes):
            candidates = []
            for v in range(num_nodes):
                if v != u and dist[u, v] > 1 and dist[u, v] != -1:
                    candidates.append((dist[u, v], v))
            candidates.sort(key=lambda x: (-x[0], x[1]))
            long_candidates[u] = [v for _, v in candidates]

        used_long = [set() for _ in range(num_nodes)]

        node_curv_sum = torch.zeros(num_nodes, dtype=torch.float, device=device)
        for i in range(edge_index.size(1)):
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()
            c = curvatures[i].item()
            node_curv_sum[u] += c
            node_curv_sum[v] += c
        current_curv_sum = node_curv_sum.clone()

        new_edges_set = set()
        for i in range(edge_index.size(1)):
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()
            new_edges_set.add((u, v))
            new_edges_set.add((v, u))

        def get_max_curv_sum_non_neighbor(node, adj_mat, curv_sum):
            neighbors = set(adj_mat[node].nonzero(as_tuple=False).squeeze(1).cpu().tolist()) if adj_mat[
                node].any() else set()
            neighbors.add(node)
            best_node = None
            best_val = -float('inf')
            for v in range(num_nodes):
                if v not in neighbors:
                    val = curv_sum[v].item()
                    if val > best_val:
                        best_val = val
                        best_node = v
            return best_node

        for idx in range(neg_edges.size(1)):
            u = neg_edges[0, idx].item()
            v = neg_edges[1, idx].item()
            c = neg_curv[idx].item()

            if use_long_range:
                long_u = None
                for cand in long_candidates[u]:
                    if cand not in used_long[u]:
                        long_u = cand
                        used_long[u].add(cand)
                        used_long[cand].add(u)
                        break
                if long_u is not None and long_u != u:
                    if (u, long_u) not in new_edges_set:
                        new_edges_set.add((u, long_u))
                        new_edges_set.add((long_u, u))
                        current_curv_sum[long_u] += c * dynamic_update_fraction

            if use_curvature_max:
                max_node_u = get_max_curv_sum_non_neighbor(u, adj, current_curv_sum)
                if max_node_u is not None and max_node_u != u:
                    if (u, max_node_u) not in new_edges_set:
                        new_edges_set.add((u, max_node_u))
                        new_edges_set.add((max_node_u, u))
                        current_curv_sum[max_node_u] += c * dynamic_update_fraction

            if use_long_range:
                long_v = None
                for cand in long_candidates[v]:
                    if cand not in used_long[v]:
                        long_v = cand
                        used_long[v].add(cand)
                        used_long[cand].add(v)
                        break
                if long_v is not None and long_v != v:
                    if (v, long_v) not in new_edges_set:
                        new_edges_set.add((v, long_v))
                        new_edges_set.add((long_v, v))
                        current_curv_sum[long_v] += c * dynamic_update_fraction

            if use_curvature_max:
                max_node_v = get_max_curv_sum_non_neighbor(v, adj, current_curv_sum)
                if max_node_v is not None and max_node_v != v:
                    if (v, max_node_v) not in new_edges_set:
                        new_edges_set.add((v, max_node_v))
                        new_edges_set.add((max_node_v, v))
                        current_curv_sum[max_node_v] += c * dynamic_update_fraction

        edge_list = list(new_edges_set)
        if edge_list:
            new_edge_index = torch.tensor(edge_list, device=device).t().contiguous()
        else:
            new_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        return new_edge_index


    def _compute_balanced_forman_curvature(
            self, adj: torch.Tensor, edge_index: torch.Tensor, deg: torch.Tensor
    ) -> torch.Tensor:

        device = adj.device
        N = adj.size(0)

        if adj.dtype != torch.bool:
            adj = adj.bool()

        A2 = torch.mm(adj.float(), adj.float()).to(torch.long)  # (N, N)

        deg_float = deg.float()
        neighbors = [adj[i].nonzero(as_tuple=True)[0] for i in range(N)]

        curvatures = torch.zeros(edge_index.size(1), device=device, dtype=torch.float)

        for e, (u, v) in enumerate(edge_index.t()):
            u, v = u.item(), v.item()
            du = deg_float[u].item()
            dv = deg_float[v].item()

            if du < 1.5 or dv < 1.5 or min(du, dv) == 1:
                curvatures[e] = 0.0
                continue

            common = A2[u, v].item()
            max_deg = max(du, dv)
            min_deg = min(du, dv)
            tri_part = (2.0 / du) + (2.0 / dv) - 2.0 + \
                       2.0 * common / max_deg + common / min_deg

            set_n_u = set(neighbors[u])
            set_n_v = set(neighbors[v])

            A_candidates = [k for k in set_n_u if k != v and k not in set_n_v]
            B_candidates = [l for l in set_n_v if l != u and l not in set_n_u]

            if not A_candidates or not B_candidates:
                curvatures[e] = tri_part
                continue

            A_idx = torch.tensor(A_candidates, device=device, dtype=torch.long)
            B_idx = torch.tensor(B_candidates, device=device, dtype=torch.long)

            sub_adj = adj[A_idx][:, B_idx].to(torch.float)

            m_k = sub_adj.sum(dim=1)
            m_l = sub_adj.sum(dim=0)

            L_count = (m_k > 0).sum().item()
            U_count = (m_l > 0).sum().item()

            max_m_k = m_k.max().item() if m_k.numel() > 0 else 0
            max_m_l = m_l.max().item() if m_l.numel() > 0 else 0
            gamma_max = max(max_m_k, max_m_l)

            if gamma_max == 0:
                quad_part = 0.0
            else:
                quad_part = (L_count + U_count) / (max_deg * gamma_max)

            curvatures[e] = tri_part + quad_part

        return curvatures

    def _compute_combinatorial_forman_curvature(self, adj: torch.Tensor, edge_index: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        common = []
        edges = edge_index.t()
        for i in range(edges.size(0)):
            u, v = edges[i, 0].item(), edges[i, 1].item()
            cn = (adj[u] & adj[v]).sum().item()
            common.append(cn)
        common = torch.tensor(common, device=adj.device)
        deg_u = deg[edge_index[0]]
        deg_v = deg[edge_index[1]]
        curvature = 2 - deg_u - deg_v + common
        return curvature.float()


class RewiredTUDataset(InMemoryDataset):
    def __init__(
            self,
            root: str,
            name: str,
            viewlist: List[Dict[str, Any]],
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
    ):
        self.name = name
        self.viewlist = viewlist
        self.augmentor = AdjacencyRewiringAugmentation(device='cpu', dtype=torch.long)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self._node_curvature_cache = {}
        self._curvature_cache = {}
        self._spectral_gap_cache = {}

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        import hashlib
        config_str = str(self.viewlist) + self.name + str(self.transform)
        hash_id = hashlib.md5(config_str.encode()).hexdigest()[:8]
        self.hash_id = hash_id
        return [f'data_{self.name}_views_{hash_id}.pt']

    def download(self):
        # 触发原始数据下载
        TUDataset(self.raw_dir, name=self.name, use_node_attr=True)

    def process(self):
        raw_dataset = TUDataset(self.raw_dir, name=self.name, use_node_attr=True)

        data_list = []
        for raw_data in raw_dataset:
            if 'x' in raw_data.keys():
                base_data = Data(x=raw_data.x.clone(), edge_index=raw_data.edge_index.clone(), y=raw_data.y.clone())
            else:
                base_data = Data(edge_index=raw_data.edge_index.clone(), y=raw_data.y.clone())

            for idx, view_cfg in enumerate(self.viewlist):
                view_name = view_cfg.get("type")
                new_x, new_ei = self._apply_view(raw_data, view_cfg)

                if new_ei.dim() == 1:
                    new_ei = new_ei.view(2, -1) if new_ei.numel() > 0 else torch.empty((2, 0), dtype=torch.long)
                elif new_ei.dim() == 2 and new_ei.size(0) != 2:
                    new_ei = new_ei.t().contiguous()
                base_data[f'edge_index_{view_name}'] = new_ei.to(torch.long)
                if new_x is not None:
                    base_data[f'x_{view_name}'] = new_x
            data_list.append(base_data)

        if not data_list:
            raise RuntimeError("No data generated from viewlist.")


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Saved {len(data_list)} samples to {self.processed_paths[0]}")

    def _apply_view(self, data: Data, view_cfg: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], torch.Tensor]:

        view_type = view_cfg.get("type")
        if view_type is None:
            raise ValueError("Each view config must have a 'type' field.")

        # 准备通用输入
        x = data.x
        edge_index = data.edge_index
        num_node = data.num_nodes

        # 根据类型调用相应方法
        if view_type == "original":
            return None, edge_index.clone()

        elif view_type == "fully_connected":
            new_edge_index = self.augmentor.build_fully_connected(num_node)
            # build_fully_connected 可能返回 (edge_index, edge_batch) 或仅 edge_index
            if isinstance(new_edge_index, tuple):
                new_edge_index = new_edge_index[0]
            return None, new_edge_index

        elif view_type == "order":
            order = view_cfg.get("order", 2)
            include_self_loops = view_cfg.get("include_self_loops", False)
            normalize = view_cfg.get("normalize", True)
            threshold = view_cfg.get("threshold", None)
            new_edge_index = self.augmentor.build_higher_order_graphs(edge_index, num_nodes=num_node, order=order,
                                                                         include_self_loops=include_self_loops,
                                                                         normalize=normalize, threshold=threshold
                                                                         )
            return None, new_edge_index

        elif view_type == "spectral_embedding":
            k = view_cfg.get("k", 1)
            include_original = view_cfg.get("include_original", True)
            new_edge_index = self.augmentor.build_using_spectral_embedding(
                x, edge_index, num_node, k=k, include_original=include_original
            )
            return None, new_edge_index

        elif view_type == "acyclic":
            method = view_cfg.get("method", "boruvka_spanning_tree")
            # 注意：edge_weights 需要是针对当前图的权重，这里简化处理，不传入
            new_edge_index = self.augmentor.build_acyclic_view(
                edge_index=edge_index,
                num_nodes=num_node,
                method=method
            )
            return None, new_edge_index

        elif view_type == "curvature_based":
            curvature_type = view_cfg.get("curvature_type", "balanced_forman")
            dynamic_update_fraction = view_cfg.get("dynamic_update_fraction", 0.5)
            use_long_range = view_cfg.get("use_long_range", True)
            use_curvature_max = view_cfg.get("use_curvature_max", True)
            new_edge_index = self.augmentor.build_curvature_based_augmentation(edge_index=edge_index,
                                                                               num_nodes=num_node,
                                                                               curvature_type=curvature_type,
                                                                               dynamic_update_fraction=dynamic_update_fraction,
                                                                               use_long_range=use_long_range,
                                                                               use_curvature_max=use_curvature_max)
            return None, new_edge_index

        else:
            raise ValueError(f"Unsupported view type: {view_type}")

    def _get_views_from_data(self, data: Data) -> List[Tuple[str, torch.Tensor, int]]:
        views = []
        for key in data.keys():  # 或者 data.keys()
            if key.startswith('edge_index_'):
                view_name = key[len('edge_index_'):]
                edge_index = data[key]
                if edge_index.numel() == 0:
                    num_nodes = 0
                else:
                    num_nodes = edge_index.max().item() + 1
                views.append((view_name, edge_index, num_nodes))
        return views
