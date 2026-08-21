from torch.nn import Parameter
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn.models import MLP
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select import SelectTopK

from torch_geometric.nn.norm import LayerNorm, GraphNorm

from models.loss import global_global_contrast_loss, global_local_contrast_loss, \
    diversity_representation_loss, specificity_node_distribution_loss_fast, perturb_scores

class MultiViewRewiringGraphPooling(torch.nn.Module):

    def __init__(self, args, in_channels, ratio=0.8, num_views=3, perturb=True, feature_fusion=True, score_fusion=True, view_types=None):
        super(MultiViewRewiringGraphPooling, self).__init__()
        if view_types is None:
            view_types = ['random'] * num_views

        self.num_views = num_views
        self.ratio = ratio
        self.args = args
        self.view_types = view_types
        self.perturb = perturb
        self.feature_fusion = feature_fusion
        self.score_fusion = score_fusion

        self.score_convs = torch.nn.ModuleList([
            GCNConv(in_channels, 1) for _ in range(num_views)
        ])

        self.feature_convs = torch.nn.ModuleList([
            GCNConv(in_channels, in_channels) for _ in range(num_views)
        ])

        self.feature_conv = GCNConv(in_channels, in_channels)
        self.score_conv = GCNConv(in_channels, 1)


        self.view_att = Parameter(torch.Tensor(self.num_views, self.num_views))
        nn.init.xavier_uniform_(self.view_att.data)
        self.view_bias = Parameter(torch.Tensor(self.num_views))
        nn.init.zeros_(self.view_bias.data)

        self.view_scale = nn.Parameter(torch.ones(self.num_views))

        self.feature_weights = torch.nn.Parameter(torch.ones(num_views))

        self.select = SelectTopK(1, ratio, None, "sigmoid")
        self.connect = FilterEdges()

    def forward(self, x, edge_index, view_list, edge_attr, batch):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        batch_size = batch.max().item() + 1

        # 存储每个视图的评分和特征
        all_scores = []
        all_features = []
        all_readout = []
        view_graph_emd = []
        init_readout = gap(x, batch)
        total_loss = 0.0

        for view_idx in range(self.num_views):
            view_edge_attr = edge_attr
            view_edge_index = view_list[self.view_types[view_idx]]
            view_x = F.relu(self.feature_convs[view_idx](x, view_edge_index, view_edge_attr))
            view_score = self.score_convs[view_idx](view_x, view_edge_index, view_edge_attr)

            all_scores.append(view_score)
            all_features.append(view_x)
            all_readout.append(torch.cat([gmp(view_x, batch), gap(view_x, batch)], dim=1))
            view_graph_emd.append(gap(view_x, batch))

        if self.perturb:
            all_scores = perturb_scores(all_scores, perturb_type=self.args.noise, ratio=0.1, gumble_tau=1.0, noise_std=0.05)

        if self.args.spec_loss_coef != 0:
            loss_spec = self.args.spec_loss_coef * specificity_node_distribution_loss_fast(all_scores, batch)
        else:
            loss_spec = 0

        score_cat = torch.cat(all_scores, dim=1)
        max_value, _ = torch.max(torch.abs(score_cat), dim=0)
        score_cat = score_cat / max_value


        if self.score_fusion:
            score_weight = torch.sigmoid(torch.matmul(score_cat, self.view_att) + self.view_bias)
            score_weight = torch.softmax(score_weight, dim=1)
            # score = torch.sigmoid(torch.sum(score_cat * score_weight, dim=1))
            score = torch.sum(score_cat * score_weight, dim=1)
            # print("score:", score)
        else:
            score = torch.sum(score_cat, dim=1)

        select_out = self.select(score, batch)

        features_stack = torch.stack(all_features, dim=1)
        if self.feature_fusion:
            fusion_weights_norm = torch.softmax(self.feature_weights, dim=0)
            fusion_weights_expanded = fusion_weights_norm.view(1, -1, 1)
            x = torch.sum(features_stack * fusion_weights_expanded, dim=1)
        else:
            x = torch.sum(features_stack, dim=1)

        if self.args.gg_loss_coef != 0:
            loss_gg = 0.0
            cnt = 0
            for i in range(len(all_features)):
                for j in range(i + 1, len(all_features)):
                    loss_gg += global_global_contrast_loss(all_readout[i], all_readout[j])
                    cnt += 1
            loss_gg /= cnt
            loss_gg = self.args.gg_loss_coef * loss_gg
        else:
            loss_gg = 0

        if self.args.gl_loss_coef != 0:
            loss_gl = 0.0
            cnt = 0
            for i in range(len(all_features)):
                for j in range(len(all_features)):
                    if i != j:
                        loss_gl += global_local_contrast_loss(view_graph_emd[i], all_features[i], batch)
                        cnt += 1
            loss_gl /= cnt
            loss_gl = self.args.gl_loss_coef * loss_gl
        else:
            loss_gl = 0

        total_loss +=  loss_gg + loss_spec + loss_gl


        perm = select_out.node_index
        score = select_out.weight
        assert score is not None

        x = x[perm] * score.view(-1, 1)
        connect_out = self.connect(select_out, edge_index, edge_attr, batch)

        for view_idx in range(self.num_views):
            view_edge_index = view_list[self.view_types[view_idx]]
            view_connect_out = self.connect(select_out, view_edge_index, edge_attr, batch)
            view_list[self.view_types[view_idx]] = view_connect_out.edge_index


        return x, connect_out.edge_index, connect_out.edge_attr, connect_out.batch, perm, score, all_readout, total_loss, view_list


class MVRPool(torch.nn.Module):
    def __init__(self, data, args):
        super(MVRPool, self).__init__()
        self.args = args
        self.task_type = args.task_type
        self.num_features = data.num_features
        self.nhid = args.hid_dim
        self.num_classes = data.num_classes if self.task_type == 'classification' else data.target_dim
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.num_pool_layers = 3
        self.JK = args.jump_connection
        self.perturb = args.perturb
        self.feature_fusion = args.feature_fusion
        self.score_fusion = args.score_fusion

        self.spec_loss_coef = args.spec_loss_coef
        self.gg_loss_coef = args.gg_loss_coef
        self.gl_loss_coef = args.gl_loss_coef
        self.norm_type = args.normalization
        self.view_types = self.build_view_types_from_args(args)

        print("Task:", self.task_type, self.num_classes)
        print("View Types:", self.view_types)
        print("Perturb Scores:",  self.perturb)

        if self.norm_type == 'Layer':
            NormLayer = LayerNorm
        elif self.norm_type == 'Graph':
            NormLayer = GraphNorm
        else:
            raise ValueError("norm_type must be one of 'layer', 'graph'")

        self.num_views = len(self.view_types)

        self.conv1 = GCNConv(self.num_features, self.nhid)

        self.pool1 = MultiViewRewiringGraphPooling(self.args, self.nhid, num_views=self.num_views,
                                                   view_types=self.view_types, perturb=self.perturb,
                                                   feature_fusion=self.feature_fusion, score_fusion=self.score_fusion,
                                                   ratio=self.pooling_ratio)
        self.norm1 = NormLayer(self.nhid)

        self.conv2 = GCNConv(self.nhid, self.nhid)

        self.pool2 = MultiViewRewiringGraphPooling(self.args, self.nhid, num_views=self.num_views,
                                                   view_types=self.view_types, perturb=self.perturb,
                                                   feature_fusion=self.feature_fusion, score_fusion=self.score_fusion,
                                                   ratio=self.pooling_ratio)
        self.norm2 = NormLayer(self.nhid)

        self.conv3 = GCNConv(self.nhid, self.nhid)

        self.pool3 = MultiViewRewiringGraphPooling(self.args, self.nhid, num_views=self.num_views,
                                                   view_types=self.view_types, perturb=self.perturb,
                                                   feature_fusion=self.feature_fusion, score_fusion=self.score_fusion,
                                                   ratio=self.pooling_ratio)
        self.norm3 = NormLayer(self.nhid)

        if self.JK in ["Sum", "Mean"]:
            mlp_embed_dim_input = self.nhid * 2
        elif self.JK == "Cat":
            mlp_embed_dim_input = self.nhid * 2 * self.num_pool_layers
        else:  # None and others
            mlp_embed_dim_input = self.nhid * 2

        view_heads = []
        view_mlp_embed_dim_input = self.nhid * 2
        for i in range(self.num_views):
            view_heads.append(
                MLP([view_mlp_embed_dim_input, view_mlp_embed_dim_input // 2, view_mlp_embed_dim_input // 4,
                     self.num_classes],
                    dropout=self.dropout_ratio))
        self.view_heads = nn.ModuleList(view_heads)

        self.lin1 = torch.nn.Linear(mlp_embed_dim_input, mlp_embed_dim_input // 2)
        self.lin2 = torch.nn.Linear(mlp_embed_dim_input // 2, mlp_embed_dim_input // 4)
        self.lin3 = torch.nn.Linear(mlp_embed_dim_input // 4, self.num_classes)

    def forward(self, data):
        x, edge_index, batch, ground_truth = data.x, data.edge_index, data.batch, data.y

        x_all = []
        view_readouts = [[] for _ in range(self.num_views)]
        gg_loss = 0.0
        view_list = {}
        for view_name in self.view_types:
            view_list[view_name] = data[f'edge_index_{view_name}']

        x = F.relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _, _, view_readout, loss_gg, view_list = self.pool1(x, edge_index, view_list, None, batch)
        x = self.norm1(x, batch)

        gg_loss += loss_gg
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x_all.append(x1)
        for i in range(self.num_views):
            view_readouts[i].append(view_readout[i])

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _, view_readout, loss_gg, view_list = self.pool2(x, edge_index, view_list, None,
                                                                                     batch)
        x = self.norm2(x, batch)

        gg_loss += loss_gg
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x_all.append(x2)
        for i in range(self.num_views):
            view_readouts[i].append(view_readout[i])

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _, view_readout, loss_gg, view_list = self.pool3(x, edge_index, view_list, None,
                                                                                     batch)
        x = self.norm3(x, batch)

        gg_loss += loss_gg
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x_all.append(x3)
        for i in range(self.num_views):
            view_readouts[i].append(view_readout[i])

        if self.JK == "Sum":
            x = torch.stack(x_all, dim=0).sum(dim=0)
        elif self.JK == "Mean":
            x = torch.stack(x_all, dim=0).sum(dim=0) / self.num_pool_layers
        elif self.JK == "Cat":
            x = torch.cat(x_all, dim=1)
        else:
            x = x_all[-1]

        view_loss = 0.0
        for i in range(self.num_views):
            view_x = torch.stack(view_readouts[i], dim=0).sum(dim=0)
            view_x = self.view_heads[i](view_x)
            if self.task_type == "classification":
                view_x = F.log_softmax(view_x, dim=-1)
                view_loss += F.nll_loss(view_x, ground_truth.view(-1))
            else:
                view_loss += F.mse_loss(view_x, ground_truth.view(-1, self.num_classes))


        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        if self.task_type == "classification":
            x = F.log_softmax(x, dim=-1)

        return x, view_loss + gg_loss

    def build_view_types_from_args(self, args):
        view_types = ["original"]
        if args.fully_connected:
            view_types.append("fully_connected")
        if args.order:
            view_types.append("order")
        if args.acyclic:
            view_types.append("acyclic")
        if args.spectral_embedding:
            view_types.append("spectral_embedding")
        if args.curvature_based:
            view_types.append("curvature_based")
        return view_types

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

        for module in [self.pool1, self.pool2, self.pool3]:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

        for i in range(self.num_views):
            self.view_heads[i].reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
