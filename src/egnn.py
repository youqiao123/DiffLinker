import math
import numpy as np
import torch
import torch.nn as nn

from src import utils
from pdb import set_trace

'''
Graph Convolution Layer

input_nf: 输入节点特征的维度 node feature dimension
output_nf: 输出节点特征的维度 node feature dimension after one GCL layer
hidden_nf: 中间层的维度 hidden dimension
normalization_factor: 用于对聚合操作进行归一化的因子 normalization factor for the aggregation
aggregation_method: 聚合方法，可以是'sum'或'mean' 
activation: 激活函数 activation function
edges_in_d: 边特征的维度 
nodes_att_dim: 节点注意力特征的维度 
attention: 是否使用注意力机制 
normalization: 是否使用归一化方法，如'batch_norm' 

edge_mlp: 用于更新边特征的多层感知机
node_mlp: 用于更新节点特征的多层感知机 
'''

class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method, activation,
                 edges_in_d=0, nodes_att_dim=0, attention=False, normalization=None):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation)

        if normalization is None:
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
                activation,
                nn.Linear(hidden_nf, output_nf)
            )
        elif normalization == 'batch_norm':
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
                nn.BatchNorm1d(hidden_nf),
                activation,
                nn.Linear(hidden_nf, output_nf),
                nn.BatchNorm1d(output_nf),
            )
        else:
            raise NotImplementedError

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask # 逐元素相乘
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


'''
EquivariantUpdate:
这个类用于实现等变更新，不更新边和节点的特征，但是会更新坐标

具体如何实现等变更新：

1. 输入特征的构造
input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
其中，h[row]和h[col]分别表示边的起始节点和终止节点的特征，edge_attr表示边的属性特征。
节点特征和边特征的拼接不会破坏等变性，因为这些特征与坐标的平移和旋转无关

2. 坐标差值的使用
trans = coord_diff * self.coord_mlp(input_tensor)
coord_diff是边的起点坐标与重点坐标的差值，是一个方向向量，表示边的几何信息
coord_diff = c[row] - c[col]
坐标差值对于平移操作不变，对于旋转操作也不变

3. 坐标更新值的计算
trans = coord_diff * self.coord_mlp(input_tensor)

    self.coord_mlp: 是一个多层感知机，输入是拼接后的特征input_tensor(拼接两个node特征和一个edge特征)，
    输出是一个标量，表示边的权重

    更新值: 坐标更新值时边的权重与坐标差值的乘积

    等变性保证: 平移时，coord_diff对平移不变；旋转时，coord_diff和trans都会随旋转矩阵R一起旋转
'''

class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, activation=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask, linker_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if linker_mask is not None:
            agg = agg * linker_mask

        coord = coord + agg
        return coord

    def forward(
        self, h, coord, edge_index, coord_diff, edge_attr=None, linker_mask=None, node_mask=None, edge_mask=None
    ):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask, linker_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord

'''
EquivariantBlock: 结合了多个GCL层和一个EquivariantUpdate层的模块
参数说明: 
edge_feat_nf: 边特征的维度，默认为 2
activation: 激活函数，默认为 SiLU
n_layers: 图卷积层的数量
coords_range: 坐标更新的范围
norm_constant: 用于坐标差值归一化的常数

主要功能:
1. 节点特征更新: 使用多个 GCL 层，逐层更新节点的特征
2. 坐标更新: 使用 EquivariantUpdate 层，基于更新后的节点特征和边的几何信息，更新节点的坐标，同时保证等变性
'''

class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', activation=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        # "gcl_%d" % i 这里的%d是一个占位符，表示将一个整数i插入到字符串中
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              activation=activation, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, activation=activation, tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        # if torch.cuda.is_available():
        #     self.to(self.device)
        # else:
        #     self.to('cpu')

    def forward(self, h, x, edge_index, node_mask=None, linker_mask=None, edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        x = self._modules["gcl_equiv"](
            h, x,
            edge_index=edge_index,
            coord_diff=coord_diff,
            edge_attr=edge_attr,
            linker_mask=linker_mask,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x

# 由多个 EquivariantBlock 组成的网络
class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', activation=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               activation=activation, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method))
        # if torch.cuda.is_available():
        #     self.to(self.device)
        # else:
        #     self.to('cpu')

    def forward(self, h, x, edge_index, node_mask=None, linker_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)

        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h, x, edge_index,
                node_mask=node_mask,
                linker_mask=linker_mask,
                edge_mask=edge_mask,
                edge_attr=distances
            )

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x


# 多个 GCL 组成的网络
class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 activation=nn.SiLU(), n_layers=4, attention=False, normalization_factor=1,
                 out_node_nf=None, normalization=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        # Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, activation=activation,
                attention=attention, normalization=normalization))

        # if torch.cuda.is_available():
        #     self.to(self.device)
        # else:
        #     self.to('cpu')

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h

'''
正弦嵌入
'''
class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result

'''
Dynamics用于在 全连接图(FC) 上对带有坐标与节点特征的分子/粒子系统进行一次前向“动力学更新”。
其核心是将输入的节点坐标 x ∈ R^{B x N x 3} 与节点特征 h ∈ R^{B x N x nf}（可附加时间与上下文特征）
经过 EGNN 或普通 GNN 的传播，得到：
    速度(或位移增量)vel ∈ R^{B x N x 3}（表示坐标的更新量），以及
    更新后的节点特征 h_final ∈ R^{B x N x nf},

并最终返回拼接后的张量 cat([vel, h_final], dim=2) ∈ R^{B x N x (3+nf)}

Dynamics 模块在扩散模型(diffusion models)尤其是用于 3D 分子 / 几何结构生成的扩散模型中，是一个常见而且合理的构建方式

'''
class Dynamics(nn.Module):
    def __init__(
            self, n_dims, in_node_nf, context_node_nf, hidden_nf=64, device='cpu', activation=nn.SiLU(),
            n_layers=4, attention=False, condition_time=True, tanh=False, norm_constant=0, inv_sublayers=2,
            sin_embedding=False, normalization_factor=100, aggregation_method='sum', model='egnn_dynamics',
            normalization=None, centering=False, graph_type='FC',
    ):
        super().__init__()
        self.device = device
        self.n_dims = n_dims
        self.context_node_nf = context_node_nf
        self.condition_time = condition_time
        self.model = model
        self.centering = centering
        self.graph_type = graph_type

        in_node_nf = in_node_nf + context_node_nf + condition_time
        if self.model == 'egnn_dynamics':
            self.dynamics = EGNN(
                in_node_nf=in_node_nf,
                in_edge_nf=1,
                hidden_nf=hidden_nf, device=device,
                activation=activation,
                n_layers=n_layers,
                attention=attention,
                tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers,
                sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )
        elif self.model == 'gnn_dynamics':
            self.dynamics = GNN(
                in_node_nf=in_node_nf+3,
                in_edge_nf=0,
                hidden_nf=hidden_nf,
                out_node_nf=in_node_nf+3,
                device=device,
                activation=activation,
                n_layers=n_layers,
                attention=attention,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                normalization=normalization,
            )
        else:
            raise NotImplementedError

        self.edge_cache = {}

    def forward(self, t, xh, node_mask, linker_mask, edge_mask, context):
        """
        - t: (B)
        - xh: (B, N, D), where D = 3 + nf
        - node_mask: (B, N, 1)
        - edge_mask: (B*N*N, 1)
        - context: (B, N, C)
        """

        assert self.graph_type == 'FC'

        bs, n_nodes = xh.shape[0], xh.shape[1]
        edges = self.get_edges(n_nodes, bs, xh_device=xh.device)  # (2, B*N)
        node_mask = node_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        if linker_mask is not None:
            linker_mask = linker_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        # Reshaping node features & adding time feature
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask  # (B*N, D)
        x = xh[:, :self.n_dims].clone()  # (B*N, 3)
        h = xh[:, self.n_dims:].clone()  # (B*N, nf)
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)  # (B*N, nf+1)
        if context is not None:
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        # Forward EGNN
        # Output: h_final (B*N, nf), x_final (B*N, 3), vel (B*N, 3)
        if self.model == 'egnn_dynamics':
            h_final, x_final = self.dynamics(
                h,
                x,
                edges,
                node_mask=node_mask,
                linker_mask=linker_mask,
                edge_mask=edge_mask
            )
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.model == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.dynamics(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]
        else:
            raise NotImplementedError

        # Slice off context size
        if context is not None:
            h_final = h_final[:, :-self.context_node_nf]

        # Slice off last dimension which represented time.
        if self.condition_time:
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)  # (B, N, 3)
        h_final = h_final.view(bs, n_nodes, -1)  # (B, N, D)
        node_mask = node_mask.view(bs, n_nodes, 1)  # (B, N, 1)

        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final)):
            raise utils.FoundNaNException(vel, h_final)

        if self.centering:
            vel = utils.remove_mean_with_mask(vel, node_mask)

        return torch.cat([vel, h_final], dim=2)

    def get_edges(self, n_nodes, batch_size, xh_device=None):
        if n_nodes in self.edge_cache:
            edges_dic_b = self.edge_cache[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                # edges = [torch.LongTensor(rows).to(self.device), torch.LongTensor(cols).to(self.device)]
                edges = [torch.LongTensor(rows).to(xh_device), torch.LongTensor(cols).to(xh_device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self.edge_cache[n_nodes] = {}
            return self.get_edges(n_nodes, batch_size)


class DynamicsWithPockets(Dynamics):
    def forward(self, t, xh, node_mask, linker_mask, edge_mask, context):
        """
        - t: (B)
        - xh: (B, N, D), where D = 3 + nf
        - node_mask: (B, N, 1)
        - edge_mask: (B*N*N, 1)
        - context: (B, N, C)
        """

        bs, n_nodes = xh.shape[0], xh.shape[1]
        node_mask = node_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        if linker_mask is not None:
            linker_mask = linker_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        fragment_only_mask = context[..., -2].view(bs * n_nodes, 1)  # (B*N, 1)
        pocket_only_mask = context[..., -1].view(bs * n_nodes, 1)  # (B*N, 1)
        assert torch.all(fragment_only_mask.bool() | pocket_only_mask.bool() | linker_mask.bool() == node_mask.bool())

        # Reshaping node features & adding time feature
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask  # (B*N, D)
        x = xh[:, :self.n_dims].clone()  # (B*N, 3)
        h = xh[:, self.n_dims:].clone()  # (B*N, nf)

        assert self.graph_type in ['4A', 'FC-4A', 'FC-10A-4A']
        if self.graph_type == '4A' or self.graph_type is None:
            # edges = self.get_dist_edges_4A(x, node_mask, edge_mask)
            edges = self.get_dist_edges_4A_new(x, node_mask, edge_mask)
        else:
            edges = self.get_dist_edges(x, node_mask, edge_mask, linker_mask, fragment_only_mask, pocket_only_mask)

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)  # (B*N, nf+1)
        if context is not None:
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        # Forward EGNN
        # Output: h_final (B*N, nf), x_final (B*N, 3), vel (B*N, 3)
        if self.model == 'egnn_dynamics':
            h_final, x_final = self.dynamics(
                h,
                x,
                edges,
                node_mask=node_mask,
                linker_mask=linker_mask,
                edge_mask=None
            )
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.model == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.dynamics(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]
        else:
            raise NotImplementedError

        # Slice off context size
        if context is not None:
            h_final = h_final[:, :-self.context_node_nf]

        # Slice off last dimension which represented time.
        if self.condition_time:
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)  # (B, N, 3)
        h_final = h_final.view(bs, n_nodes, -1)  # (B, N, D)
        node_mask = node_mask.view(bs, n_nodes, 1)  # (B, N, 1)

        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final)):
            raise utils.FoundNaNException(vel, h_final)

        if self.centering:
            vel = utils.remove_mean_with_mask(vel, node_mask)

        return torch.cat([vel, h_final], dim=2)

    @staticmethod
    def get_dist_edges_4A(x, node_mask, batch_mask):
        device = x.device
        node_mask = node_mask.to(device)
        batch_mask = batch_mask.to(device)

        node_mask = node_mask.squeeze().bool()
        batch_adj = (batch_mask[:, None] == batch_mask[None, :])
        nodes_adj = (node_mask[:, None] & node_mask[None, :])
        dists_adj = (torch.cdist(x, x) <= 4)
        rm_self_loops = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        adj = batch_adj & nodes_adj & dists_adj & rm_self_loops
        edges = torch.stack(torch.where(adj))
        return edges

    def get_dist_edges(self, x, node_mask, batch_mask, linker_mask, fragment_only_mask, pocket_only_mask):
        node_mask = node_mask.squeeze().bool()
        linker_mask = linker_mask.squeeze().bool() & node_mask
        fragment_only_mask = fragment_only_mask.squeeze().bool() & node_mask
        pocket_only_mask = pocket_only_mask.squeeze().bool() & node_mask
        ligand_mask = linker_mask | fragment_only_mask

        # General constrains:
        batch_adj = (batch_mask[:, None] == batch_mask[None, :])
        nodes_adj = (node_mask[:, None] & node_mask[None, :])
        rm_self_loops = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        constraints = batch_adj & nodes_adj & rm_self_loops

        # Ligand atoms – fully-connected graph
        ligand_adj = (ligand_mask[:, None] & ligand_mask[None, :])
        ligand_interactions = ligand_adj & constraints

        # Pocket atoms - within 4A
        pocket_adj = (pocket_only_mask[:, None] & pocket_only_mask[None, :])
        pocket_dists_adj = (torch.cdist(x, x) <= 4)
        pocket_interactions = pocket_adj & pocket_dists_adj & constraints

        # Pocket-ligand atoms - within 10A
        pocket_ligand_cutoff = 4 if self.graph_type == 'FC-4A' else 10
        pocket_ligand_adj = (ligand_mask[:, None] & pocket_only_mask[None, :])
        pocket_ligand_adj = pocket_ligand_adj | (pocket_only_mask[:, None] & ligand_mask[None, :])
        pocket_ligand_dists_adj = (torch.cdist(x, x) <= pocket_ligand_cutoff)
        pocket_ligand_interactions = pocket_ligand_adj & pocket_ligand_dists_adj & constraints

        adj = ligand_interactions | pocket_interactions | pocket_ligand_interactions
        edges = torch.stack(torch.where(adj))
        return edges

    @staticmethod
    def get_dist_edges_4A_new(x, node_mask, batch_mask):
        """
        x         : (B*N, 3)
        node_mask : (B*N, 1) or (B*N,)  -> padding 掩码
        batch_mask: (B*N,)              -> 每个节点属于哪一个 batch 的 index
        返回:
            edges : (2, E) 的 LongTensor, 表示全局 flatten 后的边索引
        """
        device = x.device
        node_mask = node_mask.squeeze().bool().to(device)
        batch_mask = batch_mask.to(device)

        # 只保留有效节点（去掉 padding）
        valid_idx = torch.where(node_mask)[0]
        if valid_idx.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=device)

        # x_valid = x[valid_idx]                  # (N_valid, 3)
        batch_valid = batch_mask[valid_idx]     # (N_valid,)

        rows_all = []
        cols_all = []

        # 按 batch 分块构造 4A 邻接
        for b in batch_valid.unique():
            idx_b = valid_idx[batch_valid == b]   # 这一 batch 内的全局索引
            if idx_b.numel() <= 1:
                continue

            x_b = x[idx_b]                        # (n_b, 3)
            # 局部 pairwise 距离矩阵 (n_b, n_b)，n_b <= num_atoms <= 1000
            dist_b = torch.cdist(x_b, x_b)

            # 4Å 内、非自环
            adj_b = (dist_b <= 4)
            adj_b.fill_diagonal_(False)

            row_local, col_local = torch.where(adj_b)   # 在局部 (n_b, n_b) 上做 where

            # 映射回全局索引
            rows_all.append(idx_b[row_local])
            cols_all.append(idx_b[col_local])

        if not rows_all:
            return torch.empty(2, 0, dtype=torch.long, device=device)

        rows = torch.cat(rows_all, dim=0)
        cols = torch.cat(cols_all, dim=0)
        edges = torch.stack([rows, cols], dim=0)        # (2, E)

        return edges