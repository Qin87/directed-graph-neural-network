import torch
from torch import nn, optim
import pytorch_lightning as pl
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear
from torch_geometric.nn import (
    SAGEConv,
    GCNConv,
    GATConv,
    JumpingKnowledge,
)

from src.datasets.data_utils import get_norm_adj
from argparse import Namespace

def get_conv(conv_type, input_dim, output_dim, alpha):
    if conv_type == "gcn":
        return GCNConv(input_dim, output_dim, add_self_loops=False)
    elif conv_type == "sage":
        return SAGEConv(input_dim, output_dim)
    elif conv_type == "gat":
        return GATConv(input_dim, output_dim, heads=1)
    elif conv_type == "dir-gcn":
        return DirGCNConv(input_dim, output_dim, alpha)
    elif conv_type == "dir-sage":
        return DirSageConv(input_dim, output_dim, alpha)
    elif conv_type == "dir-gat":
        return DirGATConv(input_dim, output_dim, heads=1, alpha=alpha)
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")


def get_conv_layer(input_dim, output_dim, args):
    # if args.conv_type == "gcn":
    #     return GCNConv(input_dim, output_dim, add_self_loops=False)
    # elif args.conv_type == "sage":
    #     return SAGEConv(input_dim, output_dim)
    # elif args.conv_type == "gat":
    #     return GATConv(input_dim, output_dim, heads=1)
    # elif args.conv_type == "dir-gcn":
    #     return DirGCNConv(input_dim, output_dim, alpha)
    # elif args.conv_type == "dir-sage":
    #     return DirSageConv(input_dim, output_dim, alpha)
    # elif args.conv_type == "dir-gat":
    #     return DirGATConv(input_dim, output_dim, heads=1, alpha=alpha)
    # else:
    #     raise ValueError(f"Convolution type {conv_type} not supported")
    if args.conv_type == 'dir-sage':
        return SAGEConv(input_dim, output_dim, root_weight=True)
    elif args.conv_type == 'dir-gcn':
        return GCNConv(input_dim, output_dim, add_self_loops=False)
    elif args.conv_type == 'dir-gat':
        heads = 1
        return GATConv(input_dim, output_dim*heads, heads=heads, add_self_loops=False)
    else:
        raise NotImplementedError

class Base2LayerGNN(torch.nn.Module):
    """Generic 2-layer GNN with configurable edge direction handling"""

    def __init__(self, input_dim, output_dim, args,
                 edge1_fn=lambda x: x, edge2_fn=lambda x: x):
        super().__init__()
        self.edge1_fn = edge1_fn
        self.edge2_fn = edge2_fn
        hidden_dim = output_dim

        self.gnn1 = get_conv_layer(input_dim, hidden_dim, args)
        self.gnn2 = get_conv_layer(hidden_dim, output_dim, args)

    def forward(self, x, edge_index):
        x1 = self.gnn1(x, edge_index)
        x2 = self.gnn2(x1, edge_index)
        return x2
def transpose_edge_index(edge_index):
    return torch.stack([edge_index[1], edge_index[0]], dim=0)

class DirConv_MixLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.coef = args.coef_agg
        self.norm_list = []
        self.conv_type = args.conv_type
        self.inci_norm = args.inci_norm

        self.alpha = nn.Parameter(torch.ones(1) * args.alpha, requires_grad=False)
        self.beta = nn.Parameter(torch.ones(1) * args.beta, requires_grad=False)
        self.gama = nn.Parameter(torch.ones(1) * args.gama, requires_grad=False)

        if args.conv_type == 'dir-gcn':
            self.lin_src_to_dst = Linear(input_dim, output_dim)
            self.lin_dst_to_src = Linear(input_dim, output_dim)
            self.adj_norm, self.adj_t_norm = None, None

            self.linx = nn.ModuleList([Linear(input_dim, output_dim) for i in range(4)])
            self.adj_norm_in_out, self.adj_norm_out_in, self.adj_norm_in_in, self.adj_norm_out_out = None, None, None, None
        elif args.conv_type == 'dir-sage':
            self.lin_src_to_dst = SAGEConv(input_dim, output_dim, root_weight=True)
            self.lin_dst_to_src = SAGEConv(input_dim, output_dim, root_weight=True)

            self.linx = nn.ModuleList([SAGEConv(input_dim, output_dim, root_weight=True) for i in range(4)])
            self.edge_in_out, self.edge_out_in, self.edge_in_in, self.edge_out_out = None, None, None, None
        elif args.conv_type == 'dir-gat':
            heads = 1
            self.lin_src_to_dst = GATConv(input_dim, output_dim * heads, heads=heads, add_self_loops=False)
            self.lin_dst_to_src = GATConv(input_dim, output_dim * heads, heads=heads, add_self_loops=False)

            self.linx = nn.ModuleList([GATConv(input_dim, output_dim * heads, heads=heads) for i in range(4)])
            self.edge_in_out, self.edge_out_in, self.edge_in_in, self.edge_out_out = None, None, None, None
        else:
            raise NotImplementedError

        # 2-layer components
        self.lin_in_in = Base2LayerGNN(input_dim, output_dim, args)
        self.lin_out_out = Base2LayerGNN(
            input_dim, output_dim, args,
            edge1_fn=transpose_edge_index,
            edge2_fn=transpose_edge_index
        )
        self.lin_in_out = Base2LayerGNN(
            input_dim, output_dim, args,
            edge2_fn=transpose_edge_index
        )
        self.lin_out_in = Base2LayerGNN(
            input_dim, output_dim, args,
            edge1_fn=transpose_edge_index
        )

        self.norm_list = [self.lin_in_out, self.lin_out_in, self.lin_in_in, self.lin_out_out]

    def forward(self, x, edge_index):
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)
        if self.conv_type == 'dir-gcn':
            if self.adj_norm is None:
                row, col = edge_index
                num_nodes = x.shape[0]

                adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
                self.adj_norm = get_norm_adj(adj, norm="dir")

                adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
                self.adj_t_norm = get_norm_adj(adj_t, norm="dir")
            out1 = self.coef*(1 + self.alpha) * ((1 - self.alpha) * self.lin_src_to_dst(self.adj_norm @ x) + self.alpha * self.lin_dst_to_src(self.adj_t_norm @ x))

            if not (self.beta == -1 and self.gama == -1):
                if self.beta != -1:
                    lin0 = self.lin_in_in
                    lin1 = self.lin_out_out
                    out2 = (1 + self.beta) * ((1 - self.beta) * lin0(x, edge_index) + self.beta * lin1(x, edge_index))
                else:
                    out2 = torch.zeros_like(out1)
                if self.gama != -1:
                    lin0 = self.lin_in_out
                    lin1 = self.lin_out_in
                    out3 = (1 + self.gama) * ((1 - self.gama) * lin0(x, edge_index) + self.gama * lin1(x, edge_index))
                else:
                    out3 = torch.zeros_like(out1)
        elif self.conv_type in ['dir-gat', 'dir-sage']:
            edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)
            if self.edge_in_in is None:
                self.edge_in_out, self.edge_out_in, self.edge_in_in, self.edge_out_out = get_higher_edge_index(edge_index, x.shape[0])

            out1 = aggregate_index(x, self.alpha, self.lin_src_to_dst, edge_index, self.lin_dst_to_src, edge_index_t)
            if not (self.beta == -1 and self.gama == -1):
                if self.beta != -1:
                    out2 = aggregate_index(x, self.beta, self.linx[0], self.edge_in_out, self.linx[1], self.edge_out_in)
                else:
                    out2 = torch.zeros_like(out1)
                if self.gama != -1:
                    out3 = aggregate_index(x, self.gama, self.linx[2], self.edge_in_in, self.linx[3], self.edge_out_out)
                else:
                    out3 = torch.zeros_like(out1)

        return out1+out2+out3

class DirConv_Mix(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.coef = args.coef_agg
        self.norm_list = []
        self.conv_type = args.conv_type
        self.inci_norm = args.inci_norm

        self.alpha = nn.Parameter(torch.ones(1) * args.alpha, requires_grad=False)
        self.beta = nn.Parameter(torch.ones(1) * args.beta, requires_grad=False)
        self.gama = nn.Parameter(torch.ones(1) * args.gama, requires_grad=False)

        if args.conv_type == 'dir-gcn':
            self.lin_src_to_dst = Linear(input_dim, output_dim)
            self.lin_dst_to_src = Linear(input_dim, output_dim)
            self.adj_norm, self.adj_t_norm = None, None

            self.linx = nn.ModuleList([Linear(input_dim, output_dim) for i in range(4)])
            self.adj_norm_in_out, self.adj_norm_out_in, self.adj_norm_in_in, self.adj_norm_out_out = None, None, None, None
        elif args.conv_type == 'dir-sage':
            self.lin_src_to_dst = SAGEConv(input_dim, output_dim, root_weight=True)
            self.lin_dst_to_src = SAGEConv(input_dim, output_dim, root_weight=True)

            self.linx = nn.ModuleList([SAGEConv(input_dim, output_dim, root_weight=True) for i in range(4)])
            self.edge_in_out, self.edge_out_in, self.edge_in_in, self.edge_out_out = None, None, None, None
        elif args.conv_type == 'dir-gat':
            heads = 1
            self.lin_src_to_dst = GATConv(input_dim, output_dim * heads, heads=heads, add_self_loops=False)
            self.lin_dst_to_src = GATConv(input_dim, output_dim * heads, heads=heads, add_self_loops=False)

            self.linx = nn.ModuleList([GATConv(input_dim, output_dim * heads, heads=heads) for i in range(4)])
            self.edge_in_out, self.edge_out_in, self.edge_in_in, self.edge_out_out = None, None, None, None
        else:
            raise NotImplementedError

    def forward(self, x, edge_index):
        if self.conv_type == 'dir-gcn':
            if self.adj_norm is None:
                row, col = edge_index
                num_nodes = x.shape[0]

                adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
                self.adj_norm = get_norm_adj(adj, norm="dir")

                adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
                self.adj_t_norm = get_norm_adj(adj_t, norm="dir")
            if not (self.gama == -1 and self.beta == -1) and len(self.norm_list) == 0:
                adj_pairs = [adj @ adj_t, adj_t @ adj,adj @ adj,adj_t @ adj_t ]
                self.norm_list = [get_norm_adj(mat, norm=self.inci_norm) for mat in adj_pairs]
            out1 = aggregate(x, self.alpha, self.lin_src_to_dst, self.adj_norm, self.lin_dst_to_src, self.adj_t_norm, self.coef)
            if not (self.beta == -1 and self.gama == -1):
                out2 = aggregate(x, self.beta, self.norm_list[0],self.adj_norm,  self.norm_list[1], self.adj_norm, self.coef)
                out3 = aggregate(x, self.gama, self.norm_list[2], self.adj_norm, self.norm_list[3], self.adj_norm, self.coef)
            else:
                out2 = torch.zeros_like(out1)
                out3 = torch.zeros_like(out1)
        elif self.conv_type in ['dir-gat', 'dir-sage']:
            edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)
            if self.edge_in_in is None:
                self.edge_in_out, self.edge_out_in, self.edge_in_in, self.edge_out_out = get_higher_edge_index(edge_index, x.shape[0])

            out1 = aggregate_index(x, self.alpha, self.lin_src_to_dst, edge_index, self.lin_dst_to_src, edge_index_t)
            if not (self.beta == -1 and self.gama == -1):
                if self.beta != -1:
                    out2 = aggregate_index(x, self.beta, self.linx[0], self.edge_in_out, self.linx[1], self.edge_out_in)
                else:
                    out2 = torch.zeros_like(out1)
                if self.gama != -1:
                    out3 = aggregate_index(x, self.gama, self.linx[2], self.edge_in_in, self.linx[3], self.edge_out_out)
                else:
                    out3 = torch.zeros_like(out1)

        return out1+out2+out3

def aggregate(x, alpha, lin0, adj0, lin1, adj1, coef=1):
    out = coef*(alpha * lin0(adj0 @ x) + (1 - alpha) * lin1(adj1 @ x))
    return out
def aggregate_layer(x, alpha, lin0, adj0_x, lin1, adj1_x, coef=1):
    out = coef*(alpha * lin0(adj0_x) + (1 - alpha) * lin1(adj1_x))
    return out
def aggregate_index(x, alpha, lin0, index0, lin1, index1, coef=1):
    out = coef*(1 + alpha) * ((1 - alpha) * lin0(x, index0) + alpha * lin1(x, index1))
    return out

def edge_index_to_adj(edge_index, num_nodes):
    row = edge_index[0]
    col = edge_index[1]
    adj = SparseTensor(row=row.contiguous(), col=col.contiguous(), sparse_sizes=(num_nodes, num_nodes))
    return adj

def get_index(adj_aat):
    row, col = adj_aat.storage._row, adj_aat.storage._col
    edge_index_aat = torch.stack([row, col], dim=0)
    return edge_index_aat

def get_higher_edge_index(edge_index, num_nodes, rm_gen_sLoop=0):
    adj = edge_index_to_adj(edge_index, num_nodes)
    adj_in_out = adj @ adj.t()
    adj_out_in =  adj.t() @ adj

    adj_aa = adj @ adj
    adj_out_out = adj.t() @ adj.t()

    return get_index(adj_in_out), get_index(adj_out_in), get_index(adj_aa), get_index(adj_out_out)


class DirGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        return self.alpha * self.lin_src_to_dst(self.adj_norm @ x) + (1 - self.alpha) * self.lin_dst_to_src(self.adj_t_norm @ x)




class DirSageConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirSageConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = SAGEConv(input_dim, output_dim, flow="source_to_target", root_weight=False)
        self.conv_dst_to_src = SAGEConv(input_dim, output_dim, flow="target_to_source", root_weight=False)
        self.lin_self = Linear(input_dim, output_dim)
        self.alpha = alpha

    def forward(self, x, edge_index):
        return (
            self.lin_self(x)
            + (1 - self.alpha) * self.conv_src_to_dst(x, edge_index)
            + self.alpha * self.conv_dst_to_src(x, edge_index)
        )


class DirGATConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heads, alpha):
        super(DirGATConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = GATConv(input_dim, output_dim, heads=heads)
        self.conv_dst_to_src = GATConv(input_dim, output_dim, heads=heads)
        self.alpha = alpha

    def forward(self, x, edge_index):
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)

        return (1 - self.alpha) * self.conv_src_to_dst(x, edge_index) + self.alpha * self.conv_dst_to_src(
            x, edge_index_t
        )

class ScaleNet(torch.nn.Module):
    '''
    done = May21
    '''
    def __init__(
        self,
            num_features: object,
            num_classes: object,
            hidden_dim: object,
            num_layers: object = 2,
            dropout: object = 0,
            conv_type: object = "dir-gcn",
            jumping_knowledge: object = False,
            normalize: object = False,
            alpha: object = 1 / 2,
            learn_alpha: object = False,
            args: Namespace = None
    ) -> object:
        super().__init__()

        output_dim = hidden_dim if jumping_knowledge else num_classes
        if num_layers == 1:
            self.convs = ModuleList([DirConv_Mix(num_features, output_dim, args)])
        else:
            self.convs = ModuleList([DirConv_Mix(num_features, hidden_dim, args)])
            for _ in range(num_layers - 2):
                self.convs.append(DirConv_Mix(hidden_dim, hidden_dim, args))
            self.convs.append(DirConv_Mix(hidden_dim, output_dim, args))

        if jumping_knowledge is not None:
            input_dim = hidden_dim * num_layers if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, num_classes)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]

        if self.jumping_knowledge is not None:
            x = self.jump(xs)
            x = self.lin(x)

        return torch.nn.functional.log_softmax(x, dim=1)
    
    
class ScaleLayer(torch.nn.Module):
    '''
    not done---May20
    '''
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        num_layers=2,
        dropout=0,
        conv_type="dir-gcn",
        jumping_knowledge=False,
        normalize=False,
        alpha=1 / 2,
        learn_alpha=False,
        args: Namespace = None
    ):
        super().__init__()

        output_dim = hidden_dim if jumping_knowledge else num_classes
        if num_layers == 1:
            self.convs = ModuleList([DirConv_MixLayer(num_features, output_dim, args)])
        else:
            self.convs = ModuleList([DirConv_MixLayer(num_features, hidden_dim, args)])
            for _ in range(num_layers - 2):
                self.convs.append(DirConv_MixLayer(hidden_dim, hidden_dim, args))
            self.convs.append(DirConv_MixLayer(hidden_dim, output_dim, args))

        if jumping_knowledge is not None:
            input_dim = hidden_dim * num_layers if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, num_classes)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]

        if self.jumping_knowledge is not None:
            x = self.jump(xs)
            x = self.lin(x)

        return torch.nn.functional.log_softmax(x, dim=1)


class GNN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        num_layers=2,
        dropout=0,
        conv_type="dir-gcn",
        jumping_knowledge=False,
        normalize=False,
        alpha=1 / 2,
        learn_alpha=False,
    ):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)
        output_dim = hidden_dim if jumping_knowledge else num_classes
        if num_layers == 1:
            self.convs = ModuleList([get_conv(conv_type, num_features, output_dim, self.alpha)])
        else:
            self.convs = ModuleList([get_conv(conv_type, num_features, hidden_dim, self.alpha)])
            for _ in range(num_layers - 2):
                self.convs.append(get_conv(conv_type, hidden_dim, hidden_dim, self.alpha))
            self.convs.append(get_conv(conv_type, hidden_dim, output_dim, self.alpha))

        if jumping_knowledge is not None:
            input_dim = hidden_dim * num_layers if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, num_classes)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]

        if self.jumping_knowledge is not None:
            x = self.jump(xs)
            x = self.lin(x)

        return torch.nn.functional.log_softmax(x, dim=1)


class LightingFullBatchModelWrapper(pl.LightningModule):
    def __init__(self, model, lr, weight_decay, train_mask, val_mask, test_mask, evaluator=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.evaluator = evaluator
        self.train_mask, self.val_mask, self.test_mask = train_mask, val_mask, test_mask

    def on_train_epoch_end(self):
        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()
    def training_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index)

        loss = nn.functional.nll_loss(out[self.train_mask], y[self.train_mask].squeeze())
        # self.log("train_loss", loss)

        y_pred = out.max(1)[1]
        train_acc = self.evaluate(y_pred=y_pred[self.train_mask], y_true=y[self.train_mask])
        # self.log("train_acc", train_acc)
        val_acc = self.evaluate(y_pred=y_pred[self.val_mask], y_true=y[self.val_mask])
        # self.log("val_acc", val_acc)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def evaluate(self, y_pred, y_true):
        if self.evaluator:
            acc = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred.unsqueeze(1)})["acc"]
        else:
            acc = y_pred.eq(y_true.squeeze()).sum().item() / y_pred.shape[0]

        return acc

    def test_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index)

        y_pred = out.max(1)[1]
        val_acc = self.evaluate(y_pred=y_pred[self.test_mask], y_true=y[self.test_mask])
        self.log("test_acc", val_acc)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


def get_model(args):
    if args.model == 'gnn':
        return GNN(
            num_features=args.num_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            dropout=args.dropout,
            conv_type=args.conv_type,
            jumping_knowledge=args.jk,
            normalize=args.normalize,
            alpha=args.alpha,
            learn_alpha=args.learn_alpha,
        )
    elif args.model == 'scalenet':
        return ScaleNet(
            num_features=args.num_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            dropout=args.dropout,
            conv_type=args.conv_type,
            jumping_knowledge=args.jk,
            normalize=args.normalize,
            alpha=args.alpha,
            learn_alpha=args.learn_alpha,
            args=args,
        )
    elif args.model == 'scalelayer':
        return ScaleLayer(
            num_features=args.num_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            dropout=args.dropout,
            conv_type=args.conv_type,
            jumping_knowledge=args.jk,
            normalize=args.normalize,
            alpha=args.alpha,
            learn_alpha=args.learn_alpha, 
            args=args,
        )
