import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, GATv2Conv, AvgPooling
import numpy as np
import os

from fasttext_model import FastTextEmbedding
from utils import log_info,cvss_task_type


class SelfAttention(nn.Module):
    """Self Attention"""

    def __init__(self, input_size, hidden_size):
        """Initilisation."""
        super(SelfAttention, self).__init__()
        self.weight_W = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_proj = nn.Parameter(torch.Tensor(hidden_size, 1))
        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, input_tensor):
        u = torch.tanh(torch.matmul(input_tensor, self.weight_W))
        # print(u.shape)
        att = torch.matmul(u, self.weight_proj)
        # print(att.shape)
        att_score = F.softmax(att, dim=1)
        # print(att_score.shape)
        scored_x = input_tensor * att_score
        # print(scored_x.shape)
        outputs = torch.sum(scored_x, dim=1)
        return outputs


class GModel_Embedding_MultiTask(nn.Module):
    """GModel model."""

    # def __init__(self,embedding_dim = 300, hidden_size=300,dev = "cpu",dropout=0.9):
    def __init__(self, gnn_type, dropout=0.1):
        """Initilisation."""
        super(GModel_Embedding_MultiTask, self).__init__()
        self.encoder = FastTextEmbedding()

        graph_input_feature = 1024
        graph_output_feature = 300
        self.gnn_type = gnn_type
        self.num_heads = 1
        self.gnn = None
        if self.gnn_type == "gatv2":
            self.num_heads = 3
            self.gnn = GATv2Conv(graph_input_feature, graph_output_feature, num_heads=self.num_heads)
        elif self.gnn_type == "gat":
            self.num_heads = 3
            self.gnn = GATConv(graph_input_feature, graph_output_feature, num_heads=self.num_heads)

        elif self.gnn_type == "gcn":
            self.gnn = GraphConv(graph_input_feature, graph_output_feature)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.ModuleDict()   # ensure classifier and model are in the same device
        for task_type,class_dict in cvss_task_type.items():
            self.classifier[task_type] = nn.Linear(self.num_heads * graph_output_feature , len(class_dict))

    def forward(self, g, tokens):
        """
            tokens: [1gram,2gram,3gram]
        """
        node_nums = tokens[0].shape[0]

        outputs = self.encoder(tokens)
        x = outputs
        h = self.dropout(x)
        h = self.gnn(g, h)

        h = self.dropout(h)
        # print(h.shape)
        if self.gnn_type in ["gatv2","gat"]:
            h = h.view(node_nums, -1)
        with g.local_scope():
            g.ndata["h"] = h
            hg = dgl.mean_nodes(g, "h")
            x = self.dropout(hg)
            x = F.relu(x)
            ans = { }
            for task_type,classifier in self.classifier.items():
                ans[task_type] = classifier(x)
            return ans


class EarlyStopping:
    """Early stops the training if validation result doesn't improve after a given patience."""

    def __init__(self, save_path, model_name, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : save file name
            patience (int): How long to wait after last time validation result improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation result improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.model_name = model_name
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.result = np.Inf
        self.delta = delta

    def __call__(self, result, model):

        score = result

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(result, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            log_info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(result, model)
            self.counter = 0

    def save_checkpoint(self, result, model):
        '''Saves model when result increase.'''
        if self.verbose:
            log_info(f'Validation result increased ({self.result:.6f} --> {result:.6f}).  Saving model ...')
        self.result = result
        path = os.path.join(self.save_path, self.model_name + '_best_network.pth')
        torch.save(model.state_dict(), path)  # save best model
        self.val_loss_min = result

    def counter_exceed_2(self) -> bool:
        return self.counter >= 2

if __name__ == "__main__":
    # just test the model
    hidden_size = 40
    model = SelfAttention(hidden_size, hidden_size)
    inputs = torch.ones((1, 15, 40))
    model(inputs)