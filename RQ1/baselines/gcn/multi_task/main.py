import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from dgl.dataloading import GraphDataLoader
import numpy as np
import sys
import pandas as pd
from utils import init_logging,log_info,cvss_task_type
from model_single import GModel_Embedding_SingleTask
from model import GModel_Embedding_MultiTask, EarlyStopping,TextCNN,LSTM_Encoder
from transformers import AutoTokenizer, AutoModel
import os
from dataset_new import GraphDataset_Token_MultiTask
from typing import Union
from time import strftime, localtime

path = "/".join(sys.path[0].split("/")[:-2])
sys.path.append(path)
import gcn

path = "/".join(sys.path[0].split("/")[:-3]) + "/script"
sys.path.append(path)

import random
import warnings

warnings.filterwarnings('ignore')
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
context_type = 'nature'# nature(One-line Context),data_flow,control_flow,sentence(Zero Context)
token_type = 'codebert' # codebert unixcoder textcnn lstm
gnn_type = "gcn"   #gat gatv2 gcn SAGEConv
early_stop_metric = "f1"
save_path = gcn.multi_task_dir()/f"result/{token_type}/{gnn_type}/{context_type}"
model_name = f"{token_type}_finetune"
# nature
# data_flow
# control_flow
# sentence

if token_type in ["codebert","unixcoder"]:
    context_type = "codebert_"+context_type
else:
    context_type = "vocab_"+context_type
context_type_dict = {
    #codebert and unixcoder token
    'codebert_nature' : 'nature_context__codebert_tokens' ,
    'codebert_data_flow' : 'DD_context_codebert_tokens' ,
    'codebert_control_flow' : 'CD_context_codebert_tokens',
    'codebert_sentence' : 'sentence_codebert_tokens',
    #textcnn
    'vocab_nature' : 'nature_feature' ,
    'vocab_data_flow' : 'data_feature' ,
    'vocab_control_flow' : 'control_feature',
    'vocab_sentence' : 'sentence_feature',
}



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

# random seed
setup_seed(3047)

def get_metrics(true_dict, prob_dict,print_metrics:bool = False):
    assert true_dict.keys() == prob_dict.keys()
    metrics = { }
    for task_type in true_dict.keys():
        true = np.array(true_dict[task_type])
        prob = np.array(prob_dict[task_type])
        pred = prob.argmax(axis=1)
        metrics_item = {}
        metrics_item["acc"] = round(accuracy_score(true, pred), 3)
        metrics_item["precision"] = round(precision_score(true, pred, average='macro'), 3)
        metrics_item["recall"] = round(recall_score(true, pred, average='macro'), 3)
        metrics_item["f1"] = round(f1_score(true, pred, average='macro'), 3)
        metrics_item["mcc"] = round(matthews_corrcoef(true, pred), 3)
        metrics[task_type] = metrics_item

    if print_metrics:
        for task_type in metrics.keys():
            log_info(f'{task_type} : {metrics[task_type]}')
    return metrics

def multi_task_criterion(pre_item:dict,true_item:dict,criterion: nn.CrossEntropyLoss,):
    loss = None
    assert pre_item.keys() == true_item.keys()
    keys = pre_item.keys()
    for task_type in keys:
        pre = pre_item[task_type]
        true = true_item[task_type]
        if loss is None:
            loss = criterion(pre,true)
        else:
            loss += criterion(pre,true)
    return loss

def init_pred_and_label():
    PRED ,LABEL = { } , { }
    for task_type in cvss_task_type.keys():
        PRED[task_type] = []
        LABEL[task_type] = []
    return PRED,LABEL

def add_to_pred_and_label(PRED:dict,pre_item:dict,LABEL:dict,true_item:dict):
    for task_type in cvss_task_type.keys():
        PRED[task_type].extend(pre_item[task_type].tolist())
        LABEL[task_type].extend(true_item[task_type].tolist())

def train(
        epoch: int, dev: torch.device,
        dl: GraphDataLoader,
        criterion: nn.CrossEntropyLoss,
        model: Union[GModel_Embedding_MultiTask,GModel_Embedding_SingleTask],
        optimizer: torch.optim.Optimizer):
    model.train()
    total_loss = 0
    PRED,LABEL = init_pred_and_label()
    for batch in dl:
        # Training
        g = batch[0].to(dev)
        # tokens = g.ndata["_Feat"].to(dev)
        tokens = g.ndata.pop("tokens").to(dev)

        attention_masks = (tokens != 1).int().to(dev)

        true_item = batch[1]    # (batch_size,dict)
        for k,v in true_item.items():
            true_item[k] = v.to(dev)
        # {'cvss2_C': tensor([2, 0, 1, 2]), 'cvss2_I': tensor([2, 0, 1, 2]), ..... }
        pre_item = model(g, tokens, attention_masks)    # (batch_size,dict)
        optimizer.zero_grad()
        loss = multi_task_criterion(pre_item,true_item,criterion)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        add_to_pred_and_label(PRED,pre_item,LABEL,true_item)

        del g
        del tokens
        del loss
        torch.cuda.empty_cache()

    log_info("\n")
    log_info(f"epoch:{epoch}------------train-----------")
    log_info(f"train_loss: {total_loss}")
    get_metrics(LABEL, PRED , print_metrics=True)

def evaluate(
        type: str, dev: torch.device,
        early_stopping: EarlyStopping,
        dl: GraphDataLoader,
        criterion: nn.CrossEntropyLoss,
        model: Union[GModel_Embedding_MultiTask,GModel_Embedding_SingleTask],
        lr_scheduler : torch.optim.lr_scheduler.StepLR
) -> bool:
    if type == 'test':
        best_model_path = os.path.join(save_path,f'{model_name}_best_network.pth')
        model.load_state_dict(torch.load(best_model_path))
    model.eval()
    PRED,LABEL = init_pred_and_label()
    total_loss = 0
    with torch.no_grad():
        for batch in dl:
            g = batch[0].to(dev)
            # tokens = g.ndata["_Feat"].to(dev)
            tokens = g.ndata.pop("tokens").to(dev)
            attention_masks = (tokens != 1).int().to(dev)
            true_item = batch[1]  # (batch_size,dict)
            for k, v in true_item.items():
                true_item[k] = v.to(dev)
            pre_item = model(g, tokens, attention_masks)

            loss = multi_task_criterion(pre_item, true_item, criterion)
            total_loss += loss.item()

            add_to_pred_and_label(PRED, pre_item, LABEL, true_item)

            del g
            del tokens
            del loss
            torch.cuda.empty_cache()
            # print(all_pred)
        log_info(f"--------------------{type}-----------")
        log_info(f"{type}_loss: {total_loss}")
        valid_mets = get_metrics(LABEL, PRED,print_metrics=True)

        if type == 'valid':
            val_f1 = 0
            for item in valid_mets.values():
                val_f1 += item[early_stop_metric]
            val_f1 = val_f1 / len(cvss_task_type.keys())
            early_stopping(val_f1, model)

            if early_stopping.early_stop:
                log_info("Early stopping")
                return True  # early break
            if early_stopping.counter_exceed_2():
                log_info('lr_scheduler adjust learning rate')
                lr_scheduler.step()
    return False

def main(lr="SGD", lr_rate=0.0005, dropout=0.1):
    lr_rate = float(lr_rate)
    dropout = float(dropout)
    torch.cuda.empty_cache()
    global save_path
    date = strftime('%m:%d:%H:%M', localtime())
    save_path = f'{save_path}/lr_{lr_rate}-{date}'
    gcn.get_dir(save_path)
    init_logging(os.path.join(save_path,'logging.log'))

    log_info(f'learning_rate: {lr_rate} dropout:{dropout} device:{os.environ["CUDA_VISIBLE_DEVICES"]} context_type:{context_type} ,early_stop:{early_stop_metric}')
    train_ds = GraphDataset_Token_MultiTask(context_type = context_type_dict[context_type] , token_type= token_type , partition="train")
    val_ds = GraphDataset_Token_MultiTask(context_type = context_type_dict[context_type] , token_type= token_type ,partition="valid")
    test_ds = GraphDataset_Token_MultiTask(context_type = context_type_dict[context_type] , token_type= token_type ,partition="test")
    """
    If you are running this file for the first time, please uncomment it,which helps to cache the graph fastly.
    """
    # train_ds.cache_features()
    # val_ds.cache_features()
    # test_ds.cache_features()
    # print(train_ds[0][0].ndata["tokens"].shape)
    # print(train_ds[0][0].ndata["tokens"])
    # return 
    #batch_size
    batch_size = None
    if token_type in ["codebert","unixcoder"]:
        batch_size = 1
    elif token_type == "textcnn":
        batch_size = 256
    elif token_type == "lstm":
        batch_size = 32
    dl_args1 = {"drop_last": False, "shuffle": True, "num_workers": 6}
    dl_args2 = {"drop_last": False, "shuffle": False, "num_workers": 6}
    train_dl = GraphDataLoader(train_ds, batch_size=batch_size, **dl_args1)
    val_dl = GraphDataLoader(val_ds, batch_size=batch_size, **dl_args2)
    test_dl = GraphDataLoader(test_ds, batch_size=batch_size, **dl_args2)

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = GModel_Embedding(embedding_dim = embedding_dim,hidden_size = hidden_size,\
    #     dropout = dropout,dev = dev)

    # codebert
    encoder = None
    if token_type=="codebert":
        encoder = AutoModel.from_pretrained(f"microsoft/codebert-base")   # microsoft/codebert-base   microsoft/unixcoder-base-nine
    elif token_type == "unixcoder":
        encoder = AutoModel.from_pretrained(f"microsoft/unixcoder-base-nine")
    elif token_type == "textcnn":
        encoder = TextCNN(vocab_size=10000, embedding_dim=300, kernel_sizes = [2,3,4], num_channels=128,dropout = 0.1)
    elif token_type == "lstm":
        encoder = LSTM_Encoder(embedding_dim = 300,hidden_size = 300,dropout = 0.8)
    
    model = GModel_Embedding_MultiTask(encoder=encoder, encoder_type = token_type,gnn_type=gnn_type, dropout=dropout)
    # model.load_state_dict(torch.load(f'result/model_multi_task/nature/lr_0.0004-10:10:10:26/codebert_finetune_best_network.pth'))
    # {'cvss2_C': tensor([[0.3777, -0.5968, 0.5857],
    #                     [0.7428, -0.5861, 0.4512],
    #                     [0.6054, -0.6138, 0.2595],
    #                     [0.5944, -0.4635, 0.6401]], device='cuda:0'  .........

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info(f"number of params: {n_parameters}")
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    # model = model.cuda(device=device_ids[0])
    model.to(dev)

    criterion = nn.CrossEntropyLoss()
    if lr == "Adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_rate)
    elif lr == "SGD":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_rate)

    early_stopping = EarlyStopping(save_path, model_name=model_name, verbose=True,patience=8)
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)
    try:
        for epoch in range(500):
            log_info(f'current learning rate {stepLR.get_lr()}')
            train(epoch, dev, train_dl, criterion, model, optimizer)
            if evaluate('valid', dev, early_stopping, val_dl, criterion, model, stepLR):
                break  # break
        evaluate('test', dev, early_stopping, test_dl, criterion, model, stepLR)
    except Exception as e:
        logging.exception(e)

if __name__ == '__main__':
    main()
    # main("")
