import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

from transformers import AutoTokenizer, AutoModel
from model import CodebertModel
from model import EarlyStopping

path = "/".join(sys.path[0].split("/")[:-2])
sys.path.append(path)
import gcn

labels = ['cvss2_C', 'cvss2_I', 'cvss2_A', 'cvss2_AV',
          'cvss2_AC', 'cvss2_AU', 'cvss2_severity']

# config
train_batch = 16
test_batch = 16
valid_batch = 16

import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


# random seed
setup_seed(20)


def get_metrics(true, prob):
    true = true.detach().cpu().numpy()
    prob = prob.detach().cpu().numpy()
    pred = prob.argmax(axis=1)
    # l = pred.tolist()
    # print(len(l))
    # print(len([i for i in l if i==0]))
    # print(len([i for i in l if i==1]))
    # print(len([i for i in l if i==2]))

    metrics = {}
    metrics["acc"] = round(accuracy_score(true, pred), 3)
    metrics["precision"] = round(precision_score(true, pred, average='macro', zero_division=1), 3)
    metrics["recall"] = round(recall_score(true, pred, average='macro'), 3)
    metrics["f1"] = round(f1_score(true, pred, average='macro'), 3)
    metrics["mcc"] = round(matthews_corrcoef(true, pred), 3)

    return metrics


def generate_output(df, label, fit=True, lb=None):
    if fit:
        lb = LabelBinarizer()
        y = lb.fit_transform(df[label].values)

        return y, lb

    y = lb.fit_transform(df[label].values)

    return y


feature_path = gcn.data_dir() / "codebert" / "codebert_tokens_data.csv"
data = pd.read_csv(feature_path)

dev = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
partition = "partition"
# print(data.count())

data["codebert_feature"] = data["codebert_feature"].apply(
    lambda r: [int(i) for i in r.split()]
)

tokens = "codebert_feature"

train_data = data[data[partition] == "train"]
test_data = data[data[partition] == "test"]
valid_data = data[data[partition] == "valid"]

train_tokens = np.array(train_data[tokens].values.tolist())
test_tokens = np.array(test_data[tokens].values.tolist())
valid_tokens = np.array(valid_data[tokens].values.tolist())

print("\n########################")
print("Extracting labels")

y_train = {}
y_valid = {}
lb_dict = {}

n_classes = {label: len(train_data[label].unique()) for label in labels}

for label in labels:
    cur_y_train, lb_dict[label] = generate_output(train_data, label, fit=True)

    cur_output = label
    y_train[cur_output] = cur_y_train

for label in labels:
    cur_y_val = generate_output(valid_data, label, fit=False, lb=lb_dict[label])

    cur_output = label
    y_valid[cur_output] = cur_y_val

y_test = {}

for label in labels:
    cur_y_test = generate_output(test_data, label, fit=False, lb=lb_dict[label])

    cur_output = label
    y_test[cur_output] = cur_y_test

# label =  labels[0]+"_output"#cvss2_C

y_train_C = torch.tensor(y_train[labels[0]])  # cvss2_C
y_train_I = torch.tensor(y_train[labels[1]])  # cvss2_I
y_train_A = torch.tensor(y_train[labels[2]])  # cvss2_A
y_train_AV = torch.tensor(y_train[labels[3]])  # cvss2_AV
y_train_AC = torch.tensor(y_train[labels[4]])  # cvss2_AC
y_train_AU = torch.tensor(y_train[labels[5]])  # cvss2_AU
y_train_severity = torch.tensor(y_train[labels[6]])  # cvss2_severity

y_valid_C = torch.tensor(y_valid[labels[0]])  # cvss2_C
y_valid_I = torch.tensor(y_valid[labels[1]])  # cvss2_I
y_valid_A = torch.tensor(y_valid[labels[2]])  # cvss2_A
y_valid_AV = torch.tensor(y_valid[labels[3]])  # cvss2_AV
y_valid_AC = torch.tensor(y_valid[labels[4]])  # cvss2_AC
y_valid_AU = torch.tensor(y_valid[labels[5]])  # cvss2_AU
y_valid_severity = torch.tensor(y_valid[labels[6]])  # cvss2_severity

y_test_C = torch.tensor(y_test[labels[0]])  # cvss2_C
y_test_I = torch.tensor(y_test[labels[1]])  # cvss2_I
y_test_A = torch.tensor(y_test[labels[2]])  # cvss2_A
y_test_AV = torch.tensor(y_test[labels[3]])  # cvss2_AV
y_test_AC = torch.tensor(y_test[labels[4]])  # cvss2_AC
y_test_AU = torch.tensor(y_test[labels[5]])  # cvss2_AU
y_test_severity = torch.tensor(y_test[labels[6]])  # cvss2_severity

train_dataloader = TensorDataset(torch.tensor(train_tokens), y_train_C, y_train_I, y_train_A, y_train_AV, y_train_AC,
                                 y_train_AU, y_train_severity)
test_dataloader = TensorDataset(torch.tensor(test_tokens), y_test_C, y_test_I, y_test_A, y_test_AV, y_test_AC,
                                y_test_AU, y_test_severity)
valid_dataloader = TensorDataset(torch.tensor(valid_tokens), y_valid_C, y_valid_I, y_valid_A, y_valid_AV, y_valid_AC,
                                 y_valid_AU, y_valid_severity)

trainData = DataLoader(train_dataloader, batch_size=train_batch, shuffle=True)
testData = DataLoader(test_dataloader, batch_size=test_batch, shuffle=False)
validData = DataLoader(valid_dataloader, batch_size=valid_batch, shuffle=False)
print("###############################")
print("model")

codebert = AutoModel.from_pretrained("microsoft/codebert-base")

model = CodebertModel(codebert)
model.to(dev)
save_path = gcn.get_dir(gcn.multi_codebert_dir()) / "save_model_cls"
gcn.get_dir(save_path)
early_stopping = EarlyStopping(save_path, verbose=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

class2id = {
    "cvss2_C": 1,
    "cvss2_I": 2,
    "cvss2_A": 3,
    "cvss2_AV": 4,
    "cvss2_AC": 5,
    "cvss2_AU": 6,
    "cvss2_severity": 7,
}

for epoch in range(50):
    print("##############################")
    print("training")
    model.train()
    PRED_C = torch.empty((0, 3)).long()
    LABEL_C = torch.empty((0)).long()
    PRED_I = torch.empty((0, 3)).long()
    LABEL_I = torch.empty((0)).long()
    PRED_A = torch.empty((0, 3)).long()
    LABEL_A = torch.empty((0)).long()
    PRED_AV = torch.empty((0, 3)).long()
    LABEL_AV = torch.empty((0)).long()
    PRED_AC = torch.empty((0, 3)).long()
    LABEL_AC = torch.empty((0)).long()
    PRED_AU = torch.empty((0, 2)).long()
    LABEL_AU = torch.empty((0)).long()
    PRED_severity = torch.empty((0, 3)).long()
    LABEL_severity = torch.empty((0)).long()
    total_loss = 0
    for step, batch in enumerate(trainData):
        true_C = batch[1].argmax(axis=1).long().to(dev)
        true_I = batch[2].argmax(axis=1).long().to(dev)
        true_A = batch[3].argmax(axis=1).long().to(dev)
        true_AV = batch[4].argmax(axis=1).long().to(dev)
        true_AC = batch[5].argmax(axis=1).long().to(dev)
        true_AU = batch[6].squeeze(1).long().to(dev)
        true_severity = batch[7].argmax(axis=1).long().to(dev)

        input_ids = batch[0].to(dev)
        attention_masks = (input_ids != 1).int().to(dev)
        pre_C, pre_I, pre_A, pre_AV, pre_AC, pre_AU, pre_severity, _ = model(input_ids, attention_masks)
        optimizer.zero_grad()
        loss_C = criterion(pre_C, true_C)
        loss_I = criterion(pre_I, true_I)
        loss_A = criterion(pre_A, true_A)
        loss_AV = criterion(pre_AV, true_AV)
        loss_AC = criterion(pre_AC, true_AC)
        loss_AU = criterion(pre_AU, true_AU)
        loss_severity = criterion(pre_severity, true_severity)
        loss = (loss_C + loss_I + loss_A + loss_AV + loss_AC + loss_AU + loss_severity)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        PRED_C = torch.cat([PRED_C, pre_C.detach().cpu()])
        LABEL_C = torch.cat([LABEL_C, true_C.detach().cpu()])
        PRED_I = torch.cat([PRED_I, pre_I.detach().cpu()])
        LABEL_I = torch.cat([LABEL_I, true_I.detach().cpu()])
        PRED_A = torch.cat([PRED_A, pre_A.detach().cpu()])
        LABEL_A = torch.cat([LABEL_A, true_A.detach().cpu()])
        PRED_AV = torch.cat([PRED_AV, pre_AV.detach().cpu()])
        LABEL_AV = torch.cat([LABEL_AV, true_AV.detach().cpu()])
        PRED_AC = torch.cat([PRED_AC, pre_AC.detach().cpu()])
        LABEL_AC = torch.cat([LABEL_AC, true_AC.detach().cpu()])
        PRED_AU = torch.cat([PRED_AU, pre_AU.detach().cpu()])
        LABEL_AU = torch.cat([LABEL_AU, true_AU.detach().cpu()])
        PRED_severity = torch.cat([PRED_severity, pre_severity.detach().cpu()])
        LABEL_severity = torch.cat([LABEL_severity, true_severity.detach().cpu()])
    # torch.cuda.empty_cache()
    # print(y_true)
    # print(outputs)

    print()
    print(f"epoch:{epoch}------------train-----------")

    print("train_loss: ", total_loss)
    train_C_mets = get_metrics(LABEL_C, PRED_C)
    print("train C metric:", train_C_mets)
    train_I_mets = get_metrics(LABEL_I, PRED_I)
    print("train I metric:", train_I_mets)
    train_A_mets = get_metrics(LABEL_A, PRED_A)
    print("train A metric:", train_A_mets)
    train_AV_mets = get_metrics(LABEL_AV, PRED_AV)
    print("train AV metric:", train_AV_mets)
    train_AC_mets = get_metrics(LABEL_AC, PRED_AC)
    print("train AC metric:", train_AC_mets)
    train_AU_mets = get_metrics(LABEL_AU, PRED_AU)
    print("train AU metric:", train_AU_mets)
    train_severity_mets = get_metrics(LABEL_severity, PRED_severity)
    print("train severity metric:", train_severity_mets)

    model.eval()
    PRED_C = torch.empty((0, 3)).long()
    LABEL_C = torch.empty((0)).long()
    PRED_I = torch.empty((0, 3)).long()
    LABEL_I = torch.empty((0)).long()
    PRED_A = torch.empty((0, 3)).long()
    LABEL_A = torch.empty((0)).long()
    PRED_AV = torch.empty((0, 3)).long()
    LABEL_AV = torch.empty((0)).long()
    PRED_AC = torch.empty((0, 3)).long()
    LABEL_AC = torch.empty((0)).long()
    PRED_AU = torch.empty((0, 2)).long()
    LABEL_AU = torch.empty((0)).long()
    PRED_severity = torch.empty((0, 3)).long()
    LABEL_severity = torch.empty((0)).long()
    total_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(validData):
            true_C = batch[1].argmax(axis=1).long().to(dev)
            true_I = batch[2].argmax(axis=1).long().to(dev)
            true_A = batch[3].argmax(axis=1).long().to(dev)
            true_AV = batch[4].argmax(axis=1).long().to(dev)
            true_AC = batch[5].argmax(axis=1).long().to(dev)
            true_AU = batch[6].squeeze(1).long().to(dev)
            true_severity = batch[7].argmax(axis=1).long().to(dev)

            input_ids = batch[0].to(dev)
            attention_masks = (input_ids != 1).int().to(dev)
            pre_C, pre_I, pre_A, pre_AV, pre_AC, pre_AU, pre_severity, _ = model(input_ids, attention_masks)

            loss_C = criterion(pre_C, true_C)
            loss_I = criterion(pre_I, true_I)
            loss_A = criterion(pre_A, true_A)
            loss_AV = criterion(pre_AV, true_AV)
            loss_AC = criterion(pre_AC, true_AC)
            loss_AU = criterion(pre_AU, true_AU)
            loss_severity = criterion(pre_severity, true_severity)
            loss = (loss_C + loss_I + loss_A + loss_AV + loss_AC + loss_AU + loss_severity)

            total_loss += loss.item()

            PRED_C = torch.cat([PRED_C, pre_C.detach().cpu()])
            LABEL_C = torch.cat([LABEL_C, true_C.detach().cpu()])
            PRED_I = torch.cat([PRED_I, pre_I.detach().cpu()])
            LABEL_I = torch.cat([LABEL_I, true_I.detach().cpu()])
            PRED_A = torch.cat([PRED_A, pre_A.detach().cpu()])
            LABEL_A = torch.cat([LABEL_A, true_A.detach().cpu()])
            PRED_AV = torch.cat([PRED_AV, pre_AV.detach().cpu()])
            LABEL_AV = torch.cat([LABEL_AV, true_AV.detach().cpu()])
            PRED_AC = torch.cat([PRED_AC, pre_AC.detach().cpu()])
            LABEL_AC = torch.cat([LABEL_AC, true_AC.detach().cpu()])
            PRED_AU = torch.cat([PRED_AU, pre_AU.detach().cpu()])
            LABEL_AU = torch.cat([LABEL_AU, true_AU.detach().cpu()])
            PRED_severity = torch.cat([PRED_severity, pre_severity.detach().cpu()])
            LABEL_severity = torch.cat([LABEL_severity, true_severity.detach().cpu()])
    # torch.cuda.empty_cache()

    print(f"--------------------valid-----------")
    print("valid_loss: ", total_loss)
    valid_C_mets = get_metrics(LABEL_C, PRED_C)
    print("valid C metric:", valid_C_mets)
    valid_I_mets = get_metrics(LABEL_I, PRED_I)
    print("valid I metric:", valid_I_mets)
    valid_A_mets = get_metrics(LABEL_A, PRED_A)
    print("valid A metric:", valid_A_mets)
    valid_AV_mets = get_metrics(LABEL_AV, PRED_AV)
    print("valid AV metric:", valid_AV_mets)
    valid_AC_mets = get_metrics(LABEL_AC, PRED_AC)
    print("valid AC metric:", valid_AC_mets)
    valid_AU_mets = get_metrics(LABEL_AU, PRED_AU)
    print("valid AU metric:", valid_AU_mets)
    valid_severity_mets = get_metrics(LABEL_severity, PRED_severity)
    print("valid severity metric:", valid_severity_mets)

    val_f1 = (valid_C_mets["f1"] + valid_I_mets["f1"] + \
              valid_A_mets["f1"] + valid_AV_mets["f1"] + valid_AC_mets["f1"] + valid_AU_mets["f1"] +
              valid_severity_mets["f1"]) / 7
    early_stopping(val_f1, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break  # early break
best_model_path = save_path / ("best_network.pth")
model.load_state_dict(torch.load(best_model_path))
model.eval()

PRED_C = torch.empty((0, 3)).long()
LABEL_C = torch.empty((0)).long()
PRED_I = torch.empty((0, 3)).long()
LABEL_I = torch.empty((0)).long()
PRED_A = torch.empty((0, 3)).long()
LABEL_A = torch.empty((0)).long()
PRED_AV = torch.empty((0, 3)).long()
LABEL_AV = torch.empty((0)).long()
PRED_AC = torch.empty((0, 3)).long()
LABEL_AC = torch.empty((0)).long()
PRED_AU = torch.empty((0, 2)).long()
LABEL_AU = torch.empty((0)).long()
PRED_severity = torch.empty((0, 3)).long()
LABEL_severity = torch.empty((0)).long()
total_loss = 0
with torch.no_grad():
    for step, batch in enumerate(validData):
        true_C = batch[1].argmax(axis=1).long().to(dev)
        true_I = batch[2].argmax(axis=1).long().to(dev)
        true_A = batch[3].argmax(axis=1).long().to(dev)
        true_AV = batch[4].argmax(axis=1).long().to(dev)
        true_AC = batch[5].argmax(axis=1).long().to(dev)
        true_AU = batch[6].squeeze(1).long().to(dev)
        true_severity = batch[7].argmax(axis=1).long().to(dev)

        input_ids = batch[0].to(dev)
        attention_masks = (input_ids != 1).int().to(dev)
        pre_C, pre_I, pre_A, pre_AV, pre_AC, pre_AU, pre_severity, _ = model(input_ids, attention_masks)

        loss_C = criterion(pre_C, true_C)
        loss_I = criterion(pre_I, true_I)
        loss_A = criterion(pre_A, true_A)
        loss_AV = criterion(pre_AV, true_AV)
        loss_AC = criterion(pre_AC, true_AC)
        loss_AU = criterion(pre_AU, true_AU)
        loss_severity = criterion(pre_severity, true_severity)
        loss = (loss_C + loss_I + loss_A + loss_AV + loss_AC + loss_AU + loss_severity)

        total_loss += loss.item()

        PRED_C = torch.cat([PRED_C, pre_C.detach().cpu()])
        LABEL_C = torch.cat([LABEL_C, true_C.detach().cpu()])
        PRED_I = torch.cat([PRED_I, pre_I.detach().cpu()])
        LABEL_I = torch.cat([LABEL_I, true_I.detach().cpu()])
        PRED_A = torch.cat([PRED_A, pre_A.detach().cpu()])
        LABEL_A = torch.cat([LABEL_A, true_A.detach().cpu()])
        PRED_AV = torch.cat([PRED_AV, pre_AV.detach().cpu()])
        LABEL_AV = torch.cat([LABEL_AV, true_AV.detach().cpu()])
        PRED_AC = torch.cat([PRED_AC, pre_AC.detach().cpu()])
        LABEL_AC = torch.cat([LABEL_AC, true_AC.detach().cpu()])
        PRED_AU = torch.cat([PRED_AU, pre_AU.detach().cpu()])
        LABEL_AU = torch.cat([LABEL_AU, true_AU.detach().cpu()])
        PRED_severity = torch.cat([PRED_severity, pre_severity.detach().cpu()])
        LABEL_severity = torch.cat([LABEL_severity, true_severity.detach().cpu()])
    # torch.cuda.empty_cache()

    print(f"--------------------valid-----------")
    print("valid_loss: ", total_loss)
    valid_C_mets = get_metrics(LABEL_C, PRED_C)
    print("valid C metric:", valid_C_mets)
    valid_I_mets = get_metrics(LABEL_I, PRED_I)
    print("valid I metric:", valid_I_mets)
    valid_A_mets = get_metrics(LABEL_A, PRED_A)
    print("valid A metric:", valid_A_mets)
    valid_AV_mets = get_metrics(LABEL_AV, PRED_AV)
    print("valid AV metric:", valid_AV_mets)
    valid_AC_mets = get_metrics(LABEL_AC, PRED_AC)
    print("valid AC metric:", valid_AC_mets)
    valid_AU_mets = get_metrics(LABEL_AU, PRED_AU)
    print("valid AU metric:", valid_AU_mets)
    valid_severity_mets = get_metrics(LABEL_severity, PRED_severity)
    print("valid severity metric:", valid_severity_mets)

PRED_C = torch.empty((0, 3)).long()
LABEL_C = torch.empty((0)).long()
PRED_I = torch.empty((0, 3)).long()
LABEL_I = torch.empty((0)).long()
PRED_A = torch.empty((0, 3)).long()
LABEL_A = torch.empty((0)).long()
PRED_AV = torch.empty((0, 3)).long()
LABEL_AV = torch.empty((0)).long()
PRED_AC = torch.empty((0, 3)).long()
LABEL_AC = torch.empty((0)).long()
PRED_AU = torch.empty((0, 2)).long()
LABEL_AU = torch.empty((0)).long()
PRED_severity = torch.empty((0, 3)).long()
LABEL_severity = torch.empty((0)).long()
total_loss = 0
with torch.no_grad():
    for step, batch in enumerate(testData):
        true_C = batch[1].argmax(axis=1).long().to(dev)
        true_I = batch[2].argmax(axis=1).long().to(dev)
        true_A = batch[3].argmax(axis=1).long().to(dev)
        true_AV = batch[4].argmax(axis=1).long().to(dev)
        true_AC = batch[5].argmax(axis=1).long().to(dev)
        true_AU = batch[6].squeeze(1).long().to(dev)
        true_severity = batch[7].argmax(axis=1).long().to(dev)

        input_ids = batch[0].to(dev)
        attention_masks = (input_ids != 1).int().to(dev)
        pre_C, pre_I, pre_A, pre_AV, pre_AC, pre_AU, pre_severity, _ = model(input_ids, attention_masks)

        loss_C = criterion(pre_C, true_C)
        loss_I = criterion(pre_I, true_I)
        loss_A = criterion(pre_A, true_A)
        loss_AV = criterion(pre_AV, true_AV)
        loss_AC = criterion(pre_AC, true_AC)
        loss_AU = criterion(pre_AU, true_AU)
        loss_severity = criterion(pre_severity, true_severity)
        loss = (loss_C + loss_I + loss_A + loss_AV + loss_AC + loss_AU + loss_severity)

        total_loss += loss.item()

        PRED_C = torch.cat([PRED_C, pre_C.detach().cpu()])
        LABEL_C = torch.cat([LABEL_C, true_C.detach().cpu()])
        PRED_I = torch.cat([PRED_I, pre_I.detach().cpu()])
        LABEL_I = torch.cat([LABEL_I, true_I.detach().cpu()])
        PRED_A = torch.cat([PRED_A, pre_A.detach().cpu()])
        LABEL_A = torch.cat([LABEL_A, true_A.detach().cpu()])
        PRED_AV = torch.cat([PRED_AV, pre_AV.detach().cpu()])
        LABEL_AV = torch.cat([LABEL_AV, true_AV.detach().cpu()])
        PRED_AC = torch.cat([PRED_AC, pre_AC.detach().cpu()])
        LABEL_AC = torch.cat([LABEL_AC, true_AC.detach().cpu()])
        PRED_AU = torch.cat([PRED_AU, pre_AU.detach().cpu()])
        LABEL_AU = torch.cat([LABEL_AU, true_AU.detach().cpu()])
        PRED_severity = torch.cat([PRED_severity, pre_severity.detach().cpu()])
        LABEL_severity = torch.cat([LABEL_severity, true_severity.detach().cpu()])
    # torch.cuda.empty_cache()

    print(f"--------------------test-----------")
    print("test_loss: ", total_loss)
    test_C_mets = get_metrics(LABEL_C, PRED_C)
    print("test C metric:", test_C_mets)
    test_I_mets = get_metrics(LABEL_I, PRED_I)
    print("test I metric:", test_I_mets)
    test_A_mets = get_metrics(LABEL_A, PRED_A)
    print("test A metric:", test_A_mets)
    test_AV_mets = get_metrics(LABEL_AV, PRED_AV)
    print("test AV metric:", test_AV_mets)
    test_AC_mets = get_metrics(LABEL_AC, PRED_AC)
    print("test AC metric:", test_AC_mets)
    test_AU_mets = get_metrics(LABEL_AU, PRED_AU)
    print("test AU metric:", test_AU_mets)
    test_severity_mets = get_metrics(LABEL_severity, PRED_severity)
    print("test severity metric:", test_severity_mets)
