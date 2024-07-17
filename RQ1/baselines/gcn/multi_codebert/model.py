import torch
import torch.nn as nn
import numpy as np
import os

class CodebertModel(nn.Module):
    def __init__(self, encoder,hidden_size = 768):# BERT output dimension is 768
        super(CodebertModel, self).__init__()
        self.encoder = encoder
        self.classifier_C = nn.Linear(hidden_size, 3)
        self.classifier_I = nn.Linear(hidden_size, 3)
        self.classifier_A = nn.Linear(hidden_size, 3)
        self.classifier_AV = nn.Linear(hidden_size, 3)
        self.classifier_AC = nn.Linear(hidden_size, 3)
        self.classifier_AU = nn.Linear(hidden_size, 2)
        self.classifier_severity = nn.Linear(hidden_size, 3)
        self.dropout = nn.Dropout(0.8)
    def forward(self,input_ids,attention_masks=None):
        outputs = self.encoder(input_ids,attention_mask=attention_masks)
        x = outputs[0][:,0,:]
        x = self.dropout(x)
        cvss_C = self.classifier_C(x)
        cvss_I = self.classifier_I(x)
        cvss_A = self.classifier_A(x)
        cvss_AV = self.classifier_AV(x)
        cvss_AC = self.classifier_AC(x)
        cvss_AU = self.classifier_AU(x)
        cvss_severity = self.classifier_severity(x)
        return cvss_C,cvss_I,cvss_A,cvss_AV,cvss_AC,cvss_AU,cvss_severity,x # x is bert output(768,)



class EarlyStopping:
    """Early stops the training if validation f1 doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : model save path
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.f1_max = np.Inf
        self.delta = delta

    def __call__(self, f1, model):

        score = f1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(f1, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(f1, model)
            self.counter = 0

    def save_checkpoint(self, f1, model):
        '''Saves model when f1 increase.'''
        if self.verbose:
            print(f'Validation f1 increased ({self.f1_max:.6f} --> {f1:.6f}).  Saving model ...')
        self.f1_max = f1
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)	# save best model
        self.val_loss_min = f1


