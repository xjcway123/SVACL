# -*- coding: utf-8 -*-
import sys

import gcn
import torch
import pandas as pd
from transformers import  AutoTokenizer, AutoModel
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import json
def extract_codebert_features(text, model, tokenizer, batch_size=8):
    max_length = 20
    text = text.split('\n')
    tokens_ids = tokenizer(text, max_length=max_length, padding=True, truncation=True, add_special_tokens=True)
    attention_masks = tokens_ids['attention_mask']
    tokens_ids = tokens_ids['input_ids']
    batch_size = batch_size

	# wrap tensors
    train_data = TensorDataset(torch.tensor(tokens_ids), torch.tensor(attention_masks),
							   torch.tensor([1] * len(tokens_ids)))

	# dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=None, batch_size=batch_size)

	# for tokens in tokens_ids:
	# 	print(tokenizer.decode(tokens))
    
    features = []
    for step,batch in enumerate(train_dataloader):
        sent_ids,masks,labels = batch
        output = model(sent_ids, attention_mask=masks)[1]
        
        # print(output[0].shape)
        if step ==0:
            features = output.squeeze().detach().cpu().numpy().tolist()

        else:
            features.extend(output.squeeze().detach().cpu().numpy().tolist())


    print(len(features), len(features[0]))

    return features
def Remove_annotation(code):
    # code = code[0]
    # print(code)
    code = code.split('\n')
    res = []
    is_annotation = False
    for cur_line in code:
        temp = ''
        
        if '//' in cur_line:
            temp = cur_line[:cur_line.find('//')]
        elif '/*' in cur_line:
            temp = cur_line[:cur_line.find('/*')]
            is_annotation = True
            if '*/' in cur_line:
                temp += cur_line[cur_line.find('/*')+2:]
                is_annotation = False
        elif '*/' in cur_line:
            temp = cur_line[cur_line.find('*/')+2:]
            is_annotation = False
        else:
            if is_annotation:
                continue
            temp = cur_line
        
        if temp.strip()!='':
            res.append(temp)
    return "\n".join(res)

if __name__ == "__main__":
    datafile = gcn.data_dir() / "combined_df_method_partition.parquet"
    df = pd.read_parquet(datafile)
    df['id'] = np.arange(df.shape[0])
    print(df.count())
    df.to_parquet(datafile)

    # df = df.head(10)
    # cvss_cols = ['cvss2_confidentiality_impact','cvss2_integrity_impact','cvss2_availability_impact',\
    #          'cvss2_access_vector','cvss2_access_complexity','cvss2_authentication','severity']
    # df = df[['code','id',"partition"]+cvss_cols]
    # df.set_index(["id"], inplace=True)
    # df['filter_code'] = df[['code']].apply(
	# lambda r: Remove_annotation(r.code), axis=1)
    # # print(df[["code",'filter_code']].head(10))
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    # model = AutoModel.from_pretrained("microsoft/codebert-base")
    # df['feature'] = df[['filter_code']].apply(
	# lambda r: extract_codebert_features(r.filter_code, model, tokenizer, batch_size=16), axis=1)
    
    # feature  = df['feature'].to_list()
    # # df.to_parquet(gcn.data_dir() / "combined_df_method_features.parquet")
    # with open(gcn.data_dir() / "combined_df_method_features.json",'w',encoding='utf-8') as f:
    #     json.dump(feature,f)
