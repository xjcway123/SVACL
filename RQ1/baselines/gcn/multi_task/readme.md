FVA in the folder.
1. run main.py, if you run main.py for the first time, please uncomment 235-240 to cache feature.Different feature extraction types(codebert,unixcoder etc.) need to be cached separately
you can choose different graph model(gcn,gat,gatv2,etc.),different context(zero context,data context,etc),different feature extraction(codebert,unixcoder,etc.)
file description
    |dataset_new.py     | define dataset class and extract_feature by codebert and unixcoder
    |extract_features.py| define vocab class to extract feature for lstm and textcnn
    |model.py           | define model
    |utils.py           | about logging