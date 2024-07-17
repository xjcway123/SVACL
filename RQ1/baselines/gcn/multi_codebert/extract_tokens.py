from transformers import AutoTokenizer

import pandas as pd
import sys

path = "/".join(sys.path[0].split("/")[:-2])
sys.path.append(path)
import gcn

max_length = 512
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")


def codebert_tokenizer(content: str):
    tokens_id = tokenizer(content, max_length=max_length, padding=True, truncation=True, add_special_tokens=True)[
        "input_ids"]

    tokens_id = tokens_id + [1] * (max_length - len(tokens_id))  # <pad> is 1
    return " ".join([str(i) for i in tokens_id])


print("###################################")
print("load data")
data_file = gcn.data_dir() / "mydata_split.csv"
data = pd.read_csv(data_file)
print(data.count())

data["codebert_feature"] = data[["func_before"]].apply(
    lambda r: codebert_tokenizer(r.func_before), axis=1
)
l = data["codebert_feature"].tolist()

for s in l:
    t = [int(i) for i in s.split()]
    print(len(t))
feature_path = gcn.get_dir(gcn.data_dir() / "codebert") / "codebert_tokens_data.csv"
data.to_csv(feature_path)
