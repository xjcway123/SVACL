import transformers.models.roberta.tokenization_roberta_fast
from torch.utils.data import TensorDataset, DataLoader,Dataset
from transformers import  AutoTokenizer
import numpy as np
import dgl
import pandas as pd
from pathlib import Path
import pickle as pkl
import scipy.sparse as sparse
import torch
import sys
import pandas as pd
path = "/".join(sys.path[0].split("/")[:-2])
sys.path.append(path)
import gcn
path = "/".join(sys.path[0].split("/")[:-3])+"/script"
sys.path.append(path)
from script.joern import get_node_edges,rdg,drop_lone_nodes
from vocab import Vocab,gen_tok_pattern,extract_features
from transformers import AutoTokenizer
from utils import cvss_task_type
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
code_token_pattern = gen_tok_pattern()
vectorizer = extract_features(start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern)
tokenizer = vectorizer.build_analyzer()
from fasttext_utils import getBiGram,gettriGram

class GraphDataset_Token_MultiTask(Dataset):

    def __init__(self,  context_type: str, token_type : str = 'codebert' , partition="train"):
        """Init."""

        datafile = gcn.data_dir() / "mydata_split.csv"
        self.df = pd.read_csv(datafile)
        self.df.set_index('id', drop=False, inplace=True)
        self.df = self.df[self.df["partition"] == partition]

        train_data = self.df[self.df["partition"] == "train"]
        func_before = train_data["func_before"].tolist()
        train_data = "\n".join(func_before)
        self.codebert_tokenizer = None
        self.vocab = None
        assert token_type in ["codebert","unixcoder","textcnn","lstm","fasttext"]
        self.token_type = token_type
        if self.token_type == "codebert":
            self.codebert_tokenizer = AutoTokenizer.from_pretrained(f"microsoft/codebert-base")
        elif self.token_type == "unixcoder":
            self.codebert_tokenizer = AutoTokenizer.from_pretrained(f"microsoft/unixcoder-base-nine")
        elif self.token_type in ["textcnn","lstm","fasttext"]:
            self.vocab = Vocab(train_data, tokenizer, min_freq=0, max_vocab=10000, max_len=128)


        self.cvss_ans = { }
        for task_type,cvss_dict in cvss_task_type.items():
            cvss_local = [cvss_dict[i] for i in self.df[task_type].to_list()]
            self.cvss_ans[task_type] = cvss_local
        assert context_type in ['DD_context_codebert_tokens', 'sentence_codebert_tokens', 'CD_context_codebert_tokens',
                                'nature_context__codebert_tokens',\
                                "sentence_feature","data_feature","control_feature","nature_feature"
                                    ]
        self.context_type = context_type
        self.idx2id = self.df.id.to_list()

    def _feat_ext_itempath(self, _id):
        """Run feature extraction with itempath."""
        filepath = gcn.graphdata_dir() / "before" / Path(str(_id) + ".c")
        try:
            feature_extraction(filepath, self.codebert_tokenizer ,self.vocab, self.token_type)
        except:

            print("error: ", _id)
            pass

    def cache_features(self):
        """Save features to disk as cache."""
        gcn.dfmp(
            self.df,
            self._feat_ext_itempath,
            "id",
            ordr=False,
            workers=8,
            desc="Cache features: ",
        )

    def __len__(self):
        """Get length of dataset."""
        return len(self.df)

    def __getitem__(self, idx):

        """Override getitem."""
        filepath = gcn.graphdata_dir() / "before" / Path(str(self.idx2id[idx]) + ".c")
        # print(self.idx2id[idx])
        n, e = feature_extraction(filepath, self.codebert_tokenizer,self.vocab,self.token_type)

        g = dgl.graph(e)
        g = dgl.add_reverse_edges(g)  # convert to undirected graph
        # we can choose sentence_codebert_tokens,DD_context_codebert_tokens,CD_context_codebert_tokens
        # nature_context__codebert_tokens
        if self.token_type in ['fasttext']:
            g.ndata["tokens_1gram"] = torch.Tensor(
                np.array(n[f"{self.context_type}_1gram"].tolist())).int()
            g.ndata["tokens_2gram"] = torch.Tensor(
                np.array(n[f"{self.context_type}_2gram"].tolist())).int()
            g.ndata["tokens_3gram"] = torch.Tensor(
                np.array(n[f"{self.context_type}_3gram"].tolist())).int()
        else:
            g.ndata["tokens"] = torch.Tensor(
                np.array(n[self.context_type].tolist())).int()
        g = dgl.add_self_loop(g)
        cvss_item = { }
        for task_type , ans in self.cvss_ans.items():
            cvss_item[task_type] = ans[idx] # { 'cvss2_C' : 0 , 'cvss2_I' : 1 , 'cvss2_A' : 0 }
        return g, cvss_item

    def __len__(self):
        return self.df.shape[0]

def codebert_tokens(codebert_tokenizer,content,max_length):
    tokens_id = codebert_tokenizer(content, max_length=max_length, padding='max_length', truncation=True, add_special_tokens=True)["input_ids"]
    return tokens_id
def feature_extraction(filepath,codebert_tokenizer:RobertaTokenizerFast,vocab:Vocab,token_type:str):
    """Extract relevant components of IVDetect Code Representation.

       DEBUGGING:
       filepath = "/home/xx/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/180189.c"
       filepath = "/home/xx/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/182480.c"

       PRINTING:
       svdj.plot_graph_node_edge_df(nodes, svdj.rdg(edges, "ast"), [24], 0)
       svdj.plot_graph_node_edge_df(nodes, svdj.rdg(edges, "reftype"))
       pd.options.display.max_colwidth = 500
       print(subseq.to_markdown(mode="github", index=0))
       print(nametypes.to_markdown(mode="github", index=0))
       print(uedge.to_markdown(mode="github", index=0))

       4/5 COMPARISON:
       Theirs: 31, 22, 13, 10, 6, 29, 25, 23
       Ours  : 40, 30, 19, 14, 7, 38, 33, 31
       Pred  : 40,   , 19, 14, 7, 38, 33, 31
       """
    if token_type in ["codebert", "unixcoder"]:
        outdir = Path(filepath).parent.parent / f"graph_embedding_{token_type}"
    elif token_type in ["textcnn", "lstm"]:
        outdir = Path(filepath).parent.parent / f"graph_embedding_vocab_token"
    elif token_type in ['fasttext']:
        outdir = Path(filepath).parent.parent / f"graph_embedding_fasttext"


    gcn.get_dir(outdir)
    outfile = outdir / Path(filepath).name.split(".")[0]

    cachefp = str(outfile) + ".pkl"

    try:
        with open(cachefp, "rb") as f:
            pdg_nodes, pdg_edges = pkl.load(f)
        return pdg_nodes, pdg_edges
    except:
        pass
    nodes, edges = get_node_edges(filepath)
    with open(filepath, "r") as f:
        code = f.readlines()
    code = {i + 1: code[i].strip() for i in range(len(code))}  # linenumber index start from 1

    nodesline = nodes[nodes.lineNumber != ""].copy()
    nodesline.lineNumber = nodesline.lineNumber.astype(int)
    nodesline = (
        nodesline.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
        .groupby("lineNumber")
        .head(1)
    )
    edgesline = edges.copy()
    edgesline.innode = edgesline.line_in
    edgesline.outnode = edgesline.line_out
    nodesline.id = nodesline.lineNumber
    edgesline = rdg(edgesline, "pdg")
    nodesline = drop_lone_nodes(nodesline, edgesline)
    # Drop duplicate edges
    edgesline = edgesline.drop_duplicates(subset=["innode", "outnode", "etype"])
    # REACHING DEF to DDG
    edgesline["etype"] = edgesline[["etype"]].apply(
        lambda x: "DDG" if x.etype == "REACHING_DEF" else x.etype, axis=1
    )
    edgesline = edgesline[edgesline.innode.apply(lambda x: isinstance(x, float))]
    edgesline = edgesline[edgesline.outnode.apply(lambda x: isinstance(x, float))]
    # get uedge to get the DDG and CDG context
    edgesline_reverse = edgesline[["innode", "outnode", "etype"]].copy()
    edgesline_reverse.columns = ["outnode", "innode", "etype"]
    uedge = pd.concat([edgesline, edgesline_reverse])
    uedge = uedge[uedge.innode != uedge.outnode]
    uedge = uedge.groupby(["innode", "etype"]).agg({"outnode": set})
    uedge = uedge.reset_index()
    if len(uedge) > 0:
        uedge = uedge.pivot("innode", "etype", "outnode")
        if "DDG" not in uedge.columns:
            uedge["DDG"] = None
        if "CDG" not in uedge.columns:
            uedge["CDG"] = None
        uedge = uedge.reset_index()[["innode", "CDG", "DDG"]]
        uedge.columns = ["lineNumber", "control", "data"]

        uedge.control = uedge[["lineNumber", "control"]].apply(
            lambda r: sorted(list(r.control) + [r.lineNumber]) if isinstance(r.control, set) else [r.lineNumber], axis=1
        )
        uedge.data = uedge[["lineNumber", "data"]].apply(
            lambda r: sorted(list(r.data) + [r.lineNumber]) if isinstance(r.data, set) else [r.lineNumber], axis=1
        )
        data = uedge.set_index("lineNumber").to_dict()["data"]
        control = uedge.set_index("lineNumber").to_dict()["control"]

        for i in nodesline.id.tolist():
            if i not in data.keys():
                data[i] = [i]
            if i not in control.keys():
                control[i] = [i]
    else:
        data = {i: [i] for i in nodesline.id.tolist()}
        control = {i: [i] for i in nodesline.id.tolist()}
        # print(data)
        # print(control)

    pdg_nodes = nodesline.copy()
    pdg_nodes = pdg_nodes[["id"]].sort_values("id")
    pdg_nodes["sentence"] = pdg_nodes.id.map(code).fillna("")
    pdg_nodes["data"] = pdg_nodes.id.map(data)
    pdg_nodes["control"] = pdg_nodes.id.map(control)
    # print(pdg_nodes[["id","data"]])
    # print(pdg_nodes.data.tolist())
    pdg_nodes.data = pdg_nodes.data.map(lambda x: "\n".join(code[i] for i in x))
    pdg_nodes.control = pdg_nodes.control.map(lambda x: "\n".join(code[i] for i in x))

    pdg_nodes["nature_context"] = pdg_nodes.id.map(
        lambda x: "\n".join(code[i] for i in [x - 1, x, x + 1] if i in code.keys()))
    pdg_edges = edgesline.copy()
    pdg_nodes = pdg_nodes.reset_index(drop=True).reset_index()
    pdg_dict = pd.Series(pdg_nodes.index.values, index=pdg_nodes.id).to_dict()
    pdg_edges.innode = pdg_edges.innode.map(pdg_dict)
    pdg_edges.outnode = pdg_edges.outnode.map(pdg_dict)
    pdg_edges = pdg_edges.dropna()
    pdg_edges = (pdg_edges.outnode.tolist(), pdg_edges.innode.tolist())

    if token_type in ["codebert", "unixcoder"]:
        pdg_nodes["sentence_codebert_tokens"] = pdg_nodes[["sentence"]].apply(
            lambda r: codebert_tokens(codebert_tokenizer, r.sentence, max_length=128)
            , axis=1)
        pdg_nodes["DD_context_codebert_tokens"] = pdg_nodes[["data"]].apply(
            lambda r: codebert_tokens(codebert_tokenizer, r.data, max_length=128)
            , axis=1)
        pdg_nodes["CD_context_codebert_tokens"] = pdg_nodes[["control"]].apply(
            lambda r: codebert_tokens(codebert_tokenizer, r.control, max_length=128)
            , axis=1)
        pdg_nodes["nature_context__codebert_tokens"] = pdg_nodes[["nature_context"]].apply(
            lambda r: codebert_tokens(codebert_tokenizer, r.nature_context, max_length=128)
            , axis=1)


    elif token_type in ["textcnn", "lstm"]:
        pdg_nodes["sentence_feature"] = pdg_nodes[["sentence"]].apply(
            lambda r: vocab.to_idxs(r.sentence)
            , axis=1)

        pdg_nodes["data_feature"] = pdg_nodes[["data"]].apply(
            lambda r: vocab.to_idxs(r.data)
            , axis=1)

        pdg_nodes["control_feature"] = pdg_nodes[["control"]].apply(
            lambda r: vocab.to_idxs(r.control)
            , axis=1)

        pdg_nodes["nature_feature"] = pdg_nodes[["nature_context"]].apply(
            lambda r: vocab.to_idxs(r.nature_context)
            , axis=1)
    elif token_type in ['fasttext']:
        pdg_nodes["sentence_feature_1gram"] = pdg_nodes[["sentence"]].apply(
            lambda r: vocab.to_idxs(r.sentence)
            , axis=1)
        pdg_nodes["sentence_feature_2gram"] = pdg_nodes[["sentence"]].apply(
            lambda r: getBiGram(vocab.to_idxs(r.sentence))
            , axis=1)
        pdg_nodes["sentence_feature_3gram"] = pdg_nodes[["sentence"]].apply(
            lambda r: gettriGram(vocab.to_idxs(r.sentence))
            , axis=1)

        pdg_nodes["data_feature_1gram"] = pdg_nodes[["data"]].apply(
            lambda r: vocab.to_idxs(r.data)
            , axis=1)
        pdg_nodes["data_feature_2gram"] = pdg_nodes[["data"]].apply(
            lambda r: getBiGram(vocab.to_idxs(r.data))
            , axis=1)
        pdg_nodes["data_feature_3gram"] = pdg_nodes[["data"]].apply(
            lambda r: gettriGram(vocab.to_idxs(r.data))
            , axis=1)

        pdg_nodes["control_feature_1gram"] = pdg_nodes[["control"]].apply(
            lambda r: vocab.to_idxs(r.control)
            , axis=1)
        pdg_nodes["control_feature_2gram"] = pdg_nodes[["control"]].apply(
            lambda r: getBiGram(vocab.to_idxs(r.control))
            , axis=1)
        pdg_nodes["control_feature_3gram"] = pdg_nodes[["control"]].apply(
            lambda r: gettriGram(vocab.to_idxs(r.control))
            , axis=1)

        pdg_nodes["nature_feature_1gram"] = pdg_nodes[["nature_context"]].apply(
            lambda r: vocab.to_idxs(r.nature_context)
            , axis=1)
        pdg_nodes["nature_feature_2gram"] = pdg_nodes[["nature_context"]].apply(
            lambda r: getBiGram(vocab.to_idxs(r.nature_context))
            , axis=1)
        pdg_nodes["nature_feature_3gram"] = pdg_nodes[["nature_context"]].apply(
            lambda r: gettriGram(vocab.to_idxs(r.nature_context))
            , axis=1)

    # Cache
    with open(cachefp, "wb") as f:
        pkl.dump([pdg_nodes, pdg_edges], f)
    return pdg_nodes, pdg_edges