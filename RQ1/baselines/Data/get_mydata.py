import pandas as pd
import random
import os 
from pathlib import Path

import sys
sys.path.append(str((Path(__file__).parent.parent)))
import gcn
import gcn.helper.git as git
from script.joern import get_node_edges,rdg,drop_lone_nodes
from utils import remove_comments,remove_empty_lines,remove_space_after_newline,remove_space_before_newline
from sklearn.model_selection import train_test_split

def data_processing():
    """
    get the changed functions from bigvul
    """
    df = pd.read_csv("MSR_data_cleaned.csv")
    print("the initial number of bigvul:",df.shape[0])
    df.rename(columns={"Unnamed: 0": "id"},inplace=True)
    
    df = df[df["func_after"]!=df["func_before"]]
    print("we need just the function changed")
    print("the number of changed_functions:",df.shape[0])
    """
    get the labels(CVSS)
    """
    # metric.csv file is a mapping from cve id to labels,which is gotten by crawling the https://nvd.nist.gov/vuln/detail/
    print("get cvss label")
    metric = pd.read_csv("metric.csv")
    metric.rename(columns={"Unnamed: 0": "id"},inplace = True)
    df["cvss2_severity"] = df[["id"]].apply(
        lambda r: metric[metric["id"]==r.id].label.values[0], axis=1
    )
    df["cvss2_AV"] = df[["id"]].apply(
        lambda r: metric[metric["id"]==r.id].AV.values[0], axis=1
    )
    df["cvss2_AC"] = df[["id"]].apply(
        lambda r: metric[metric["id"]==r.id].AC.values[0], axis=1
    )
    df["cvss2_AU"] = df[["id"]].apply(
        lambda r: metric[metric["id"]==r.id].AU.values[0], axis=1
    )
    df["cvss2_C"] = df[["id"]].apply(
        lambda r: metric[metric["id"]==r.id].C.values[0], axis=1
    )
    df["cvss2_I"] = df[["id"]].apply(
        lambda r: metric[metric["id"]==r.id].I.values[0], axis=1
    )
    df["cvss2_A"] = df[["id"]].apply(
        lambda r: metric[metric["id"]==r.id].A.values[0], axis=1
    )
    #drop failed crawl
    df = df[df["cvss2_A"]!="-1"]
    print("the number of getting cvss labels successfully:",df.shape[0])
    df.to_csv("diff_data_cleaned_labels.csv",index=False)





def cleaned_code(func_code):
    func_code = remove_empty_lines(func_code)
    func_code = remove_comments(func_code)
    func_code = remove_space_before_newline(func_code)
    func_code = remove_space_after_newline(func_code)

    return func_code
def cleaned_dataset():
    data = pd.read_csv("diff_data_cleaned_labels.csv")
    data = data[data["del_lines"]!=0]#we need the function which has deleted line
    print("we need just the function which has deleted lines")
    print('Data shape:', data.shape)
    print('Data columns:', data.columns)
    print('Cleaning Code...')
    data['func_before'] = data['func_before'].apply(lambda x: cleaned_code(x))
    data['func_after'] = data['func_after'].apply(lambda x: cleaned_code(x))
    data = data[~data['func_before'].duplicated(keep=False)] # need to remove duplicate

    #we drop_large_line_code
    data["func_before"] = data["func_before"].apply(
        lambda x:x if len(x.split("\n")) <= 200 else -1#we need the function whose line number is less than 200
    )
    data = data[data["func_before"]!=-1]
    # print(data.count())


    print('Cleaning Code Done!')


    data = data.reset_index(drop=True).reset_index().rename(columns={'index': '_id'})

    cols = ["func_before", "func_after", "id"]

    data["delete_lines"] = data[cols].apply(
            lambda r:git.c2dhelper_del(r), axis=1
        )

    data["add_lines"] = data[cols].apply(
            lambda r:git.c2dhelper_add(r), axis=1
        )

    keepcol = ["id","CVE ID","CWE ID","file_name","files_changed","codeLink","Publish Date","Score","Summary","Update Date","commit_id","commit_message",\
        "func_after","func_before","delete_lines","add_lines","vul"]
    cvss = ["cvss2_severity","cvss2_AV","cvss2_AC","cvss2_AU","cvss2_C","cvss2_I","cvss2_A"]
    keepcol+=cvss
    data = data[keepcol]
    data.dropna(how='all',subset=["delete_lines"],inplace=True)
    data.to_csv("mydata.csv",index=False)

def drop_no_joern():
    #before run the function, you need run the joern script(script/getgraphs.py)
    #drop no graph 
    datafile = gcn.data_dir() / "mydata.csv"
    data = pd.read_csv(datafile)
    # print(data.count())
    id_list = data["id"].tolist()
    error = []
    for idx in id_list:
        filepath = gcn.graphdata_dir() / "before" / Path(str(idx)+".c")
        try:
            nodes,edges = get_node_edges(filepath)
        except:
            error.append(idx)
            continue
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
        if edgesline.shape[0] == 0 or nodesline.shape[0]==0:
            error.append(idx)

    print(error)
    data.set_index('id',drop=False,inplace=True)
    data.drop(index = error,inplace=True)

    drop_no_graph_datafile = gcn.data_dir() / "mydata_drop_no_graph.csv"
    data.to_csv(drop_no_graph_datafile,index=False)

def merge_blaming():
    #before running the function,you need run blaming to get blaming_result.csv.
    #firstly, merge blaming commit and big vul
    data = pd.read_csv(gcn.data_dir()/"mydata_drop_no_graph.csv")
    blaming_data = pd.read_csv("blaming_result.csv")
    del_col = ["codeLink","commit_id","file_name","func_after","func_before","project","diff","blaming"]
    for col in del_col:
        del blaming_data[col]
    data = pd.merge(data, blaming_data, on=["id"])
    print("the number of merging:",data.shape[0])

    data["blaming_func_before"] = data["blaming_func_before"].fillna("")# change nan to ""
    # we need clean the code
    data['blaming_func_before'] = data['blaming_func_before'].apply(lambda x: cleaned_code(x))
    data['blaming_func_after'] = data['blaming_func_after'].apply(lambda x: cleaned_code(x))

    cols = ["blaming_func_before", "blaming_func_after", "id"]

    data["blaming_delete_lines"] = data[cols].apply(
            lambda r:git.c2dhelper_del_blaming(r), axis=1
        )

    data["blaming_add_lines"] = data[cols].apply(
            lambda r:git.c2dhelper_add_blaming(r), axis=1
        )
    cols = ["blaming_func_before", "blaming_func_after", "id","blaming_delete_lines","blaming_add_lines"]
    
    data["pre_context"] = data[cols].apply(
            lambda r:git.CES(r.blaming_func_before,r.blaming_delete_lines), axis=1
        )

    data["cur_context"] = data[cols].apply(
            lambda r:git.CES(r.blaming_func_after,r.blaming_add_lines), axis=1
        )
    print(data.count())
    data.to_csv("final_data.csv")
    


def split_data():
    #split dataset
    filename = gcn.data_dir()/"final_data.csv"
    df = pd.read_csv(filename)
    print(df.count())
    X_train,X_test = train_test_split(df,test_size=0.2, random_state=0)
    X_test,X_valid= train_test_split(X_test,test_size=0.5, random_state=0)
    
    X_train["partition"] = "train"
    X_test["partition"] = "test"
    X_valid["partition"] = "valid"
    
    df = pd.concat([X_train,X_test,X_valid])
    partition = df["partition"].tolist()

    train = [i for i in partition if i=="train"]
    test = [i for i in partition if i=="test"]
    valid = [i for i in partition if i=="valid"]
    print(len(train)," ",len(test)," ",len(valid))
    datafile = gcn.data_dir() / "mydata_split.csv"
    df.to_csv(datafile,index=False)
    print(df.count())
def test():
    df = pd.read_csv("final_data.csv").head(10)
    df.to_csv("just_valid.csv",index=False)
if __name__ == "__main__":
    # data_processing()
    # cleaned_dataset()
    # Here, you need run script/getgraphs.py and continue.
    # drop_no_joern()
    # merge_blaming()
    split_data()
    # test()
    
    

