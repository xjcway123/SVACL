import os
import sys
from pathlib import Path
import gcn
import my_utils as utils
import pandas as pd
from tqdm import tqdm
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
def preprocess(row):
    """Parallelise  functions.

    """
    savedir_before = gcn.get_dir(gcn.graphdata_dir() / "combined_df_method" / "before")

    # Write C Files
    fpath1 = savedir_before / f"{row['id']}.class"
    with open(fpath1, "w") as f:
        f.write(row["filter_code"])

    utils.full_run_joern(fpath1, verbose=3)
    # Run Joern on "before" code
    if not os.path.exists(f"{fpath1}.edges.json"):
        utils.full_run_joern(fpath1, verbose=3)



    # Run SAST extraction
    # fpath3 = savedir_before / f"{row['method_change_id']}.java.sast.pkl"
    # if not os.path.exists(fpath3):
    #     sast_before = sast.run_sast(row["code"])
    #     with open(fpath3, "wb") as f:
    #         pkl.dump(sast_before, f)
if __name__ == "__main__":
    # print(splits[0].info())
    # print(splits[1].info())
    # print(splits[-1].info())
    # pass
    datafile = gcn.data_dir() / "combined_df_method_partition.parquet"
    df = pd.read_parquet(datafile)
    df['filter_code'] = df[['code']].apply(
	lambda r: Remove_annotation(r.code), axis=1)
    df = df.head(10)
    for index,row in tqdm(df.iterrows()):
        preprocess(row)
    # gcn.dfmp(df, preprocess, ordr=False, workers=8)
