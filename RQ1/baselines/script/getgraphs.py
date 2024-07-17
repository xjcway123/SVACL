import os
import pickle as pkl
import sys

import numpy as np

import joern
import pandas as pd
import sys
path = "/".join(sys.path[0].split("/")[:-1])
sys.path.append(path)
import gcn




def preprocess(row):

    savedir_before = gcn.get_dir(gcn.graphdata_dir() / "before")


    # Write C Files
    fpath1 = savedir_before / f"{row['id']}.c"
    with open(fpath1, "w") as f:
        f.write(row["func_before"])

    # Run Joern on "before" code
    if not os.path.exists(f"{fpath1}.edges.json"):
        joern.full_run_joern(fpath1, verbose=3)




if __name__ == "__main__":



    # Read Data
    data_file = gcn.data_dir() / "mydata.csv"
    data = pd.read_csv(data_file)
    print(data.info())
    print(data.count())
   

    gcn.dfmp(data, preprocess, ordr=False, workers=8)

    
