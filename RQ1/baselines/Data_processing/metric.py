import pandas as pd

cvss = pd.read_csv("cvss.csv")


SCORE = dict()
LABEL = dict()
AV = dict()
AC = dict()
AU = dict()
C = dict()
I = dict()
A = dict()

for index,row in cvss.iterrows():
    # print(type(row["vector"]))
    # break
    if(type(row["vector"])!=type(" ")):
        continue
    s,l = row["score"].split()
    s = float(s)
    SCORE[row["cve id"]] = s
    LABEL[row["cve id"]] = l
    
    six_metric = [i[-1] for i in row['vector'][1:-1].split("/")]
    AV[row["cve id"]] = six_metric[0]
    AC[row["cve id"]] = six_metric[1]
    AU[row["cve id"]] = six_metric[2]
    C[row["cve id"]] = six_metric[3]
    I[row["cve id"]] = six_metric[4]
    A[row["cve id"]] = six_metric[5]
    
df = pd.read_csv("id_CVE.csv")
score = []
label = []
av = []
ac = []
au = []
c = []
i = []
a = []



for index,row in df.iterrows():
    if row["CVE ID"] in SCORE.keys():
        score.append(SCORE[row["CVE ID"]])
        label.append(LABEL[row["CVE ID"]])
        av.append(AV[row["CVE ID"]])
        ac.append(AC[row["CVE ID"]])
        au.append(AU[row["CVE ID"]])
        c.append(C[row["CVE ID"]])
        i.append(I[row["CVE ID"]])
        a.append(A[row["CVE ID"]])
    else:
        score.append(-1)
        label.append(-1)
        av.append(-1)
        ac.append(-1)
        au.append(-1)
        c.append(-1)
        i.append(-1)
        a.append(-1)
        
df["SCORE"] = pd.Series(score)
df["label"] = pd.Series(label)
df["AV"] = pd.Series(av)
df["AC"] = pd.Series(ac)
df["AU"] = pd.Series(au)
df["C"] = pd.Series(c)
df["I"] = pd.Series(i)
df["A"] = pd.Series(a)

df.to_csv("metric.csv")



















