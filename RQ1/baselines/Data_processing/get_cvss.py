
import requests
import json
import re
from bs4 import BeautifulSoup
import pandas as pd

def get_cve_id():
    
    df = pd.read_csv("MSR_data_cleaned.csv")
    # print(df.columns)
    # df.to_csv("test.csv",index=None)
    # df = df[df["vul"]==1]
    # print(df["CVE ID"])
    df = df[['Unnamed: 0','CVE ID']]

    df.to_csv("id_CVE.csv",index=False)
    
def get_Complexity_id():
    
    df = pd.read_csv("MSR_data_cleaned.csv")
    # print(df.columns)
    # df.to_csv("test.csv",index=None)
    # df = df[df["vul"]==1]
    # print(df["CVE ID"])
    df = df[['Unnamed: 0','Complexity']]

    df.to_csv("id_Complexity.csv",index=False)
            
def getHTMLText(url):


    try:
        res = requests.get(url)
        res.raise_for_status()
        res.encoding = 'utf-8'
        return res.text
    except:
        return 'error'

def getkey(newurl):
    tag = []
    res = getHTMLText(newurl)
    print(res)
    soup = BeautifulSoup(res, 'html.parser')
    score = soup.select("#Cvss2CalculatorAnchor")
    vector = soup.select('.tooltipCvss2NistMetrics')
    # print(res)
    # print(tag_all[0].text)
    return score[0].text,vector[0].text
def getCVSS():
    df = pd.DataFrame()
    url = "https://nvd.nist.gov/vuln/detail/"
    with open("cveid.txt",'r',encoding="utf-8") as f:
        content = f.readlines()
    cveids = [cve.strip() for cve in content]
    scores = []
    vectors = []
    count = 0
    for cveid in cveids:
        try:
            s,v = getkey(url+cveid)
            scores.append(s)
            vectors.append(v)
        except:
            print(count)
            count+=1
            scores.append("NaN")
            vectors.append("NaN")
    df["cve id"] = pd.Series(cveids)
    df["score"] = pd.Series(scores)
    df["vector"] = pd.Series(vectors)
    df.to_csv("cvss.cvs")
        
        
    # score,vector = getkey(url)
# getkey("https://nvd.nist.gov/vuln/detail/CVE-2022-32202")


get_Complexity_id()
getCVSS()
    
















