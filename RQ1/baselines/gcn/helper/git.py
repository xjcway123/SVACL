import os
from unidiff import PatchSet
import pickle as pkl
import uuid
import gcn
import numpy as np
import re

def gitdiff(old: str, new: str):
    """Git diff between two strings."""
    cachedir = gcn.data_dir()
    oldfile = cachedir / uuid.uuid4().hex
    newfile = cachedir / uuid.uuid4().hex
    with open(oldfile, "w") as f:
        f.write(old)
    with open(newfile, "w") as f:
        f.write(new)
    cmd = " ".join(
        [
            "git",
            "diff",
            "--no-index",
            "--no-prefix",
            f"-U{len(old.splitlines()) + len(new.splitlines())}",
            str(oldfile),
            str(newfile),
        ]
    )
    process = gcn.subprocess_cmd(cmd)
    os.remove(oldfile)
    os.remove(newfile)
    return process[0].decode()


def md_lines(patch: str):
    r"""get delete line nums,namely vul lines(index start from 0 for old)
        and add lines nums(index start from 0 for new)

    old = "bool asn1_write_GeneralString(struct asn1_data *data, const char *s)\n\
    {\n\
       asn1_push_tag(data, ASN1_GENERAL_STRING);\n\
       asn1_write_LDAPString(data, s);\n\
       asn1_pop_tag(data);\n\
       return !data->has_error;\n\
    }\n\
    \n\
    "

    new = "bool asn1_write_GeneralString(struct asn1_data *data, const char *s)\n\
    {\n\
        if (!asn1_push_tag(data, ASN1_GENERAL_STRING)) return false;\n\
        if (!asn1_write_LDAPString(data, s)) return false;\n\
        return asn1_pop_tag(data);\n\
    }\n\
    \n\
    int test() {\n\
        return 1;\n\
    }\n\
    "

    patch = gitdiff(old, new)
    """
    parsed_patch = PatchSet(patch)
    ret = {"delete": [],"add":[], "diff": ""}
    if len(parsed_patch) == 0:
        return ret
    parsed_file = parsed_patch[0]
    hunks = list(parsed_file)
    assert len(hunks) == 1
    hunk = hunks[0]
    ret["diff"] = str(hunk).split("\n", 1)[1]
    add_count = 0
    del_count = 0
    for idx, ad in enumerate([i for i in ret["diff"].splitlines()], start=1):
        if len(ad) > 0:
            ad = ad[0]
            if ad == "+":
                add_count += 1
                ret["add"].append(idx-del_count)
            elif ad == "-":
                del_count += 1
                ret["delete"].append(idx-add_count)
            elif ad == "\\" :
                add_count += 1
                del_count += 1
    return ret


def code2diff(old: str, new: str):
    """Get added and removed lines from old and new string."""
    patch = gitdiff(old, new)
    return md_lines(patch)
def c2dhelper_del(item):
    """get the delete line nums"""
    savedir = gcn.get_dir(gcn.data_dir() / "gitdiff")
    savepath = savedir / f"{item['id']}.git.pkl"
    if item["func_before"] == item["func_after"]:
        return
    try:
        with open(savepath, "rb") as f:
            ret = pkl.load( f)
    except:
        ret = code2diff(item["func_before"], item["func_after"])
        ret["delete"] = [str(i) for i in ret["delete"]]
        # ret["vul"] = np.asarray(ret["vul"])
        with open(savepath, "wb") as f:
            pkl.dump(ret, f)
    return " ".join(ret["delete"])

def c2dhelper_add(item):
    """get the add line nums"""
    savedir = gcn.get_dir(gcn.data_dir() / "gitdiff")
    savepath = savedir / f"{item['id']}.git.pkl"
    if item["func_before"] == item["func_after"]:
        return
    try:
        with open(savepath, "rb") as f:
            ret = pkl.load( f)
    except:
        ret = code2diff(item["func_before"], item["func_after"])
        ret["add"] = [str(i) for i in ret["add"]]
        with open(savepath, "wb") as f:
            pkl.dump(ret, f)

    ret["add"] = [str(i) for i in ret["add"]]
    return " ".join(ret["add"])


def c2dhelper_del_blaming(item):
    """get the delete line nums"""
    savedir = gcn.get_dir(gcn.data_dir() / "blaming_gitdiff")
    savepath = savedir / f"{item['id']}.git.pkl"
    if item["blaming_func_before"] == item["blaming_func_after"]:
        return
    try:
        with open(savepath, "rb") as f:
            ret = pkl.load( f)
    except:
        ret = code2diff(item["blaming_func_before"], item["blaming_func_after"])
        ret["delete"] = [str(i) for i in ret["delete"]]
        # ret["vul"] = np.asarray(ret["vul"])
        with open(savepath, "wb") as f:
            pkl.dump(ret, f)
    return " ".join(ret["delete"])

def c2dhelper_add_blaming(item):
    """get the add line nums"""
    savedir = gcn.get_dir(gcn.data_dir() / "blaming_gitdiff")
    savepath = savedir / f"{item['id']}.git.pkl"
    if item["blaming_func_before"] == item["blaming_func_after"]:
        return
    try:
        with open(savepath, "rb") as f:
            ret = pkl.load( f)
    except:
        ret = code2diff(item["blaming_func_before"], item["blaming_func_after"])
        ret["add"] = [str(i) for i in ret["add"]]
        with open(savepath, "wb") as f:
            pkl.dump(ret, f)

    ret["add"] = [str(i) for i in ret["add"]]
    return " ".join(ret["add"])
def c2dhelper_diff(item):
    """get the diff """
    savedir = gcn.get_dir(gcn.data_dir() / "gitdiff")
    savepath = savedir / f"{item['id']}.git.pkl"
    if item["func_before"] == item["func_after"]:
        return

    with open(savepath, "rb") as f:
        ret = pkl.load( f)
    s = "\n".join([i for i in ret["diff"].splitlines() if i[0]!="\\"])
    return s

def CES(code:str,diff_line:str):
    try:
        diff_line = [int(i) for i in diff_line.split()]
    except:
        diff_line = []
    lines = code.splitlines()
    right = 0
    context = []
    idx = -1
    for diff in diff_line:
        
        diff = diff-1# line number start with 1 , index start with 0.
        # if diff>=len(lines) or diff<0:
        #     continue
        right = 0
        idx = -1
        for i in range(diff,-1,-1):
            line = lines[i]
            line = re.sub('``.*``','<STR>',line)
            line = re.sub("'.*'",'<STR>',line)
            line = re.sub('".*"','<STR>',line)
            line = re.sub('{.*}','<STR>',line)
            left_bracket = line.find('{')
            if(left_bracket!=-1):
                right -= 1
                if right < 0:
                    context.append(i)
                    idx = i
                    break
            right_bracket = line.find('}')
            if(right_bracket!=-1):
                right+=1
                
        begin_line = lines[idx]
        begin_line = re.sub('``.*``','<STR>',begin_line)
        begin_line = re.sub("'.*'",'<STR>',begin_line)
        begin_line = re.sub('".*"','<STR>',begin_line)
        begin_line = re.sub('{.*}','<STR>',begin_line)
        if idx>=0 and begin_line.find("}")==-1:
            for i in range(idx-1,-1,-1):
                line = lines[i]
                line = re.sub('``.*``','<STR>',line)
                line = re.sub("'.*'",'<STR>',line)
                line = re.sub('".*"','<STR>',line)
                if line.find(';')==-1 and line.find('}')==-1 and line.find('{')==-1:
                    context.append(i)
                else:
                    break
                
        left = 0
        for i in range(diff,len(lines)):
            line = lines[i]
            line = re.sub('``.*``','<STR>',line)
            line = re.sub("'.*'",'<STR>',line)
            line = re.sub('".*"','<STR>',line)
            line = re.sub('{.*}','<STR>',line)
            
            right_bracket = line.find('}')
            if(right_bracket!=-1):
                left_bracket = line[:right_bracket].find('{')
                if(left_bracket!=-1):
                    left+=1
                left-=1
                if left<0:
                    context.append(i)
                    break
                
                left_bracket = line[right_bracket:].find('{')
                if(left_bracket!=-1):
                    left+=1
            else:
                left_bracket = line.find('{')
                if(left_bracket!=-1):
                    left+=1
    context = list(set(context))
    context.sort()
    
    return "\n".join([lines[i] for i in context])

