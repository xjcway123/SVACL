from pydriller import Git
import pandas as pd
import re




project_name = ["libpng","native","libvpx","av","libmpeg2","bt","core","v8","tremolo","libhevc","bluedroid",\
                "minikin","libavc","media","libxaac","sonivox","base","ril","wifi","common","audio","ex",\
                    "dalvik","Bluetooth","wlan","libxml2","recovery","dhcpcd","netd","dragon","libopus","skia"]

    

def preprocess():
    #get_android_project
    filename = "data/need_blaming_new_Android.csv"
    data = pd.read_csv(filename)
    print(data.count())
    def extract_name(s):
        pro_name = s[:s.index('+')-1].split("/")[-1]
        if pro_name == "platform%2Fsystem%2Fcore":
            pro_name = "core"
        elif pro_name == "kernel%2Fcommon":
            pro_name = "common"
        elif pro_name == "platform%2Fframeworks%2Fav":
            pro_name = "av"
        return pro_name
    data["android_project"] = data[["codeLink"]].apply(
        lambda r:extract_name(r.codeLink),axis = 1)
    #Separate the dataset by project for later operation

    for cur_project in project_name:
        cur_data = data[data["android_project"]==cur_project]
        cur_data.to_csv(("data/Android/need_blaming_new_"+cur_project+".csv"),index=False)

def cache_blaming(index):

    index = index  #project index
    
    #read project file, including commit hash and the file name in which  changed function is
    filename = f"data/Android/need_blaming_new_{project_name[index]}.csv"
    data = pd.read_csv(filename)
    print(data.count())
    commit_id_list = data["commit_id"].tolist()
    file_name_list = data["file_name"].tolist()
    
    #read project
    project_path = 'XX'+project_name[index]
    gr = Git(project_path)

    blaming_commit = []
    
    for cur_commit,cur_file_name in zip(commit_id_list,file_name_list):
        try:#to avoid we can't find the commit in the repo
            print("cur_commit:",cur_commit)
            commit = gr.get_commit(cur_commit)
            # we just need function name, however the file_name in big_val is a path
            cur_file_name = cur_file_name[cur_file_name.rfind('/')+1:]
            # t represent a file, if we can't find it in the commit, it will is None
            # when we use gr.get_commits_last_modified_lines()，if modification is None，it will traversal all the files
            t = None
            for m in commit.modified_files:
                # print(m.filename)
                if m.filename==cur_file_name:
                    t = m
            if t is None:
                print("no_file")
                # blaming_commit.append("no_file")
            #when we use pydriller to track back to buggy commit,it will return a dict,key is the file name，value is buggy commit
            buggy_commits = gr.get_commits_last_modified_lines(commit = commit,modification=t)
            # break
            tmp_commit_list = []
            for i in buggy_commits.values():
                tmp_commit_list.extend(list(i))
            tmp_commit_list = list(set(tmp_commit_list))
            # print(tmp_commit_list)
            # print("--------------------")
            blaming_commit.append(",".join(tmp_commit_list))
            
        except:
            print("error:",cur_commit)
            blaming_commit.append("error")

    data["blaming"] = pd.Series(blaming_commit)
    print(data.count())
    filename = "data/Android/"+project_name[index]+".csv"
    data.to_csv(filename,index=False)
    

def blaming(index):
    index = index
    filename = f"data/Android/{project_name[index]}.csv"
    project_path = f'XXX/{project_name[index]}'
    data = pd.read_csv(filename)
    print(data.count())
    commit_id_list = data["commit_id"].tolist()
    file_name_list = data["file_name"].tolist()
    func_before = data["func_before"].tolist()
    blaming_commit = data["blaming"].tolist()
    error = 0
    for i in range(len(blaming_commit)):
        if  blaming_commit[i]!="error" and blaming_commit[i]!="no_file":
            try:
                blaming_commit[i] = blaming_commit[i].split(',')
            except:
                blaming_commit[i] = []
        else:
            error+=1
            blaming_commit[i] = []
    print(error)
    # blaming_commit = [i.split(",") for i in blaming_commit if i!="error" else [] ]
    func_name_list = []
    for cur_func in func_before:
        #get the function name
        func_name = cur_func[:cur_func.find('(')].strip()
        func_name = func_name.split()[-1]
        func_name = func_name[re.search("[_a-zA-Z]",func_name).start():]#remove*& in the begin of string
        #add the  formal parameter
        func_name_and_par = func_name +cur_func[cur_func.find('('):cur_func.find(')')+1]
        func_name_list.append(func_name_and_par)
    gr = Git(project_path)
    count = 0
    cur = 0
    
    filter_commit = []
    blaming_func_before = []
    blaming_func_after = []
    for cur_commit,cur_file_name,cur_func_name,cur_blaming_commit in zip(commit_id_list,file_name_list,func_name_list,blaming_commit):
        print("----------------")
        print("cur_commit:",cur_commit)

        cur_file_name = cur_file_name[cur_file_name.rfind('/')+1:]
        print(cur_file_name)
        print(cur_blaming_commit)
        # print(cur_func_name)
        has_find = False#whether to find the buggy function of the function
        method_code_before = ""
        method_code_after = ""
        find_commit = ""
        
        
        #Matches by function names which take formal parameter

        tmp_func_name = "".join(cur_func_name.split())#to avoid the effect of space and carriage return
        print(tmp_func_name)
        for cur_buggy_commit in cur_blaming_commit:
            buggy_commit = gr.get_commit(cur_buggy_commit)
            for m in buggy_commit.modified_files:
                if m.filename == cur_file_name:#function will only be made in the file it is in
                    for method in m.changed_methods:
                        # to avoid the effect of space and carriage return
                        method_name = "".join(method.long_name.split())
                        if tmp_func_name == method_name:
                            has_find = True
                            find_commit = cur_buggy_commit
                            #we have founded the function, we get the whole function by commit code.
                            for m_before in m.methods_before:
                                if m_before.long_name == method.long_name:
                                    start = m_before.start_line
                                    end = m_before.end_line
                                    method_code_before = '\n'.join(m.source_code_before.splitlines()[start - 1: end])
                                    break#we have founded the whole before function,so we break
                            for m_after in m.methods:
                                if m_after.long_name == method.long_name:
                                    start = m_after.start_line
                                    end = m_after.end_line
                                    method_code_after = '\n'.join(m.source_code.splitlines()[start - 1: end])
                                    break#we have founded the whole after function,so we break
                            break
                        if has_find:
                            print("success")
                            break#we had founded the buggy_function of the function,so we break
                    break#function will only be made in the file it is in，so we break
            if has_find:# we had find the buggy function,so break
                break
        #if we can't find buggy function, we matches by function names which not take formal parameter
        if not has_find:
            
            tmp_func_name = cur_func_name[:cur_func_name.find('(')]
            print(tmp_func_name)
            for cur_buggy_commit in cur_blaming_commit:
                buggy_commit = gr.get_commit(cur_buggy_commit)
                for m in buggy_commit.modified_files:
                    if m.filename == cur_file_name:#changed function will only be made in the file it is in
                        for method in m.changed_methods:
                            method_name = method.name
                            if tmp_func_name in method_name:#has found it
                                has_find = True
                                find_commit = cur_buggy_commit
                                for m_before in m.methods_before:
                                    if m_before.long_name == method.long_name:
                                        start = m_before.start_line
                                        end = m_before.end_line
                                        method_code_before = '\n'.join(m.source_code_before.splitlines()[start - 1: end])
                                        break#we have found the whole before function,so we break
                                for m_after in m.methods:
                                    if m_after.long_name == method.long_name:
                                        start = m_after.start_line
                                        end = m_after.end_line
                                        method_code_after = '\n'.join(m.source_code.splitlines()[start - 1: end])
                                        break#we have found the whole after function,so we break
                                break 
                        break#function will only be made in the file it is in，so we break
                if has_find:# we had find the buggy function,so break
                    break
        blaming_func_before.append(method_code_before)
        blaming_func_after.append(method_code_after)
        filter_commit.append(find_commit)
        if has_find:
            count+=1

        cur+=1
        print(f"count{count} cur{cur}")
    print(count)

    
    
    data["blaming_commit"] = pd.Series(filter_commit)
    data["blaming_func_before"] = pd.Series(blaming_func_before)
    data["blaming_func_after"] = pd.Series(blaming_func_after)
    print(data.count())
    save_file = "Deepcva_data/Android/"+project_name[index]+".csv"
    data.to_csv(save_file,index=False)
def concat_df():
    df = pd.DataFrame()
    for cur_project in project_name:
        # cur_project = "file"
        save_file = "Deepcva_data/Android/"+cur_project+".csv"
        data = pd.read_csv(save_file)
        # print(data.count())
        data = data[data["blaming"].notnull()]
        data = data[data["blaming_func_after"].notnull()]

        
        # print(data.shape[0])
        
        col_name_list = data.columns.tolist()
        for col_name in col_name_list:
            if col_name[:7] == "Unnamed":
                del data[col_name]
        del data["files_changed"]
        del data["android_project"]
        if df.shape[0]==0:
            df = data
        else:
            df = pd.concat([df,data])
    print(df.count())
    
    
    df.to_csv("result/blaming_result_Android.csv",index = False)
    data = pd.read_csv("blaming_result_except_Android.csv")
    print(data.count())
    df = pd.concat([data,df])
    print(df.count())
    df.to_csv("result/blaming_result.csv",index = False)

    
if __name__ == '__main__':
    # preprocess()
    # for i in range(len(project_name)):
    #     cache_blaming(i)
    #     blaming(i)
    concat_df()