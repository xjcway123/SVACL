import pandas as pd


def save_php_filename():
    filename = "php_file.csv"
    file_df = pd.read_csv(filename)
    print(file_df)
    file_list = file_df["file"]
    
    php_file = "need_blaming_new_php.csv"
    df = pd.read_csv(php_file)
    df["file_name"] = file_list
    print(df.count())
    df.to_csv(php_file,index = False)
    
    
save_php_filename()
    