import glob
import os
import sys

from pandas import DataFrame, Series

application_path = ''
# 确定存放目录的相对位置
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(__file__)

# base_dir = os.path.abspath(os.path.join(application_path, os.path.pardir))
base_dir = application_path

data_dir = os.path.join(base_dir, 'data')

files_path=glob.glob(os.path.join(data_dir,"*.txt"))

def line_reader(df):
    for file_path in files_path:
        file = open(file_path, encoding='UTF-8')
        while True:
            data = file.readline()
            if not data:
                break
            # 如果读到空行则读取下一行
            if data == '\n':
                file.readline()

            label = data[0: 2]
            text = data[2: -1]
            new_series = Series([label, text])
            df = df.append(new_series, ignore_index=True)

    return df

if __name__=="__main__":
    df = DataFrame()
    df = line_reader(df)
    df.columns = ["label", "text"]

    input_seq = ''
    target_seq = ''
    new_pair = 0
    train_seq = []
    # 对都进来的excel进行行遍历
    for index, row in df.iterrows():
        if row['label'] == "A:":
            input_seq = row[1]
        else:
            target_seq = row[1]
            new_pair = 1
        if new_pair:
            train_seq.append([input_seq, target_seq])
            new_pair = 0
    df = DataFrame(train_seq)
    df.columns = ['input', 'output']
    # df.to_excel("train.xlsx")