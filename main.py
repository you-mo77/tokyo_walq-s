import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns

# ファイルパス
data_path = "調査データ.csv"

# データ読み込み(csv -> DataFrame) なお、１行目は空白なので２行目(属性)をcolumns(列のインデックス)に指定しながら読み込んでいる
df = pd.read_csv(data_path, header=1)
#print(df.columns)

# 街路番号をindex(行のインデックス)に指定
df.set_index("街路番号", inplace=True)

# データが"NaN"の行を削除(チェックしていない街路)
df = df.dropna(subset=["3分あたり滞在者"])

# 削除する列を取得
s_col = "分類計"
e_col = "国籍計"
drop_columns = df.loc[:, s_col:e_col].columns

# 打ち込み確認行、分類計~国籍計列　削除
df = df.drop(index="打込確認",  columns=drop_columns)

########### ここまでデータを加工 以降は分析###########
 


