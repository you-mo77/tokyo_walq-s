import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# 時間文字列をint[秒]に変換
def time_convert_into_int(time_str:str):
    # timedelta型に変換
    delta = datetime.strptime(time_str, "%H:%M:%S") - datetime(1900, 1, 1)

    # 秒数に変換
    time_int = int(delta.total_seconds())

    return time_int

# データ加工
def edit_data():
    # ファイルパス
    data_path = "調査データ.csv"

    # データ読み込み(csv -> DataFrame) なお、１行目は空白なので２行目(属性)をcolumns(列のインデックス)に指定しながら読み込んでいる
    df = pd.read_csv(data_path, header=1)
    #print(df.columns)

    # 街路番号をindex(行のインデックス)に指定
    df.set_index("街路番号", inplace=True)

    # データが"NaN"の行を削除(チェックしていない街路)
    df = df.dropna(subset=["点字ブロックや音の鳴る信号がある"])

    # 削除する列を取得
    s_col = "分類計"
    e_col = "国籍計"
    drop_columns = df.loc[:, s_col:e_col].columns.tolist()
    drop_columns.extend(["調査時間"])

    # 打ち込み確認行、分類計~国籍計列　削除
    df = df.drop(index="打込確認",  columns=drop_columns)

    # 経過時間->int型へ
    df["通行者20人到達時間"] = df["通行者20人到達時間"].apply(time_convert_into_int)

    return df

# 主成分分析
def pca(df:pd.DataFrame):
    # データの標準化 iloc -> dataframeの各行、列を指定するメソッド apply -> 各行列に対して指定関数を使用 axis=0 -> 列操作
    dfs = df.iloc[:, 0:].apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    # ここでstdが0の属性は全値が一致 -> 情報量0 -> 削除してしまって問題ない？
    dfs_cleaned = dfs.dropna(axis=1, how="any")

    # 主成分分析の実行(PCAモデル作成) 第10主成分まで抽出 主成分最大数は行数(サンプル数)らしい　なぜ？
    pca = PCA(n_components=10)

    # データをPCAモデルに適用
    pca.fit(dfs_cleaned)

    # データを主成分空間に写像(要は10主成分に変換) 各主成分がどのような意味を持つかは負荷率から考察するしかない。
    feature = pca.transform(dfs_cleaned)

    # 主成分得点(各主成分軸状での各データの変換後の値)
    pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(10)])

    # 主成分プロット(とりあえず第1, 第2主成分)
    plt.scatter(feature[0:feature.shape[0]-1, 0], feature[0:feature.shape[0]-1, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.show()

# メイン関数
def main():
    df = edit_data()
    pca(df)

# 実行部分
if __name__ == "__main__":
    main()