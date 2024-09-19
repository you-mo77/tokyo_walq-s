import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import japanize_matplotlib
import PySimpleGUI as sg
import bar_plot

# gui作成前段階
#def init_gui():

# K-means 
def k_means(df:pd.DataFrame, score:pd.DataFrame, pc1:str, pc2:str, n:int):
    # 街路番号抽出(-> リスト)
    load_num = df.index
    load_num = load_num.tolist()

    kmeans = KMeans(n_clusters=n, max_iter=30, init="random")

    plt.scatter(score[pc1], score[pc2])
    plt.show()

# メイン関数
def main():
    df = bar_plot.edit_data()

    # 第n主成分まで生成
    n = 8
    (score, eigenvalue, ratio, eigen, v_name) = bar_plot.pca(df, n)

    k_means(df, score, "PC1", "PC2", 4)


    

# 実行部分
if __name__ == "__main__":
    main()
