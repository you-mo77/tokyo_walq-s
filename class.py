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

# gui(どの主成分を使うかを2つのプルダウンメニューとかで指定　クラスタ数も指定 -> GO!)
#def init_gui():

# K-means 
def k_means(df:pd.DataFrame, score:pd.DataFrame, pc1:str, pc2:str, n:int):
    # 街路番号抽出してscoreのインデックスへ
    load_num = df.index
    #load_num = load_num.tolist()
    score.index = load_num

    # Kmeansインスタンス作成
    kmeans = KMeans(n_clusters=n, max_iter=30)
    
    # モデルを学習 -> クラスタリング
    class_label = kmeans.fit_predict(score[[pc1, pc2]])

    # クラスcolumns作成
    score["class"] = class_label

    #print(score)

    # グラフ作成
    plt.xlabel(pc1)
    plt.ylabel(pc2)
    plt.title("the relationship between " + pc1 + " and "+ pc2)

    for i in np.sort(score["class"].unique()):
        clasted = score[score["class"]==i]
        plt.scatter(clasted[pc1], clasted[pc2])
        #print(clasted)
        for j in clasted.index:
            plt.text(clasted.loc[j, pc1], clasted.loc[j, pc2], j)
        

    # グラフ表示
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
