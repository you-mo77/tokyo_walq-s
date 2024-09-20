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
import tempfile

# gui(どの主成分を使うかを2つのプルダウンメニューとかで指定　クラスタ数も指定 -> GO!)
def gui(score:pd.DataFrame, df:pd.DataFrame):
    #コンボボックス用リスト
    pc_name = score.columns.tolist()

    # ウィンドウレイアウト
    layout = [[sg.Text("横軸"), sg.Combo(pc_name, key="-x-"), sg.Text("   "), sg.Text("縦軸"), sg.Combo(pc_name, key="-y-"), sg.Text("   "), sg.Text("クラスタ数"), sg.Input(key="-cluster-")],
              [sg.Image(filename="", key="-image-")],
              [sg.Button("Run")]]
    
    # ウィンドウ作成
    window = sg.Window("主成分プロット", layout)

    # イベントループ
    while True:
        event, values = window.read()

        # 終了
        if event == sg.WIN_CLOSED:
            break

        # 画像更新
        if event == "Run":
            # 画像作成
            k_means(df, score, values["-x-"], values["-y-"], int(values["-cluster-"]))

            # 画像表示
            window["-image-"].update(r"k_means\\" + values["-x-"] + "_" + values["-y-"] + "_" + values["-cluster-"] + ".png")




# K-means 
def k_means(df:pd.DataFrame, score:pd.DataFrame, pc1:str, pc2:str, n:int):
    plt.clf()
    print(f" *****pc1:{pc1} pc2:{pc2} cluster:{n}***** ")
    
    # 街路番号抽出してscoreのインデックスへ
    load_num = df.index

    # load_num = load_num.tolist()
    score.index = load_num

    # Kmeansインスタンス作成
    kmeans = KMeans(n_clusters=n, max_iter=300)
    
    # モデルを学習 -> クラスタリング
    class_label = kmeans.fit_predict(score[[pc1, pc2]])

    # クラスcolumns作成
    score["class"] = class_label

    print(score["class"])

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
        
    # グラフ保存
    plt.savefig(r"k_means\\" + pc1 + "_" + pc2 + "_" + str(n) + ".png")


# メイン関数
def main():
    df = bar_plot.edit_data()

    # 第n主成分まで生成
    n = 8
    (score, eigenvalue, ratio, eigen) = bar_plot.pca(df, n)

    gui(score, df)

    

# 実行部分
if __name__ == "__main__":
    main()