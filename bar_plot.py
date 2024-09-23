import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import japanize_matplotlib

# 時間文字列をint[秒]に変換
def time_convert_into_int(time_str:str):
    # timedelta型に変換
    delta = datetime.strptime(time_str, "%H:%M:%S") - datetime(1900, 1, 1)

    # 秒数に変換
    time_int = int(delta.total_seconds())

    return time_int

# グラフ出力
def img_output(eigen:pd.DataFrame):

    # columns各値ががタプルで格納されているので、戻す。
    eigen.columns = [tup[0] for tup in eigen.columns]

    # 表示データ制限
    
    output_eigen = eigen.loc[:, "1人":"国籍不明"]

    # 各変数を横棒グラフへ
    count = 1

    # 表題用辞書
    title = {1:"のんびり交流・\n喜びシェア軸", 2:"人にやさしい街路軸", 3:"多様性輝き軸", 4:"にぎわい喜び軸", 5:"グローバル絆づくり軸", 6:"都市づくり軸", 7:"女性が輝く交流軸", 8:"多文化ふれあい軸"}

    for index in output_eigen.index:

        # サブプロット
        plt.subplot(1, 8, count)
        
        # 各表題
        plt.title(title[count], fontsize = 12)

        # 文字大きさ
        plt.yticks(fontsize=15)

        # メモリ
        plt.tick_params(labelbottom=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        if count != 1:
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)

        # 配置調整
        plt.subplots_adjust(wspace=0, hspace=0, left=0.3)

        # 本ループのプロットデータを抽出 -> ndarray
        plot_data = output_eigen.loc[index].to_numpy()
        color = [("#ee7800" if i > 0 else "b") for i in plot_data]

        # 表示文字
        v_name = output_eigen.columns.tolist()
        print(v_name)

        # グラフ作成
        plt.barh(v_name, plot_data, color=color)

        # 数値表示(以降消すかも　PC4に表示バグあり)
        for idx, data in enumerate(plot_data):
            plt.text(0.1, idx, str(round(data, 3)), size = "x-large")


        # カウント
        count += 1

    # グラフ表示
    #plt.savefig("aaaa.png")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

# データ加工
def edit_data():
    # ファイルパス
    data_path = "調査データ36.csv"

    # データ読み込み(csv -> DataFrame) なお、１行目は空白なので２行目(属性)をcolumns(列のインデックス)に指定しながら読み込んでいる
    df = pd.read_csv(data_path, encoding="shift-jis")
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

    # テスト削除(15変数に縛る -> 結構いい感じに表示される)
    #df = df.drop(columns=df.loc[:, "通行者20人到達時間":"写真や動画を撮っている人がいる"].columns, axis=1)

    # 経過時間->int型へ
    df["通行者20人到達時間"] = df["通行者20人到達時間"].apply(time_convert_into_int)

    # 抽出
    #df = df.loc[:, "1人":"国籍不明"]

    return df

# 主成分分析
def pca(df:pd.DataFrame, n:int):
    # データの標準化 iloc -> dataframeの各行、列を指定するメソッド apply -> 各行列に対して指定関数を使用 axis=0 -> 列操作
    dfs = df.iloc[:, 0:].apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    # ここでstdが0の属性は全値が一致 -> 情報量0 -> 削除してしまって問題ない？
    dfs_cleaned = dfs.dropna(axis=1, how="any")

    # 主成分分析の実行(PCAモデル作成) 第10主成分まで抽出 主成分最大数は行数(サンプル数)らしい　なぜ？
    pca = PCA(n_components=n)

    # データをPCAモデルに適用
    pca.fit(dfs_cleaned)

    # データを主成分空間に写像(要は10主成分に変換) 各主成分がどのような意味を持つかは負荷率から考察するしかない。
    feature = pca.transform(dfs_cleaned)

    # 主成分得点(各主成分軸状での各データの変換後の値)
    score = pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(0, n)])

    # 主成分得点表に街路番号追加
    sub = score.round(6)
    sub["街路番号"] = dfs_cleaned.index.to_list()

    # 主成分プロット(とりあえず第1, 第2主成分)
    #plt.scatter(feature[0:feature.shape[0]-1, 0], feature[0:feature.shape[0]-1, 1])
    #plt.xlabel("PC1")
    #plt.ylabel("PC2")
    #plt.show()
    #exit()

    # PCA の固有値
    #print("******固有値******")
    eigenvalue = pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(0, n)])
    #for index, row in eigenvalue.iterrows():
        #print(f"{index}固有値：{row[0]}")

    # 寄与率(各主成分についてどれだけ説明できてるか -> 累積寄与率は最終的には1になる)
    #print("******寄与率******")
    ratio = pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(0, n)])
    sum = 0
    #for index, row in ratio.iterrows():
        #sum += row[0]
        #print(f"{index}寄与率：{row[0]} (累積寄与率：{sum})")

    # 負荷率(各主成分に対して、各変数がどの程度影響しているか)
    #print("******負荷率******")
    eigen_vector = pca.components_
    loadings = eigen_vector * np.sqrt(eigenvalue.values.reshape(-1, 1))
    eigen =pd.DataFrame(loadings,
                        columns=[dfs_cleaned.columns],
                        index = ["PC{}".format(x+1) for x in range(0, n)])
    #print(eigen)

    # 要素名抽出(以降のグラフ表示の際に使用)
    #v_name = eigen.columns.to_numpy()
    #v_name = [i[0] for i in v_name]
    #v_name = np.array(v_name)
    #print(v_name)

    # excelデータへ変換
    with pd.ExcelWriter("PCA_OUTPUT.xlsx") as writer:
        eigenvalue.to_excel(writer, sheet_name="固有値")
        ratio.to_excel(writer, sheet_name="寄与率")
        eigen.to_excel(writer, sheet_name="負荷率")
        score.to_excel(writer, sheet_name="主成分得点表")

    # csv化
    sub.to_csv(r"主成分得点表.csv")

    return score, eigenvalue, ratio, eigen
    
# メイン関数
def main():
    df = edit_data()
    # 第n主成分まで生成
    n = 8
    (score, eigenvalue, ratio, eigen) = pca(df, n)
    img_output(eigen)

# 実行部分
if __name__ == "__main__":
    main()
