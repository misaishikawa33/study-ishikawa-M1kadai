<<<<<<< HEAD
# AICS(Advanced Information and Communication Systems)
情報通信システム特論Ⅰのリポジトリです。

## インストール
1. このリポジトリをクローンします。
    ```bash
    git clone https://github.com/kohama-yujin/AICS.git
    ```
2. 必要なライブラリをインストールします。
    ```bash
    cd AICS
    pip install -r requirements.txt
    ```

    ```
    pip install openpyxl
　　```
## sample_wcl
授業資料中のWCLに関する記述に従い、重みと座標を返すサンプルですが、まだ計算が合わないです。
メソッド作成時は参考にしてください。

## 参考文献
- [【Python】Pandasの基本的な使い方のまとめ](https://qiita.com/k-keita/items/953bd334d4da8b944a0b)
- [Pandas DataFrameを徹底解説！](https://ai-inter1.com/pandas-dataframe_basic/)


=======
# AICS(Advanced Information and Communication Systems)
情報通信システム特論Ⅰのリポジトリです。

## インストール
1. このリポジトリをクローンします。
    ```bash
    git clone https://github.com/kohama-yujin/AICS.git
    ```
2. 必要なライブラリをインストールします。
    ```bash
    cd AICS
    pip install -r requirements.txt
    ```

    ```
    pip install openpyxl
　　```
## sample_wcl
授業資料中のWCLに関する記述に従い、重みと座標を返すサンプルです。
メソッド作成時は参考にしてください。

## overlap_ap.py
同じアクセスポイントが上位にランクインした場合，重複したアクセスポイントは除く





## 参考文献
- [【Python】Pandasの基本的な使い方のまとめ](https://qiita.com/k-keita/items/953bd334d4da8b944a0b)
- [Pandas DataFrameを徹底解説！](https://ai-inter1.com/pandas-dataframe_basic/)


##  wcl.py
### x座標の計算:
- x = Σ(重み[i] × 座標[i][0]) / Σ(重み[i])
### y座標の計算:
- y = Σ(重み[i] × 座標[i][1]) / Σ(重み[i])

### 処理の流れ
- 各基準点の重みと座標を受け取る
- 重み付き平均により、x座標とy座標をそれぞれ計算
- 計算された座標をタプル (x, y) として返す
>>>>>>> ishikawa
