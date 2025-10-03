# 同じアクセスポイントの削除
import numpy as np
import pandas as pd


# 授業資料内のWCLを実装
class OverlapapCount:
    def __init__(self, ap, rssi):
        self.ap = ap
        self.rssi = rssi

    def get_weight_and_coords(self, floor, p, l):
        """
        授業資料内のWCLに従い、重みと座標を返すメソッド
        ※未完成（計算が合わない）

        args:
            floor: 階数
            p: Location index P
            l: アンカーノードの個数 L

        return:
            weight: 重み
            coorinate: 座標 (x, y)
        """

        # 指定階数で計測したデータを抽出
        rssi_floor_data = self.rssi[str(floor)]

        """
        以下、授業資料より抜粋
        3階では3階に設置されたAPからのRSSIのみを、4階では4階に設置されたRSSIのみを位置推定に用いる。
        """
        # 1. 指定階数のAPを抽出
        check_str = str(floor) + "F"
        rssi_floor_only = rssi_floor_data[
            rssi_floor_data["AP_name"].str.contains(check_str)
        ]
        # print(f"{floor}階のAP\n{rssi_floor_only}\n")  # デバッグ用

        # 2. 位置PのRSSIデータを抽出
        rssi_p = rssi_floor_only[rssi_floor_only["Location index P"] == p]
        # print(f"位置PのRSSIデータ\n{rssi_p}\n")  # デバッグ用

        """
        以下、授業資料より抜粋
        RSSIの中央値が大きい順に上位三つのAPを選択する。
        """
        # 3. 'Counts (/100)' の値が多く、かつ 'MED (dBm)' の値が高い順にソート
        rssi_sorted = rssi_p.sort_values(
            by=["Counts (/100)", "MED (dBm)"], ascending=[False, False]
        )
        # 上位三つ(L個)のRSSI値を取得（AP_nameが重複しないようにする）
        rssi_med = []
        seen_ap_names = set()  # 重複チェック用のセット

        for _, row in rssi_sorted.iterrows():
            ap_name = row["AP_name"]
            if ap_name not in seen_ap_names:  # AP_nameがまだ選ばれていない場合
                rssi_med.append(row)  # 重複していない場合のみ追加
                seen_ap_names.add(ap_name)  # AP_nameを記録
            if len(rssi_med) == l:  # 上位L個を選び終えたら終了
                break

        # DataFrameに変換
        rssi_med = pd.DataFrame(rssi_med)
        # print(f"上位{l}個のRSSI値（重複なし）\n{rssi_med}\n")  # デバッグ用

        """
        以下、授業資料より抜粋
        重みは、上位三つの中でRSSIの中央値が最小のもの、つまり大きさ3番目の値との差[dB]を求め、その真値とする。
        """
        # 4. 重みを計算(相対的な重み)
        weight = []
        min_rssi = float(rssi_med["MED (dBm)"].min())  # 最小の中央値を取得
        for rssi in rssi_med["MED (dBm)"]:
            rssi = float(rssi)
            if rssi == min_rssi:
                weight.append(1.0)  # 最小の中央値の重みを1に設定
            else:
                weight.append(
                    float(10 ** ((rssi - min_rssi) / 10))
                )  # 最小中央値との差を基に計算
        # print(f"重み\n{weight}\n")  # デバッグ用

        # 5. 座標を取得
        coordinate = []
        for rssi_name in rssi_med["AP_name"]:
            coordinate.append(
                (
                    float(self.ap[self.ap["AP_name"] == rssi_name]["x"].iloc[0]),
                    float(self.ap[self.ap["AP_name"] == rssi_name]["y"].iloc[0]),
                )
            )
        # print(f"座標\n{coordinate}\n")  # デバッグ用

        return weight, coordinate
