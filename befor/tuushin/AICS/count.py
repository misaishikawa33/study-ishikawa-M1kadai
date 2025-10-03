import numpy as np
import pandas as pd


class Count:
    def __init__(self, ap, rssi):
        self.ap = ap
        self.rssi = rssi

    def get_weight_and_coords(self, floor, p, l):

        rssi_floor_data = self.rssi[str(floor)]

        # 指定階数に設置されたAPからのRSSIのみを抽出
        check_str = str(floor) + "F"
        rssi_floor_only = rssi_floor_data[
            rssi_floor_data["AP_name"].str.contains(check_str)
        ]
        # print(f"{floor}階のAP\n{rssi_floor_only}\n")  # デバッグ用

        # 特定のLocation index PのRSSIデータを抽出
        rssi_p = rssi_floor_only[rssi_floor_only["Location index P"] == p]
        # print(f"位置PのRSSIデータ\n{rssi_p}\n")  # デバッグ用

        # 3. 'Counts (/100)' の値が多く、かつ 'MED (dBm)' の値が高い順にソート
        rssi_sorted = rssi_p.sort_values(
            by=["Counts (/100)", "MED (dBm)"], ascending=[False, False]
        )

        # 4. 上位L個をそのまま取得（重複を許す）
        rssi_med = rssi_sorted[0:l]
        # print(f"上位{l}個のRSSI値\n{rssi_med}\n")  # デバッグ用

        # 5. 重みを計算
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

        # 6. 座標を取得
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
