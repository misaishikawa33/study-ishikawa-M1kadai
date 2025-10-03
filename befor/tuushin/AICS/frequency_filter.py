import pandas as pd


class FrequencyFilter:
    def __init__(self, data, threshold_mhz=5000):
        """
        周波数フィルタクラス

        Args:
            data: RSSIデータ（dict型 - 各階のシートがDataFrame）
            threshold_mhz: しきい値周波数（MHz）
        """
        self.data = data
        self.threshold = threshold_mhz

    def filter_5ghz(self):
        """
        5GHz帯（5000MHz以上）のデータのみ残す

        Returns:
            dict: フィルタリング後のデータ
        """
        filtered = {}
        for floor, df in self.data.items():
            if "Center Freq(MHz)" not in df.columns:
                print(
                    f"\033[31m警告: Floor {floor} に 'Center Freq(MHz)' カラムがありません。\033[0m"
                )
                filtered[floor] = pd.DataFrame()  # 空のDataFrameを返す
                continue

            # 5GHz帯のデータをフィルタリング
            filtered_df = df[df["Center Freq(MHz)"] >= 5000].copy()
            filtered[floor] = filtered_df

            # print(f"Floor {floor}: {len(df)} -> {len(filtered_df)} レコード (5GHz帯)")

        return filtered

    def filter_24ghz(self):
        """
        2.4GHz帯（3000MHz以下）のデータのみ残す

        Returns:
            dict: フィルタリング後のデータ
        """
        filtered = {}
        for floor, df in self.data.items():
            if "Center Freq(MHz)" not in df.columns:
                print(
                    f"\033[31m警告: Floor {floor} に 'Center Freq(MHz)' カラムがありません。\033[0m"
                )
                filtered[floor] = pd.DataFrame()  # 空のDataFrameを返す
                continue

            # 2.4GHz帯のデータをフィルタリング（3000MHz以下に変更）
            filtered_df = df[df["Center Freq(MHz)"] <= 3000].copy()
            filtered[floor] = filtered_df

            # print(f"Floor {floor}: {len(df)} -> {len(filtered_df)} レコード (2.4GHz帯)")

        return filtered


# import pandas as pd

# class FrequencyFilter:
#     def __init__(self, data, threshold_mhz=5000):
#         self.data = data
#         self.threshold = threshold_mhz

#     def filter_5ghz(self):
#         """
#         5GHz帯（例：5000MHz以上）のデータのみ残す
#         data: dict型（各階のシートがDataFrame）
#         """
#         filtered = {}
#         for floor, df in self.data.items():
#             if "Center Freq(MHz)" not in df.columns:
#                 print(f"Floor {floor} has no 'Center Freq(MHz)' column.")
#                 continue
#             filtered[floor] = df[df["Center Freq(MHz)"] >= self.threshold]
#         return filtered

#     def filter_24ghz(self):
#         """
#         2.4GHz帯（2500MHz以下）のデータのみ残す
#         """
#         filtered = {}
#         for floor, df in self.data.items():
#             if "Center Freq(MHz)" not in df.columns:
#                 print(f"Floor {floor} has no 'Center Freq(MHz)' column.")
#                 continue
#             filtered[floor] = df[df["Center Freq(MHz)"] <= self.threshold]
#         return filtered


# # # 5GHzフィルタをかける
# #     freq_filter = FrequencyFilter(rssi.data)
# #     rssi.data = freq_filter.filter_5ghz()
