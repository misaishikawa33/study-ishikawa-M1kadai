from aics_excel_loader import AicsExcelLoader
from sample_wcl import SampleWCL
from wcl import WCL
from frequency_filter import FrequencyFilter
from overlap_ap import Overlapap
from count import Count
from overlap_ap_count import OverlapapCount
import traceback
import csv
import math
import numpy as np
from itertools import zip_longest


def main():
    data_folda = "./dataset"  # フォルダ名

    # インスタンス化
    ap = AicsExcelLoader(data_folda + "/AP_coordinate.xlsx")
    location = AicsExcelLoader(data_folda + "/location_coordinate.xlsx")
    rssi = AicsExcelLoader(data_folda + "/measured_RSSI_Bdelete.xlsx")

    # データの確認
    # print(ap.data)  # シートが1枚の時：データフレーム型
    # print(location.data["3"])  # シートが複数枚の時：辞書型（keyはシート名）
    # print(location.data["4"])  # 処理しやすいよう、シート名を階数に変更している
    # print(rssi.data["3"])  # key指定したdata["3"]等は、データフレーム型
    # print(rssi.data["4"])

    # カラムの確認
    # print(location.data["3"].columns)  # .columnsでカラム名を確認できる
    # print(location.data["3"]["x"])

    # 周波数フィルタ
    freq_filter = FrequencyFilter(rssi.data)
    filtered_5ghz_rssi_data = freq_filter.filter_5ghz()
    filtered_24ghz_rssi_data = freq_filter.filter_24ghz()

    # 授業資料中のWCLを実装
    sampleWcl = SampleWCL(ap.data, rssi.data)
    # 5GHzフィルタを適用して授業資料中のWCLを実装
    filtered_5ghz_sampleWcl = SampleWCL(ap.data, filtered_5ghz_rssi_data)
    # 2.4GHzフィルタを適用して授業資料中のWCLを実装
    filtered_24ghz_sampleWcl = SampleWCL(ap.data, filtered_24ghz_rssi_data)
    # 重複削除
    overlap = Overlapap(ap.data, rssi.data)
    # カウント数を優先
    count = Count(ap.data, rssi.data)
    # 2.4GHzフィルタ & 重複削除
    filtered_24ghz_overlap = Overlapap(ap.data, filtered_24ghz_rssi_data)
    # 2.4GHzフィルタ & カウント
    filtered_24ghz_count = Count(ap.data, filtered_24ghz_rssi_data)
    # 5GHzフィルタ & 重複削除
    filtered_5ghz_overlap = Overlapap(ap.data, filtered_5ghz_rssi_data)
    # 5GHzフィルタ & カウント
    filtered_5ghz_count = Count(ap.data, filtered_5ghz_rssi_data)
    # 重複削除 & カウント
    overlap_count = OverlapapCount(ap.data, rssi.data)
    # 2.4GHzフィルタ & 重複削除 & カウント
    filtered_24ghz_overlap_count = OverlapapCount(ap.data, filtered_24ghz_rssi_data)
    # 5GHzフィルタ & 重複削除 & カウント
    filtered_5ghz_overlap_count = OverlapapCount(ap.data, filtered_5ghz_rssi_data)

    # 結果を格納するリスト
    all_results = []
    method_results = []
    floor_results = []
    # 実測値リスト
    all_correct = []
    method_correct = []
    floor_correct = []
    # 手法リスト
    methods = [
        "SampleWCL_L3",
        "SampleWCL_L4",
        "SampleWCL_L5",
        "5GHz",
        "24GHz",
        "Overlap",
        "Count",
        "24GHz-Overlap",
        "24GHz-Count",
        "5GHz-Overlap",
        "5GHz-Count",
        "Overlap-Count",
        "24GHz-Overlap-Count",
        "5GHz-Overlap-Count",
    ]
    # 階数リスト
    floors = [3, 4]

    for method in methods:
        method_results = []  # 初期化
        method_correct = []
        for floor in floors:
            floor_results = []  # 初期化
            floor_correct = []
            for p in range(1, 60):
                try:
                    if method == "SampleWCL_L3":
                        weight, coordinate = sampleWcl.get_weight_and_coords(
                            floor, p, 3
                        )
                    elif method == "SampleWCL_L4":
                        weight, coordinate = sampleWcl.get_weight_and_coords(
                            floor, p, 4
                        )
                    elif method == "SampleWCL_L5":
                        weight, coordinate = sampleWcl.get_weight_and_coords(
                            floor, p, 5
                        )
                    elif method == "5GHz":
                        weight, coordinate = (
                            filtered_5ghz_sampleWcl.get_weight_and_coords(floor, p, 3)
                        )
                    elif method == "24GHz":
                        weight, coordinate = (
                            filtered_24ghz_sampleWcl.get_weight_and_coords(floor, p, 3)
                        )
                    elif method == "Overlap":
                        weight, coordinate = overlap.get_weight_and_coords(floor, p, 3)
                    elif method == "Count":
                        weight, coordinate = count.get_weight_and_coords(floor, p, 3)
                    elif method == "24GHz-Overlap":
                        weight, coordinate = (
                            filtered_24ghz_overlap.get_weight_and_coords(floor, p, 3)
                        )
                    elif method == "24GHz-Count":
                        weight, coordinate = filtered_24ghz_count.get_weight_and_coords(
                            floor, p, 3
                        )
                    elif method == "5GHz-Overlap":
                        weight, coordinate = (
                            filtered_5ghz_overlap.get_weight_and_coords(floor, p, 3)
                        )
                    elif method == "5GHz-Count":
                        weight, coordinate = filtered_5ghz_count.get_weight_and_coords(
                            floor, p, 3
                        )
                    elif method == "Overlap-Count":
                        weight, coordinate = overlap_count.get_weight_and_coords(
                            floor, p, 3
                        )
                    elif method == "24GHz-Overlap-Count":
                        weight, coordinate = (
                            filtered_24ghz_overlap_count.get_weight_and_coords(
                                floor, p, 3
                            )
                        )
                    elif method == "5GHz-Overlap-Count":
                        weight, coordinate = (
                            filtered_5ghz_overlap_count.get_weight_and_coords(
                                floor, p, 3
                            )
                        )

                    # 推定位置座標 T を計算
                    wcl = WCL(weight, coordinate)
                    T = wcl.calculate_coordinate()
                    rounded_T = tuple(round(x, 3) for x in T)
                    # 推定結果を階数結果リストに追加
                    floor_results.append(
                        {
                            "position": p,
                            "estimated_coordinate": rounded_T,
                        }
                    )
                    # 実測値
                    floor_correct.append(
                        (
                            int(location.data[str(floor)]["x"][p - 1]),
                            int(location.data[str(floor)]["y"][p - 1]),
                        )
                    )
                # 例外処理を追加
                except Exception as e:
                    print(
                        f"\033[31mSampleWCL 位置P={p}: エラーが発生しました - {e}\033[0m"
                    )
                    traceback.print_exc()
                    continue
            # 階数ごとの結果を手法結果リストに追加
            method_results.append(
                {
                    "floor": floor,
                    "results": floor_results,
                }
            )
            method_correct.append(floor_correct)
            print(
                f"\033[36m{floor}階での処理完了: {len(floor_results)}個の位置で推定が成功しました\033[0m"
            )
        # 手法ごとの推定結果をすべての結果リストに追加
        all_results.append({"name": method, "results": method_results})
        all_correct.append(method_correct)
        print(f"\033[32m{method}の推定が終わりました\033[0m")

    # CSVファイルに結果を出力
    for method_index, method in enumerate(all_results):
        csv_filename = f'results/{method["name"]}.csv'
        # ファイルオープン
        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            # タイトルを作成
            title = []
            for floor in method["results"]:
                title.extend(
                    [
                        f"{floor['floor']}F-{method['name']}",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
            # タイトルを書き込み
            writer.writerow(title)
            # ヘッダーを作成
            header = []
            for floor in method["results"]:
                header.extend(
                    [
                        "",
                        "estimated",
                        "",
                        "correct",
                        "",
                        "",
                    ]
                )
            # ヘッダーを書き込み
            writer.writerow(header)
            # サブヘッダーを作成
            subheader = []
            for floor in method["results"]:
                subheader.extend(
                    [
                        "Index P",
                        "x",
                        "y",
                        "x",
                        "y",
                        "error",
                    ]
                )
            # サブヘッダーを書き込み
            writer.writerow(subheader)

            # データを作成
            total_error = []
            for positions_index, positions in enumerate(
                zip_longest(*[floor["results"] for floor in method["results"]]), start=1
            ):
                row = []
                position_error = []
                for floor_index, floor_result in enumerate(positions):
                    if floor_result is None:
                        row += ["", "", "", "", "", ""]
                    else:
                        position = floor_result["position"]
                        estimated_coordinate = list(
                            floor_result["estimated_coordinate"]
                        )
                        correct_coordinate = list(
                            all_correct[method_index][floor_index][positions_index - 1]
                        )
                        # 誤差計算
                        error = math.sqrt(
                            pow(
                                estimated_coordinate[0] - correct_coordinate[0],
                                2,
                            )
                            + pow(
                                estimated_coordinate[1] - correct_coordinate[1],
                                2,
                            )
                        )
                        row += (
                            [position]
                            + estimated_coordinate
                            + correct_coordinate
                            + [round(error, 3)]
                        )
                        position_error.append(error)
                total_error.append(position_error)
                # データを書き込み
                writer.writerow(row)
            # フッターの計算
            total_error = list(map(list, zip_longest(*total_error)))
            average_error = []  # 平均
            max_error = []  # 最大値
            min_error = []  # 最小値
            std_population = []  # 標準偏差
            variance_population = []  # 分散
            for error in total_error:
                # Noneを除外
                error_not_none = [x for x in error if x is not None]
                average_error.append(round(np.mean(error_not_none), 3))
                max_error.append(round(np.max(error_not_none), 3))
                min_error.append(round(np.min(error_not_none), 3))
                std_population.append(round(np.std(error_not_none), 3))
                variance_population.append(round(np.var(error_not_none), 3))
            # フッターを作成
            footer_ave = []
            footer_max = []
            footer_min = []
            footer_std = []
            footer_var = []
            for i, floor in enumerate(method["results"]):
                footer_ave.extend(["", "", "", "", "Ave", average_error[i]])
                footer_max.extend(["", "", "", "", "Max", max_error[i]])
                footer_min.extend(["", "", "", "", "Min", min_error[i]])
                footer_std.extend(["", "", "", "", "Std", std_population[i]])
                footer_var.extend(["", "", "", "", "Var", variance_population[i]])
            # フッターを書き込み
            writer.writerow(footer_ave)
            writer.writerow(footer_max)
            writer.writerow(footer_min)
            writer.writerow(footer_std)
            writer.writerow(footer_var)
        print(f"\033[32m結果をCSVファイル '{csv_filename}' に保存しました\033[0m")


if __name__ == "__main__":
    main()
