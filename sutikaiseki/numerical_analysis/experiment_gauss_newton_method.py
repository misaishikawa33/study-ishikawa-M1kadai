"""
【概要】
入力画像と相似変換によって変換した出力画像から回転角度θとスケールパラメータsをガウス・ニュートン法によって推定するプログラム
【使用方法】
入力：
・使用する画像
・真値（角度、スケール）
・初期値（角度、スケール）
・終了条件（閾値、最大反復回数）
・ガウシアンフィルタのパラメータ（カーネルサイズ、シグマ）
出力：
・回転角度
・スケールパラメータ
実行：
python experiment_gauss_newton_method.py {画像のパス}  {真値のスケール} {真値の角度(deg)} --theta_init {初期値の角度(deg)} --scale_init {初期値のスケール} --threshold {収束判定の閾値} --max_loop {最大反復回数} --kernel_size {ガウシアンフィルタのカーネルサイズ} --sigma {ガウシアンフィルタのシグマ}
python3 experiment_gauss_newton_method.py input/color/Lenna.bmp 1 5 --scale_init 1 --theta_init 0 --threshold 1e-5 --max_loop 1000 --kernel_size 5 --sigma 2
（ 「--」の引数は省略可。初期値はプログラムを参照）
【情報】
作成者：勝田尚樹
作成日：2025/7/23
"""
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import similarity_transform as st
import plot_objective_function as pof
import os
import pandas as pd

# 日本語フォントの設定
plt.rcParams['font.family'] = 'DejaVu Sans'
# 日本語フォントが利用可能な場合は設定
try:
    # 一般的な日本語フォントを試す
    for font_name in ['Noto Sans CJK JP', 'IPAexGothic', 'Takao Gothic', 'Hiragino Sans', 'Yu Gothic']:
        if font_name in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = font_name
            break
except:
    # フォントが見つからない場合は英語表記にフォールバック
    pass

# x方向とy方向に平滑微分フィルタを適用する
def apply_smoothing_differrential_filter(img, kernel_size=3, sigma=1):
    #平滑化なし
    # kernel_dx = np.array([[-1, 0, 1]], dtype=np.float32)
    # kernel_dy = np.array([[-1], [0], [1]], dtype=np.float32)
    # dx = cv2.filter2D(img, cv2.CV_64F, kernel_dx)
    # dy = cv2.filter2D(img, cv2.CV_64F, kernel_dy)
    # dx_disp = cv2.convertScaleAbs(dx)
    # dy_disp = cv2.convertScaleAbs(dy)
    # # 平滑化＋微分フィルタ
    # # 平滑化
    # img_blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma)
    # #微分
    # kernel_dx = np.array([[-1, 0, 1]], dtype=np.float32)
    # kernel_dy = np.array([[-1], [0], [1]], dtype=np.float32)
    # dx = cv2.filter2D(img_blurred, cv2.CV_64F, kernel_dx)
    # dy = cv2.filter2D(img_blurred, cv2.CV_64F, kernel_dy)
    # dx_disp = cv2.convertScaleAbs(dx)
    # dy_disp = cv2.convertScaleAbs(dy)
    # 平滑微分フィルタ
    dx_disp = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)  # x方向の微分
    dy_disp = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)  # y方向の微分
    # cv2.imshow("img_blurred", img_blurred)
    # cv2.waitKey(0)
    # cv2.imshow("dx", dx_disp)
    # cv2.imshow("dy", dy_disp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dx_disp, dy_disp

# ガウスニュートン法によりパラメータを推定する
def estimate_by_gauss_newton_method(img_input, img_output, *, scale_init=1, theta_init=0, threshold=1e-6, max_loop=1000, kernel_size=3, sigma=1):
    # 初期値設定
    theta = np.deg2rad(theta_init)
    scale = scale_init
    I_prime_org = img_input
    I = img_output
    theta_history = []
    scale_history = []
    error_history = []
    H, W = I.shape[:2]
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x_coords = x_coords - W / 2
    y_coords = y_coords - H / 2
    for i in range(max_loop):
        # 推定値を使って画像を相似変換
        M = st.compute_M(scale, theta, 0, 0)
        I_prime = st.apply_similarity_transform_reverse(I_prime_org, M)
        I_prime = st.crop_img_into_circle(I_prime)
        # cv2.imshow("I_prime", I_prime)
        # cv2.waitKey(1)
        I_prime_dx, I_prime_dy = apply_smoothing_differrential_filter(I_prime, kernel_size=kernel_size, sigma=sigma)
        # JθとJθθの計算
        dxprime_dtheta = -scale * (x_coords * np.sin(theta) + y_coords * np.cos(theta))
        dyprime_dtheta = scale * (x_coords * np.cos(theta) - y_coords * np.sin(theta))
        J_theta_mat = (I_prime - I) * (I_prime_dx * dxprime_dtheta + I_prime_dy * dyprime_dtheta)
        J_theta = np.sum(J_theta_mat)
        J_theta_theta_mat = (I_prime_dx * dxprime_dtheta + I_prime_dy * dyprime_dtheta) ** 2 
        J_theta_theta = np.sum(J_theta_theta_mat)
        # JSとJSSの計算
        dxprime_dscale = x_coords * np.cos(theta) - y_coords * np.sin(theta)
        dyprime_dscale = x_coords * np.sin(theta) + y_coords * np.cos(theta)
        J_scale_mat = (I_prime - I) * (I_prime_dx * dxprime_dscale + I_prime_dy * dyprime_dscale)
        J_scale = np.sum(J_scale_mat)
        J_scale_scale_mat = (I_prime_dx * dxprime_dscale + I_prime_dy * dyprime_dscale) ** 2 
        J_scale_scale = np.sum(J_scale_scale_mat)
        # JθSの計算
        J_theta_scale_mat = (I_prime_dx * dxprime_dtheta + I_prime_dy * dyprime_dtheta) * (I_prime_dx * dxprime_dscale + I_prime_dy * dyprime_dscale)
        # J_theta_scale_mat = (I_prime_dx**2 * dxprime_dtheta * dxprime_dscale) + (I_prime_dx * I_prime_dy * dxprime_dtheta * dyprime_dscale) + (I_prime_dy * I_prime_dx * dyprime_dtheta * dxprime_dscale) + (I_prime_dy**2 * dyprime_dtheta * dyprime_dscale)    
        J_theta_scale = np.sum(J_theta_scale_mat)
        objective_func_val = 0.5 * np.sum((I_prime - I) ** 2)

        nabla_u_J = np.array([J_theta, J_scale])
        H_u = np.array([[J_theta_theta, J_theta_scale],
                        [J_theta_scale, J_scale_scale]])
        H_u_inv = np.linalg.inv(H_u)
        delta_theta, delta_scale =  - H_u_inv @ nabla_u_J
        # delta_theta, delta_scale = np.linalg.solve(H_u, nabla_u_J)
        if np.abs(delta_theta) < threshold and np.abs(delta_scale) < threshold:
            print(f"delta_theta:{delta_theta},\tdelta_scale:{delta_scale}")
            break
        theta += delta_theta
        scale += delta_scale
        theta_history.append(np.rad2deg(theta))
        scale_history.append(scale)
        error_history.append(objective_func_val)
        print(f"{i}, delta_theta:{delta_theta},\tdelta_scale:{delta_scale},\ttheta:{np.rad2deg(theta)},\tscale:{scale},\terror:{objective_func_val}")
    return np.rad2deg(theta), scale, theta_history, scale_history, error_history, i

def main():
    # データ準備
    parser = argparse.ArgumentParser(description="ガウス・ニュートン法の実験パラメータ設定")
    parser.add_argument("image_path", type=str, help="入力画像のパス")
    parser.add_argument("scale_true", type=float, help="真値のスケール")
    parser.add_argument("theta_true", type=float, help="真値の角度(deg)")
    parser.add_argument("--scale_init", type=float, default=1, help="初期値のスケール")
    parser.add_argument("--theta_init", type=float, default=0, help="初期値の角度(deg)")
    parser.add_argument("--threshold", type=float, default=1e-6, help="収束判定の閾値")
    parser.add_argument("--max_loop", type=int, default=1000, help="最大反復回数")
    parser.add_argument("--kernel_size", type=int, default=3, help="ガウシアンフィルタのカーネルサイズ")
    parser.add_argument("--sigma", type=float, default=1, help="ガウシアンフィルタのシグマ")
    parser.add_argument("--output_path", type=str, default="output", help="実験結果の出力先のフォルダパス")
    args = parser.parse_args()
    img_path = args.image_path
    scale_true = args.scale_true
    theta_true_deg = args.theta_true
    scale_init = args.scale_init
    theta_init_deg = args.theta_init
    threshold = args.threshold
    max_loop = args.max_loop
    kernel_size = args.kernel_size
    sigma = args.sigma
    output_path = args.output_path
    # 画像読み込みと相似変換の適用
    img_input = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_input_cropped = st.crop_img_into_circle(img_input)
    M = st.compute_M(scale_true, np.deg2rad(theta_true_deg), 0, 0)
    img_output = st.apply_similarity_transform_reverse(img_input, M)
    img_output_cropped = st.crop_img_into_circle(img_output)
    # cv2.imshow("input", img_input_cropped)
    # cv2.imshow("output", img_output_cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # ガウスニュートン法によりパラメータを推定
    theta_est, scale_est, theta_history, scale_history, error_history, iteration = estimate_by_gauss_newton_method(img_input, img_output_cropped, 
                                                                                        scale_init=scale_init, 
                                                                                        theta_init=theta_init_deg, 
                                                                                        threshold=threshold, 
                                                                                        max_loop=max_loop, 
                                                                                        kernel_size=kernel_size, 
                                                                                        sigma=sigma)
    print(f"推定結果 角度(deg):{theta_est},\t スケール:{scale_est},\t 反復回数{iteration}")

    # 新規追加: 回転角度θに対する目的関数の一階微分 ∂J/∂θ
    theta_range_grad = np.linspace(theta_true_deg - 10, theta_true_deg + 10, 100)
    dJ_dtheta_range = []
    
    for theta_test in theta_range_grad:
        M_test = st.compute_M(scale_true, np.deg2rad(theta_test), 0, 0)
        I_prime_test = st.apply_similarity_transform_reverse(img_input, M_test)
        I_prime_test = st.crop_img_into_circle(I_prime_test)
        
        I_prime_dx, I_prime_dy = apply_smoothing_differrential_filter(I_prime_test, kernel_size=kernel_size, sigma=sigma)
        
        H, W = I_prime_test.shape[:2]
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        x_coords = x_coords - W / 2
        y_coords = y_coords - H / 2
        
        dxprime_dtheta = -scale_true * (x_coords * np.sin(np.deg2rad(theta_test)) + y_coords * np.cos(np.deg2rad(theta_test)))
        dyprime_dtheta = scale_true * (x_coords * np.cos(np.deg2rad(theta_test)) - y_coords * np.sin(np.deg2rad(theta_test)))
        
        J_theta_grad = np.sum((I_prime_test - img_output_cropped) * (I_prime_dx * dxprime_dtheta + I_prime_dy * dyprime_dtheta))
        dJ_dtheta_range.append(J_theta_grad)
    
    plt.figure(figsize=(8, 5))
    plt.plot(theta_range_grad, dJ_dtheta_range, 'purple', linewidth=2, label='∂J/∂θ')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.axvline(x=theta_true_deg, color='r', linestyle='--', linewidth=2, label=f'True value {theta_true_deg}°')
    plt.axvline(x=theta_est, color='g', linestyle=':', linewidth=2, label=f'Estimated {theta_est:.2f}°')
    plt.title("First derivative of objective function w.r.t. rotation angle θ")
    plt.xlabel("Rotation angle θ (degrees)")
    plt.ylabel("∂J/∂θ")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gradient_vs_theta.png"), dpi=300)
    plt.close()

    # 新規追加: スケールsに対する目的関数の一階微分 ∂J/∂s
    scale_range_grad = np.linspace(max(0.1, scale_true - 0.3), scale_true + 0.3, 100)
    dJ_dscale_range = []
    
    for scale_test in scale_range_grad:
        M_test = st.compute_M(scale_test, np.deg2rad(theta_true_deg), 0, 0)
        I_prime_test = st.apply_similarity_transform_reverse(img_input, M_test)
        I_prime_test = st.crop_img_into_circle(I_prime_test)
        
        I_prime_dx, I_prime_dy = apply_smoothing_differrential_filter(I_prime_test, kernel_size=kernel_size, sigma=sigma)
        
        H, W = I_prime_test.shape[:2]
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        x_coords = x_coords - W / 2
        y_coords = y_coords - H / 2
        
        dxprime_dscale = x_coords * np.cos(np.deg2rad(theta_true_deg)) - y_coords * np.sin(np.deg2rad(theta_true_deg))
        dyprime_dscale = x_coords * np.sin(np.deg2rad(theta_true_deg)) + y_coords * np.cos(np.deg2rad(theta_true_deg))
        
        J_scale_grad = np.sum((I_prime_test - img_output_cropped) * (I_prime_dx * dxprime_dscale + I_prime_dy * dyprime_dscale))
        dJ_dscale_range.append(J_scale_grad)
    
    plt.figure(figsize=(8, 5))
    plt.plot(scale_range_grad, dJ_dscale_range, 'orange', linewidth=2, label='∂J/∂s')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.axvline(x=scale_true, color='r', linestyle='--', linewidth=2, label=f'True value {scale_true}')
    plt.axvline(x=scale_est, color='g', linestyle=':', linewidth=2, label=f'Estimated {scale_est:.3f}')
    plt.title("First derivative of objective function w.r.t. scale s")
    plt.xlabel("Scale s")
    plt.ylabel("∂J/∂s")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gradient_vs_scale.png"), dpi=300)
    plt.close()
    
    # 保存
    img_name = os.path.basename(img_path)
    output_dir = os.path.join(output_path, f"{img_name}_true_s{scale_true}_t{theta_true_deg}_init_s{scale_init}_t{theta_init_deg}")
    # 初期値と推定結果の画像を保存
    M = st.compute_M(scale_init, np.deg2rad(theta_init_deg), 0, 0)
    img_init = st.apply_similarity_transform_reverse(img_input, M)
    img_init_cropped = st.crop_img_into_circle(img_init)
    M = st.compute_M(scale_est, np.deg2rad(theta_est), 0, 0)
    img_est = st.apply_similarity_transform_reverse(img_input, M)
    img_est_cropped = st.crop_img_into_circle(img_est)
    os.makedirs(output_dir, exist_ok=True)
    # 入力画像、出力画像、推定画像の保存
    cv2.imwrite(os.path.join(output_dir, "input.jpg"), img_input_cropped)
    cv2.imwrite(os.path.join(output_dir, "output.jpg"), img_output_cropped)
    cv2.imwrite(os.path.join(output_dir, "init.jpg"), img_init_cropped)
    cv2.imwrite(os.path.join(output_dir, "est.jpg"), img_est_cropped)

    # cv2.imshow("est", img_est_cropped)
    # cv2.imshow("true", img_output_cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 推定結果の変化をグラフに描画
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # 横幅を広めに設定
    axs[0].plot(scale_history)
    axs[0].set_title("Scale History")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Scale")
    axs[0].grid(True)
    axs[1].plot(theta_history)
    axs[1].set_title("Theta History")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Theta")
    axs[1].grid(True)
    plt.tight_layout()
    # 描画結果の保存
    fig.savefig(os.path.join(output_dir, "scale_theta_history.png")) 
    # plt.show()
    
    # 微分結果のグラフを保存
    plt.figure(figsize=(8, 5))
    plt.plot(theta_range_grad, dJ_dtheta_range, 'purple', linewidth=2, label='∂J/∂θ')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.axvline(x=theta_true_deg, color='r', linestyle='--', linewidth=2, label=f'True value {theta_true_deg}°')
    plt.axvline(x=theta_est, color='g', linestyle=':', linewidth=2, label=f'Estimated {theta_est:.2f}°')
    plt.title("First derivative of objective function w.r.t. rotation angle θ")
    plt.xlabel("Rotation angle θ (degrees)")
    plt.ylabel("∂J/∂θ")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gradient_vs_theta.png"), dpi=300)
    plt.close()
    
    # 結果をCSV形式で保存
    result_summary = pd.DataFrame([{
        "真値 角度(deg)": theta_true_deg,
        "推定 角度(deg)": theta_est,
        "真値 スケール": scale_true,
        "推定 スケール": scale_est,
        "反復回数": iteration
    }])
    result_summary.to_csv(os.path.join(output_dir, "result_summary.csv"), index=False, encoding="utf-8-sig")
    # 推定結果変化化をCSV形式で保存
    history_length = max(len(theta_history), len(scale_history))
    theta_history = np.pad(theta_history, (0, history_length - len(theta_history)))
    scale_history = np.pad(scale_history, (0, history_length - len(scale_history)))
    error_history = np.pad(error_history, (0, history_length - len(error_history)))
    history_df = pd.DataFrame({
        "theta_history": theta_history,
        "scale_history": scale_history,
        "error_history": error_history
    })
    history_df.to_csv(os.path.join(output_dir, "history.csv"), index=False, encoding="utf-8-sig")
    
    # 微分結果をCSV形式で保存
    gradient_df = pd.DataFrame({
        "theta_range": theta_range_grad,
        "dJ_dtheta": dJ_dtheta_range,
        "scale_range": np.pad(scale_range_grad, (0, max(0, len(theta_range_grad) - len(scale_range_grad)))),
        "dJ_dscale": np.pad(dJ_dscale_range, (0, max(0, len(theta_range_grad) - len(dJ_dscale_range))))
    })
    gradient_df.to_csv(os.path.join(output_dir, "gradient_data.csv"), index=False, encoding="utf-8-sig")


    # 目的関数を可視化
    # pof.visualize_objective_function(img_input_cropped, img_output_cropped,
    #                                  theta_max=10,
    #                                  theta_min=0,
    #                                  sigma_max=2,
    #                                  simga_min=0.1)
if __name__ == "__main__":
    main()