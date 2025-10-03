"""
【概要】
入力画像と相似変換によって変換した出力画像から回転角度θとスケールパラメータsをガウス・ニュートン法によって推定するプログラム
【使用方法】
入力：
・元画像
・相似変換した画像
出力：
・回転角度
・スケールパラメータ
実行：
python gauss_newton_method.py input.jpg output.jpg

【情報】
作成者：勝田尚樹
作成日：2025/7/23
"""
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import similarity_transform as st


# x方向とy方向に平滑微分フィルタを適用する
def apply_smoothing_differrential_filter(img, kernel_size=3, sigma=1):
    # 平滑化(ガウシアンフィルタ)
    img_blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma)
    # cv2.imshow("img_blurred", img_blurred)
    # cv2.waitKey(0)

    # 微分
    # 単純な差分フィルタ
    kernel_dx = np.array([[-1, 0, 1]], dtype=np.float32)
    kernel_dy = np.array([[-1], [0], [1]], dtype=np.float32)
    # フィルタ適用
    dx = cv2.filter2D(img_blurred, cv2.CV_64F, kernel_dx)
    dy = cv2.filter2D(img_blurred, cv2.CV_64F, kernel_dy)


    # 表示用に変換
    dx_disp = cv2.convertScaleAbs(dx)
    dy_disp = cv2.convertScaleAbs(dy)
    # cv2.imshow("dx", dx_disp)
    # cv2.imshow("dy", dy_disp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dx_disp, dy_disp

# ガウスニュートン法によりパラメータを推定する
def estimate_by_gauss_newton_method(img_input, img_output, theta_init=0, scale_init=1, threshold=1e-5, max_loop=1000):
    # 初期値設定
    theta = np.deg2rad(theta_init) # 初期角度:角度はラジアンで扱う
    scale = scale_init             # 初期スケール

    I_prime_org = img_input # 元画像(回転済み)
    I = img_output          # 推定画像

    theta_history = []
    scale_history = []
    
    H, W = I.shape[:2]
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x_coords = x_coords - W / 2
    y_coords = y_coords - H / 2
    # breakpoint()
    for i in range(max_loop):
        M = st.compute_M(scale, theta, 0, 0)
        I_prime = st.apply_similarity_transform_reverse(I_prime_org, M) # 逆変換
        I_prime = st.crop_img_into_circle(I_prime)
        I_prime_dx, I_prime_dy = apply_smoothing_differrential_filter(I_prime, kernel_size=5, sigma=2) # 微分
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
        # dxprime_dthetascale = - x_coords * np.sin(theta) - y_coords * np.cos(theta)
        # dyprime_dthetascale = x_coords * np.cos(theta) - y_coords * np.sin(theta)
        # J_theta_scale_mat = (I_prime - I) * (I_prime_dx * dxprime_dthetascale + I_prime_dy * dyprime_dthetascale)
        J_theta_scale_mat = (I_prime_dx * dxprime_dtheta + I_prime_dy * dyprime_dtheta) * (I_prime_dx * dxprime_dscale + I_prime_dy * dyprime_dscale)
        J_theta_scale = np.sum(J_theta_scale_mat)

        nabla_u_J = np.array([J_theta, J_scale])
        H_u = np.array([[J_theta_theta, J_theta_scale],
                        [J_theta_scale, J_scale_scale]])
        # H_u_inv = np.linalg.inv(H_u)
        # delta_theta, delta_scale =  - H_u_inv @ nabla_u_J
        # print(H_u, nabla_u_J)
        delta_theta, delta_scale = np.linalg.solve(H_u, nabla_u_J)
        if np.abs(delta_theta) < threshold and np.abs(delta_scale) < threshold:
            break
        theta -= delta_theta
        scale -= delta_scale
        theta_history.append(np.rad2deg(theta))
        scale_history.append(scale)
        print(f"delta_theta;{delta_theta},\tdelta_scale:{delta_scale},\ttheta:{np.rad2deg(theta)},\tscale:{scale}")
    print(f"反復回数：{i}")
    return theta, scale, theta_history, scale_history

# 目的関数を3次元空間にプロットする。極小値確認用。ただし、めっちゃ実行時間かかる
def visualize_objective_function(img_input, img_output):
    # パラメータ範囲
    I_prime_org = img_input
    I = img_output
    theta_values = np.arange(0, 10, 1)  
    scale_values = np.arange(0.5, 1.6, 0.1)    
    # Jの結果格納用 (scale x theta の2次元配列)
    J_values = np.zeros((len(scale_values), len(theta_values)))
    # I と I_prime は事前に用意されているものとする
    for i, scale in enumerate(scale_values):
        for j, theta in enumerate(theta_values):
            # 角度をラジアンに変換
            theta_rad = np.deg2rad(theta)
            # 相似変換を適用
            M = st.compute_M(scale, theta_rad, 0, 0)
            I_prime = st.apply_similarity_transform_reverse(I_prime_org, M)
            I_prime_cropped = st.crop_img_into_circle(I_prime)
            # 目的関数Jを計算
            J = 0.5 * np.sum((I_prime_cropped - I) ** 2)
            J_values[i, j] = J
    # すでに計算済みの J_values, theta_values, scale_values を使用
    Theta, Scale = np.meshgrid(theta_values, scale_values)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    # 3Dサーフェスプロット
    surf = ax.plot_surface(Theta, Scale, J_values, cmap='viridis', edgecolor='none')
    # 軸ラベル
    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel('Scale')
    ax.set_zlabel('Objective Function J')
    ax.set_title('3D Plot of J(Theta, Scale)')
    # カラーバー
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='J')
    plt.show()

def main():
    # データ準備
    if len(sys.argv) != 3:
        print("Usage: python gauss_newton_method.py {元画像のパス} {相似変換した画像のパス}")
        sys.exit(1)
    img_input_path = sys.argv[1]
    img_output_path = sys.argv[2]
    img_input = cv2.imread(img_input_path, cv2.IMREAD_GRAYSCALE)
    img_output = cv2.imread(img_output_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("input", img_input)
    cv2.imshow("output", img_output)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #ガウスニュートン法によりパラメータを推定
    theta, scale, theta_history, scale_history = estimate_by_gauss_newton_method(img_input, img_output)
    #可視化
    print(f"(deg):{np.rad2deg(theta)},\t (rad):{theta},\t (scale):{scale}")
    plt.plot(theta_history)
    plt.plot(scale_history)
    plt.grid(True)
    plt.show()
    # 目的関数を可視化
    visualize_objective_function(img_input, img_output)
if __name__ == "__main__":
    main()