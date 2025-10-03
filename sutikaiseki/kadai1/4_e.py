# とりあえず動く

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, eigh

# --- パラメータ定義 ---
a = 300             # 楕円の長半径（X方向）
b = 200             # 楕円の短半径（Y方向）
N = 1000             # 点の数
sigma_max = 2.0     # ノイズの最大標準偏差
sigma_values = np.arange(0.1, sigma_max + 0.05, 0.1)  # σ = 0.1〜2.0を0.1刻みで
num_trials = 100   # 各σごとに繰り返す試行回数

# --- 楕円上の真の点列を生成 ---
i_vals = np.arange(N)
theta = -np.pi / 4 + (11 * np.pi) / (12 * N) * i_vals  # θ_i を定義
x_true = a * np.cos(theta)  # 楕円上のx座標
y_true = b * np.sin(theta)  # 楕円上のy座標
X_true = np.vstack((x_true, y_true)).T  # (N,2)の行列にまとめる

# --- KCR下界のRMSを計算 ---
def kcr_lower_bound(X, u_true, sigma2):
    N = len(X)
    M = np.zeros((6,6))
    for x, y in X:
        xi = phi(x, y)
        V = calc_V_xi(x, y, sigma2)
        denom = u_true @ V @ u_true
        if denom < 1e-12:
            denom = 1e-12  # 0除算対策
        M += np.outer(xi, xi) / denom

    # 疑似逆行列でランク落ち対応
    M_inv = np.linalg.pinv(M)

    # 射影行列
    P = np.eye(6) - np.outer(u_true, u_true)
    cov_proj = P @ M_inv @ P

    # RMS誤差（KCR下界）の計算
    rms_kcr = np.sqrt(np.trace(cov_proj))

    # 小さすぎる値を補正
    rms_kcr = max(rms_kcr, 1e-10)

    return rms_kcr


# --- 特徴ベクトルφ(x,y)を定義（6次元） ---
def phi(x, y):
    # 式: φ(x) = (x², 2xy, y², 2x, 2y, 1)^T
    return np.array([
        x**2,
        2*x*y,
        y**2,
        2*x,
        2*y,
        1.0
    ])

# --- 最小二乗法による楕円パラメータ推定 ---
def estimate_least_squares(X):
    Phi = np.array([phi(x, y) for x, y in X])       # 各点にφ(x,y)を適用して行列化
    M = (Phi.T @ Phi) / len(X)                      # M = Σ φ φ^T / N
    _, _, Vh = np.linalg.svd(M)                     # SVDで最小固有値のベクトルを取得
    u_est = Vh[-1]                                  # 最小特異値の右特異ベクトル
    u_est /= norm(u_est)                            # ノルム正規化
    return u_est

# --- 特徴ベクトルφに対する共分散行列V_ξを計算 ---
def calc_V_xi(x, y, sigma2):
    Sigma_xy = sigma2 * np.eye(2)  # x, yに独立な同一分散のノイズ
    # φ(x,y) のヤコビ行列（偏導成分）
    J = np.array([
        [2*x, 0],
        [2*y, 2*x],
        [0, 2*y],
        [2, 0],
        [0, 2],
        [0, 0]
    ])
    V_xi = J @ Sigma_xy @ J.T      # 共分散の伝播 V = J Σ J^T
    return V_xi

# --- 反復最尤推定法 ---
def estimate_maximum_likelihood_iterative(X, sigma2, tol=1e-6, max_iter=100):
    N = len(X)
    xi_list = np.array([phi(x,y) for x,y in X])     # φ(x,y) を全点に適用
    V_list = np.array([calc_V_xi(x, y, sigma2) for x, y in X])  # 各点のV_ξ

    # 初期推定値: 最小二乗法のM行列から始める
    M_init = np.sum([np.outer(xi, xi) for xi in xi_list], axis=0)
    eigvals, eigvecs = eigh(M_init)
    u_old = eigvecs[:, 0]
    u_old /= norm(u_old)

    # 反復処理（EMに近い）
    for iteration in range(max_iter):
        M = np.zeros((6,6))
        L = np.zeros((6,6))

        for alpha in range(N):
            xi = xi_list[alpha]
            V = V_list[alpha]

            denom = u_old @ V @ u_old               # 分母：スカラ値
            if denom < 1e-15: denom = 1e-15         # 0除算回避

            M += np.outer(xi, xi) / denom           # 情報行列のようなもの
            numerator = (u_old @ xi)**2
            L += numerator * V / (denom**2)         # 分散項

        A = M - L                                   # A行列の固有ベクトルをとる
        eigvals, eigvecs = eigh(A)
        u_new = eigvecs[:, 0]
        u_new /= norm(u_new)

        if norm(u_new - u_old) < tol:               # 収束判定
            break
        u_old = u_new

    return u_new

# --- RMS誤差を計算 ---
def rms_error(u_true, u_est_list):
    P = np.eye(len(u_true)) - np.outer(u_true, u_true)  # 射影行列
    norms = [norm(P @ u_est)**2 for u_est in u_est_list]
    return np.sqrt(np.mean(norms))

# --- 真のパラメータベクトルを計算（ノイズなし最小二乗） ---
u_true = estimate_least_squares(X_true)

# --- 各σに対して、誤差を計算 ---
rms_ls_list = []  # 最小二乗法のRMS誤差
rms_ml_list = []  # 最尤推定法のRMS誤差
rms_kcr_list = []

for sigma in sigma_values:
    u_estimates_ls = []
    u_estimates_ml = []

    for _ in range(num_trials):
        np.random.seed()

        noise = np.random.normal(0, sigma, X_true.shape)
        X_noisy = X_true + noise

        u_ls = estimate_least_squares(X_noisy)
        if np.dot(u_ls, u_true) < 0:
            u_ls = -u_ls
        u_estimates_ls.append(u_ls)

        u_ml = estimate_maximum_likelihood_iterative(X_noisy, sigma**2)
        if np.dot(u_ml, u_true) < 0:
            u_ml = -u_ml
        u_estimates_ml.append(u_ml)

    rms_ls = rms_error(u_true, u_estimates_ls)
    rms_ml = rms_error(u_true, u_estimates_ml)
    rms_kcr = kcr_lower_bound(X_true, u_true, sigma**2)  # ← KCR下界の追加

    rms_ls_list.append(rms_ls)
    rms_ml_list.append(rms_ml)
    rms_kcr_list.append(rms_kcr)

    print(f"σ={sigma:.2f}, LS RMS={rms_ls:.6f}, ML RMS={rms_ml:.6f}, KCR={rms_kcr:.6f}")

# --- 結果をグラフに描画 ---
plt.figure(figsize=(8, 5))
plt.plot(sigma_values, np.array(rms_ls_list)*1e4, marker='o', label='LS')
plt.plot(sigma_values, np.array(rms_ml_list)*1e4, marker='x', label='ML')
plt.plot(sigma_values, np.array(rms_kcr_list)*1e4, linestyle='--', color='black', label='KCR')
plt.xlabel("σ")
plt.ylabel("RMS × $10^{-4}$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()