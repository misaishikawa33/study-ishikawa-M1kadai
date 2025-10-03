#課題5

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, eigh

# --- パラメータ定義 ---
a = 300             # 楕円の長半径
b = 200             # 楕円の短半径
N = 100              # 推定・誤差評価用の点の数
N_kcr = 100        # KCR用の点の数
sigma_max = 3.0     # ノイズの最大標準偏差
sigma_values = np.arange(0.1, sigma_max + 0.05, 0.1)
num_trials = 1000   # 各σに対しての試行回数

# --- 楕円上の真の点列（推定・評価用）を生成 ---
i_vals = np.arange(N)
theta = -np.pi / 4 + (11 * np.pi) / (12 * N) * i_vals
x_true = a * np.cos(theta)
y_true = b * np.sin(theta)
X_true = np.vstack((x_true, y_true)).T

# --- KCR用の楕円点列を別に生成 ---
i_vals_kcr = np.arange(N_kcr)
theta_kcr = -np.pi / 4 + (11 * np.pi) / (12 * N_kcr) * i_vals_kcr
x_kcr = a * np.cos(theta_kcr)
y_kcr = b * np.sin(theta_kcr)
X_kcr = np.vstack((x_kcr, y_kcr)).T

# --- φ(x,y)：6次元特徴ベクトル ---
def phi(x, y):
    return np.array([x**2, 2*x*y, y**2, 2*x, 2*y, 1.0])

# --- φに対するヤコビアンから共分散を計算 ---
def calc_V_xi(x, y, sigma2):
    Sigma_xy = sigma2 * np.eye(2)
    J = np.array([
        [2*x,     0],
        [2*y, 2*x],
        [0,   2*y],
        [2,     0],
        [0,     2],
        [0,     0]
    ])
    return J @ Sigma_xy @ J.T

# --- 最小二乗法 ---
def estimate_least_squares(X):
    Phi = np.array([phi(x, y) for x, y in X])
    M = (Phi.T @ Phi) / len(X)
    _, _, Vh = np.linalg.svd(M)
    u_est = Vh[-1]
    u_est /= norm(u_est)
    return u_est

# --- 最尤推定法（反復） ---
def estimate_maximum_likelihood_iterative(X, sigma2, tol=1e-6, max_iter=100):
    xi_list = np.array([phi(x,y) for x,y in X])
    V_list = np.array([calc_V_xi(x, y, sigma2) for x, y in X])
    M_init = np.sum([np.outer(xi, xi) for xi in xi_list], axis=0)
    eigvals, eigvecs = eigh(M_init)
    u_old = eigvecs[:, 0]
    u_old /= norm(u_old)

    for _ in range(max_iter):
        M = np.zeros((6,6))
        L = np.zeros((6,6))
        for xi, V in zip(xi_list, V_list):
            denom = u_old @ V @ u_old
            denom = max(denom, 1e-12)
            M += np.outer(xi, xi) / denom
            L += ((u_old @ xi)**2) * V / (denom**2)
        A = M - L
        eigvals, eigvecs = eigh(A)
        u_new = eigvecs[:, 0]
        u_new /= norm(u_new)
        if norm(u_new - u_old) < tol:
            break
        u_old = u_new
    return u_new

# --- RMS誤差（射影誤差） ---
def rms_error(u_true, u_est_list):
    P = np.eye(6) - np.outer(u_true, u_true)
    errors = [(P @ u_est) for u_est in u_est_list]
    norms = [norm(e)**2 for e in errors]
    return np.sqrt(np.mean(norms))

# --- KCR下界のRMS ---
def kcr_lower_bound(X, u_true, sigma2):
    M = np.zeros((6,6))
    for x, y in X:
        xi = phi(x, y)
        V = calc_V_xi(x, y, sigma2)
        denom = max(u_true @ V @ u_true, 1e-12)
        M += np.outer(xi, xi) / denom

    eigvals = np.linalg.eigvalsh(M)
    eigvals = eigvals[eigvals > 1e-6]  # カットオフを1e-6に戻して安定化
    if len(eigvals) < 1:
        return 1e10
    D_kcr = np.sqrt(np.sum(1.0 / eigvals))
    return max(D_kcr, 1e-10)

# --- 真のパラメータ ---
u_true = estimate_least_squares(X_true)
u_true_kcr = estimate_least_squares(X_kcr)  # KCR専用に再推定

# --- σごとにRMSを計算 ---
rms_ls_list = []
rms_ml_list = []
rms_kcr_list = []

for sigma in sigma_values:
    u_estimates_ls = []
    u_estimates_ml = []

    for _ in range(num_trials):
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

    # ← KCRだけ X_kcr, u_true_kcr を使って安定化！
    rms_kcr = kcr_lower_bound(X_kcr, u_true_kcr, sigma**2)

    rms_ls_list.append(rms_ls)
    rms_ml_list.append(rms_ml)
    rms_kcr_list.append(rms_kcr)

    print(f"σ={sigma:.2f}, LS RMS={rms_ls:.6f}, ML RMS={rms_ml:.6f}, KCR={rms_kcr:.6f}")

# --- グラフ描画 ---
plt.figure(figsize=(8, 5))
plt.plot(sigma_values, np.array(rms_ls_list)*1e4, marker='o', label='LSM', color='red')
plt.plot(sigma_values, np.array(rms_ml_list)*1e4, marker='x', label='MSE', color='blue')
plt.plot(sigma_values, np.array(rms_kcr_list)*1e4, linestyle='--', color='green', label='KCR')
plt.xlabel("standard deviation")
plt.ylabel("RMS-error ($\\times 10^{-4}$)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.figtext(0.5, -0.05, '図4　最小2乗法と最尤推定法のRMS誤差とKCR下界', ha='center', fontsize=12)
plt.show()
