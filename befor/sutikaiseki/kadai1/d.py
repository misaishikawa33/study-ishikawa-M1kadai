import numpy as np
import matplotlib.pyplot as plt

# --- パラメータ定義 ---
a = 300  # 楕円の長半径
b = 200  # 楕円の短半径
N = 100  # 点の数
sigma_max = 2.0
sigma_values = np.arange(0.1, sigma_max + 0.05, 0.1)
num_trials = 1000  # ノイズ付きデータの生成回数

# --- 楕円上の真の点列を生成 ---
i_vals = np.arange(N)
theta = -np.pi / 4 + (11 * np.pi) / (12 * N) * i_vals
x_true = a * np.cos(theta)
y_true = b * np.sin(theta)
X_true = np.vstack((x_true, y_true)).T

# --- 特徴ベクトルの作成（スケール調整なし） ---
def phi(x, y):
    return np.array([
        x**2,
        2*x*y,
        y**2,
        2*x,
        2*y,
        1.0
    ])

# --- 最小二乗法による推定 ---
def estimate_least_squares(X):
    Phi = np.array([phi(x, y) for x, y in X])
    M = (Phi.T @ Phi) / len(X)
    _, _, Vh = np.linalg.svd(M)
    u_est = Vh[-1]
    u_est /= np.linalg.norm(u_est)
    return u_est

# --- RMS誤差計算 ---
def rms_error(u_true, u_est_list):
    errors = [(np.eye(6) - np.outer(u_true, u_true)) @ u_est for u_est in u_est_list]
    norms = [np.linalg.norm(err)**2 for err in errors]
    return np.sqrt(np.mean(norms))

# --- 真のパラメータベクトルを計算（ノイズなしデータから推定） ---
u_true = estimate_least_squares(X_true)

# --- 各σで試行しRMSを記録 ---
rms_list = []

for sigma in sigma_values:
    u_estimates = []
    for _ in range(num_trials):
        noise = np.random.normal(0, sigma, X_true.shape)
        X_noisy = X_true + noise
        u_est = estimate_least_squares(X_noisy)

        # 推定ベクトルの向きを真値に揃える（符号合わせ）
        if np.dot(u_est, u_true) < 0:
            u_est = -u_est

        u_estimates.append(u_est)

    rms = rms_error(u_true, u_estimates)
    rms_list.append(rms)
    print(f"σ={sigma:.2f}, RMS={rms:.6f}")

# --- グラフ描画 ---
plt.figure(figsize=(8, 5))
plt.plot(sigma_values, np.array(rms_list) * 1e4, marker='o')  # 10^4倍して表示
plt.xlabel("σ (ノイズの標準偏差)")
plt.ylabel("RMS誤差 × $10^{-4}$")
plt.title("ノイズレベルに対するRMS誤差（スケール調整なし）")
plt.grid(True)
plt.tight_layout()
plt.show()
