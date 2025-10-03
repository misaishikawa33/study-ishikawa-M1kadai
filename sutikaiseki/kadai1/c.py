import numpy as np
import matplotlib.pyplot as plt

# 真値ベクトル u（正規化された楕円パラメータ）
u_true = np.array([1/300**2, 0, 1/200**2, 0, 0, -1], dtype=float)
u_true = u_true / np.linalg.norm(u_true)

# ξ(x, y) を返す関数
def xi(x, y):
    return np.array([x**2, 2*x*y, y**2, 2*x, 2*y, 1], dtype=float).reshape(6, 1)

# V[ξ] を生成（ここでは単位行列 × σ^2 とする）
def V_xi(sigma2):
    return sigma2 * np.identity(6)

# 最小二乗法：最小固有値に対応する固有ベクトルを返す
def estimate_ls(x, y):
    M = np.zeros((6, 6))
    for xi_vec in map(lambda xy: xi(xy[0], xy[1]), zip(x, y)):
        M += xi_vec @ xi_vec.T
    eigvals, eigvecs = np.linalg.eigh(M)
    u = eigvecs[:, np.argmin(eigvals)]
    return u / np.linalg.norm(u)

# 最尤推定法（反復）
def estimate_mle(x, y, tol=1e-6, max_iter=100):
    u = np.ones(6)
    u = u / np.linalg.norm(u)
    sigma2 = np.var(x) + np.var(y)  # 初期共分散推定（簡易）

    for _ in range(max_iter):
        M = np.zeros((6, 6))
        L = np.zeros((6, 6))
        for xi_vec in map(lambda xy: xi(xy[0], xy[1]), zip(x, y)):
            V = V_xi(sigma2)
            denom = float(u.T @ V @ u)
            M += (xi_vec @ xi_vec.T) / denom
            num = float((u_true @ xi_vec.flatten())**2)
            L += num * V / (denom**2)
        u_new = np.linalg.solve(M - L + 1e-8 * np.eye(6), np.zeros(6) + 1e-8)  # 解がゼロのときは微小値で安定化
        u_new = u_new / np.linalg.norm(u_new)
        if np.linalg.norm(u_new - u) < tol:
            break
        u = u_new
    return u

# RMS誤差を計算
def compute_rms(us, u_true):
    P = np.identity(6) - np.outer(u_true, u_true)
    deltas = np.array([P @ u_i for u_i in us])
    return np.sqrt(np.mean(np.sum(deltas**2, axis=1)))

# メインループ
sigmas = np.arange(0.1, 2.1, 0.1)
num_trials = 1000
rms_ls = []
rms_mle = []

for sigma in sigmas:
    us_ls = []
    us_mle = []
    for _ in range(num_trials):
        theta = -np.pi / 4 + (11 * np.pi) / (12 * 100) * np.arange(100)
        x = 300 * np.cos(theta)
        y = 200 * np.sin(theta)
        x += np.random.normal(0, sigma, size=100)
        y += np.random.normal(0, sigma, size=100)
        u_ls = estimate_ls(x, y)
        u_mle = estimate_mle(x, y)
        us_ls.append(u_ls)
        us_mle.append(u_mle)
    rms_ls.append(compute_rms(us_ls, u_true))
    rms_mle.append(compute_rms(us_mle, u_true))

# グラフ描画
plt.plot(sigmas, rms_ls, label="Least Squares", marker='o')
plt.plot(sigmas, rms_mle, label="Maximum Likelihood", marker='x')
plt.xlabel("Noise standard deviation σ")
plt.ylabel("RMS error")
plt.title("RMS Error vs σ for LS and MLE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
