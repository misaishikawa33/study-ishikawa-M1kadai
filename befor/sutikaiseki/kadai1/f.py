import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, norm

# 楕円上の点を生成
def generate_ellipse_points(a, b, center, num_points=100):
    t = np.linspace(0, 2*np.pi, num_points)
    x = a * np.cos(t) + center[0]
    y = b * np.sin(t) + center[1]
    return x, y

# 特徴ベクトル ξ(x) を生成
def make_xi(x, y):
    return np.vstack([x**2, 2*x*y, y**2, 2*x, 2*y, np.ones_like(x)]).T

# 最小二乗法による推定
def estimate_ols(x, y):
    X = make_xi(x, y)
    M = np.zeros((6, 6))
    for xi in X:
        M += np.outer(xi, xi)
    eigvals, eigvecs = eig(M)
    u = eigvecs[:, np.argmin(eigvals)]
    return u / norm(u)

# 最尤推定法（簡略版）
def estimate_mle(x, y, tol=1e-6, max_iter=100):
    X = make_xi(x, y)
    u = np.ones(6)
    u = u / norm(u)
    epsilon = 1e-12

    for _ in range(max_iter):
        M = np.zeros((6, 6))
        L = np.zeros((6, 6))
        for xi in X:
            xi = xi.reshape(-1, 1)
            xiTxiu = float((xi.T @ u) ** 2)
            denom = float((u.T @ xi @ xi.T @ u)) + epsilon
            M += xi @ xi.T
            L += (xiTxiu / denom) * (xi @ xi.T)

        try:
            eigvals, eigvecs = eig(M - L)
            min_idx = np.argmin(np.abs(eigvals))
            new_u = eigvecs[:, min_idx]
            new_u = new_u / norm(new_u)
        except np.linalg.LinAlgError:
            break

        if np.linalg.norm(u - new_u) < tol:
            break
        u = new_u

    return u

# RMS誤差の計算（課題の定義に基づく）
def rms_error(u_est, u_true):
    Pu = np.eye(len(u_true)) - np.outer(u_true, u_true)
    residual = Pu @ u_est
    return norm(residual)

# メイン関数
def main():
    a, b = 300, 200
    center = (0, 0)
    x_true, y_true = generate_ellipse_points(a, b, center)
    u_true = estimate_ols(x_true, y_true)

    sigma_values = np.arange(0.1, 2.1, 0.1)
    rms_ols = []
    rms_mle = []
    trials = 1000

    for sigma in sigma_values:
        err_ols, err_mle = 0, 0

        for _ in range(trials):
            x_noisy = x_true + np.random.normal(0, sigma, size=x_true.shape)
            y_noisy = y_true + np.random.normal(0, sigma, size=y_true.shape)

            u_ols = estimate_ols(x_noisy, y_noisy)
            u_mle = estimate_mle(x_noisy, y_noisy)

            err_ols += rms_error(u_ols, u_true) ** 2
            err_mle += rms_error(u_mle, u_true) ** 2

        rms_ols.append(np.sqrt(err_ols / trials))
        rms_mle.append(np.sqrt(err_mle / trials))
        print(f"σ={sigma:.1f}, OLS_RMS={rms_ols[-1]:.5f}, MLE_RMS={rms_mle[-1]:.5f}")

    # グラフ描画
    plt.figure(figsize=(10, 6))
    plt.plot(sigma_values, rms_ols, label="最小二乗法 (OLS)", marker='o', color='red')
    plt.plot(sigma_values, rms_mle, label="最尤推定法 (MLE)", marker='x', color='blue')
    plt.xlabel("σ（ノイズの標準偏差）")
    plt.ylabel("RMS誤差")
    plt.title("σ に対する RMS 誤差の比較")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("rms_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
