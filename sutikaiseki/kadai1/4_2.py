import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, norm

# 楕円の点を生成
def generate_ellipse_points(a, b, center, num_points=100):
    t = np.linspace(0, 2*np.pi, num_points)
    x = a * np.cos(t) + center[0]
    y = b * np.sin(t) + center[1]
    return x, y

# 特徴ベクトル xi を生成
def make_xi(x, y):
    return np.vstack([x**2, 2*x*y, y**2, 2*x, 2*y, np.ones_like(x)]).T

# 最小二乗法でパラメータを推定
def estimate_ols(x, y):
    X = make_xi(x, y)
    M = np.zeros((6, 6))
    for xi in X:
        M += np.outer(xi, xi)
    eigvals, eigvecs = eig(M)
    u = eigvecs[:, np.argmin(eigvals)]
    return u / norm(u)

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
            M += xi @ xi.T
            L += (xiTxiu / (float((u.T @ xi @ xi.T @ u)) + epsilon)) * (xi @ xi.T)

        try:
            eigvals, eigvecs = eig(M - L)
            min_idx = np.argmin(np.abs(eigvals))
            new_u = eigvecs[:, min_idx]
            new_u = new_u / norm(new_u)
        except np.linalg.LinAlgError:
            break  # 固有値計算が失敗したら中止

        if np.linalg.norm(u - new_u) < tol:
            break
        u = new_u

    return u

# RMS誤差を計算
def rms_error(u_est, u_true):
    diff = u_est - u_true * np.sign(np.dot(u_est, u_true))
    return np.sqrt(np.mean(diff**2))

# メイン関数
def main():
    # 真の楕円パラメータ
    a, b = 300, 200
    center = (0, 0)
    x_true, y_true = generate_ellipse_points(a, b, center)

    # 真のパラメータを計算（理想データでOLSを使う）
    u_true = estimate_ols(x_true, y_true)

    # σの範囲
    sigma_values = np.arange(0.1, 2.1, 0.1)
    rms_ols = []
    rms_mle = []

    trials = 1000
    for sigma in sigma_values:
        errors_ols = []
        errors_mle = []
        for _ in range(trials):
            x_noisy = x_true + np.random.normal(0, sigma, size=x_true.shape)
            y_noisy = y_true + np.random.normal(0, sigma, size=y_true.shape)

            u_ols = estimate_ols(x_noisy, y_noisy)
            u_mle = estimate_mle(x_noisy, y_noisy)

            errors_ols.append(rms_error(u_ols, u_true))
            errors_mle.append(rms_error(u_mle, u_true))

        rms_ols.append(np.mean(errors_ols))
        rms_mle.append(np.mean(errors_mle))
        print(f"σ={sigma:.1f}, OLS_RMS={rms_ols[-1]:.4f}, MLE_RMS={rms_mle[-1]:.4f}")

    # グラフ描画
    plt.figure(figsize=(10, 6))
    plt.plot(sigma_values, rms_ols, label="最小二乗法 (OLS)", marker='o')
    plt.plot(sigma_values, rms_mle, label="最尤推定法 (MLE)", marker='x')
    plt.xlabel("σ（ノイズの標準偏差）")
    plt.ylabel("RMS誤差")
    plt.title("ノイズσに対する楕円パラメータ推定誤差")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
