import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, norm

# 楕円の点を生成（u_true からでなく単に形状を定義）
def generate_ellipse_points(a, b, center, num_points=100):
    t = np.linspace(0, 2*np.pi, num_points)
    x = a * np.cos(t) + center[0]
    y = b * np.sin(t) + center[1]
    return x, y

def rms_error_projected(u_true, u_est_list):
    P = np.eye(len(u_true)) - np.outer(u_true, u_true)  # 射影行列
    norms = [norm(P @ u_est)**2 for u_est in u_est_list]
    return np.sqrt(np.mean(norms))

# 特徴ベクトル ξ(x, y)
def make_xi(x, y):
    return np.vstack([x**2, 2*x*y, y**2, 2*x, 2*y, np.ones_like(x)]).T

# 最小二乗法（LSM）
def estimate_lsm(x, y):
    X = make_xi(x, y)
    M = np.zeros((6, 6))
    for xi in X:
        xi = xi.reshape(6, 1)
        M += xi @ xi.T
    eigvals, eigvecs = eig(M)
    u = eigvecs[:, np.argmin(eigvals)]
    return u / norm(u)

# 最尤推定法（MSE）
def estimate_mse(x, y, tol=1e-6, max_iter=100):
    X = make_xi(x, y)
    u = np.ones(6)
    u = u / norm(u)
    epsilon = 1e-12

    for _ in range(max_iter):
        M = np.zeros((6, 6))
        L = np.zeros((6, 6))
        for xi in X:
            xi = xi.reshape(6, 1)
            num = float((xi.T @ u) ** 2)
            denom = float(u.T @ xi @ xi.T @ u + epsilon)
            M += xi @ xi.T
            L += (num / denom) * (xi @ xi.T)

        try:
            eigvals, eigvecs = eig(M - L)
            u_new = eigvecs[:, np.argmin(np.abs(eigvals))]
            u_new = u_new / norm(u_new)
        except np.linalg.LinAlgError:
            break

        if np.linalg.norm(u_new - u) < tol:
            break
        u = u_new

    return u

# 点が楕円からどれだけズレているか（目的関数残差）
def ellipse_residual_error(x, y, u):
    X = make_xi(x, y)
    residuals = X @ u
    return np.sqrt(np.mean(residuals**2))

# メイン関数
def main():
    a, b = 300, 200
    center = (0, 0)
    x_true, y_true = generate_ellipse_points(a, b, center)

    # 理想的な楕円式（x^2/90000 + y^2/40000 = 1）
    # → 4e-5 x^2 + 2.5e-5 y^2 - 1 = 0
    u_true = np.array([4e-5, 0, 2.5e-5, 0, 0, -1])

    sigma_values = np.arange(0.1, 2.1, 0.1)
    rms_lsm = []
    rms_mse = []

    trials = 1000  # 課題条件：1000回試行

    for sigma in sigma_values:
        errors_lsm = []
        errors_mse = []
        for _ in range(trials):
            x_noisy = x_true + np.random.normal(0, sigma, size=x_true.shape)
            y_noisy = y_true + np.random.normal(0, sigma, size=y_true.shape)

            u_lsm = estimate_lsm(x_noisy, y_noisy)
            u_mse = estimate_mse(x_noisy, y_noisy)

            errors_lsm.append(ellipse_residual_error(x_noisy, y_noisy, u_lsm))
            errors_mse.append(ellipse_residual_error(x_noisy, y_noisy, u_mse))

        rms_ls = rms_error_projected(u_true, u_estimates_ls)
        rms_ml = rms_error_projected(u_true, u_estimates_ml)
        print(f"σ={sigma:.1f}, LSM_RMS={rms_lsm[-1]:.5f}, MSE_RMS={rms_mse[-1]:.5f}")

    # グラフ描画
    plt.figure(figsize=(10, 6))
    plt.plot(sigma_values, rms_lsm, label="最小二乗法 (LSM)", marker='o')
    plt.plot(sigma_values, rms_mse, label="最尤推定法 (MSE)", marker='x')
    plt.xlabel("σ（ノイズの標準偏差）")
    plt.ylabel("RMS残差（点が楕円式からずれる量）")
    plt.title("楕円パラメータ推定精度（LSM vs MSE）")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
