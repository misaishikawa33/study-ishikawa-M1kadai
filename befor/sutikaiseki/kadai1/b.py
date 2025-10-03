import numpy as np
from numpy.linalg import eig, norm
import matplotlib.pyplot as plt

# 真の楕円パラメータ（中心: 原点、傾き0、a=300, b=200）
a, b = 300, 200
true_params = np.array([1 / a**2, 0, 1 / b**2, 0, 0, -1])  # A, B, C, D, E, F

# 点を楕円上に生成
def generate_ellipse_points(N=100):
    theta = -np.pi / 4 + (11 * np.pi) / (12 * N) * np.arange(N)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    return x, y

# 特徴ベクトル ξ(x, y)
def make_xi(x, y):
    return np.vstack([x**2, 2*x*y, y**2, 2*x, 2*y, np.ones_like(x)]).T

# 最小二乗法
def estimate_lsm(x, y):
    X = make_xi(x, y)
    M = np.zeros((6, 6))
    for xi in X:
        xi = xi.reshape(6, 1)
        M += xi @ xi.T
    eigvals, eigvecs = eig(M)
    u = eigvecs[:, np.argmin(eigvals)]
    return u / norm(u)

# 最尤推定法
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
            dot = (xi.T @ u).item()
            num = dot ** 2
            denom = (u.T @ xi @ xi.T @ u).item() + epsilon
            M += (xi @ xi.T).real
            L += ((num / denom) * (xi @ xi.T)).real
        try:
            eigvals, eigvecs = eig(M - L)
            u_new = eigvecs[:, np.argmin(np.abs(eigvals))].real
            u_new = u_new / norm(u_new)
        except np.linalg.LinAlgError:
            break
        if np.linalg.norm(u_new - u) < tol:
            break
        u = u_new
    return u

# 符号の一致を無視した角度誤差（ラジアン）
def angle_error(params, true_params):
    # スケーリングを無視して正規化
    p = params / norm(params)
    t = true_params / norm(true_params)

    # コサイン類似度の絶対値をとることで、±方向どちらでもOKに
    cos_sim = np.clip(np.dot(p, t), -1.0, 1.0)
    cos_sim = abs(cos_sim)  # 方向の逆転を許容

    angle = np.arccos(cos_sim)  # ラジアンで返す
    return angle

# メイン処理
def main():
    np.random.seed(42)  # 再現性のため
    x_true, y_true = generate_ellipse_points(N=100)
    sigma_list = np.arange(0.1, 2.1, 0.1)
    num_trials = 1000

    lsm_errors = []
    mse_errors = []

    for sigma in sigma_list:
        err_lsm_list = []
        err_mse_list = []
        for _ in range(num_trials):
            x_noisy = x_true + np.random.normal(0, sigma, size=x_true.shape)
            y_noisy = y_true + np.random.normal(0, sigma, size=y_true.shape)

            u_lsm = estimate_lsm(x_noisy, y_noisy)
            u_mse = estimate_mse(x_noisy, y_noisy)

            err_lsm_list.append(angle_error(u_lsm, true_params))
            err_mse_list.append(angle_error(u_mse, true_params))

        avg_lsm = np.mean(err_lsm_list)
        avg_mse = np.mean(err_mse_list)
        lsm_errors.append(avg_lsm)
        mse_errors.append(avg_mse)

        print(f"σ={sigma:.1f}, LSM={np.degrees(avg_lsm):.2f}°, MSE={np.degrees(avg_mse):.2f}°")

    # グラフ表示
    plt.figure(figsize=(10, 6))
    plt.plot(sigma_list, np.degrees(lsm_errors), label="LSM（最小二乗法）", marker="o")
    plt.plot(sigma_list, np.degrees(mse_errors), label="MSE（最尤推定法）", marker="x")
    plt.xlabel("σ（ノイズ標準偏差）")
    plt.ylabel("角度誤差（度）")
    plt.title("楕円パラメータの推定誤差（角度）")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
